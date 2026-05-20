"""Shared building blocks for BRep neural network models.

This module consolidates reusable architectural primitives (residual blocks,
positional encodings, attention layers, token encoders/decoders, and intersection
modules) that are shared across the autoencoder and diffusion pipelines.
"""

import copy
import math

import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def inv_sigmoid(x: Tensor) -> Tensor:
    """Inverse sigmoid (logit) function.

    Maps values in (0, 1) back to the real line via log(x / (1 - x)).
    """
    return torch.log(x / (1 - x))


def sincos_embedding(input: Tensor, dim: int, max_period: int = 10000) -> Tensor:
    """Create sinusoidal positional/timestep embeddings.

    Encodes N-D indices into a ``dim``-dimensional sinusoidal representation,
    following the scheme from *Attention Is All You Need* and DDPM.

    Args:
        input: An N-D tensor of indices (may be fractional).
        dim: The dimension of the output embedding.
        max_period: Controls the minimum frequency of the embeddings.

    Returns:
        A tensor of shape ``[*input.shape, dim]`` containing positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=input.dtype, device=input.device)
        / half
    )
    for _ in range(len(input.size())):
        freqs = freqs[None]
    args = input.unsqueeze(-1).float() * freqs
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
        )
    return embedding


def make_uv_grid(h: int, w: int, device, dtype=torch.float32) -> Tensor:
    """Create a flattened 2-D UV coordinate grid in [-1, 1].

    Args:
        h: Number of grid points along U.
        w: Number of grid points along V.
        device: Torch device for the output tensor.
        dtype: Torch dtype for the output tensor.

    Returns:
        Tensor of shape ``(h*w, 2)`` with (u, v) coordinates.
    """
    u = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    v = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    uu, vv = torch.meshgrid(u, v, indexing="ij")
    return torch.stack((uu, vv), dim=-1).reshape(h * w, 2)


def make_t_grid(n: int, device, dtype=torch.float32) -> Tensor:
    """Create a 1-D parameter grid in [-1, 1].

    Args:
        n: Number of grid points.
        device: Torch device for the output tensor.
        dtype: Torch dtype for the output tensor.

    Returns:
        Tensor of shape ``(n, 1)``.
    """
    return torch.linspace(-1.0, 1.0, n, device=device, dtype=dtype)[:, None]


# ---------------------------------------------------------------------------
# Residual blocks
# ---------------------------------------------------------------------------


class ResLinear(nn.Module):
    """Residual linear block with optional normalization.

    Applies a linear projection with a skip connection, followed by
    normalization and LeakyReLU activation.

    Args:
        dim_in: Input (and residual) feature dimension.
        dim_out: Output feature dimension. Must equal ``dim_in`` for the
            residual addition to be valid.
        norm: Normalization type -- ``"none"``, ``"layer"``, or ``"batch"``.
    """

    def __init__(self, dim_in: int, dim_out: int, norm: str = "none"):
        super(ResLinear, self).__init__()
        self.conv = nn.Linear(dim_in, dim_out)
        self.act = nn.LeakyReLU()
        if norm == "none":
            self.norm = nn.Identity()
        elif norm == "layer":
            self.norm = nn.LayerNorm(dim_out)
        elif norm == "batch":
            self.norm = nn.BatchNorm1d(dim_out)
        else:
            raise ValueError("Norm type not supported")

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.conv(x)
        x = self.norm(x)
        return self.act(x)


class ResBlock1D(nn.Module):
    """1-D convolutional residual block with optional normalization.

    Args:
        dim_in: Number of input channels.
        dim_out: Number of output channels (must equal ``dim_in`` for skip).
        ks: Kernel size.
        st: Stride.
        pa: Padding.
        norm: Normalization type -- ``"none"``, ``"layer"``, or ``"batch"``.
    """

    def __init__(self, dim_in: int, dim_out: int, ks: int = 3, st: int = 1, pa: int = 1, norm: str = "none"):
        super(ResBlock1D, self).__init__()
        self.conv = nn.Conv1d(dim_in, dim_out, kernel_size=ks, stride=st, padding=pa)
        self.act = nn.ReLU()
        if norm == "none":
            self.norm = nn.Identity()
        elif norm == "layer":
            self.norm = nn.Sequential(
                Rearrange('... c h -> ... h c'),
                nn.LayerNorm(dim_out),
                Rearrange('... h c -> ... c h'),
            )
        elif norm == "batch":
            self.norm = nn.BatchNorm1d(dim_out)
        else:
            raise ValueError("Norm type not supported")

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.conv(x)
        x = self.norm(x)
        return self.act(x)


class ResBlock2D(nn.Module):
    """2-D convolutional residual block with optional normalization.

    Args:
        dim_in: Number of input channels.
        dim_out: Number of output channels (must equal ``dim_in`` for skip).
        ks: Kernel size.
        st: Stride.
        pa: Padding.
        norm: Normalization type -- ``"none"``, ``"layer"``, or ``"batch"``.
    """

    def __init__(self, dim_in: int, dim_out: int, ks: int = 3, st: int = 1, pa: int = 1, norm: str = "none"):
        super(ResBlock2D, self).__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=ks, stride=st, padding=pa)
        self.act = nn.ReLU()
        if norm == "none":
            self.norm = nn.Identity()
        elif norm == "layer":
            self.norm = nn.Sequential(
                Rearrange('... c h w -> ... h w c'),
                nn.LayerNorm(dim_out),
                Rearrange('... h w c -> ... c h w'),
            )
        elif norm == "batch":
            self.norm = nn.BatchNorm2d(dim_out)
        else:
            raise ValueError("Norm type not supported")

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.conv(x)
        x = self.norm(x)
        return self.act(x)


class ResBlockXD(nn.Module):
    """Generic N-dimensional residual block (N in {0, 1, 2, 3}).

    Implements a two-layer residual block where the convolution dimension is
    selected by the ``dim`` argument:
      - ``dim=0``: fully-connected (Linear)
      - ``dim=1``: Conv1d
      - ``dim=2``: Conv2d
      - ``dim=3``: Conv3d

    A learned downsampling shortcut is added when ``dim_in != dim_out``.

    Args:
        dim: Spatial dimensionality (0, 1, 2, or 3).
        dim_in: Number of input features/channels.
        dim_out: Number of output features/channels.
        kernel_size: Kernel size for convolutions.
        stride: Stride for convolutions.
        padding: Padding for convolutions.
        v_norm: Normalization type -- ``None``/``"none"`` or ``"layer"``.
        v_norm_shape: Shape argument passed to ``nn.LayerNorm`` when
            ``v_norm="layer"``.
    """

    def __init__(
        self,
        dim: int,
        dim_in: int,
        dim_out: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        v_norm=None,
        v_norm_shape=None,
    ):
        super(ResBlockXD, self).__init__()
        self.downsample = None
        if v_norm is None or v_norm == "none":
            norm = nn.Identity()
        elif v_norm == "layer":
            norm = nn.LayerNorm(v_norm_shape)

        if dim == 0:
            self.conv1 = nn.Linear(dim_in, dim_out)
            self.norm1 = copy.deepcopy(norm)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Linear(dim_out, dim_out)
            self.norm2 = copy.deepcopy(norm)
            if dim_in != dim_out:
                self.downsample = nn.Sequential(
                    nn.Linear(dim_in, dim_out),
                    copy.deepcopy(norm),
                )
        if dim == 1:
            self.conv1 = nn.Conv1d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding)
            self.norm1 = copy.deepcopy(norm)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv1d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding)
            self.norm2 = copy.deepcopy(norm)
            if dim_in != dim_out:
                self.downsample = nn.Sequential(
                    nn.Conv1d(dim_in, dim_out, kernel_size=1, stride=1, bias=False),
                    copy.deepcopy(norm),
                )
        elif dim == 2:
            self.conv1 = nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding)
            self.norm1 = copy.deepcopy(norm)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding)
            self.norm2 = copy.deepcopy(norm)
            if dim_in != dim_out:
                self.downsample = nn.Sequential(
                    nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, bias=False),
                    copy.deepcopy(norm),
                )
        elif dim == 3:
            self.conv1 = nn.Conv3d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding)
            self.norm1 = copy.deepcopy(norm)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv3d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding)
            self.norm2 = copy.deepcopy(norm)
            if dim_in != dim_out:
                self.downsample = nn.Sequential(
                    nn.Conv3d(dim_in, dim_out, kernel_size=1, stride=1, bias=False),
                    copy.deepcopy(norm),
                )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------


class FourierPE(nn.Module):
    """Fourier positional encoding using multiple frequency bands.

    Maps each scalar coordinate to a higher-dimensional representation by
    concatenating sin/cos of geometrically-spaced frequencies.

    Args:
        in_dim: Dimensionality of the input coordinates.
        num_freqs: Number of frequency octaves.

    Attributes:
        out_dim: Total output dimensionality = ``in_dim + 2 * in_dim * num_freqs``.
    """

    def __init__(self, in_dim: int, num_freqs: int = 6):
        super().__init__()
        self.in_dim = in_dim
        self.num_freqs = num_freqs
        self.out_dim = in_dim + 2 * in_dim * num_freqs

    def forward(self, x: Tensor) -> Tensor:
        freqs = 2.0 ** torch.arange(self.num_freqs, device=x.device, dtype=x.dtype)
        xb = x[..., None, :] * freqs[:, None]
        xb = xb.flatten(-2)
        return torch.cat((x, torch.sin(torch.pi * xb), torch.cos(torch.pi * xb)), dim=-1)


# ---------------------------------------------------------------------------
# Attention blocks
# ---------------------------------------------------------------------------


class CrossAttnBlock(nn.Module):
    """Pre-norm cross-attention block with a feed-forward network.

    Performs multi-head cross-attention (query attends to memory) followed by
    a two-layer GELU MLP, both with residual connections and LayerNorm.

    Args:
        dim: Model/embedding dimension.
        heads: Number of attention heads.
        mlp_ratio: Hidden-dimension expansion factor for the FFN.
        dropout: Dropout probability.
    """

    def __init__(self, dim: int, heads: int = 8, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.q_norm = nn.LayerNorm(dim)
        self.mem_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, q: Tensor, mem: Tensor) -> Tensor:
        mem = self.mem_norm(mem)
        out, _ = self.attn(self.q_norm(q), mem, mem, need_weights=False)
        q = q + out
        return q + self.ffn(q)


class SelfAttnBlock(nn.Module):
    """Pre-norm self-attention block with a feed-forward network.

    Performs multi-head self-attention followed by a two-layer GELU MLP,
    both with residual connections and LayerNorm.

    Args:
        dim: Model/embedding dimension.
        heads: Number of attention heads.
        mlp_ratio: Hidden-dimension expansion factor for the FFN.
        dropout: Dropout probability.
    """

    def __init__(self, dim: int, heads: int = 8, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        x_norm = self.norm(x)
        out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + out
        return x + self.ffn(x)


# ---------------------------------------------------------------------------
# Token encoders
# ---------------------------------------------------------------------------


class SurfaceTokenEncoder(nn.Module):
    """Encode a surface patch (N, C, H, W) into a flat latent vector.

    The encoder flattens spatial positions, augments them with Fourier UV
    positional encoding, projects to a hidden space, cross-attends from
    learnable queries, refines via self-attention layers, and projects to
    the final latent dimensionality.

    Args:
        in_channels: Number of input feature channels per spatial point.
        out_dim: Total output latent dimension (split across ``num_tokens``).
        num_tokens: Number of learnable query tokens.
        hidden_dim: Internal transformer hidden dimension.
        depth: Number of self-attention refinement layers.
        heads: Number of attention heads.
        pe_freqs: Number of Fourier frequency octaves for UV encoding.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        out_dim: int,
        num_tokens: int = 4,
        hidden_dim: int = 128,
        depth: int = 4,
        heads: int = 8,
        pe_freqs: int = 6,
        dropout: float = 0.0,
    ):
        super().__init__()
        if out_dim % num_tokens != 0:
            raise ValueError(f"out_dim={out_dim} must be divisible by num_tokens={num_tokens}")
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.num_tokens = num_tokens
        self.token_dim = out_dim // num_tokens

        self.uv_pe = FourierPE(2, pe_freqs)
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels + self.uv_pe.out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.queries = nn.Parameter(torch.randn(1, num_tokens, hidden_dim) * 0.02)
        self.cross_attn = CrossAttnBlock(hidden_dim, heads=heads, dropout=dropout)
        self.self_attn = nn.ModuleList([
            SelfAttnBlock(hidden_dim, heads=heads, dropout=dropout) for _ in range(depth)
        ])
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.token_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        n, c, h, w = x.shape
        if c != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {c}")
        pts = rearrange(x, "n c h w -> n (h w) c")
        uv = make_uv_grid(h, w, x.device, x.dtype)
        uv = self.uv_pe(uv)[None].expand(n, -1, -1)
        mem = self.input_proj(torch.cat((pts, uv), dim=-1))

        tokens = self.queries.expand(n, -1, -1)
        tokens = self.cross_attn(tokens, mem)
        for block in self.self_attn:
            tokens = block(tokens)
        tokens = self.out_proj(tokens)
        return tokens.reshape(n, self.out_dim)


class CurveTokenEncoder(nn.Module):
    """Encode a curve signal (N, C, L) into a flat latent vector.

    Similar to :class:`SurfaceTokenEncoder` but operates on 1-D sequences.
    Uses Fourier positional encoding over the curve parameter *t*.

    Args:
        in_channels: Number of input feature channels per sample point.
        out_dim: Total output latent dimension (split across ``num_tokens``).
        num_tokens: Number of learnable query tokens.
        hidden_dim: Internal transformer hidden dimension.
        depth: Number of self-attention refinement layers.
        heads: Number of attention heads.
        pe_freqs: Number of Fourier frequency octaves for t encoding.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        out_dim: int,
        num_tokens: int = 4,
        hidden_dim: int = 128,
        depth: int = 4,
        heads: int = 8,
        pe_freqs: int = 6,
        dropout: float = 0.0,
    ):
        super().__init__()
        if out_dim % num_tokens != 0:
            raise ValueError(f"out_dim={out_dim} must be divisible by num_tokens={num_tokens}")
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.num_tokens = num_tokens
        self.token_dim = out_dim // num_tokens

        self.t_pe = FourierPE(1, pe_freqs)
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels + self.t_pe.out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.queries = nn.Parameter(torch.randn(1, num_tokens, hidden_dim) * 0.02)
        self.cross_attn = CrossAttnBlock(hidden_dim, heads=heads, dropout=dropout)
        self.self_attn = nn.ModuleList([
            SelfAttnBlock(hidden_dim, heads=heads, dropout=dropout) for _ in range(depth)
        ])
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.token_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        n, c, l = x.shape
        if c != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {c}")
        pts = rearrange(x, "n c l -> n l c")
        t = make_t_grid(l, x.device, x.dtype)
        t = self.t_pe(t)[None].expand(n, -1, -1)
        mem = self.input_proj(torch.cat((pts, t), dim=-1))

        tokens = self.queries.expand(n, -1, -1)
        tokens = self.cross_attn(tokens, mem)
        for block in self.self_attn:
            tokens = block(tokens)
        tokens = self.out_proj(tokens)
        return tokens.reshape(n, self.out_dim)


# ---------------------------------------------------------------------------
# Query decoders
# ---------------------------------------------------------------------------


class SurfaceQueryDecoder(nn.Module):
    """Decode a flat latent vector into a surface patch (N, H, W, C_out).

    The decoder splits the latent into tokens, refines them with self-attention,
    then cross-attends from Fourier-encoded UV query positions to produce
    per-point output features.

    Args:
        in_dim: Total input latent dimension (split across ``num_tokens``).
        out_channels: Number of output feature channels per spatial point.
        num_tokens: Number of latent tokens.
        hidden_dim: Internal transformer hidden dimension.
        depth: Number of self-attention refinement layers.
        heads: Number of attention heads.
        pe_freqs: Number of Fourier frequency octaves for UV encoding.
        out_h: Output grid height.
        out_w: Output grid width.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_dim: int,
        out_channels: int,
        num_tokens: int = 4,
        hidden_dim: int = 128,
        depth: int = 4,
        heads: int = 8,
        pe_freqs: int = 6,
        out_h: int = 16,
        out_w: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        if in_dim % num_tokens != 0:
            raise ValueError(f"in_dim={in_dim} must be divisible by num_tokens={num_tokens}")
        self.in_dim = in_dim
        self.out_channels = out_channels
        self.num_tokens = num_tokens
        self.token_dim = in_dim // num_tokens
        self.out_h = out_h
        self.out_w = out_w

        self.uv_pe = FourierPE(2, pe_freqs)
        self.token_proj = nn.Sequential(
            nn.Linear(self.token_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.latent_attn = nn.ModuleList([
            SelfAttnBlock(hidden_dim, heads=heads, dropout=dropout) for _ in range(depth)
        ])
        self.query_proj = nn.Sequential(
            nn.Linear(self.uv_pe.out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.query_attn = CrossAttnBlock(hidden_dim, heads=heads, dropout=dropout)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_channels),
        )

    def forward(self, z: Tensor) -> Tensor:
        n = z.shape[0]
        if n == 0:
            return z.new_empty((0, self.out_h, self.out_w, self.out_channels))
        tokens = z.reshape(n, self.num_tokens, self.token_dim)
        mem = self.token_proj(tokens)
        for block in self.latent_attn:
            mem = block(mem)

        uv = make_uv_grid(self.out_h, self.out_w, z.device, z.dtype)
        q = self.query_proj(self.uv_pe(uv))[None].expand(n, -1, -1)
        q = self.query_attn(q, mem)
        return self.out(q).reshape(n, self.out_h, self.out_w, -1)


class CurveQueryDecoder(nn.Module):
    """Decode a flat latent vector into a curve signal (N, L, C_out).

    Similar to :class:`SurfaceQueryDecoder` but produces a 1-D output
    sequence by querying at Fourier-encoded *t* positions.

    Args:
        in_dim: Total input latent dimension (split across ``num_tokens``).
        out_channels: Number of output feature channels per sample point.
        num_tokens: Number of latent tokens.
        hidden_dim: Internal transformer hidden dimension.
        depth: Number of self-attention refinement layers.
        heads: Number of attention heads.
        pe_freqs: Number of Fourier frequency octaves for t encoding.
        out_l: Output sequence length.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_dim: int,
        out_channels: int,
        num_tokens: int = 4,
        hidden_dim: int = 128,
        depth: int = 4,
        heads: int = 8,
        pe_freqs: int = 6,
        out_l: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        if in_dim % num_tokens != 0:
            raise ValueError(f"in_dim={in_dim} must be divisible by num_tokens={num_tokens}")
        self.in_dim = in_dim
        self.out_channels = out_channels
        self.num_tokens = num_tokens
        self.token_dim = in_dim // num_tokens
        self.out_l = out_l

        self.t_pe = FourierPE(1, pe_freqs)
        self.token_proj = nn.Sequential(
            nn.Linear(self.token_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.latent_attn = nn.ModuleList([
            SelfAttnBlock(hidden_dim, heads=heads, dropout=dropout) for _ in range(depth)
        ])
        self.query_proj = nn.Sequential(
            nn.Linear(self.t_pe.out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.query_attn = CrossAttnBlock(hidden_dim, heads=heads, dropout=dropout)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_channels),
        )

    def forward(self, z: Tensor) -> Tensor:
        n = z.shape[0]
        if n == 0:
            return z.new_empty((0, self.out_l, self.out_channels))
        tokens = z.reshape(n, self.num_tokens, self.token_dim)
        mem = self.token_proj(tokens)
        for block in self.latent_attn:
            mem = block(mem)

        t = make_t_grid(self.out_l, z.device, z.dtype)
        q = self.query_proj(self.t_pe(t))[None].expand(n, -1, -1)
        q = self.query_attn(q, mem)
        return self.out(q).reshape(n, self.out_l, -1)


# ---------------------------------------------------------------------------
# Intersection module
# ---------------------------------------------------------------------------


class AttnIntersection(nn.Module):
    """Attention-based intersection reasoning module.

    Uses a Transformer decoder to model the interaction between two input
    tokens (e.g., two adjacent surface/curve embeddings) via cross-attention,
    and outputs a fused representation.

    Args:
        dim_in: Input feature dimension per token.
        dim_latent: Internal transformer latent dimension.
        num_layers: Number of transformer decoder layers.
    """

    def __init__(
        self,
        dim_in: int,
        dim_latent: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers

        self.attn_proj_in = nn.Linear(dim_in, dim_latent)
        layer = nn.TransformerDecoderLayer(
            dim_latent, 8, dim_feedforward=2048, dropout=0.1,
            batch_first=True, norm_first=True)
        self.layers = ModuleList([copy.deepcopy(layer) for i in range(num_layers)])
        self.attn_proj_out = nn.Linear(dim_latent, dim_in * 2)

        self.pos_encoding = nn.Parameter(torch.randn(1, 2, dim_latent))

    def forward(self, src: Tensor) -> Tensor:
        output = self.attn_proj_in(src) + self.pos_encoding
        tgt = output[:, 0:1]
        mem = output[:, 1:2]

        for mod in self.layers:
            tgt = mod(tgt, mem)

        output = self.attn_proj_out(tgt)[:, 0]
        return output


# ---------------------------------------------------------------------------
# Backward-compatible aliases (legacy snake_case / old class names)
# ---------------------------------------------------------------------------

res_linear = ResLinear
res_block_1D = ResBlock1D
res_block_2D = ResBlock2D
res_block_xd = ResBlockXD
AttnIntersection3 = AttnIntersection
