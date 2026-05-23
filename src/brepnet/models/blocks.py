"""Shared building blocks for BRep neural network models.

This module consolidates reusable architectural primitives (residual blocks,
sinusoidal embeddings, and intersection modules) that are shared across the
autoencoder and diffusion pipelines.
"""

import copy
import math

import torch
from torch import nn, Tensor
from torch.nn import ModuleList

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
# Intersection module
# ---------------------------------------------------------------------------


class AttnIntersection(nn.Module):
    """Attention-based intersection reasoning module.

    Uses a Transformer decoder to model the interaction between two input
    tokens (e.g., two adjacent surface/curve embeddings) via cross-attention,
    and outputs a fused representation.

    Args:
        dim_in: Input feature dimension per token.
        hidden_dim: Internal transformer dimension.
        num_layers: Number of transformer decoder layers.
    """

    def __init__(
        self,
        dim_in: int,
        hidden_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers

        self.attn_proj_in = nn.Linear(dim_in, hidden_dim)
        layer = nn.TransformerDecoderLayer(
            hidden_dim, 8, dim_feedforward=2048, dropout=0.1,
            batch_first=True, norm_first=True)
        self.layers = ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.attn_proj_out = nn.Linear(hidden_dim, dim_in * 2)

        self.pos_encoding = nn.Parameter(torch.randn(1, 2, hidden_dim))

    def forward(self, src: Tensor) -> Tensor:
        output = self.attn_proj_in(src) + self.pos_encoding
        tgt = output[:, 0:1]
        mem = output[:, 1:2]

        for mod in self.layers:
            tgt = mod(tgt, mem)

        output = self.attn_proj_out(tgt)[:, 0]
        return output


class AttnIntersection3(AttnIntersection):
    """Checkpoint-compatible intersection block name."""
