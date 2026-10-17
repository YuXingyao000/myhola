"""Shared building blocks for BRep neural network models.

This module consolidates reusable architectural primitives (residual blocks,
sinusoidal embeddings, and intersection modules) that are shared across the
autoencoder and diffusion pipelines.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING
import copy
import math

import torch
from torch import nn, Tensor
from torch.nn import ModuleList
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import add_self_loops, remove_self_loops

from einops.layers.torch import Rearrange

if TYPE_CHECKING:
    from src.brepnet.models.vae import AutoEncoder_light


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
# Face geometry token encoder
# ---------------------------------------------------------------------------


class UVFaceEncoder(nn.Module):
    """Encode a 16 x 16 XYZ/normal grid without intermediate spatial pooling.

    This optional replacement for the face CNN keeps all 256 parameter-grid
    positions through its self-attention layers. A single learned query then
    reads the complete face into one vector. Faces are independent batch items;
    the existing downstream face attention/GAT performs cross-face fusion.

    The UV coordinates are normalized grid indices, not additional CAD samples
    or a claim that the samples have equal spacing in world coordinates.
    Input: ``[M, 6, 16, 16]``. Output: ``[M, out_channels]``.
    """

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 32,
        hidden_dim: int = 512,
        embedding_dim: int = 128,
        num_layers: int = 6,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if isinstance(in_channels, bool) or not isinstance(in_channels, int) or in_channels != 6:
            raise ValueError("UVFaceEncoder requires in_channels=6 (XYZ and normal)")
        dimensions = {
            "out_channels": out_channels,
            "hidden_dim": hidden_dim,
            "embedding_dim": embedding_dim,
            "num_layers": num_layers,
            "nhead": nhead,
            "dim_feedforward": dim_feedforward,
        }
        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if hidden_dim % nhead:
            raise ValueError("hidden_dim must be divisible by nhead")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
            raise ValueError("dropout must be a number in [0, 1)")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim

        self.geometry_embedding = nn.Sequential(
            nn.Linear(in_channels, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, hidden_dim),
        )
        self.uv_embedding = nn.Sequential(
            nn.Linear(2, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, hidden_dim),
        )
        coordinates = torch.arange(16, dtype=torch.float32) / 15
        u, v = torch.meshgrid(coordinates, coordinates, indexing="ij")
        # Deterministic for every checkpoint; follows flatten(2)'s i-major order.
        self.register_buffer(
            "uv_grid", torch.stack((u, v), dim=-1).reshape(1, 256, 2), persistent=False
        )
        # Instantiate each layer separately: no copied/shared initial parameters.
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.query = nn.Parameter(torch.empty(1, 1, hidden_dim))
        nn.init.normal_(self.query, std=0.02)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.readout_attention = nn.MultiheadAttention(
            hidden_dim, nhead, dropout=dropout, batch_first=True
        )
        self.readout_dropout = nn.Dropout(dropout)
        self.readout_ffn_norm = nn.LayerNorm(hidden_dim)
        self.readout_ffn = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dim),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, out_channels)

    def forward(self, face_points: Tensor) -> Tensor:
        if not isinstance(face_points, Tensor) or face_points.layout != torch.strided:
            raise ValueError("face_points must be a dense floating tensor [M, 6, 16, 16]")
        if (face_points.ndim != 4 or tuple(face_points.shape[1:]) != (6, 16, 16)
                or face_points.shape[0] == 0 or not face_points.is_floating_point()):
            raise ValueError("face_points must have floating shape [M, 6, 16, 16] with M > 0")

        # [M, 6, 16, 16] -> [M, 256, 6] -> [M, 256, hidden_dim].
        points = face_points.flatten(2).transpose(1, 2)
        tokens = self.geometry_embedding(points) + self.uv_embedding(self.uv_grid)
        for layer in self.layers:
            tokens = layer(tokens)
        memory = self.memory_norm(tokens)

        # Q: [M, 1, hidden_dim]; K/V: all [M, 256, hidden_dim] positions.
        query = self.query.expand(face_points.shape[0], -1, -1)
        readout, _ = self.readout_attention(
            self.query_norm(query), memory, memory, need_weights=False
        )
        summary = query + self.readout_dropout(readout)
        summary = summary + self.readout_ffn(self.readout_ffn_norm(summary))
        return self.output_projection(self.output_norm(summary)).squeeze(1)


# ---------------------------------------------------------------------------
# Encoder graph message values
# ---------------------------------------------------------------------------


class EdgeValueGATv2Conv(GATv2Conv):
    """Keep GATv2 attention and add an edge-conditioned message value.

    This experiment supports the encoder's dense, homogeneous, single-head
    graph. The original GATv2 output and parameter names are preserved. A new
    ``value_net`` reads ``[target_face, source_face, edge]`` and contributes an
    attention-weighted residual. Its final layer starts at zero, so importing
    the old GAT weights preserves the initial function exactly.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        *,
        edge_dim: int,
        value_hidden_dim: int = 64,
        **kwargs,
    ) -> None:
        if not isinstance(in_channels, int) or in_channels <= 0:
            raise ValueError("EdgeValueGATv2Conv requires a positive homogeneous in_channels")
        if not isinstance(out_channels, int) or out_channels <= 0:
            raise ValueError("out_channels must be a positive integer")
        if heads != 1:
            raise ValueError("EdgeValueGATv2Conv supports heads=1 only")
        if not isinstance(edge_dim, int) or edge_dim <= 0:
            raise ValueError("edge_dim must be a positive integer")
        if not isinstance(value_hidden_dim, int) or value_hidden_dim <= 0:
            raise ValueError("value_hidden_dim must be a positive integer")
        if kwargs.get("flow", "source_to_target") != "source_to_target":
            raise ValueError("EdgeValueGATv2Conv requires source_to_target flow")
        if kwargs.get("aggr", "add") not in ("add", "sum"):
            raise ValueError("EdgeValueGATv2Conv requires sum aggregation")
        if kwargs.get("decomposed_layers", 1) != 1:
            raise ValueError("EdgeValueGATv2Conv requires decomposed_layers=1")
        super().__init__(
            in_channels, out_channels, heads=heads, edge_dim=edge_dim, **kwargs
        )
        self.value_hidden_dim = value_hidden_dim
        # Initializing the optional branch must not advance the old model's RNG.
        devices = [self.att.device.index] if self.att.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            factory_kwargs = {"device": self.att.device, "dtype": self.att.dtype}
            self.value_net = nn.Sequential(
                nn.Linear(2 * in_channels + edge_dim, value_hidden_dim, **factory_kwargs),
                nn.SiLU(),
                nn.Linear(value_hidden_dim, out_channels, **factory_kwargs),
            )
            nn.init.zeros_(self.value_net[-1].weight)
            nn.init.zeros_(self.value_net[-1].bias)

    def reset_parameters(self) -> None:
        super().reset_parameters()
        # GATv2's constructor calls this before value_net exists.
        if hasattr(self, "value_net"):
            devices = [self.att.device.index] if self.att.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                self.value_net[0].reset_parameters()
                self.value_net[-1].reset_parameters()
                nn.init.zeros_(self.value_net[-1].weight)
                nn.init.zeros_(self.value_net[-1].bias)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
        return_attention_weights: bool | None = None,
    ) -> Tensor | tuple[Tensor, tuple[Tensor, Tensor]]:
        if not isinstance(x, Tensor) or x.layout != torch.strided:
            raise ValueError("x must be a dense homogeneous node tensor, not a bipartite pair")
        if x.ndim != 2 or x.shape[1] != self.in_channels or not x.is_floating_point():
            raise ValueError(f"x must have floating shape [N, {self.in_channels}]")
        if not isinstance(edge_index, Tensor) or edge_index.layout != torch.strided:
            raise ValueError("edge_index must be a dense [2, E] index tensor; sparse graphs are unsupported")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.dtype != torch.long:
            raise ValueError("edge_index must have shape [2, E] and dtype torch.long")
        if not isinstance(edge_attr, Tensor) or edge_attr.layout != torch.strided:
            raise ValueError("edge_attr is required as a dense [E, edge_dim] tensor")
        if edge_attr.shape != (edge_index.shape[1], self.edge_dim) or not edge_attr.is_floating_point():
            raise ValueError(f"edge_attr must have floating shape [E, {self.edge_dim}]")
        if x.device != edge_index.device or x.device != edge_attr.device:
            raise ValueError("x, edge_index and edge_attr must be on the same device")

        out, (attention_edges, alpha) = super().forward(
            x, edge_index, edge_attr, return_attention_weights=True
        )
        # Match PyG's self-loop removal, ordering and mean-filled attributes.
        # Keep these tensors local so checkpoint recomputation is reentrant.
        if self.add_self_loops:
            loop_edges, edge_attr = remove_self_loops(edge_index, edge_attr)
            _, edge_attr = add_self_loops(
                loop_edges, edge_attr, fill_value=self.fill_value, num_nodes=x.shape[0]
            )
        source, target = attention_edges
        values = self.value_net(torch.cat([x[target], x[source], edge_attr], dim=-1))
        weighted_values = (values * alpha).to(dtype=out.dtype)
        residual = out.new_zeros(out.shape).index_add(0, target, weighted_values)
        out = out + residual
        # PyG returns attention for either boolean value (including False).
        if isinstance(return_attention_weights, bool):
            return out, (attention_edges, alpha)
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


# ---------------------------------------------------------------------------
# Decoder tensor contracts
# ---------------------------------------------------------------------------


@dataclass
class CandidatePairs:
    """All ordered, distinct face pairs, partitioned by input object.

    indices: [P, 2] packed face indices; reverse: [P] reciprocal row indices.
    object_ids: [P]; face_counts: [B]. No GT connectivity enters this object.
    """

    indices: Tensor
    reverse: Tensor
    object_ids: Tensor
    face_counts: Tensor


@dataclass
class PrimitiveGeometry:
    """Local XYZ+direction samples, center/scale, and global XYZ samples."""

    local_points: Tensor
    center_scale: Tensor
    points: Tensor


@dataclass
class DecoderStage:
    """A complete coarse or refined prediction, before thresholding edges."""

    face_features: Tensor
    edge_features: Tensor
    face_geometry: PrimitiveGeometry
    edge_geometry: PrimitiveGeometry
    edge_logits: Tensor


@dataclass
class RefinementOutput:
    pairs: CandidatePairs
    coarse: DecoderStage
    refined: DecoderStage
    delta_face: Tensor
    delta_edge: Tensor


# ---------------------------------------------------------------------------
# Differentiable geometry projection
# ---------------------------------------------------------------------------


def _validate_geometry_pair(points: Tensor, geometry: Tensor) -> torch.dtype:
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("points must have shape [B, L, 3].")
    if points.shape[0] != geometry.shape[0]:
        raise ValueError("points and geometry must have the same batch size.")
    if points.device != geometry.device:
        raise ValueError("points and geometry must be on the same device.")
    if not points.is_floating_point() or not geometry.is_floating_point():
        raise TypeError("Geometry coordinates must be floating point tensors.")
    return torch.float64 if torch.float64 in (points.dtype, geometry.dtype) else torch.float32


def _project_to_segments(points: Tensor, starts: Tensor, ends: Tensor) -> tuple[Tensor, Tensor]:
    """Broadcast over leading dimensions; collapsed segments project to starts."""
    directions = ends - starts
    squared_lengths = directions.square().sum(dim=-1)
    valid = squared_lengths > torch.finfo(points.dtype).tiny
    denominators = torch.where(valid, squared_lengths, torch.ones_like(squared_lengths))
    numerators = ((points - starts) * directions).sum(dim=-1)
    with torch.no_grad():
        fractions = numerators / denominators
        interior = valid & (fractions >= 0.0) & (fractions <= 1.0)
        fractions = torch.where(valid, fractions.clamp(0.0, 1.0), torch.zeros_like(fractions))
    if torch.is_grad_enabled():
        # A clamped endpoint has no derivative through its fraction. Mask before
        # division: masking afterwards can still produce 0 * inf in backward
        # for a tiny segment and a distant point. Keep endpoint clamp semantics.
        interior_fractions = torch.where(interior, numerators, 0.0) / torch.where(
            interior, denominators, 1.0
        )
        fractions = torch.where(interior, interior_fractions, fractions)
    return starts + fractions[..., None] * directions, fractions


def _project_to_triangles(points: Tensor, triangles: Tensor) -> Tensor:
    """Choose a plane projection inside the triangle or the nearest edge point.

    Degenerate triangles use their three segments. Relative area guards prevent
    divisions by zero without imposing an absolute minimum triangle size.
    """
    vertex_a, vertex_b, vertex_c = triangles.unbind(dim=-2)
    side_b = vertex_b - vertex_a
    side_c = vertex_c - vertex_a
    normal = torch.linalg.cross(side_b, side_c, dim=-1)
    area_squared = normal.square().sum(dim=-1)
    side_product = side_b.square().sum(dim=-1) * side_c.square().sum(dim=-1)
    dtype_info = torch.finfo(points.dtype)
    valid = area_squared > (dtype_info.eps ** 2 * side_product).clamp_min(dtype_info.tiny)
    denominator = torch.where(valid, area_squared, torch.ones_like(area_squared))

    relative = points - vertex_a
    numerator_b = (torch.linalg.cross(relative, side_c, dim=-1) * normal).sum(dim=-1)
    numerator_c = (torch.linalg.cross(side_b, relative, dim=-1) * normal).sum(dim=-1)
    with torch.no_grad():
        weight_b = numerator_b / denominator
        weight_c = numerator_c / denominator
        inside = valid & (weight_b >= 0.0) & (weight_c >= 0.0) & (weight_b + weight_c <= 1.0)
    if torch.is_grad_enabled():
        # The chosen plane remains differentiable. Unused plane projections can
        # have huge barycentric weights on thin triangles; exclude their unsafe
        # divisions from backward instead of relying on the final torch.where.
        selected_denominator = torch.where(inside, denominator, 1.0)
        weight_b = torch.where(inside, numerator_b, 0.0) / selected_denominator
        weight_c = torch.where(inside, numerator_c, 0.0) / selected_denominator
    plane_point = vertex_a + weight_b[..., None] * side_b + weight_c[..., None] * side_c

    edge_ab, _ = _project_to_segments(points, vertex_a, vertex_b)
    edge_bc, _ = _project_to_segments(points, vertex_b, vertex_c)
    edge_ca, _ = _project_to_segments(points, vertex_c, vertex_a)
    edge_points = torch.stack((edge_ab, edge_bc, edge_ca), dim=-2)
    nearest_edge = (points[..., None, :] - edge_points).square().sum(dim=-1).argmin(dim=-1)
    nearest_edge_point = edge_points.gather(
        -2, nearest_edge[..., None, None].expand(*nearest_edge.shape, 1, 3)
    ).squeeze(-2)
    return torch.where(inside[..., None], plane_point, nearest_edge_point)


def _surface_triangle_indices(num_u: int, num_v: int, device: torch.device) -> Tensor:
    """Split each UV cell along its (u,v) to (u+1,v+1) diagonal."""
    grid = torch.arange(num_u * num_v, device=device).reshape(num_u, num_v)
    corner_a = grid[:-1, :-1].reshape(-1)
    corner_b = grid[1:, :-1].reshape(-1)
    corner_c = grid[:-1, 1:].reshape(-1)
    corner_d = grid[1:, 1:].reshape(-1)
    first = torch.stack((corner_a, corner_b, corner_d), dim=-1)
    second = torch.stack((corner_a, corner_d, corner_c), dim=-1)
    return torch.cat((first, second), dim=0)


def closest_surface_points(
    points: Tensor,
    surfaces: Tensor,
    chunk_size: int = 32,
) -> tuple[Tensor, Tensor]:
    """Project [B,L,3] points onto triangulated [B,U,V,3] surface grids.

    Returns closest points [B,L,3] and Euclidean distances [B,L]. These are exact
    for the piecewise linear grid, not for the unknown smooth CAD surface.
    ``chunk_size`` bounds the number of batch items in each nearest-triangle
    search; its largest intermediate is proportional to chunk_size * L * U * V.
    Half/bfloat16 inputs are evaluated and returned in float32, even under AMP.
    Gradients propagate through the selected triangle to both inputs; discrete
    triangle changes are not differentiated. Zero distances have finite gradients.
    """
    if surfaces.ndim != 4 or surfaces.shape[-1] != 3 or min(surfaces.shape[1:3]) < 2:
        raise ValueError("surfaces must have shape [B, U, V, 3] with U,V >= 2.")
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    dtype = _validate_geometry_pair(points, surfaces)
    with torch.autocast(device_type=points.device.type, enabled=False):
        points = points.to(dtype=dtype)
        surfaces = surfaces.to(dtype=dtype)
        if points.shape[0] == 0 or points.shape[1] == 0:
            closest = points + surfaces.sum() * 0.0
            return closest, torch.linalg.vector_norm(points - closest, dim=-1)

        triangle_indices = _surface_triangle_indices(surfaces.shape[1], surfaces.shape[2], points.device)
        surface_vertices = surfaces.flatten(1, 2)
        closest_chunks = []
        for start in range(0, points.shape[0], chunk_size):
            point_chunk = points[start:start + chunk_size]
            vertex_chunk = surface_vertices[start:start + chunk_size]
            with torch.no_grad():
                triangles = vertex_chunk[:, triangle_indices]
                projections = _project_to_triangles(point_chunk[:, :, None], triangles[:, None])
                squared_distances = (point_chunk[:, :, None] - projections).square().sum(dim=-1)
                nearest_triangle = squared_distances.argmin(dim=-1)
                selected_indices = triangle_indices[nearest_triangle]

            selected_vertices = vertex_chunk.gather(
                1, selected_indices.flatten(1, 2)[..., None].expand(-1, -1, 3)
            ).reshape(*point_chunk.shape[:2], 3, 3)
            closest_chunks.append(_project_to_triangles(point_chunk, selected_vertices))

        closest = torch.cat(closest_chunks, dim=0)
        distances = torch.linalg.vector_norm(points - closest, dim=-1)
        return closest, distances


def closest_polyline_points(points: Tensor, polylines: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Project [B,L,3] points onto [B,K,3] polylines, K >= 2.

    Returns closest points, segment start indices [B,L] (long), and fractions
    [B,L] in [0,1]. This correspondence can be computed on GT under no_grad and
    applied to predicted vertices by linear interpolation. Polylines are not
    automatically closed: a closed polyline must repeat its first endpoint.
    Precision and gradients follow ``closest_surface_points``.
    """
    if polylines.ndim != 3 or polylines.shape[-1] != 3 or polylines.shape[1] < 2:
        raise ValueError("polylines must have shape [B, K, 3] with K >= 2.")
    dtype = _validate_geometry_pair(points, polylines)
    with torch.autocast(device_type=points.device.type, enabled=False):
        points = points.to(dtype=dtype)
        polylines = polylines.to(dtype=dtype)
        if points.shape[0] == 0 or points.shape[1] == 0:
            closest = points + polylines.sum() * 0.0
            return closest, torch.empty(points.shape[:2], dtype=torch.long, device=points.device), points[..., 0]

        closest_chunks, index_chunks, fraction_chunks = [], [], []
        for start in range(0, points.shape[0], 32):
            point_chunk = points[start:start + 32]
            line_chunk = polylines[start:start + 32]
            with torch.no_grad():
                projections, _ = _project_to_segments(
                    point_chunk[:, :, None], line_chunk[:, None, :-1], line_chunk[:, None, 1:]
                )
                squared_distances = (point_chunk[:, :, None] - projections).square().sum(dim=-1)
                nearest_segment = squared_distances.argmin(dim=-1)

            selected_indices = nearest_segment[..., None].expand(-1, -1, 3)
            starts = line_chunk.gather(1, selected_indices)
            ends = line_chunk.gather(1, selected_indices + 1)
            closest, fractions = _project_to_segments(point_chunk, starts, ends)
            closest_chunks.append(closest)
            index_chunks.append(nearest_segment)
            fraction_chunks.append(fractions)

        return torch.cat(closest_chunks), torch.cat(index_chunks), torch.cat(fraction_chunks)


# ---------------------------------------------------------------------------
# Local face-pair intersection (retained experiment)
# ---------------------------------------------------------------------------


class LocalIntersectionBlock(nn.Module):
    """Update a single pair query while keeping the face memory unchanged."""

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, query: Tensor, memory: Tensor) -> Tensor:
        update, _ = self.attention(
            self.query_norm(query), memory, memory, need_weights=False,
        )
        query = query + self.residual_dropout(update)
        return query + self.residual_dropout(self.ffn(self.ffn_norm(query)))


class LocalPairIntersection(nn.Module):
    """One ordered face pair produces one edge feature, without topology slots.

    The pair's global face features produce a single query. Each face contributes
    its last CNN feature map and predicted object-space XYZ to a fixed memory.
    Relative side embeddings distinguish the first and second owner face; they
    do not encode coedge topology. Candidate gathering happens inside checkpoint
    regions so training does not retain all [pairs, 2*H*W, hidden_dim] memories.
    """

    def __init__(
        self,
        face_dim: int,
        edge_dim: int,
        local_channels: int,
        cfg: Mapping,
    ):
        super().__init__()
        self.face_dim = int(face_dim)
        self.edge_dim = int(edge_dim)
        self.local_channels = int(local_channels)
        # Runs before the capacity correction omit block_type. Preserve their
        # exact module/state layout instead of silently upgrading checkpoints.
        self.block_type = str(cfg.get("block_type", "cross_attention"))
        if self.block_type not in {"cross_attention", "transformer_decoder"}:
            raise ValueError(f"Unknown local intersection block_type: {self.block_type!r}.")
        full_capacity = self.block_type == "transformer_decoder"
        hidden_dim = int(cfg.get("hidden_dim", 512 if full_capacity else 128))
        num_heads = int(cfg.get("num_heads", 8 if full_capacity else 4))
        num_layers = int(cfg.get("num_layers", 8 if full_capacity else 2))
        ffn_dim = int(cfg.get("ffn_dim", 2048 if full_capacity else 256))
        dropout = float(cfg.get("dropout", 0.1 if full_capacity else 0.0))
        self.chunk_size = int(cfg.get("candidate_chunk_size", 64))
        self.checkpoint_candidates = bool(cfg.get("checkpoint_candidates", True))
        if min(
            self.face_dim, self.edge_dim, self.local_channels, hidden_dim,
            num_heads, num_layers, ffn_dim, self.chunk_size,
        ) <= 0:
            raise ValueError("Local intersection dimensions and chunk size must be positive.")
        if hidden_dim % num_heads:
            raise ValueError("Local intersection hidden_dim must be divisible by num_heads.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Local intersection dropout must be in [0, 1).")

        self.query_projection = nn.Sequential(
            nn.Linear(2 * self.face_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.memory_projection = nn.Sequential(
            nn.Linear(self.local_channels + 3, hidden_dim), nn.LayerNorm(hidden_dim),
        )
        self.side_embedding = nn.Parameter(torch.empty(2, hidden_dim))
        nn.init.normal_(self.side_embedding, std=0.02)
        if full_capacity:
            activation = str(cfg.get("activation", "relu"))
            if activation not in {"relu", "gelu"}:
                raise ValueError(f"Unsupported TransformerDecoder activation: {activation!r}.")
            # Match the retained AttnIntersection layer, including query self
            # attention, pre-normalization, activation and layer initialization.
            layer = nn.TransformerDecoderLayer(
                hidden_dim, num_heads, dim_feedforward=ffn_dim, dropout=dropout,
                activation=activation, batch_first=True, norm_first=True,
            )
            self.blocks = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        else:
            if str(cfg.get("activation", "gelu")) != "gelu":
                raise ValueError("Historical cross_attention blocks require activation=gelu.")
            self.blocks = nn.ModuleList([
                LocalIntersectionBlock(hidden_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_layers)
            ])
        self.output_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, self.edge_dim),
        )

    def _run_chunk(self, faces: Tensor, face_memory: Tensor, pairs: Tensor) -> Tensor:
        query = self.query_projection(faces[pairs].flatten(1)).unsqueeze(1)
        # Gather only this chunk. Each query reads both complete owner maps.
        memory = face_memory[pairs]
        memory = memory + self.side_embedding.to(memory.dtype)[None, :, None, :]
        memory = memory.flatten(1, 2)
        for block in self.blocks:
            query = block(query, memory)
        return self.output_projection(query[:, 0])

    def forward(
        self,
        face_features: Tensor,
        local_features: Tensor,
        face_points: Tensor,
        pair_indices: Tensor,
    ) -> Tensor:
        """Read [F,C,H,W] maps and [F,H,W,3] XYZ for [P,2] ordered pairs."""
        if face_features.ndim != 2 or face_features.shape[1] != self.face_dim:
            raise ValueError(f"Expected face_features [F,{self.face_dim}].")
        if (
            local_features.ndim != 4
            or local_features.shape[:2] != (face_features.shape[0], self.local_channels)
        ):
            raise ValueError(f"Expected local_features [F,{self.local_channels},H,W].")
        expected_points = (face_features.shape[0], *local_features.shape[2:], 3)
        if tuple(face_points.shape) != expected_points:
            raise ValueError("face_points must match local feature positions as [F,H,W,3].")
        if pair_indices.ndim != 2 or pair_indices.shape[1] != 2:
            raise ValueError("pair_indices must have shape [P,2].")
        if pair_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("pair_indices must contain integers.")
        if pair_indices.shape[0] == 0:
            return face_features.new_empty((0, self.edge_dim))

        local_tokens = local_features.flatten(2).transpose(1, 2)
        xyz_tokens = face_points.flatten(1, 2).to(local_tokens.dtype)
        # This projection is per face, shared by all pairs containing that face.
        # Neither CNN activations nor predicted XYZ are detached.
        face_memory = self.memory_projection(torch.cat((local_tokens, xyz_tokens), dim=-1))
        outputs = []
        for pairs in pair_indices.split(self.chunk_size):
            if self.checkpoint_candidates and self.training and torch.is_grad_enabled():
                edges = checkpoint(
                    self._run_chunk, face_features, face_memory, pairs, use_reentrant=False,
                )
            else:
                edges = self._run_chunk(face_features, face_memory, pairs)
            outputs.append(edges)
        return torch.cat(outputs, dim=0)


# ---------------------------------------------------------------------------
# Geometry feedback and feature refinement
# ---------------------------------------------------------------------------


def weighted_incident_mean(
    messages: Tensor,
    probabilities: Tensor,
    face_pairs: Tensor,
    num_faces: int,
) -> Tensor:
    """Aggregate [P,H] messages at both incident faces, using predicted weights.

    Ordered candidates ij and ji remain distinct messages. Both are incident
    to faces i and j; normalization accounts for both, with no hard threshold.
    Accumulation is float32 even under mixed precision.
    """
    values = messages.float() * probabilities.float().unsqueeze(-1)
    weights = probabilities.float().unsqueeze(-1)
    sums = values.new_zeros((num_faces, messages.shape[-1]))
    totals = weights.new_zeros((num_faces, 1))
    for side in range(2):
        sums = sums.index_add(0, face_pairs[:, side], values)
        totals = totals.index_add(0, face_pairs[:, side], weights)
    return (sums / totals.clamp_min(1e-6)).to(messages.dtype)


def curve_surface_observations(
    curves: Tensor, surfaces: Tensor, face_pairs: Tensor, chunk_size: int,
) -> Tensor:
    """Read 17 FP32 geometric values per curve sample from predicted owners."""
    with torch.autocast(device_type=curves.device.type, enabled=False):
        curves = curves.float()
        first_closest, first_distances = closest_surface_points(
            curves, surfaces[face_pairs[:, 0]].float(), chunk_size,
        )
        second_closest, second_distances = closest_surface_points(
            curves, surfaces[face_pairs[:, 1]].float(), chunk_size,
        )
        return torch.cat(
            (
                curves, first_closest, second_closest,
                curves - first_closest, curves - second_closest,
                first_distances.unsqueeze(-1), second_distances.unsqueeze(-1),
            ),
            dim=-1,
        )


class GeometryRead(nn.Module):
    """Read each candidate curve against its two predicted owner surfaces.

    Each sample contributes curve XYZ, closest XYZ on each surface, two
    point-to-surface offsets and two distances: 3 + 6 + 6 + 2 = 17 channels.
    Owners are ordered; neither edge features nor geometric reads are averaged
    with the reciprocal candidate. Output is [P, geometry_dim].
    """

    def __init__(self, geometry_dim: int, hidden_dim: int, chunk_size: int):
        super().__init__()
        self.chunk_size = chunk_size
        self.point_encoder = nn.Sequential(
            nn.Linear(17, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, geometry_dim),
            nn.SiLU(),
        )
        self.pool_projection = nn.Linear(geometry_dim * 2, geometry_dim)

    def forward(self, curves: Tensor, surfaces: Tensor, face_pairs: Tensor) -> Tensor:
        # Geometric differences must not be rounded to bf16 before encoding.
        observations = curve_surface_observations(curves, surfaces, face_pairs, self.chunk_size)
        point_features = self.point_encoder(observations)
        pooled = torch.cat(
            (point_features.mean(dim=1), point_features.amax(dim=1)), dim=-1
        )
        return self.pool_projection(pooled)


class CurvePointFeedback(nn.Module):
    """Added 2026-10-14: final curve XYZ feedback without spatial pooling.

    Read final predicted surfaces at every final curve point, concatenate the
    ordered owner/edge features, and predict a global XYZ displacement. The
    shared MLP has no point index or dropout: coincident curve samples with
    identical conditions receive identical updates. GT never enters forward.
    """

    def __init__(
        self, face_dim: int, edge_dim: int, hidden_dim: int = 128,
        feature_dim: int = 64, geometry_chunk_size: int = 32,
    ):
        super().__init__()
        if min(face_dim, edge_dim, hidden_dim, feature_dim, geometry_chunk_size) <= 0:
            raise ValueError("Curve feedback dimensions and chunk size must be positive.")
        self.geometry_chunk_size = geometry_chunk_size
        self.point_encoder = nn.Sequential(
            nn.Linear(17 + edge_dim + 2 * face_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim), nn.SiLU(),
        )
        self.delta = nn.Linear(feature_dim, 3)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(
        self, edge_geometry: PrimitiveGeometry, face_geometry: PrimitiveGeometry,
        edge_features: Tensor, face_features: Tensor, pair_indices: Tensor,
    ) -> PrimitiveGeometry:
        if edge_features.shape[0] == 0:
            return edge_geometry
        # BF16 bbox outputs can be exactly zero, including valid candidates.
        # With the original bbox held fixed, such a collapsed curve cannot
        # represent a nonzero XYZ delta. Leave its new residual zero; the old
        # geometry/bbox supervision still updates the original decoder. This
        # is an exact degeneracy mask, never an epsilon clamp on small curves.
        scale = edge_geometry.center_scale[:, None, 3:].float()
        if not bool(torch.isfinite(scale).all()):
            raise ValueError("Curve feedback requires finite predicted bbox scales.")
        invertible = scale != 0
        denominator = torch.where(invertible, scale, torch.ones_like(scale))
        deltas = []
        for start in range(0, edge_features.shape[0], self.geometry_chunk_size):
            stop = start + self.geometry_chunk_size
            owners = pair_indices[start:stop]
            observations = curve_surface_observations(
                edge_geometry.points[start:stop], face_geometry.points,
                owners, self.geometry_chunk_size,
            )
            condition = torch.cat(
                (edge_features[start:stop], face_features[owners].flatten(1)), dim=-1,
            )
            point_inputs = torch.cat(
                (observations, condition[:, None].expand(-1, observations.shape[1], -1)), dim=-1,
            )
            # Small final coordinate corrections must not be quantized by AMP.
            with torch.autocast(device_type=point_inputs.device.type, enabled=False):
                deltas.append(self.delta(self.point_encoder(point_inputs.float())))
        global_delta = torch.cat(deltas) * invertible
        with torch.autocast(device_type=global_delta.device.type, enabled=False):
            local_points = torch.cat(
                (
                    edge_geometry.local_points[..., :3].float() + global_delta / denominator,
                    edge_geometry.local_points[..., 3:].float(),
                ),
                dim=-1,
            )
            points = edge_geometry.points.float() + global_delta
        return PrimitiveGeometry(
            local_points=local_points, center_scale=edge_geometry.center_scale, points=points,
        )


class IntersectionRefiner(nn.Module):
    """Exchange candidate messages and write zero-initialized ΔF and ΔE.

    p0 is obtained from the symmetric coarse classifier logits. It controls
    soft routing only; no GT adjacency, true edge counts or GT geometry enter
    this forward pass. The final classifier remains outside this module.
    """

    def __init__(self, face_dim: int, edge_dim: int, cfg: dict):
        super().__init__()
        hidden_dim = int(cfg["hidden_dim"])
        geometry_dim = int(cfg["geometry_dim"])
        self.geometry_read = GeometryRead(
            geometry_dim, hidden_dim, int(cfg["geometry_chunk_size"])
        )
        self.message_encoder = nn.Sequential(
            nn.Linear(edge_dim + geometry_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.relation_encoder = nn.Sequential(
            nn.Linear(edge_dim + geometry_dim + 2 * face_dim + 2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.face_update = nn.Sequential(
            nn.Linear(face_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.edge_delta = nn.Linear(hidden_dim, edge_dim)
        self.face_delta = nn.Linear(hidden_dim, face_dim)
        for head in (self.edge_delta, self.face_delta):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, coarse: DecoderStage, pairs: CandidatePairs) -> tuple[Tensor, Tensor]:
        faces = coarse.face_features
        edges = coarse.edge_features
        if pairs.indices.shape[0] == 0:
            return torch.zeros_like(faces), torch.zeros_like(edges)

        first_owner, second_owner = pairs.indices.unbind(dim=-1)
        geometry = self.geometry_read(
            coarse.edge_geometry.points, coarse.face_geometry.points, pairs.indices
        )
        probabilities = coarse.edge_logits.float().sigmoid()

        # Read: each face gathers messages from all its candidate boundaries.
        messages = self.message_encoder(torch.cat((edges, geometry), dim=-1))
        face_context = weighted_incident_mean(
            messages, probabilities, pairs.indices, faces.shape[0]
        )

        # Exchange: a candidate sees both owners and their boundary contexts.
        relations = self.relation_encoder(
            torch.cat(
                (
                    edges, geometry,
                    faces[first_owner], faces[second_owner],
                    face_context[first_owner], face_context[second_owner],
                ),
                dim=-1,
            )
        )
        delta_edge = self.edge_delta(relations)

        # Write: each face receives the probability-weighted relation messages.
        face_messages = weighted_incident_mean(
            relations, probabilities, pairs.indices, faces.shape[0]
        )
        face_hidden = self.face_update(torch.cat((faces, face_messages), dim=-1))
        delta_face = self.face_delta(face_hidden)
        # A single-face object has no boundary information to write back.
        has_candidate = torch.zeros(faces.shape[0], dtype=torch.bool, device=faces.device)
        has_candidate[pairs.indices.reshape(-1)] = True
        delta_face = delta_face * has_candidate.unsqueeze(-1)
        return delta_face, delta_edge


# ---------------------------------------------------------------------------
# VAE geometry heads and intersection decoder
# ---------------------------------------------------------------------------


def enumerate_candidate_pairs(face_counts: Tensor) -> CandidatePairs:
    """Enumerate ordered i != j pairs inside each object, without GT inputs."""
    if face_counts.ndim != 1 or face_counts.numel() == 0:
        raise ValueError("face_counts must be a nonempty [B] tensor.")
    if face_counts.dtype not in (torch.int32, torch.int64):
        raise ValueError("face_counts must contain integers.")
    if bool((face_counts <= 0).any()):
        raise ValueError("Each object must contain at least one face.")
    device = face_counts.device
    all_pairs, all_reverse, all_objects = [], [], []
    face_offset = 0
    pair_offset = 0
    for object_id, count in enumerate(face_counts.tolist()):
        face_ids = torch.arange(count, device=device)
        first, second = torch.meshgrid(face_ids, face_ids, indexing="ij")
        distinct = first != second
        local_pairs = torch.stack((first[distinct], second[distinct]), dim=-1)
        count_pairs = local_pairs.shape[0]
        # A lookup makes the reciprocal mapping independent of formula tricks.
        lookup = torch.full((count, count), -1, dtype=torch.long, device=device)
        lookup[local_pairs[:, 0], local_pairs[:, 1]] = torch.arange(count_pairs, device=device)
        reverse = lookup[local_pairs[:, 1], local_pairs[:, 0]]
        all_pairs.append(local_pairs + face_offset)
        all_reverse.append(reverse + pair_offset)
        all_objects.append(torch.full((count_pairs,), object_id, dtype=torch.long, device=device))
        face_offset += count
        pair_offset += count_pairs
    return CandidatePairs(
        indices=torch.cat(all_pairs),
        reverse=torch.cat(all_reverse),
        object_ids=torch.cat(all_objects),
        face_counts=face_counts,
    )


def symmetric_pair_logits(logits: Tensor, reverse: Tensor) -> Tensor:
    """Average ij/ji logits only; their directional features remain separate."""
    logits = logits.float()
    return 0.5 * (logits + logits[reverse])


class FaceFeatureDecoder(nn.Module):
    """Run the original face Transformer once with a packed object mask."""

    def __init__(self, input_projection: nn.Module, backbone: nn.Module, output_projection: nn.Module):
        super().__init__()
        self.input_projection = input_projection
        self.backbone = backbone
        self.output_projection = output_projection

    def forward(self, face_latents: Tensor, face_counts: Tensor) -> Tensor:
        object_ids = torch.repeat_interleave(
            torch.arange(face_counts.numel(), device=face_latents.device), face_counts
        )
        attention_mask = object_ids[:, None] != object_ids[None, :]
        hidden = self.input_projection(face_latents)
        hidden = self.backbone(hidden, mask=attention_mask)
        return self.output_projection(hidden)


class FacePairConditioning(nn.Module):
    """Condition the final curve heads; zero outputs preserve the old heads.

    Added for curve_conditioning: a shared MLP reads [E, F_i, F_j], then
    separate zero-initialized heads produce channel-wise scale and shift for
    the curve CNN and center/scale MLP. No normalization is inserted.
    """

    def __init__(self, condition_dim: int, hidden_dim: int, point_channels: int, bbox_channels: int):
        super().__init__()
        if min(condition_dim, hidden_dim, point_channels, bbox_channels) <= 0:
            raise ValueError("Curve conditioning dimensions must be positive.")
        self.condition_dim = condition_dim
        self.projection = nn.Sequential(nn.Linear(condition_dim, hidden_dim), nn.SiLU())
        self.point_affine = nn.Linear(hidden_dim, 2 * point_channels)
        self.bbox_affine = nn.Linear(hidden_dim, 2 * bbox_channels)
        for head in (self.point_affine, self.bbox_affine):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, condition: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if condition.ndim != 2 or condition.shape[1] != self.condition_dim:
            raise ValueError(f"Expected curve condition [P,{self.condition_dim}].")
        hidden = self.projection(condition)
        point_scale, point_shift = self.point_affine(hidden).chunk(2, dim=-1)
        bbox_scale, bbox_shift = self.bbox_affine(hidden).chunk(2, dim=-1)
        return point_scale, point_shift, bbox_scale, bbox_shift


class PrimitiveDecoder(nn.Module):
    """Retained local point and center/scale heads, with explicit global XYZ."""

    def __init__(
        self,
        points_decoder: nn.Module,
        bbox_decoder: nn.Module,
        spatial_shape: tuple[int, ...],
        point_channels: int,
        condition_dim: int | None = None,
        condition_hidden_dim: int = 128,
    ):
        super().__init__()
        self.points_decoder = points_decoder
        self.bbox_decoder = bbox_decoder
        self.spatial_shape = spatial_shape
        self.point_channels = point_channels
        self.conditioning = None
        if condition_dim is not None:
            if spatial_shape != (16,) or not isinstance(points_decoder, nn.Sequential):
                raise ValueError("Curve conditioning requires the retained Sequential curve CNN.")
            indices = [i for i, layer in enumerate(points_decoder) if isinstance(layer, nn.Conv1d)]
            if not indices:
                raise ValueError("Curve conditioning requires a final Conv1d output layer.")
            self.point_condition_index = indices[-1]
            output_layer = points_decoder[self.point_condition_index]
            if output_layer.out_channels != point_channels or output_layer.kernel_size != (1,):
                raise ValueError("Curve conditioning expects the original final 1x1 Conv1d.")
            if not isinstance(bbox_decoder, nn.Sequential) or not isinstance(bbox_decoder[-1], nn.Linear):
                raise ValueError("Curve conditioning requires the retained Sequential bbox MLP.")
            if bbox_decoder[-1].out_features != 4:
                raise ValueError("The retained bbox head must output center 3 + scale 1.")
            self.bbox_condition_index = len(bbox_decoder) - 1
            self.conditioning = FacePairConditioning(
                condition_dim, condition_hidden_dim, output_layer.in_channels,
                bbox_decoder[-1].in_features,
            )

    def _empty_geometry(self, features: Tensor) -> PrimitiveGeometry:
        return PrimitiveGeometry(
            local_points=features.new_empty((0, *self.spatial_shape, self.point_channels)),
            center_scale=features.new_empty((0, 4)),
            points=features.new_empty((0, *self.spatial_shape, 3), dtype=torch.float32),
        )

    def _with_global_points(
        self, features: Tensor, local_points: Tensor, center_scale: Tensor | None = None,
    ) -> PrimitiveGeometry:
        if center_scale is None:
            center_scale = self.bbox_decoder(features)
        with torch.autocast(device_type=features.device.type, enabled=False):
            broadcast_shape = (features.shape[0],) + (1,) * len(self.spatial_shape)
            center = center_scale[:, :3].float().reshape(*broadcast_shape, 3)
            scale = center_scale[:, 3:].float().reshape(*broadcast_shape, 1)
            points = local_points[..., :3].float() * scale + center
        return PrimitiveGeometry(local_points=local_points, center_scale=center_scale, points=points)

    def _local_output_layer_index(self) -> int:
        if not isinstance(self.points_decoder, nn.Sequential):
            raise ValueError("Local intersection requires a Sequential face points decoder.")
        indices = [
            index for index, module in enumerate(self.points_decoder)
            if isinstance(module, nn.Conv2d)
        ]
        if not indices:
            raise ValueError("Local intersection requires a final Conv2d face output layer.")
        index = indices[-1]
        layer = self.points_decoder[index]
        if layer.out_channels != self.point_channels or layer.kernel_size != (1, 1):
            raise ValueError("Expected a final 1x1 Conv2d projecting local face features to points.")
        return index

    @property
    def local_feature_channels(self) -> int:
        return self.points_decoder[self._local_output_layer_index()].in_channels

    def forward_with_local_features(self, features: Tensor) -> tuple[PrimitiveGeometry, Tensor]:
        """Decode once and expose the map immediately before the output Conv2d.

        Iteration preserves all original module registrations and checkpoint
        names. The captured map stays connected to the face CNN's gradients.
        """
        output_index = self._local_output_layer_index()
        if features.shape[0] == 0:
            local_map = features.new_empty((0, self.local_feature_channels, *self.spatial_shape))
            return self._empty_geometry(features), local_map
        points = features
        local_map = None
        for index, module in enumerate(self.points_decoder):
            if index == output_index:
                local_map = points
            points = module(points)
        assert local_map is not None
        return self._with_global_points(features, points), local_map

    def forward(self, features: Tensor, condition: Tensor | None = None) -> PrimitiveGeometry:
        if features.shape[0] == 0:
            return self._empty_geometry(features)
        if condition is None:
            # Coarse geometry and the encoder-edge teacher keep this path.
            local_points = self.points_decoder(features)
            return self._with_global_points(features, local_points)
        if self.conditioning is None or condition.shape[0] != features.shape[0]:
            raise ValueError("Curve conditioning must be enabled and match the edge batch.")

        point_scale, point_shift, bbox_scale, bbox_shift = self.conditioning(condition)
        local_points = features
        for index, layer in enumerate(self.points_decoder):
            if index == self.point_condition_index:
                local_points = (1 + point_scale[..., None]) * local_points + point_shift[..., None]
            local_points = layer(local_points)
        center_scale = features
        for index, layer in enumerate(self.bbox_decoder):
            if index == self.bbox_condition_index:
                center_scale = (1 + bbox_scale) * center_scale + bbox_shift
            center_scale = layer(center_scale)
        return self._with_global_points(features, local_points, center_scale)


class IntersectionDecoder(nn.Module):
    """Decode coarse geometry, then apply one read/exchange/write refinement.

    Direction ij retains the existing ordered face-pair curve/normal target
    semantics, using either the legacy inter or local cross-attention. A
    reciprocal ji candidate is always present. Only logits become symmetric.

    cfg is the refinement mapping itself. intersection_cfg selects the legacy
    inter (default) or local face-memory cross-attention. The legacy model is
    not registered here: named decoder modules are transferred directly.
    """

    def __init__(
        self, legacy: AutoEncoder_light, cfg: dict, intersection_cfg: dict | None = None,
        curve_conditioning_cfg: dict | None = None,
        curve_point_feedback_cfg: dict | None = None,
    ):
        super().__init__()
        self.face_dim = legacy.face_feature_dim
        self.edge_dim = legacy.edge_feature_dim
        self.chunk_size = int(cfg["candidate_chunk_size"])
        self.checkpoint_candidates = bool(cfg["checkpoint_candidates"])
        self.refinement_enabled = bool(cfg["enabled"])
        conditioning_cfg = {} if curve_conditioning_cfg is None else curve_conditioning_cfg
        self.curve_conditioning_enabled = bool(conditioning_cfg.get("enabled", False))
        feedback_cfg = curve_point_feedback_cfg or {}
        self.curve_point_feedback_enabled = bool(feedback_cfg.get("enabled", False))
        if self.curve_conditioning_enabled and (
            not self.refinement_enabled or (intersection_cfg or {}).get("kind", "legacy") != "legacy"
        ):
            raise ValueError("Curve conditioning requires legacy intersection with refinement enabled.")
        if self.curve_point_feedback_enabled and not self.curve_conditioning_enabled:
            raise ValueError("Curve point feedback requires the retained curve conditioning path.")
        if self.chunk_size <= 0 or int(cfg["geometry_chunk_size"]) <= 0:
            raise ValueError("Candidate and geometry chunk sizes must be positive.")
        if int(cfg["hidden_dim"]) <= 0 or int(cfg["geometry_dim"]) <= 0:
            raise ValueError("Refinement feature dimensions must be positive.")

        self.face_features = FaceFeatureDecoder(
            legacy.face_attn_proj_in2, legacy.face_attn2, legacy.face_attn_proj_out2
        )
        self.face_geometry = PrimitiveDecoder(
            legacy.face_points_decoder, legacy.face_center_scale_decoder,
            spatial_shape=(16, 16), point_channels=legacy.in_channels,
        )
        self.edge_geometry = PrimitiveDecoder(
            legacy.edge_points_decoder, legacy.edge_center_scale_decoder,
            spatial_shape=(16,), point_channels=legacy.in_channels,
            condition_dim=self.edge_dim + 2 * self.face_dim if self.curve_conditioning_enabled else None,
            condition_hidden_dim=int(conditioning_cfg.get("hidden_dim", 128)),
        )
        intersection_cfg = {} if intersection_cfg is None else intersection_cfg
        self.intersection_kind = str(intersection_cfg.get("kind", "legacy"))
        if self.intersection_kind == "legacy":
            self.inter = legacy.inter
        elif self.intersection_kind == "local_attention":
            self.inter = LocalPairIntersection(
                self.face_dim, self.edge_dim, self.face_geometry.local_feature_channels,
                intersection_cfg,
            )
        else:
            raise ValueError(f"Unknown intersection kind: {self.intersection_kind!r}.")
        self.classifier = legacy.classifier
        self.refiner = (
            IntersectionRefiner(self.face_dim, self.edge_dim, cfg)
            if self.refinement_enabled else None
        )
        # Default-off registers no new state keys in historical checkpoints.
        self.curve_point_feedback = (
            CurvePointFeedback(
                self.face_dim, self.edge_dim,
                hidden_dim=int(feedback_cfg.get("hidden_dim", 128)),
                feature_dim=int(feedback_cfg.get("feature_dim", 64)),
                geometry_chunk_size=int(cfg["geometry_chunk_size"]),
            )
            if self.curve_point_feedback_enabled else None
        )

    def _run_candidates(self, feature_pairs: Tensor) -> Tensor:
        if feature_pairs.shape[0] == 0:
            return feature_pairs.new_empty((0, self.edge_dim))
        chunks = []
        for chunk in feature_pairs.split(self.chunk_size):
            if self.checkpoint_candidates and self.training and torch.is_grad_enabled():
                values = checkpoint(self.inter, chunk, use_reentrant=False)
            else:
                values = self.inter(chunk)
            chunks.append(values)
        return torch.cat(chunks)

    def _decode_edge_chunk(
        self, features: Tensor, condition: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        geometry = self.edge_geometry(features, condition)
        return geometry.local_points, geometry.center_scale, geometry.points

    def _decode_edges(self, features: Tensor, condition: Tensor | None = None) -> PrimitiveGeometry:
        if features.shape[0] == 0:
            return self.edge_geometry(features, condition)
        chunks = []
        for start in range(0, features.shape[0], self.chunk_size):
            chunk = features[start:start + self.chunk_size]
            condition_chunk = None if condition is None else condition[start:start + self.chunk_size]
            if self.checkpoint_candidates and self.training and torch.is_grad_enabled():
                geometry = checkpoint(self._decode_edge_chunk, chunk, condition_chunk, use_reentrant=False)
            else:
                geometry = self._decode_edge_chunk(chunk, condition_chunk)
            chunks.append(geometry)
        return PrimitiveGeometry(
            local_points=torch.cat([chunk[0] for chunk in chunks]),
            center_scale=torch.cat([chunk[1] for chunk in chunks]),
            points=torch.cat([chunk[2] for chunk in chunks]),
        )

    def _decode_stage(
        self,
        faces: Tensor,
        edges: Tensor,
        pairs: CandidatePairs,
        face_geometry: PrimitiveGeometry | None = None,
        condition_curves: bool = False,
    ) -> DecoderStage:
        logits = self.classifier(edges).squeeze(-1)
        # Only the final refined edge geometry reads the direct owner features.
        condition = None
        if condition_curves and self.curve_conditioning_enabled:
            condition = torch.cat((edges, faces[pairs.indices].flatten(1)), dim=-1)
        return DecoderStage(
            face_features=faces,
            edge_features=edges,
            face_geometry=self.face_geometry(faces) if face_geometry is None else face_geometry,
            edge_geometry=self._decode_edges(edges, condition),
            edge_logits=symmetric_pair_logits(logits, pairs.reverse),
        )

    def forward(self, face_latents: Tensor, face_counts: Tensor) -> RefinementOutput:
        if face_latents.ndim != 2 or face_latents.shape[1] != self.face_dim:
            raise ValueError(f"Expected packed face_latents [F,{self.face_dim}].")
        if not bool(torch.isfinite(face_latents).all()):
            raise ValueError("face_latents must be finite.")
        face_counts = face_counts.to(device=face_latents.device)
        pairs = enumerate_candidate_pairs(face_counts)
        if int(face_counts.sum()) != face_latents.shape[0]:
            raise ValueError("Sum of face_counts must equal the packed latent count.")

        # The original Transformer runs once. No geometry/GT enters this stage.
        faces = self.face_features(face_latents, face_counts)
        if self.intersection_kind == "local_attention":
            coarse_faces, local_features = self.face_geometry.forward_with_local_features(faces)
            edges = self.inter(faces, local_features, coarse_faces.points, pairs.indices)
            coarse = self._decode_stage(faces, edges, pairs, face_geometry=coarse_faces)
        else:
            edges = self._run_candidates(faces[pairs.indices])
            coarse = self._decode_stage(faces, edges, pairs)

        if self.refiner is None:
            delta_face, delta_edge = torch.zeros_like(faces), torch.zeros_like(edges)
            refined = coarse
        else:
            delta_face, delta_edge = self.refiner(coarse, pairs)
            # Reuse exactly the same geometry heads and linear classifier.
            refined = self._decode_stage(
                faces + delta_face, edges + delta_edge, pairs, condition_curves=True,
            )

        if self.curve_point_feedback is not None:
            # Final local/global geometry is updated together, so reconstruction,
            # incidence, twin supervision and exported curves all see this result.
            refined.edge_geometry = self.curve_point_feedback(
                refined.edge_geometry, refined.face_geometry,
                refined.edge_features, refined.face_features, pairs.indices,
            )

        return RefinementOutput(
            pairs=pairs, coarse=coarse, refined=refined,
            delta_face=delta_face, delta_edge=delta_edge,
        )


# ---------------------------------------------------------------------------
# Shared-boundary decoder (retained experiment)
# ---------------------------------------------------------------------------


class BoundaryInteraction(nn.Module):
    """One face -> boundary -> face update, followed by updated ownership."""

    def __init__(self, dim: int, heads: int, dropout: float, edge_to_face: bool):
        super().__init__()
        self.edge_to_face = edge_to_face
        self.face_norm = nn.LayerNorm(dim)
        self.edge_norm = nn.LayerNorm(dim)
        self.face_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.edge_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.side_query = nn.Linear(dim, 2 * dim)
        self.face_key = nn.Linear(dim, dim)
        self.edge_update = nn.Sequential(
            nn.Linear(3 * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim)
        )
        # Remove these parameters entirely for the unidirectional ablation.
        if edge_to_face:
            self.side_message = nn.Linear(dim, 2 * dim)
            self.face_update = nn.Sequential(
                nn.Linear(2 * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim)
            )

    def ownership(self, faces: Tensor, edges: Tensor, valid: Tensor) -> Tensor:
        batch, queries, dim = edges.shape
        query = self.side_query(edges).reshape(batch, queries, 2, dim)
        logits = torch.einsum("bmsd,bnd->bmsn", query, self.face_key(faces)) / math.sqrt(dim)
        return logits.masked_fill(~valid[:, None, None, :], -torch.inf)

    def forward(
        self, faces: Tensor, edges: Tensor, valid: Tensor, existence_head: nn.Module
    ) -> tuple[Tensor, Tensor]:
        normalized = self.face_norm(faces)
        faces = faces + self.face_attention(
            normalized, normalized, normalized, key_padding_mask=~valid, need_weights=False
        )[0]
        faces = faces.masked_fill(~valid[..., None], 0)
        normalized = self.edge_norm(edges)
        edges = edges + self.edge_attention(normalized, normalized, normalized, need_weights=False)[0]

        assignment = self.ownership(faces, edges, valid).softmax(dim=-1)
        read = torch.einsum("bmsn,bnd->bmsd", assignment, faces).flatten(-2)
        edges = edges + self.edge_update(torch.cat((edges, read), dim=-1))

        if self.edge_to_face:
            # Recompute after the edge update. The SAME ownership weights both
            # express incidence and route edge information to each face.
            assignment = self.ownership(faces, edges, valid).softmax(dim=-1)
            mass = assignment * existence_head(edges).sigmoid()[..., None]
            messages = self.side_message(edges).reshape(*edges.shape[:2], 2, edges.shape[-1])
            write = torch.einsum("bmsn,bmsd->bnd", mass, messages)
            write = write / mass.sum(dim=(1, 2))[..., None].clamp_min(1.0)
            faces = faces + self.face_update(torch.cat((faces, write), dim=-1))
            faces = faces.masked_fill(~valid[..., None], 0)
        return faces, edges


class JointBoundaryDecoder(nn.Module):
    """Decode a variable face set and at most ``num_queries`` boundaries.

    Only batching is padded. There are no duplicate face slots or face positional
    embeddings. Global XYZ heads deliberately differ from the legacy local-grid
    plus bounding-box heads; this must be accounted for in research comparisons.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 256,
        num_queries: int = 128,
        num_rounds: int = 3,
        num_heads: int = 8,
        dropout: float = 0.0,
        edge_to_face: bool = True,
        face_resolution: int = 16,
        curve_resolution: int = 16,
    ):
        super().__init__()
        if min(latent_dim, hidden_dim, num_queries, num_rounds, num_heads) < 1:
            raise ValueError("Decoder dimensions, query capacity, and round count must be positive")
        if hidden_dim % num_heads:
            raise ValueError("joint_hidden_dim must be divisible by joint_num_heads")
        self.num_queries = num_queries
        self.face_resolution = face_resolution
        self.curve_resolution = curve_resolution
        self.face_input = nn.Linear(latent_dim, hidden_dim)
        self.edge_queries = nn.Parameter(torch.randn(num_queries, hidden_dim) / math.sqrt(hidden_dim))
        self.rounds = nn.ModuleList([
            BoundaryInteraction(hidden_dim, num_heads, dropout, edge_to_face)
            for _ in range(num_rounds)
        ])
        self.existence_head = nn.Linear(hidden_dim, 1)
        self.face_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, face_resolution * face_resolution * 3),
        )
        self.curve_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, curve_resolution * 3),
        )

    def forward(self, face_latents: Tensor, face_counts: Tensor) -> dict[str, Tensor]:
        if face_latents.ndim != 2:
            raise ValueError("face_latents must be a flat [sum_faces, latent_dim] tensor")
        if face_counts.ndim != 1 or face_counts.dtype not in (torch.int32, torch.int64):
            raise ValueError("face_counts must be a one-dimensional integer tensor")
        counts = face_counts.detach().cpu().tolist()
        if not counts or min(counts) < 1 or sum(counts) != len(face_latents):
            raise ValueError("face_counts must partition the input into nonempty objects")
        if not torch.isfinite(face_latents).all():
            raise ValueError("face_latents contains nonfinite values")
        padded = pad_sequence(face_latents.split(counts), batch_first=True)
        counts_tensor = face_counts.to(face_latents.device)
        valid = torch.arange(padded.shape[1], device=padded.device)[None, :] < counts_tensor[:, None]
        faces = self.face_input(padded).masked_fill(~valid[..., None], 0)
        edges = self.edge_queries[None].expand(len(counts), -1, -1)
        for interaction in self.rounds:
            faces, edges = interaction(faces, edges, valid, self.existence_head)
        ownership_logits = self.rounds[-1].ownership(faces, edges, valid)
        surfaces = self.face_head(faces).reshape(
            len(counts), padded.shape[1], self.face_resolution, self.face_resolution, 3
        ).masked_fill(~valid[..., None, None, None], 0)
        curves = self.curve_head(edges).reshape(len(counts), self.num_queries, self.curve_resolution, 3)
        return {
            "face_points": surfaces,
            "curves": curves,
            "existence_logits": self.existence_head(edges).squeeze(-1),
            "ownership_logits": ownership_logits,
            "face_valid": valid,
            "face_counts": counts_tensor,
            "face_states": faces,
            "edge_states": edges,
        }
