"""Face-owned boundary reading, exchange, and two explicit output choices."""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from .conditional_intersection import ConditionalIntersection
from .refinement import GeometryRead
from ..state import CandidatePairs, DecoderStage


class OwnerIntersection(ConditionalIntersection):
    """An ordered pair initializes a query that reads the FIRST face's chart."""

    memory_side = 0

    def __init__(self, legacy_inter: nn.Module, local_channels: int, cfg: Mapping):
        super().__init__(legacy_inter, local_channels, cfg)
        face_dim = self.attn_proj_in.in_features
        width = self.attn_proj_in.out_features
        hidden = int(cfg.get("reference_hidden_dim", 128))
        self.pair_query = nn.Sequential(
            nn.Linear(2 * face_dim, hidden), nn.SiLU(), nn.Linear(hidden, self.edge_dim),
        )
        self.query_projection = nn.Linear(self.edge_dim, width)

    def _inputs(self, faces: Tensor, local: Tensor, pairs: Tensor) -> tuple[Tensor, ...]:
        pair_query = self.pair_query(faces[pairs].flatten(1))  # [P,256] -> [P,64]
        query = self.query_projection(pair_query)[:, None] + self.pos_encoding[:, 0:1]
        projected = self.attn_proj_in(faces)
        memory = self.spatial_memory(projected + self.pos_encoding[:, 1], local)
        # ReferenceUVHead predicts its SECOND input's chart: (j,i) means chart i.
        reference = self.reference_uv_head(faces[pairs.flip(-1)])
        reference_position = self.uv_position(reference).to(query.dtype)
        grid = self.uv_position.grid(*local.shape[-2:])
        return query, reference, reference_position, memory, self.uv_position(grid).to(query.dtype)


class OwnerBoundaryAttention(nn.Module):
    """Per-owner Q/K/V; confidence-weighted keys plus a zero-value null source.

    alpha_j = p_j exp(score_j) / (1 + sum_k p_k exp(score_k)).
    All candidates remain queries. p=0 cannot send; all p=0 gives exactly zero.
    Packed face ids isolate both owners and objects. No GT routing is used.
    """

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        # No bias: a null-only read must stay zero after projection.
        self.output = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, messages: Tensor, probabilities: Tensor, owners: Tensor, num_faces: int) -> Tensor:
        if messages.shape[0] == 0:
            return messages
        counts = torch.bincount(owners, minlength=num_faces)
        length = int(counts.max())
        order = owners.argsort(stable=True)
        sorted_owners = owners[order]
        starts = counts.cumsum(0) - counts
        slots = torch.arange(len(owners), device=owners.device) - starts[sorted_owners]
        padded = messages.new_zeros((num_faces, length, messages.shape[-1]))
        padded[sorted_owners, slots] = messages[order]
        confidence = probabilities.detach().float().new_zeros((num_faces, length))
        confidence[sorted_owners, slots] = probabilities.detach().float()[order]
        qkv = self.qkv(self.norm(padded)).reshape(
            num_faces, length, 3, self.num_heads, -1,
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        scores = (query @ key.transpose(-1, -2)).float() / query.shape[-1] ** 0.5
        log_confidence = confidence.clamp_min(torch.finfo(torch.float32).tiny).log()
        log_confidence = log_confidence.masked_fill(confidence == 0, -torch.inf)
        scores = scores + log_confidence[:, None, None, :]
        # The fixed null logit is zero; its value is zero, hence omitted below.
        scores = torch.cat((scores, scores.new_zeros((*scores.shape[:-1], 1))), dim=-1)
        weights = scores.softmax(-1)[..., :-1].to(value.dtype)
        context = (weights @ value).transpose(1, 2).reshape_as(padded)
        context = self.output(context)
        sorted_context = context[sorted_owners, slots]
        return sorted_context[order.argsort()]  # [P,128], original candidate order


def owner_pool(messages: Tensor, probabilities: Tensor, owners: Tensor, num_faces: int) -> Tensor:
    """Owner-only confidence pooling with one zero-message unit of mass."""
    weights = probabilities.detach().float()[:, None]
    sums = messages.new_zeros((num_faces, messages.shape[-1]), dtype=torch.float32)
    mass = weights.new_ones((num_faces, 1))
    sums = sums.index_add(0, owners, messages.float() * weights)
    mass = mass.index_add(0, owners, weights)
    return (sums / mass).to(messages.dtype)


class BoundaryResidualWriteback(nn.Module):
    """Retained Relation MLP and face/edge residual writes; no extra refiner."""

    def __init__(self, face_dim: int, edge_dim: int, geometry_dim: int, hidden_dim: int):
        super().__init__()
        self.relation_encoder = nn.Sequential(
            nn.Linear(edge_dim + geometry_dim + 2 * face_dim + 2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.edge_delta = nn.Linear(hidden_dim, edge_dim)
        self.face_update = nn.Sequential(
            nn.Linear(face_dim + hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
        )
        self.face_delta = nn.Linear(hidden_dim, face_dim)
        for head in (self.edge_delta, self.face_delta):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, coarse: DecoderStage, geometry: Tensor, context: Tensor,
                probabilities: Tensor, pairs: CandidatePairs) -> tuple[Tensor, Tensor]:
        faces, edges = coarse.face_features, coarse.edge_features
        first, second = pairs.indices.unbind(-1)
        relation = self.relation_encoder(torch.cat((
            edges, geometry, faces[first], faces[second], context, context[pairs.reverse],
        ), dim=-1))  # [P,640] -> [P,128]
        edge_delta = self.edge_delta(relation)
        face_context = owner_pool(relation, probabilities, first, len(faces))
        face_delta = self.face_delta(self.face_update(torch.cat((faces, face_context), dim=-1)))
        has_owner = torch.bincount(first, minlength=len(faces)) > 0
        return faces + face_delta * has_owner[:, None], edges + edge_delta


class OwnerBoundaryExchange(nn.Module):
    """Shared geometry/message/attention trunk; direct OR residual output."""

    def __init__(self, face_dim: int, edge_dim: int, cfg: Mapping):
        super().__init__()
        hidden, geometry = int(cfg["hidden_dim"]), int(cfg["geometry_dim"])
        self.geometry_read = GeometryRead(geometry, hidden, int(cfg["geometry_chunk_size"]))
        self.message_encoder = nn.Sequential(
            nn.Linear(edge_dim + geometry, hidden), nn.LayerNorm(hidden), nn.SiLU(),
        )
        self.attention = OwnerBoundaryAttention(hidden, int(cfg.get("owner_num_heads", 4)))
        self.readout_mode = str(cfg["owner_readout"])
        if self.readout_mode == "direct":
            self.direct_readout = nn.Linear(2 * hidden, edge_dim)
            self.writeback = None
        elif self.readout_mode == "residual":
            self.direct_readout = None
            self.writeback = BoundaryResidualWriteback(face_dim, edge_dim, geometry, hidden)
        else:
            raise ValueError(f"Unknown owner_readout: {self.readout_mode!r}")

    def forward(self, coarse: DecoderStage, pairs: CandidatePairs) -> tuple[Tensor, Tensor]:
        if len(pairs.indices) == 0:
            return coarse.face_features, coarse.edge_features
        geometry = self.geometry_read(
            coarse.edge_geometry.points, coarse.face_geometry.points, pairs.indices,
        )  # r [P,64]
        messages = self.message_encoder(torch.cat((coarse.edge_features, geometry), dim=-1))
        probabilities = coarse.edge_logits.float().sigmoid().detach()
        context = self.attention(messages, probabilities, pairs.indices[:, 0], len(coarse.face_features))
        if self.direct_readout is not None:
            edges = self.direct_readout(torch.cat((messages, context), dim=-1))
            return coarse.face_features, edges
        return self.writeback(coarse, geometry, context, probabilities, pairs)
