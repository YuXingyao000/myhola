"""Directed face-pair intersection reasoning and the retained decoder assembly."""

from __future__ import annotations

from collections.abc import Mapping
import copy
from typing import TYPE_CHECKING

import torch
from torch import nn, Tensor
from torch.nn import ModuleList
from torch.utils.checkpoint import checkpoint

from ..state import CandidatePairs, DecoderStage, PrimitiveGeometry, RefinementOutput
from .primitives import (
    FaceFeatureDecoder, PrimitiveDecoder, UVPathHead,
    SharedCurveFusion, CurveConditionedUVPathHead,
)
from .refinement import CurvePointFeedback, IntersectionRefiner

if TYPE_CHECKING:
    from ..legacy import AutoEncoder_light


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


def enumerate_candidate_pairs(face_counts: Tensor) -> CandidatePairs:
    """Enumerate ordered i != j pairs inside each object, without GT inputs."""
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
        uv_boundary_cfg: dict | None = None,
        shared_curve_cfg: dict | None = None,
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
        shared_options = shared_curve_cfg or {}
        self.shared_curve_enabled = bool(shared_options.get("enabled", False))
        if self.shared_curve_enabled and self.curve_point_feedback_enabled:
            raise ValueError("Per-half-edge point feedback would break shared curve geometry.")
        if self.curve_conditioning_enabled and (
            not self.refinement_enabled or (intersection_cfg or {}).get("kind", "legacy") != "legacy"
        ):
            raise ValueError("Curve conditioning requires legacy intersection with refinement enabled.")
        if self.curve_point_feedback_enabled and not self.curve_conditioning_enabled:
            raise ValueError("Curve point feedback requires the retained curve conditioning path.")
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
        uv_options = dict(uv_boundary_cfg or {})
        uv_enabled = bool(uv_options.pop("enabled", False))
        self.uv_boundary = None
        self.shared_curve = None
        if self.shared_curve_enabled:
            with torch.random.fork_rng(devices=[]):
                self.shared_curve = SharedCurveFusion(self.edge_dim, int(shared_options.get("hidden_dim", 128)))
        if uv_enabled:
            # Auxiliary initialization must not perturb retained-model RNG/state.
            with torch.random.fork_rng(devices=[]):
                head = CurveConditionedUVPathHead if self.shared_curve_enabled else UVPathHead
                self.uv_boundary = head(self.face_dim, self.edge_dim, **uv_options)

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

    def _decode_edges(
        self, features: Tensor, condition: Tensor | None = None,
    ) -> PrimitiveGeometry:
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

    def _decode_shared_stage(
        self, faces: Tensor, edges: Tensor, pairs: CandidatePairs,
    ) -> tuple[DecoderStage, Tensor]:
        # P directed candidates -> K=P/2 shared curves -> P half-edge views.
        canonical = (pairs.indices[:, 0] < pairs.indices[:, 1]).nonzero().flatten()
        reciprocal = pairs.reverse[canonical]
        shared = self.shared_curve(edges[canonical], edges[reciprocal])  # [K,64]
        condition = None
        if self.curve_conditioning_enabled:
            owner_features = faces[pairs.indices[canonical]]  # [K,2,32]
            condition = torch.cat((shared, owner_features.flatten(1)), dim=-1)
        geometry = self._decode_edges(shared, condition)  # one CNN call per curve
        inverse = torch.empty(len(edges), device=edges.device, dtype=torch.long)
        curve_ids = torch.arange(len(canonical), device=edges.device)
        inverse[canonical], inverse[reciprocal] = curve_ids, curve_ids
        reversed_rows = pairs.indices[:, 0] > pairs.indices[:, 1]

        def expand(points: Tensor) -> Tensor:
            directed = points[inverse]
            return torch.where(reversed_rows[:, None, None], directed.flip(1), directed)

        stage = DecoderStage(
            face_features=faces, edge_features=edges,
            face_geometry=self.face_geometry(faces),
            edge_geometry=PrimitiveGeometry(
                local_points=expand(geometry.local_points),
                center_scale=geometry.center_scale[inverse], points=expand(geometry.points),
            ),
            edge_logits=symmetric_pair_logits(self.classifier(edges).squeeze(-1), pairs.reverse),
        )
        return stage, shared[inverse]

    def forward(self, face_latents: Tensor, face_counts: Tensor) -> RefinementOutput:
        # Packed [sum_faces, face_dim] latents; candidate pairs stay within objects.
        face_counts = face_counts.to(device=face_latents.device)
        pairs = enumerate_candidate_pairs(face_counts)

        # The original Transformer runs once. No geometry/GT enters this stage.
        faces = self.face_features(face_latents, face_counts)
        if self.intersection_kind == "local_attention":
            coarse_faces, local_features = self.face_geometry.forward_with_local_features(faces)
            edges = self.inter(faces, local_features, coarse_faces.points, pairs.indices)
            coarse = self._decode_stage(faces, edges, pairs, face_geometry=coarse_faces)
        else:
            edges = self._run_candidates(faces[pairs.indices])
            coarse = self._decode_stage(faces, edges, pairs)

        shared_features = None
        if self.refiner is None:
            delta_face, delta_edge = torch.zeros_like(faces), torch.zeros_like(edges)
            refined = coarse
        else:
            delta_face, delta_edge = self.refiner(coarse, pairs)
            # Reuse exactly the same geometry heads and linear classifier.
            if self.shared_curve_enabled:
                refined, shared_features = self._decode_shared_stage(faces + delta_face, edges + delta_edge, pairs)
            else:
                refined = self._decode_stage(
                    faces + delta_face, edges + delta_edge, pairs, condition_curves=True,
                )
        if self.shared_curve_enabled and self.refiner is None:
            refined, shared_features = self._decode_shared_stage(faces, edges, pairs)

        if self.curve_point_feedback is not None:
            # Final local/global geometry is updated together, so reconstruction,
            # incidence, twin supervision and exported curves all see this result.
            refined.edge_geometry = self.curve_point_feedback(
                refined.edge_geometry, refined.face_geometry,
                refined.edge_features, refined.face_features, pairs.indices,
            )

        edge_uv = None
        if self.uv_boundary is not None:
            if self.shared_curve_enabled:
                owner_uv = self.uv_boundary(
                    refined.edge_features, shared_features, refined.face_features,
                    pairs.indices, refined.edge_geometry.points,
                )  # [P,16,2], each row's first owner
                edge_uv = torch.stack((owner_uv, owner_uv[pairs.reverse].flip(1)), dim=1)
            else:
                edge_uv = self.uv_boundary(refined.edge_features, refined.face_features, pairs.indices)
        return RefinementOutput(
            pairs=pairs, coarse=coarse, refined=refined,
            delta_face=delta_face, delta_edge=delta_edge,
            edge_uv=edge_uv,
        )
