"""Legacy geometry heads with one explicit intersection refinement pass."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from src.brepnet.models.intersection_refiner import IntersectionRefiner
from src.brepnet.models.intersection_state import (
    CandidatePairs,
    DecoderStage,
    PrimitiveGeometry,
    RefinementOutput,
)

if TYPE_CHECKING:
    from src.brepnet.models.vae import AutoEncoder_light


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


class PrimitiveDecoder(nn.Module):
    """Retained local point and center/scale heads, with explicit global XYZ."""

    def __init__(
        self,
        points_decoder: nn.Module,
        bbox_decoder: nn.Module,
        spatial_shape: tuple[int, ...],
        point_channels: int,
    ):
        super().__init__()
        self.points_decoder = points_decoder
        self.bbox_decoder = bbox_decoder
        self.spatial_shape = spatial_shape
        self.point_channels = point_channels

    def forward(self, features: Tensor) -> PrimitiveGeometry:
        if features.shape[0] == 0:
            return PrimitiveGeometry(
                local_points=features.new_empty((0, *self.spatial_shape, self.point_channels)),
                center_scale=features.new_empty((0, 4)),
                points=features.new_empty((0, *self.spatial_shape, 3), dtype=torch.float32),
            )
        local_points = self.points_decoder(features)
        center_scale = self.bbox_decoder(features)
        with torch.autocast(device_type=features.device.type, enabled=False):
            broadcast_shape = (features.shape[0],) + (1,) * len(self.spatial_shape)
            center = center_scale[:, :3].float().reshape(*broadcast_shape, 3)
            scale = center_scale[:, 3:].float().reshape(*broadcast_shape, 1)
            points = local_points[..., :3].float() * scale + center
        return PrimitiveGeometry(local_points=local_points, center_scale=center_scale, points=points)


class IntersectionDecoder(nn.Module):
    """Preserve legacy modules and add coarse→read/exchange/write→refined.

    Direction ij means the original inter([F_i, F_j]) feature, retaining its
    original directional curve/normal semantics. A reciprocal ji candidate is
    always present. Only connection logits are made symmetric.

    cfg is the refinement mapping itself. The legacy model is not registered
    here: only its named decoder modules are transferred, preserving weights.
    """

    def __init__(self, legacy: AutoEncoder_light, cfg: dict):
        super().__init__()
        self.face_dim = legacy.face_feature_dim
        self.edge_dim = legacy.edge_feature_dim
        self.chunk_size = int(cfg["candidate_chunk_size"])
        self.checkpoint_candidates = bool(cfg["checkpoint_candidates"])
        self.refinement_enabled = bool(cfg["enabled"])
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
        )
        self.inter = legacy.inter
        self.classifier = legacy.classifier
        self.refiner = (
            IntersectionRefiner(self.face_dim, self.edge_dim, cfg)
            if self.refinement_enabled else None
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

    def _decode_edge_chunk(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        geometry = self.edge_geometry(features)
        return geometry.local_points, geometry.center_scale, geometry.points

    def _decode_edges(self, features: Tensor) -> PrimitiveGeometry:
        if features.shape[0] == 0:
            return self.edge_geometry(features)
        chunks = []
        for chunk in features.split(self.chunk_size):
            if self.checkpoint_candidates and self.training and torch.is_grad_enabled():
                geometry = checkpoint(self._decode_edge_chunk, chunk, use_reentrant=False)
            else:
                geometry = self._decode_edge_chunk(chunk)
            chunks.append(geometry)
        return PrimitiveGeometry(
            local_points=torch.cat([chunk[0] for chunk in chunks]),
            center_scale=torch.cat([chunk[1] for chunk in chunks]),
            points=torch.cat([chunk[2] for chunk in chunks]),
        )

    def _decode_stage(self, faces: Tensor, edges: Tensor, pairs: CandidatePairs) -> DecoderStage:
        logits = self.classifier(edges).squeeze(-1)
        return DecoderStage(
            face_features=faces,
            edge_features=edges,
            face_geometry=self.face_geometry(faces),
            edge_geometry=self._decode_edges(edges),
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
        edges = self._run_candidates(faces[pairs.indices])
        coarse = self._decode_stage(faces, edges, pairs)

        if self.refiner is None:
            delta_face, delta_edge = torch.zeros_like(faces), torch.zeros_like(edges)
            refined = coarse
        else:
            delta_face, delta_edge = self.refiner(coarse, pairs)
            # Reuse exactly the same geometry heads and linear classifier.
            refined = self._decode_stage(faces + delta_face, edges + delta_edge, pairs)

        return RefinementOutput(
            pairs=pairs, coarse=coarse, refined=refined,
            delta_face=delta_face, delta_edge=delta_edge,
        )
