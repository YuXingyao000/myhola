"""Geometry reads and residual message exchange for intersection refinement."""

from __future__ import annotations

import torch
from torch import nn, Tensor

from ..geometry import closest_surface_points
from ..state import CandidatePairs, DecoderStage, PrimitiveGeometry


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
