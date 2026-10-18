"""Tensor containers exchanged between VAE decoder stages."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


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
    # Auxiliary parameter-space paths; disabled models retain no new state.
    edge_uv: Tensor | None = None
