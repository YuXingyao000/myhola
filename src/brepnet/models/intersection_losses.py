"""Supervision for intersection refinement; GT never constructs decoder edges."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.brepnet.models.intersection_geometry import (
    closest_polyline_points,
    closest_surface_points,
)
from src.brepnet.models.intersection_state import (
    DecoderStage,
    PrimitiveGeometry,
    RefinementOutput,
)


@dataclass
class DirectedTargets:
    """GT rows mapped onto the independently constructed candidate set."""

    candidate_rows: Tensor
    edge_ids: Tensor
    reciprocal_rows: Tensor
    reciprocal_edge_ids: Tensor
    labels: Tensor


def _require_integer(value: Tensor, name: str) -> None:
    if value.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must be an integer tensor")


def _require_shape(value: Tensor, expected: tuple[int, ...], name: str) -> None:
    if tuple(value.shape) != tuple(expected):
        raise ValueError(f"{name} must have shape {tuple(expected)}, got {tuple(value.shape)}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains nonfinite values")


def _mean_l1(prediction: Tensor, target: Tensor) -> Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"Reconstruction shape mismatch: {prediction.shape} vs {target.shape}")
    if not prediction.numel():
        return prediction.sum() * 0.0
    return F.l1_loss(prediction.float(), target.float())


def balanced_binary_cross_entropy(logits: Tensor, labels: Tensor) -> Tensor:
    """Give each present class equal weight, including empty-candidate batches."""
    if logits.shape != labels.shape:
        raise ValueError("Classification logits and labels must have identical shapes")
    if not logits.numel():
        return logits.sum() * 0.0
    losses = F.binary_cross_entropy_with_logits(logits.float(), labels.float(), reduction="none")
    positive = labels.bool()
    class_losses = []
    if positive.any():
        class_losses.append(losses[positive].mean())
    if (~positive).any():
        class_losses.append(losses[~positive].mean())
    return torch.stack(class_losses).mean()


def _map_directed_targets(output: RefinementOutput, batch: Mapping) -> DirectedTargets:
    pairs = output.pairs
    device = pairs.indices.device
    counts = batch["face_counts"].to(device=device)
    _require_integer(counts, "face_counts")
    if counts.ndim != 1 or not len(counts) or (counts <= 0).any():
        raise ValueError("face_counts must contain positive counts for a nonempty batch")
    if not torch.equal(counts, pairs.face_counts.to(device=device)):
        raise ValueError("Decoder and target face_counts disagree")
    face_count = int(counts.sum())
    edge_count = len(batch["edge_points"])
    candidate_count = len(pairs.indices)
    _require_integer(pairs.indices, "candidate indices")
    _require_integer(pairs.reverse, "candidate reverse")
    _require_integer(pairs.object_ids, "candidate object_ids")
    _require_shape(pairs.indices, (candidate_count, 2), "candidate indices")
    _require_shape(pairs.reverse, (candidate_count,), "candidate reverse")
    _require_shape(pairs.object_ids, (candidate_count,), "candidate object_ids")
    if candidate_count != int((counts * (counts - 1)).sum()):
        raise ValueError("Decoder candidates must include all ordered within-object face pairs")
    face_objects = torch.repeat_interleave(torch.arange(len(counts), device=device), counts)
    if candidate_count:
        if (pairs.indices < 0).any() or (pairs.indices >= face_count).any():
            raise ValueError("Candidate face index is outside the packed face array")
        if (pairs.indices[:, 0] == pairs.indices[:, 1]).any():
            raise ValueError("Candidate seam/self pairs are unsupported")
        owners = face_objects[pairs.indices]
        if (owners[:, 0] != owners[:, 1]).any() or not torch.equal(owners[:, 0], pairs.object_ids):
            raise ValueError("Candidate pairs cross object boundaries or have wrong object_ids")
        if (pairs.reverse < 0).any() or (pairs.reverse >= candidate_count).any():
            raise ValueError("Candidate reciprocal index is outside the candidate array")
        if not torch.equal(pairs.indices[pairs.reverse], pairs.indices.flip(-1)):
            raise ValueError("Candidate reciprocal mapping is incorrect")

    candidate_keys = pairs.indices[:, 0] * face_count + pairs.indices[:, 1]
    sorted_keys, sorted_rows = candidate_keys.sort()
    if candidate_count > 1 and (sorted_keys[1:] == sorted_keys[:-1]).any():
        raise ValueError("Duplicate directed candidate face pair")

    connectivity = batch["edge_face_connectivity"].to(device=device)
    _require_integer(connectivity, "edge_face_connectivity")
    if connectivity.ndim != 2 or connectivity.shape[1] != 3:
        raise ValueError("edge_face_connectivity must have shape [E,3]")
    edge_ids = connectivity[:, 0].long()
    owner_pairs = connectivity[:, 1:].long()
    if len(connectivity) != edge_count:
        raise ValueError("GT connectivity must cover every edge sample exactly once")
    if len(connectivity):
        if (edge_ids < 0).any() or (edge_ids >= edge_count).any():
            raise ValueError("GT edge index is outside the edge sample array")
        if len(edge_ids.unique()) != edge_count:
            raise ValueError("Duplicate GT edge index or missing edge coverage")
        if (owner_pairs < 0).any() or (owner_pairs >= face_count).any():
            raise ValueError("GT face index is outside the packed face array")
        if (owner_pairs[:, 0] == owner_pairs[:, 1]).any():
            raise ValueError("GT seam/self pairs are unsupported")
        if (face_objects[owner_pairs[:, 0]] != face_objects[owner_pairs[:, 1]]).any():
            raise ValueError("GT connectivity crosses object boundaries")
        target_keys = owner_pairs[:, 0] * face_count + owner_pairs[:, 1]
        if len(target_keys.unique()) != len(target_keys):
            raise ValueError("Duplicate GT directed face pair")
        locations = torch.searchsorted(sorted_keys, target_keys)
        if not candidate_count or (locations >= candidate_count).any():
            raise ValueError("GT directed pair is missing from decoder candidates")
        if not torch.equal(sorted_keys[locations], target_keys):
            raise ValueError("GT directed pair is missing from decoder candidates")
        candidate_rows = sorted_rows[locations]
    else:
        candidate_rows = torch.empty(0, device=device, dtype=torch.long)

    pair_to_edge = torch.full((candidate_count,), -1, device=device, dtype=torch.long)
    pair_to_edge[candidate_rows] = edge_ids
    reciprocal_rows = pairs.reverse[candidate_rows]
    reciprocal_edge_ids = pair_to_edge[reciprocal_rows]
    if (reciprocal_edge_ids < 0).any():
        raise ValueError("Every GT directed edge must have a reciprocal edge")
    labels = output.refined.edge_logits.new_zeros(candidate_count)
    labels[candidate_rows] = 1.0
    return DirectedTargets(candidate_rows, edge_ids, reciprocal_rows, reciprocal_edge_ids, labels)


class IntersectionLoss(nn.Module):
    """Explicit raw losses, weighted training terms, and detached diagnostics."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.coarse_weight          = float(cfg.get("coarse_weight", 0.25))
        self.classification_weight  = float(cfg.get("classification_weight", 0.1))
        self.feature_weight         = float(cfg.get("feature_weight", 1.0))
        self.encoder_edge_weight    = float(cfg.get("encoder_edge_weight", 1.0))
        self.xyz_weight             = float(cfg.get("xyz_weight", 1.0))
        self.incidence_weight       = float(cfg.get("incidence_weight", 0.1))
        self.twin_weight            = float(cfg.get("twin_weight", 0.1))
        self.geometry_tolerance     = float(cfg.get("geometry_tolerance", 1e-5))
        self.geometry_chunk_size    = int(cfg.get("geometry_chunk_size", 32))
        for name in (
            "coarse_weight", "classification_weight", "feature_weight", "encoder_edge_weight",
            "xyz_weight", "incidence_weight", "twin_weight", "geometry_tolerance",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.geometry_chunk_size <= 0:
            raise ValueError("geometry_chunk_size must be positive")

    def _validate_geometry(
        self,
        output: RefinementOutput,
        batch: Mapping,
        teacher_edge_features: Tensor,
        teacher_geometry: PrimitiveGeometry,
    ) -> None:
        face_count = int(output.pairs.face_counts.sum())
        edge_count = len(batch["edge_points"])
        candidate_count = len(output.pairs.indices)
        if output.coarse.face_features.ndim != 2 or output.coarse.edge_features.ndim != 2:
            raise ValueError("Decoder features must be rank-two tensors")
        face_feature_dim = output.coarse.face_features.shape[-1]
        edge_feature_dim = output.coarse.edge_features.shape[-1]
        face_shape, edge_shape = batch["face_norm"].shape, batch["edge_norm"].shape
        if len(face_shape) != 4 or face_shape[0] != face_count or face_shape[-1] != 6:
            raise ValueError("face_norm must have shape [sum_faces,U,V,6]")
        if len(edge_shape) != 3 or edge_shape[0] != edge_count or edge_shape[-1] != 6:
            raise ValueError("edge_norm must have shape [GT_edges,L,6]")
        if min(face_shape[1:3]) < 2 or edge_shape[1] < 2:
            raise ValueError("Geometry must contain at least 2x2 face samples and 2 edge samples")
        expected = {
            "face_norm": face_shape, "edge_norm": edge_shape,
            "face_bbox": (face_count, 4), "edge_bbox": (edge_count, 4),
            "face_points": face_shape, "edge_points": edge_shape,
        }
        for name, shape in expected.items():
            _require_shape(batch[name], shape, name)
        for name, stage in (("coarse", output.coarse), ("refined", output.refined)):
            _require_shape(stage.face_geometry.local_points, face_shape, f"{name} face local")
            _require_shape(stage.face_geometry.center_scale, (face_count, 4), f"{name} face bbox")
            _require_shape(stage.face_geometry.points, (*face_shape[:-1], 3), f"{name} face XYZ")
            _require_shape(stage.edge_geometry.local_points, (candidate_count, *edge_shape[1:]), f"{name} edge local")
            _require_shape(stage.edge_geometry.center_scale, (candidate_count, 4), f"{name} edge bbox")
            _require_shape(stage.edge_geometry.points, (candidate_count, edge_shape[1], 3), f"{name} edge XYZ")
            _require_shape(stage.face_features, (face_count, face_feature_dim), f"{name} face features")
            _require_shape(stage.edge_features, (candidate_count, edge_feature_dim), f"{name} edge features")
            _require_shape(stage.edge_logits, (candidate_count,), f"{name} edge logits")
        _require_shape(teacher_edge_features, (edge_count, edge_feature_dim), "encoder edge features")
        _require_shape(teacher_geometry.local_points, edge_shape, "encoder edge local")
        _require_shape(teacher_geometry.center_scale, (edge_count, 4), "encoder edge bbox")
        _require_shape(teacher_geometry.points, (*edge_shape[:-1], 3), "encoder edge XYZ")
        _require_shape(output.delta_face, (face_count, face_feature_dim), "delta face")
        _require_shape(output.delta_edge, (candidate_count, edge_feature_dim), "delta edge")

    def _reconstruction(self, stage: DecoderStage, batch: Mapping, targets: DirectedTargets) -> dict[str, Tensor]:
        return {
            "face_local": _mean_l1(stage.face_geometry.local_points, batch["face_norm"]),
            "face_bbox": _mean_l1(stage.face_geometry.center_scale, batch["face_bbox"]),
            "edge_local": _mean_l1(stage.edge_geometry.local_points[targets.candidate_rows], batch["edge_norm"][targets.edge_ids]),
            "edge_bbox": _mean_l1(stage.edge_geometry.center_scale[targets.candidate_rows], batch["edge_bbox"][targets.edge_ids]),
            "classification": balanced_binary_cross_entropy(stage.edge_logits, targets.labels),
        }

    def _incidence(self, output: RefinementOutput, batch: Mapping, targets: DirectedTargets) -> Tensor:
        geometry = output.refined
        if not len(targets.candidate_rows):
            return (geometry.edge_geometry.points.sum() + geometry.face_geometry.points.sum()) * 0.0
        curves = geometry.edge_geometry.points[targets.candidate_rows].float()
        owners = output.pairs.indices[targets.candidate_rows]
        gt_curves = batch["edge_points"][targets.edge_ids, :, :3].float()
        side_losses = []
        for side in range(2):
            _, distances = closest_surface_points(
                curves, geometry.face_geometry.points[owners[:, side]].float(), self.geometry_chunk_size,
            )
            with torch.no_grad():
                _, baseline = closest_surface_points(
                    gt_curves, batch["face_points"][owners[:, side], ..., :3].float(), self.geometry_chunk_size,
                )
            side_losses.append(F.relu(distances - baseline.detach() - self.geometry_tolerance).mean())
        return torch.stack(side_losses).mean()

    def _twin(self, output: RefinementOutput, batch: Mapping, targets: DirectedTargets) -> Tensor:
        curves = output.refined.edge_geometry.points.float()
        if not len(targets.candidate_rows):
            return curves.sum() * 0.0
        gt_curves = batch["edge_points"][targets.edge_ids, :, :3].float()
        gt_twins = batch["edge_points"][targets.reciprocal_edge_ids, :, :3].float()
        with torch.no_grad():
            closest, segments, fractions = closest_polyline_points(gt_curves, gt_twins)
            baseline = (gt_curves - closest).abs()
        twins = curves[targets.reciprocal_rows]
        gather_indices = segments[..., None].expand(-1, -1, 3)
        starts = twins.gather(1, gather_indices)
        ends = twins.gather(1, gather_indices + 1)
        mapped_twins = starts + fractions[..., None] * (ends - starts)
        residuals = (curves[targets.candidate_rows] - mapped_twins).abs()
        # Every directed GT row participates, hence both correspondence directions.
        return F.relu(residuals - baseline.detach() - self.geometry_tolerance).mean()

    @staticmethod
    def _diagnostics(output: RefinementOutput, targets: DirectedTargets) -> dict[str, Tensor]:
        result = {}
        with torch.no_grad():
            positives = targets.labels.bool()
            for name, stage in (("coarse", output.coarse), ("refined", output.refined)):
                probabilities = stage.edge_logits.float().sigmoid()
                selected = probabilities > 0.5
                true_positive = (selected & positives).sum().float()
                result[f"{name}_precision"] = true_positive / selected.sum().clamp_min(1)
                result[f"{name}_recall"] = true_positive / positives.sum().clamp_min(1)
                result[f"{name}_positive_probability"] = (
                    probabilities[positives].mean() if positives.any() else probabilities.sum() * 0.0
                )
            result["delta_face_mean"] = output.delta_face.float().abs().mean()
            result["delta_edge_mean"] = (
                output.delta_edge.float().abs().mean() if output.delta_edge.numel() else output.delta_edge.sum() * 0.0
            )
        return result

    def forward(
        self,
        output: RefinementOutput,
        batch: Mapping,
        teacher_edge_features: Tensor,
        teacher_geometry: PrimitiveGeometry,
        kl_loss: Tensor,
        encoder_frozen: bool,
    ) -> dict[str, Tensor]:
        targets = _map_directed_targets(output, batch)
        self._validate_geometry(output, batch, teacher_edge_features, teacher_geometry)
        _require_shape(kl_loss, (), "kl_loss")
        losses, weighted = {}, {}
        for name, stage, stage_weight in (
            ("coarse", output.coarse, self.coarse_weight),
            ("refined", output.refined, 1.0),
        ):
            for term, value in self._reconstruction(stage, batch, targets).items():
                key = f"{name}_{term}"
                weight = stage_weight * (self.classification_weight if term == "classification" else 1.0)
                losses[key] = value
                weighted[f"weighted_{key}"] = weight * value

        losses.update({
            "edge_feature": _mean_l1(output.coarse.edge_features[targets.candidate_rows], teacher_edge_features[targets.edge_ids].detach()),
            "encoder_edge_local": _mean_l1(teacher_geometry.local_points, batch["edge_norm"]),
            "encoder_edge_bbox": _mean_l1(teacher_geometry.center_scale, batch["edge_bbox"]),
            "face_xyz": _mean_l1(output.refined.face_geometry.points, batch["face_points"][..., :3]),
            "edge_xyz": _mean_l1(output.refined.edge_geometry.points[targets.candidate_rows], batch["edge_points"][targets.edge_ids, :, :3]),
            "incidence": self._incidence(output, batch, targets),
            "twin": self._twin(output, batch, targets),
            "kl_loss": kl_loss.detach() if encoder_frozen else kl_loss,
        })
        weights = {
            "edge_feature": self.feature_weight,
            "encoder_edge_local": self.encoder_edge_weight,
            "encoder_edge_bbox": self.encoder_edge_weight,
            "face_xyz": self.xyz_weight,
            "edge_xyz": self.xyz_weight,
            "incidence": self.incidence_weight,
            "twin": self.twin_weight,
            "kl_loss": 0.0 if encoder_frozen else 1.0,
        }
        weighted.update({f"weighted_{name}": weight * losses[name] for name, weight in weights.items()})
        losses.update(weighted)
        losses["total_loss"] = torch.stack(tuple(weighted.values())).sum()
        losses.update(self._diagnostics(output, targets))
        return losses
