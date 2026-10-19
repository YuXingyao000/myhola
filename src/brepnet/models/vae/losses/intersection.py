"""Directed intersection targets, reconstruction and UV boundary supervision."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from ..geometry import closest_polyline_points, closest_surface_points
from ..state import DecoderStage, PrimitiveGeometry, RefinementOutput


@dataclass
class DirectedTargets:
    """GT rows mapped onto the independently constructed candidate set."""

    candidate_rows: Tensor
    edge_ids: Tensor
    reciprocal_rows: Tensor
    reciprocal_edge_ids: Tensor
    labels: Tensor


def _mean_l1(prediction: Tensor, target: Tensor) -> Tensor:
    if not prediction.numel():
        return prediction.sum() * 0.0
    return F.l1_loss(prediction.float(), target.float())


def balanced_binary_cross_entropy(logits: Tensor, labels: Tensor) -> Tensor:
    """Give each present class equal weight, including empty-candidate batches."""
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
    """Map validated GT connection rows onto the decoder's ordered face pairs."""
    pairs = output.pairs
    device = pairs.indices.device
    counts = batch["face_counts"].to(device=device)
    face_count = int(counts.sum())
    candidate_count = len(pairs.indices)
    # Flatten [face_i, face_j] into keys; storage order need not match GT rows.
    candidate_keys = pairs.indices[:, 0] * face_count + pairs.indices[:, 1]
    sorted_keys, sorted_rows = candidate_keys.sort()
    connectivity = batch["edge_face_connectivity"].to(device=device)
    edge_ids = connectivity[:, 0].long()
    owner_pairs = connectivity[:, 1:].long()
    if len(connectivity):
        target_keys = owner_pairs[:, 0] * face_count + owner_pairs[:, 1]
        locations = torch.searchsorted(sorted_keys, target_keys)
        candidate_rows = sorted_rows[locations]
    else:
        candidate_rows = torch.empty(0, device=device, dtype=torch.long)

    pair_to_edge = torch.full((candidate_count,), -1, device=device, dtype=torch.long)
    pair_to_edge[candidate_rows] = edge_ids
    reciprocal_rows = pairs.reverse[candidate_rows]
    reciprocal_edge_ids = pair_to_edge[reciprocal_rows]
    labels = output.refined.edge_logits.new_zeros(candidate_count)
    labels[candidate_rows] = 1.0
    return DirectedTargets(candidate_rows, edge_ids, reciprocal_rows, reciprocal_edge_ids, labels)


class IntersectionLoss(nn.Module):
    """Explicit raw losses, weighted training terms, and detached diagnostics."""

    def __init__(self, cfg: dict, uv_boundary_enabled: bool = False, shared_curve_enabled: bool = False):
        super().__init__()
        self.uv_boundary_enabled = uv_boundary_enabled
        self.shared_curve_enabled = shared_curve_enabled
        self.uv_weight = float(cfg.get("uv_weight", 0.01))
        self.uv_anchor_weight = float(cfg.get("uv_anchor_weight", 0.1))
        self.coarse_weight          = float(cfg.get("coarse_weight", 0.25))
        self.classification_weight  = float(cfg.get("classification_weight", 0.1))
        self.feature_weight         = float(cfg.get("feature_weight", 1.0))
        self.encoder_edge_weight    = float(cfg.get("encoder_edge_weight", 1.0))
        self.xyz_weight             = float(cfg.get("xyz_weight", 1.0))
        self.incidence_weight       = float(cfg.get("incidence_weight", 0.1))
        self.twin_weight            = float(cfg.get("twin_weight", 0.1))
        self.geometry_tolerance     = float(cfg.get("geometry_tolerance", 1e-5))
        self.geometry_chunk_size    = int(cfg.get("geometry_chunk_size", 32))

    @staticmethod
    def _shared_curve_batch(output: RefinementOutput, batch: Mapping, targets: DirectedTargets) -> dict:
        """One GT parameterization per pair, preserving its two UV chart branches.

        Only final-head targets change. Original directed targets still train
        the coarse decoder and encoder teacher. No GT enters the decoder.
        """
        rows = targets.candidate_rows
        owners = output.pairs.indices[rows]
        reversed_rows = owners[:, 0] > owners[:, 1]
        representatives = torch.where(reversed_rows, output.pairs.reverse[rows], rows)
        row_lookup = torch.full_like(output.pairs.reverse, -1)
        row_lookup[rows] = torch.arange(len(rows), device=rows.device)
        source_rows = row_lookup[representatives]
        source_edges = targets.edge_ids[source_rows]
        aligned = dict(batch)
        for name in ("edge_points", "edge_norm", "edge_bbox"):
            values = batch[name][source_edges]
            if name != "edge_bbox":
                values = torch.where(reversed_rows[:, None, None], values.flip(1), values)
            aligned[name] = batch[name].clone()
            aligned[name][targets.edge_ids] = values
        if "edge_uv_targets" in batch:
            aligned_uv = {}
            for name, values in batch["edge_uv_targets"].items():
                selected = values[source_rows]
                mask = reversed_rows.reshape(-1, *([1] * (selected.ndim - 1)))
                aligned_uv[name] = torch.where(mask, selected.flip((1, 2)), selected)
            aligned["edge_uv_targets"] = aligned_uv
        return aligned

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

    def _uv_boundary(
        self, output: RefinementOutput, batch: Mapping, targets: DirectedTargets,
    ) -> dict[str, Tensor]:
        """Anchor geometry at fixed GT UV; predicted UV cannot slide the anchor.

        Sidecars follow connectivity ROW order, while edge_points use edge IDs.
        Bilinear interpolation reads [v,u] grids with normalized (u,v) inputs.
        The GT residual removes the exact same grid interpolation discrepancy.
        """
        labels = batch["edge_uv_targets"]
        rows = len(batch["edge_face_connectivity"])
        # UV [R,2,16,2] and valid [R,2,16] retain connectivity-row order.
        uv, valid = labels["uv"], labels["valid"]
        prediction = output.edge_uv[targets.candidate_rows].float()
        geometry = output.refined
        curves = geometry.edge_geometry.points[targets.candidate_rows].float()
        surfaces = geometry.face_geometry.points.float()
        coverage = valid.float().mean() if valid.numel() else prediction.detach().sum() * 0
        if not valid.any():
            zero = prediction.sum() * 0 + curves.sum() * 0 + surfaces.sum() * 0
            return {"uv": zero, "uv_anchor": zero, "uv_valid_fraction": coverage.detach()}
        # Invalid positions are sanitized before grid_sample, then excluded.
        coordinates = torch.where(valid[..., None], uv.float(), 0.0).detach()
        owners = output.pairs.indices[targets.candidate_rows]

        def sample(grid: Tensor) -> Tensor:
            values = grid[owners].reshape(rows * 2, *grid.shape[1:]).permute(0, 3, 1, 2)
            locations = (coordinates * 2 - 1).reshape(rows * 2, 1, 16, 2)
            sampled = F.grid_sample(values, locations, mode="bilinear", padding_mode="border", align_corners=True)
            return sampled.squeeze(2).transpose(1, 2).reshape(rows, 2, 16, 3)

        with torch.autocast(device_type=curves.device.type, enabled=False):
            predicted_surface = sample(surfaces)
            with torch.no_grad():
                gt_curves = batch["edge_points"][targets.edge_ids, :, :3].float()
                residual = gt_curves[:, None] - sample(batch["face_points"][..., :3].float())
            discrepancy = curves[:, None] - predicted_surface - residual
            return {
                "uv": (prediction - coordinates).abs()[valid].mean(),
                "uv_anchor": discrepancy.abs()[valid].mean(),
                "uv_valid_fraction": coverage.detach(),
            }

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
        refined_batch = self._shared_curve_batch(output, batch, targets) if self.shared_curve_enabled else batch
        losses, weighted = {}, {}
        for name, stage, stage_weight in (
            ("coarse", output.coarse, self.coarse_weight),
            ("refined", output.refined, 1.0),
        ):
            stage_batch = batch if name == "coarse" else refined_batch
            for term, value in self._reconstruction(stage, stage_batch, targets).items():
                key = f"{name}_{term}"
                weight = stage_weight * (self.classification_weight if term == "classification" else 1.0)
                losses[key] = value
                weighted[f"weighted_{key}"] = weight * value

        losses.update({
            "edge_feature": _mean_l1(output.coarse.edge_features[targets.candidate_rows], teacher_edge_features[targets.edge_ids].detach()),
            "encoder_edge_local": _mean_l1(teacher_geometry.local_points, batch["edge_norm"]),
            "encoder_edge_bbox": _mean_l1(teacher_geometry.center_scale, batch["edge_bbox"]),
            "face_xyz": _mean_l1(output.refined.face_geometry.points, batch["face_points"][..., :3]),
            "edge_xyz": _mean_l1(output.refined.edge_geometry.points[targets.candidate_rows], refined_batch["edge_points"][targets.edge_ids, :, :3]),
            "incidence": self._incidence(output, refined_batch, targets),
            "twin": self._twin(output, refined_batch, targets),
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
        if self.uv_boundary_enabled:
            losses.update(self._uv_boundary(output, refined_batch, targets))
            weights.update({"uv": self.uv_weight, "uv_anchor": self.uv_anchor_weight})
        weighted.update({f"weighted_{name}": weight * losses[name] for name, weight in weights.items()})
        losses.update(weighted)
        losses["total_loss"] = torch.stack(tuple(weighted.values())).sum()
        losses.update(self._diagnostics(output, targets))
        return losses
