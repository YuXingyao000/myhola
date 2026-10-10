"""Set matching and losses for shared boundary queries (no GT decoding input)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from torch.nn import functional as F


def boundary_costs(curves: Tensor, ownership_logits: Tensor, target: Mapping) -> tuple[Tensor, Tensor]:
    """Return [queries, targets] reversal/side-swap invariant L1 and NLL costs."""
    curves = curves.float()
    truth = target["curves"].to(device=curves.device, dtype=torch.float32)
    pairs = target["face_pairs"].to(device=curves.device, dtype=torch.long)
    direct = (curves[:, None] - truth[None]).abs().mean(dim=(-1, -2))
    reverse = (curves[:, None] - truth.flip(1)[None]).abs().mean(dim=(-1, -2))
    geometry = torch.minimum(direct, reverse)
    closed = target.get("is_closed")
    if closed is not None:
        # Closed targets include the repeated endpoint. Their sample origin is
        # arbitrary, so search both directions and every discrete cyclic origin.
        closed = closed.to(device=curves.device, dtype=torch.bool)
        closed_indices = closed.nonzero(as_tuple=True)[0]
        if len(closed_indices):
            unique = truth[closed_indices, :-1]
            positions = torch.arange(unique.shape[1], device=unique.device)
            forward_indices = (positions[None, :] - positions[:, None]) % unique.shape[1]
            # Preserve the original order: forward shifts 0..L-2, then reverse
            # shifts 0..L-2. The order matters for min() gradients at exact ties.
            indices = torch.cat((forward_indices, unique.shape[1] - 1 - forward_indices), dim=0)
            indices = torch.cat((indices, indices[:, :1]), dim=1)
            variants = unique[:, indices]
            closed_cost = (curves[:, None, None] - variants[None]).abs().mean(dim=(-1, -2)).min(dim=-1).values
            geometry = geometry.clone()
            geometry[:, closed_indices] = closed_cost
    log_prob = ownership_logits.float().log_softmax(-1)
    direct_owner = -(log_prob[:, 0, pairs[:, 0]] + log_prob[:, 1, pairs[:, 1]]) / 2
    swap_owner = -(log_prob[:, 0, pairs[:, 1]] + log_prob[:, 1, pairs[:, 0]]) / 2
    return geometry, torch.minimum(direct_owner, swap_owner)


def validate_boundary_target(target: Mapping, faces: int, capacity: int, resolution: int, name: str) -> None:
    curves, pairs = target["curves"], target["face_pairs"]
    if curves.ndim != 3 or curves.shape[1:] != (resolution, 3):
        raise ValueError(f"{name}: boundary curves must have shape [K, {resolution}, 3]")
    if pairs.shape != (len(curves), 2) or pairs.dtype != torch.long:
        raise ValueError(f"{name}: face_pairs must be int64 [K, 2]")
    if len(curves) > capacity:
        raise ValueError(f"{name}: {len(curves)} target boundaries exceed query capacity {capacity}")
    if not torch.isfinite(curves).all():
        raise ValueError(f"{name}: nonfinite target boundary geometry")
    if "is_closed" in target:
        closed = target["is_closed"]
        if closed.shape != (len(curves),) or closed.dtype != torch.bool:
            raise ValueError(f"{name}: is_closed must be bool [K]")
    if pairs.numel() and (pairs.min() < 0 or pairs.max() >= faces):
        raise ValueError(f"{name}: boundary face indices outside [0, {faces})")
    if pairs.numel() and (pairs[:, 0] == pairs[:, 1]).any():
        raise ValueError(f"{name}: seam targets are unsupported by shared_boundary_v1")
    if len(pairs) and len(torch.unique(pairs.sort(dim=-1).values, dim=0)) != len(pairs):
        raise ValueError(f"{name}: multiple target boundaries for one face pair are unsupported")


def _match_costs(
    geometry: Tensor, ownership: Tensor, curve_weight: float, ownership_weight: float
) -> tuple[Tensor, Tensor]:
    """Solve detached matching while retaining the original differentiable costs."""
    with torch.no_grad():
        cost = curve_weight * geometry + ownership_weight * ownership
        if not torch.isfinite(cost).all():
            raise ValueError("Nonfinite boundary matching cost")
        rows, columns = linear_sum_assignment(cost.detach().float().cpu().numpy())
    return (
        torch.as_tensor(rows, device=geometry.device, dtype=torch.long),
        torch.as_tensor(columns, device=geometry.device, dtype=torch.long),
    )


def match_boundaries(
    curves: Tensor, ownership_logits: Tensor, target: Mapping, curve_weight: float, ownership_weight: float
) -> tuple[Tensor, Tensor]:
    if len(target["curves"]) > len(curves):
        raise ValueError("Target boundary count exceeds query capacity")
    with torch.no_grad():
        geometry, ownership = boundary_costs(curves, ownership_logits, target)
        return _match_costs(geometry, ownership, curve_weight, ownership_weight)


def surface_boundary_consistency(surfaces: Tensor, curves: Tensor, ownership_logits: Tensor) -> Tensor:
    """Expected squared distance to each side's sampled surface (no UV claim).

    Distances are evaluated per face BEFORE weighting by ownership. Measuring
    distance to an average of unrelated surfaces would allow ghost boundaries.
    """
    if not len(curves):
        return curves.sum() * 0
    face_count = len(surfaces)
    resolution = curves.shape[1]
    distances = torch.cdist(curves.float().flatten(0, 1), surfaces.float().reshape(-1, 3)).square()
    distances = distances.reshape(len(curves), resolution, face_count, -1).min(dim=-1).values.mean(dim=1)
    return (ownership_logits.float().softmax(-1) * distances[:, None, :]).sum(-1).mean()


def joint_boundary_loss(
    state: Mapping[str, Tensor],
    targets: Sequence[Mapping],
    target_faces: Tensor,
    weights: Mapping[str, float],
    match_weights: Mapping[str, float],
    no_boundary_weight: float = 0.1,
    existence_threshold: float = 0.5,
    sample_names: Sequence[str] | None = None,
) -> dict[str, Tensor]:
    batch = len(state["face_counts"])
    if len(targets) != batch:
        raise ValueError("boundary_targets must contain one dictionary per object")
    if target_faces.shape != state["face_points"][state["face_valid"]].shape:
        raise ValueError("Target faces must be flat [sum_faces, 16, 16, 3] in unchanged face order")
    if not torch.isfinite(target_faces).all():
        raise ValueError("Target face geometry contains nonfinite values")
    face_loss = F.l1_loss(state["face_points"][state["face_valid"]].float(), target_faces.float())
    zero = state["curves"].sum() * 0
    curve_losses, owner_losses, existence_losses, consistency_losses = [], [], [], []
    count_errors, prediction_counts, ownership_accuracy, duplicate_counts, invalid_counts = [], [], [], [], []
    for item, target in enumerate(targets):
        count = int(state["face_counts"][item])
        curves = state["curves"][item]
        ownership = state["ownership_logits"][item, :, :, :count]
        name = str(sample_names[item]) if sample_names is not None else f"batch item {item}"
        validate_boundary_target(target, count, len(curves), curves.shape[1], name)
        geometry, owner = boundary_costs(curves, ownership, target)
        rows, columns = _match_costs(geometry, owner, **{
            "curve_weight": float(match_weights["curve"]),
            "ownership_weight": float(match_weights["ownership"]),
        })
        exists = torch.zeros_like(state["existence_logits"][item])
        exists[rows] = 1
        existence_weight = torch.where(exists.bool(), 1.0, no_boundary_weight)
        existence_losses.append((F.binary_cross_entropy_with_logits(
            state["existence_logits"][item], exists, reduction="none"
        ) * existence_weight).mean())
        if len(rows):
            curve_losses.append(geometry[rows, columns].mean())
            owner_losses.append(owner[rows, columns].mean())
            consistency_losses.append(surface_boundary_consistency(
                state["face_points"][item, :count], curves[rows], ownership[rows]
            ))
            predicted_pair = ownership[rows].argmax(-1).sort(-1).values
            target_pair = target["face_pairs"].to(curves.device)[columns].sort(-1).values
            ownership_accuracy.append((predicted_pair == target_pair).all(-1).float().mean())
        else:
            curve_losses.append(zero)
            owner_losses.append(zero)
            consistency_losses.append(zero)
            ownership_accuracy.append(zero.detach())
        with torch.no_grad():
            active = state["existence_logits"][item].sigmoid() >= existence_threshold
            predicted = ownership[active].argmax(-1).sort(-1).values
            invalid = (predicted[:, 0] == predicted[:, 1])
            valid_pairs = predicted[~invalid]
            duplicate_counts.append(curves.new_tensor(len(valid_pairs) - len(torch.unique(valid_pairs, dim=0))))
            invalid_counts.append(invalid.sum().to(curves.dtype))
            prediction_counts.append(active.sum().to(curves.dtype))
            count_errors.append((active.sum() - len(target["curves"])).abs().to(curves.dtype))
    loss = {
        "face": face_loss,
        "curve": torch.stack(curve_losses).mean(),
        "ownership": torch.stack(owner_losses).mean(),
        "existence": torch.stack(existence_losses).mean(),
        "consistency": torch.stack(consistency_losses).mean(),
    }
    loss["total_loss"] = sum(float(weights[key]) * value for key, value in loss.items())
    loss.update({
        "prediction_count": torch.stack(prediction_counts).mean(),
        "boundary_count_error": torch.stack(count_errors).mean(),
        "matched_ownership_accuracy": torch.stack(ownership_accuracy).mean().detach(),
        "raw_duplicate_count": torch.stack(duplicate_counts).mean(),
        "raw_invalid_incidence_count": torch.stack(invalid_counts).mean(),
    })
    return loss
