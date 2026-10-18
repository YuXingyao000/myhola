"""Numerical regression against the pre-optimization loss, on CPU only.

The small reference functions retain the old roll/cat variants and the two
separate matching/loss cost evaluations. They intentionally do not call the
optimized matching/cost implementation, so tie-breaking and derivatives are
checked independently. These are equivalence tests, not throughput benchmarks.
"""

import pytest
import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F

from src.brepnet.models.vae.losses import joint as joint_losses


WEIGHTS = dict(face=1.0, curve=1.0, ownership=0.1, existence=0.1, consistency=0.1)
MATCH_WEIGHTS = dict(curve=1.0, ownership=0.25)


def legacy_costs(curves, ownership, target):
    curves = curves.float()
    truth = target["curves"].to(curves.device, torch.float32)
    pairs = target["face_pairs"].to(curves.device, torch.long)
    geometry = torch.minimum(
        (curves[:, None] - truth[None]).abs().mean((-1, -2)),
        (curves[:, None] - truth.flip(1)[None]).abs().mean((-1, -2)),
    )
    indices = target["is_closed"].nonzero(as_tuple=True)[0]
    if len(indices):
        unique = truth[indices, :-1]
        variants = []
        for direction in (unique, unique.flip(1)):
            for shift in range(unique.shape[1]):
                shifted = direction.roll(shift, dims=1)
                variants.append(torch.cat((shifted, shifted[:, :1]), dim=1))
        variants = torch.stack(variants, dim=1)
        closed_cost = (curves[:, None, None] - variants[None]).abs().mean((-1, -2)).min(-1).values
        geometry = geometry.clone()
        geometry[:, indices] = closed_cost
    probability = ownership.float().log_softmax(-1)
    direct = -(probability[:, 0, pairs[:, 0]] + probability[:, 1, pairs[:, 1]]) / 2
    reverse = -(probability[:, 0, pairs[:, 1]] + probability[:, 1, pairs[:, 0]]) / 2
    return geometry, torch.minimum(direct, reverse)


def legacy_loss(state, targets, target_faces):
    zero = state["curves"].sum() * 0
    terms = {name: [] for name in ("curve", "ownership", "existence", "consistency")}
    metrics = {name: [] for name in (
        "prediction_count", "boundary_count_error", "matched_ownership_accuracy",
        "raw_duplicate_count", "raw_invalid_incidence_count",
    )}
    for item, target in enumerate(targets):
        count = int(state["face_counts"][item])
        curves = state["curves"][item]
        owner = state["ownership_logits"][item, :, :, :count]
        with torch.no_grad():
            geometry, ownership = legacy_costs(curves, owner, target)
            row, col = linear_sum_assignment((geometry + 0.25 * ownership).cpu().numpy())
            row, col = torch.as_tensor(row), torch.as_tensor(col)
        labels = torch.zeros_like(state["existence_logits"][item])
        labels[row] = 1
        terms["existence"].append((F.binary_cross_entropy_with_logits(
            state["existence_logits"][item], labels, reduction="none"
        ) * torch.where(labels.bool(), 1.0, 0.1)).mean())
        if len(row):
            geometry, ownership = legacy_costs(curves, owner, target)
            terms["curve"].append(geometry[row, col].mean())
            terms["ownership"].append(ownership[row, col].mean())
            terms["consistency"].append(joint_losses.surface_boundary_consistency(
                state["face_points"][item, :count], curves[row], owner[row]
            ))
            correct = (owner[row].argmax(-1).sort(-1).values == target["face_pairs"][col].sort(-1).values).all(-1)
            metrics["matched_ownership_accuracy"].append(correct.float().mean())
        else:
            for key in ("curve", "ownership", "consistency"):
                terms[key].append(zero)
            metrics["matched_ownership_accuracy"].append(zero.detach())
        with torch.no_grad():
            active = state["existence_logits"][item].sigmoid() >= 0.5
            pairs = owner[active].argmax(-1).sort(-1).values
            invalid = pairs[:, 0] == pairs[:, 1]
            valid = pairs[~invalid]
            metrics["prediction_count"].append(active.sum().to(curves.dtype))
            metrics["boundary_count_error"].append((active.sum() - len(target["curves"])).abs().to(curves.dtype))
            metrics["raw_duplicate_count"].append(curves.new_tensor(len(valid) - len(torch.unique(valid, dim=0))))
            metrics["raw_invalid_incidence_count"].append(invalid.sum().to(curves.dtype))
    result = {name: torch.stack(values).mean() for name, values in terms.items()}
    result["face"] = F.l1_loss(state["face_points"][state["face_valid"]].float(), target_faces.float())
    # Match the original summation order as well as its weighting.
    result["total_loss"] = sum(WEIGHTS[name] * result[name] for name in ("face", "curve", "ownership", "existence", "consistency"))
    result.update({name: torch.stack(values).mean().detach() for name, values in metrics.items()})
    return result


def inputs(case):
    torch.manual_seed(817)
    state = {
        "face_points": torch.randn(2, 3, 16, 16, 3),
        "curves": torch.randn(2, 4, 16, 3),
        "ownership_logits": torch.randn(2, 4, 2, 3),
        "existence_logits": torch.tensor([[-1.0, 1.0, 0.0, 2.0], [0.0, -2.0, 1.0, -1.0]]),
        "face_counts": torch.tensor([2, 3]),
        "face_valid": torch.tensor([[True, True, False], [True, True, True]]),
    }
    state["ownership_logits"][0, :, :, 2] = -torch.inf
    targets = []
    for item, pairs in enumerate(([[0, 1]], [[0, 2], [1, 2]])):
        curves = torch.randn(len(pairs), 16, 3)
        closed = torch.zeros(len(pairs), dtype=torch.bool)
        if case in ("closed", "mixed_empty"):
            curves[0, -1] = curves[0, 0]
            closed[0] = True
        if case == "reversed":
            curves = curves.flip(1)
            pairs = [pair[::-1] for pair in pairs]
        if case == "empty" or case == "mixed_empty" and item == 1:
            curves, pairs, closed = curves[:0], [], closed[:0]
        targets.append({"curves": curves, "face_pairs": torch.tensor(pairs, dtype=torch.long).reshape(-1, 2), "is_closed": closed})
    return state, targets, torch.randn(5, 16, 16, 3)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("case", ["open", "reversed", "closed", "empty", "mixed_empty"])
def test_all_loss_metrics_and_gradients_match_legacy(case):
    state, targets, faces = inputs(case)
    keys = ("face_points", "curves", "ownership_logits", "existence_logits")
    legacy_state = {key: value.clone().requires_grad_(key in keys) for key, value in state.items()}
    current_state = {key: value.clone().requires_grad_(key in keys) for key, value in state.items()}
    expected = legacy_loss(legacy_state, targets, faces)
    actual = joint_losses.joint_boundary_loss(current_state, targets, faces, WEIGHTS, MATCH_WEIGHTS)
    assert expected.keys() == actual.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0, msg=key)
    expected_grad = torch.autograd.grad(expected["total_loss"], [legacy_state[key] for key in keys], allow_unused=True)
    actual_grad = torch.autograd.grad(actual["total_loss"], [current_state[key] for key in keys], allow_unused=True)
    for key, actual_value, expected_value in zip(keys, actual_grad, expected_grad):
        if expected_value is None:
            assert actual_value is None, key
        else:
            torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0, msg=key)


def test_closed_variant_order_preserves_min_tie_gradients():
    torch.manual_seed(818)
    curves = torch.randn(3, 16, 3)
    ownership = torch.zeros(3, 2, 2)  # also tie the side-swap cost
    target = {"curves": torch.zeros(1, 16, 3), "face_pairs": torch.tensor([[0, 1]]), "is_closed": torch.tensor([True])}
    values, gradients = [], []
    for function in (legacy_costs, joint_losses.boundary_costs):
        prediction, logits = curves.clone().requires_grad_(), ownership.clone().requires_grad_()
        truth = target["curves"].clone().requires_grad_()
        geometry, owner = function(prediction, logits, {**target, "curves": truth})
        values.append((geometry, owner))
        gradients.append(torch.autograd.grad(geometry.sum() + owner.sum(), (prediction, logits, truth)))
    for first, second in zip(values[0] + gradients[0], values[1] + gradients[1]):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_joint_loss_evaluates_full_cost_once_per_object(monkeypatch):
    state, targets, faces = inputs("closed")
    original = joint_losses.boundary_costs
    calls = []

    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(joint_losses, "boundary_costs", counted)
    joint_losses.joint_boundary_loss(state, targets, faces, WEIGHTS, MATCH_WEIGHTS)
    assert len(calls) == len(targets)
