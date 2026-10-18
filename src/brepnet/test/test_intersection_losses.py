"""Loss contracts and gradients, not reconstruction-quality claims."""

from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from src.brepnet.models.vae import IntersectionLoss, balanced_binary_cross_entropy
from src.brepnet.models.blocks import (
    CandidatePairs,
    DecoderStage,
    PrimitiveGeometry,
    RefinementOutput,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def leaf(value):
    return value.clone().detach().requires_grad_(True)


def geometry(local_points, center_scale, points=None):
    return PrimitiveGeometry(
        leaf(local_points), leaf(center_scale), leaf(local_points[..., :3] if points is None else points),
    )


def make_case(face_counts=(2,), connected=True, curve_length=5):
    counts = torch.tensor(face_counts)
    num_faces = sum(face_counts)
    axis = torch.linspace(-1.0, 1.0, 3)
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    plane_xy = torch.stack((xx, yy, torch.zeros_like(xx)), dim=-1)
    plane_xz = torch.stack((xx, torch.zeros_like(xx), yy), dim=-1)
    faces = torch.stack([plane_xy if index % 2 == 0 else plane_xz for index in range(num_faces)])
    face_local = torch.cat((faces, torch.ones_like(faces)), dim=-1)
    face_bbox = torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(num_faces, 1)

    pair_rows, objects = [], []
    offset = 0
    for object_id, count in enumerate(face_counts):
        pair_rows.extend((offset + i, offset + j) for i in range(count) for j in range(count) if i != j)
        objects.extend([object_id] * (count * (count - 1)))
        offset += count
    indices = torch.tensor(pair_rows, dtype=torch.long).reshape(-1, 2)
    lookup = {pair: row for row, pair in enumerate(pair_rows)}
    reverse = torch.tensor([lookup[j, i] for i, j in pair_rows], dtype=torch.long)
    pairs = CandidatePairs(indices, reverse, torch.tensor(objects, dtype=torch.long), counts)
    curve = torch.stack((torch.linspace(-0.8, 0.8, curve_length), torch.zeros(curve_length), torch.zeros(curve_length)), dim=-1)
    if connected:
        assert face_counts[0] >= 2
        truth_curves = torch.stack((curve, curve.flip(0)))
        connectivity = torch.tensor([[0, 0, 1], [1, 1, 0]])
    else:
        truth_curves = torch.empty(0, curve_length, 3)
        connectivity = torch.empty(0, 3, dtype=torch.long)
    edge_local = torch.cat((truth_curves, torch.ones_like(truth_curves)), dim=-1)
    edge_bbox = torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(len(truth_curves), 1)
    candidate_curves = torch.stack([curve if i < j else curve.flip(0) for i, j in pair_rows]) if pair_rows else torch.empty(0, curve_length, 3)
    candidate_local = torch.cat((candidate_curves, torch.ones_like(candidate_curves)), dim=-1)
    candidate_bbox = torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(len(indices), 1)

    def stage():
        return DecoderStage(
            leaf(torch.zeros(num_faces, 32)), leaf(torch.zeros(len(indices), 64)),
            geometry(face_local, face_bbox), geometry(candidate_local, candidate_bbox),
            leaf(torch.full((len(indices),), 1.0)),
        )

    output = RefinementOutput(pairs, stage(), stage(), leaf(torch.zeros(num_faces, 32)), leaf(torch.zeros(len(indices), 64)))
    batch = {
        "face_counts": counts,
        "face_norm": face_local,
        "face_points": face_local.clone(),
        "face_bbox": face_bbox,
        "edge_norm": edge_local,
        "edge_points": edge_local.clone(),
        "edge_bbox": edge_bbox,
        "edge_face_connectivity": connectivity,
    }
    return output, batch, leaf(torch.ones(len(truth_curves), 64)), geometry(edge_local, edge_bbox)


def evaluate(case, *, cfg=None, frozen=True, kl=None):
    return IntersectionLoss(cfg or {})(*case, torch.tensor(0.3, requires_grad=True) if kl is None else kl, frozen)


def test_gt_geometry_has_no_added_consistency_penalty_and_total_counts_once():
    case = make_case(face_counts=(3,))
    result = evaluate(case)
    assert result["incidence"] == 0
    assert result["twin"] == 0
    assert result["face_xyz"] == 0
    assert result["edge_xyz"] == 0
    weights = [value for name, value in result.items() if name.startswith("weighted_")]
    torch.testing.assert_close(result["total_loss"], torch.stack(weights).sum())
    torch.testing.assert_close(result["weighted_coarse_classification"], 0.025 * result["coarse_classification"])
    torch.testing.assert_close(result["weighted_refined_classification"], 0.1 * result["refined_classification"])
    assert result["weighted_kl_loss"] == 0
    assert result["kl_loss"] == 0.3
    for name in ("coarse_precision", "refined_recall", "refined_positive_probability", "delta_edge_mean"):
        assert not result[name].requires_grad


def test_incidence_gradients_reach_both_surfaces_and_curves_without_probability_escape():
    case = make_case()
    output = case[0]
    shifted = output.refined.edge_geometry.points.detach() + torch.tensor([0.0, 0.2, 0.3])
    output.refined.edge_geometry.points = leaf(shifted)
    result = evaluate(case)
    assert result["incidence"] > 0
    result["incidence"].backward()
    curves = output.refined.edge_geometry.points
    surfaces = output.refined.face_geometry.points
    assert torch.isfinite(curves.grad).all() and curves.grad.abs().sum() > 0
    assert torch.isfinite(surfaces.grad).all()
    assert (surfaces.grad.abs().flatten(1).sum(-1) > 0).all()
    assert output.refined.edge_logits.grad is None


def test_geometry_floor_does_not_penalize_discretization_error_already_in_gt():
    case = make_case()
    output, batch = case[:2]
    for tensor in (batch["edge_points"], batch["edge_norm"]):
        tensor[..., 2] += 0.1
    for stage in (output.coarse, output.refined):
        with torch.no_grad():
            stage.edge_geometry.points[..., 2] += 0.1
            stage.edge_geometry.local_points[..., 2] += 0.1
    assert evaluate(case)["incidence"] == 0
    with torch.no_grad():
        output.refined.edge_geometry.points[..., 2] += 0.05
    assert evaluate(case)["incidence"] > 0


def test_twin_uses_gt_correspondence_reversal_and_backpropagates_to_both_curves():
    case = make_case()
    output = case[0]
    with torch.no_grad():
        output.refined.edge_geometry.points[0, :, 2] += 0.2
    loss = evaluate(case)["twin"]
    assert loss > 0
    loss.backward()
    gradients = output.refined.edge_geometry.points.grad
    assert torch.isfinite(gradients).all()
    assert (gradients.abs().flatten(1).sum(-1) > 0).all()


def test_closed_twin_phase_and_different_parameter_spacing_are_not_forced_equal():
    case = make_case(curve_length=5)
    output, batch = case[:2]
    closed = torch.tensor([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0], [-0.5, -0.5, 0.0]])
    unique = closed[:-1].flip(0).roll(1, 0)
    reciprocal = torch.cat((unique, unique[:1]))
    truth = torch.stack((closed, reciprocal))
    batch["edge_points"][..., :3] = truth
    output.refined.edge_geometry.points = leaf(truth)
    assert evaluate(case)["twin"] == 0

    first = torch.tensor([-0.8, -0.6, 0.0, 0.4, 0.8])
    second = torch.tensor([0.8, 0.3, -0.2, -0.5, -0.8])
    truth.zero_()
    truth[0, :, 0], truth[1, :, 0] = first, second
    batch["edge_points"][..., :3] = truth
    output.refined.edge_geometry.points = leaf(truth)
    assert evaluate(case)["twin"] == 0


def test_teacher_detaches_only_matching_target_and_kl_depends_on_encoder_mode():
    case = make_case()
    output, _, teacher_features, teacher_geometry = case
    kl = torch.tensor(0.3, requires_grad=True)
    result = evaluate(case, frozen=True, kl=kl)
    result["edge_feature"].backward()
    assert teacher_features.grad is None
    assert output.coarse.edge_features.grad.abs().sum() > 0
    assert output.refined.edge_features.grad is None
    with torch.no_grad():
        teacher_geometry.local_points.add_(0.1)
    result = evaluate(case, frozen=False, kl=kl)
    result["total_loss"].backward()
    assert teacher_geometry.local_points.grad.abs().sum() > 0
    assert kl.grad == 1


@pytest.mark.parametrize("face_counts", [(1,), (3,), (1, 2)])
def test_no_edges_and_empty_candidate_classes_are_finite(face_counts):
    case = make_case(face_counts=face_counts, connected=False)
    result = evaluate(case)
    assert all(torch.isfinite(value).all() for value in result.values())
    for name in ("edge_feature", "edge_xyz", "twin", "incidence", "encoder_edge_local"):
        assert result[name] == 0
    result["total_loss"].backward()
    logits = case[0].refined.edge_logits
    if len(logits):
        assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


def test_balanced_bce_equal_class_weight_and_one_or_no_class():
    logits = leaf(torch.tensor([1.0, -0.5, -0.5, -0.5]))
    labels = torch.tensor([1.0, 0.0, 0.0, 0.0])
    raw = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    torch.testing.assert_close(balanced_binary_cross_entropy(logits, labels), (raw[0] + raw[1:].mean()) / 2)
    for value in (0.0, 1.0):
        one_class = torch.full_like(logits, value)
        torch.testing.assert_close(balanced_binary_cross_entropy(logits, one_class), F.binary_cross_entropy_with_logits(logits, one_class))
    empty = leaf(torch.empty(0))
    zero = balanced_binary_cross_entropy(empty, torch.empty(0))
    zero.backward()
    assert zero == 0 and empty.grad is not None


@pytest.mark.parametrize("permutation", ["gt_rows", "gt_edges", "candidates"])
def test_gt_and_candidate_storage_order_do_not_change_loss(permutation):
    case = make_case(face_counts=(3,))
    baseline = evaluate(case)
    changed = list(deepcopy(case))
    output, batch, teacher_features, teacher_geometry = changed
    if permutation == "gt_rows":
        batch["edge_face_connectivity"] = batch["edge_face_connectivity"].flip(0)
    elif permutation == "gt_edges":
        for name in ("edge_points", "edge_norm", "edge_bbox"):
            batch[name] = batch[name].flip(0)
        batch["edge_face_connectivity"][:, 0] = 1 - batch["edge_face_connectivity"][:, 0]
        changed[2] = teacher_features.flip(0)
        for name in ("local_points", "center_scale", "points"):
            setattr(teacher_geometry, name, getattr(teacher_geometry, name).flip(0))
    else:
        order = torch.tensor([5, 0, 3, 2, 4, 1])
        inverse = order.argsort()
        pairs = output.pairs
        pairs.indices = pairs.indices[order]
        pairs.object_ids = pairs.object_ids[order]
        pairs.reverse = inverse[pairs.reverse[order]]
        for stage in (output.coarse, output.refined):
            stage.edge_features = stage.edge_features[order]
            stage.edge_logits = stage.edge_logits[order]
            for name in ("local_points", "center_scale", "points"):
                setattr(stage.edge_geometry, name, getattr(stage.edge_geometry, name)[order])
        output.delta_edge = output.delta_edge[order]
    result = evaluate(changed)
    for name in baseline:
        torch.testing.assert_close(result[name], baseline[name])


def test_loss_supports_smaller_feature_widths_without_changing_geometry_contract():
    case = list(make_case())
    output = case[0]
    for stage in (output.coarse, output.refined):
        stage.face_features = stage.face_features[:, :4]
        stage.edge_features = stage.edge_features[:, :8]
    output.delta_face = output.delta_face[:, :4]
    output.delta_edge = output.delta_edge[:, :8]
    case[2] = case[2][:, :8]
    result = evaluate(case)
    result["total_loss"].backward()
    assert torch.isfinite(result["total_loss"])
