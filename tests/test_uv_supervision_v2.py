"""Behavioral checks for the resumed UV stage's geometry and label contract."""

import pytest
import torch

from src.models.vae.uv_intersection import (
    CommonCurveOrientation,
    CurveGeometryLoss,
    CurveTangentLoss,
    MaskedCurveUVLoss,
    RelativeCurveXYZLoss,
    SupervisedCurveLoss,
    UVDecoderLoss,
)


def curve_targets(count=2):
    t = torch.linspace(0, 1, 16)
    xyz = torch.stack((t, t.square(), 0.2 * t), -1)[None].repeat(count, 1, 1)
    uv = torch.stack((t, 0.2 + 0.4 * t), -1)[None, None].repeat(count, 2, 1, 1)
    uv[:, 1] = 1 - uv[:, 1]
    valid = torch.ones(count, 2, 16, dtype=torch.bool)
    return {"edge_xyz": xyz, "edge_uv": uv, "edge_uv_valid": valid}


def decoder_case(num_faces=3, true_pairs=((0, 1),)):
    pairs = torch.triu_indices(num_faces, num_faces, 1).T
    targets = curve_targets(len(true_pairs))
    targets.update({
        "face_local": torch.zeros(num_faces, 16, 16, 6),
        "face_center_scale": torch.zeros(num_faces, 4),
        "face_xyz": torch.zeros(num_faces, 16, 16, 3),
        "edge_pairs": torch.tensor(true_pairs, dtype=torch.long).reshape(-1, 2),
    })
    prediction = {
        "face_local": torch.full_like(targets["face_local"], 0.2, requires_grad=True),
        "face_center_scale": torch.full_like(targets["face_center_scale"], 0.3, requires_grad=True),
        "face_xyz": torch.full_like(targets["face_xyz"], 0.1, requires_grad=True),
        "edge_pairs": pairs,
        "edge_logits": torch.full((len(pairs),), -100.0, requires_grad=True),
        "edge_xyz": torch.full((len(pairs), 2, 16, 3), 0.1, requires_grad=True),
        "edge_uv": torch.full((len(pairs), 2, 16, 2), 0.4, requires_grad=True),
    }
    return prediction, targets


def test_one_orientation_reverses_xyz_uv_and_valid_together():
    target = curve_targets()
    target["edge_uv_valid"][0, 0, :5] = False
    target["edge_uv_valid"][0, 1, 10:] = False
    prediction = target["edge_xyz"][:, None].repeat(1, 2, 1, 1)
    prediction[0] = prediction[0].flip(1)
    aligned = CommonCurveOrientation()(prediction, target)
    for name, axis in (("edge_xyz", 0), ("edge_uv", 1), ("edge_uv_valid", 1)):
        torch.testing.assert_close(aligned[name][0], target[name][0].flip(axis))
        torch.testing.assert_close(aligned[name][1], target[name][1])
    predicted_uv = aligned["edge_uv"].clone()
    predicted_uv[~aligned["edge_uv_valid"]] = -100
    losses = SupervisedCurveLoss()(prediction, predicted_uv, target)
    for name in ("curve_xyz_l1", "curve_xyz_relative", "curve_uv", "curve_tangent", "curve_incidence"):
        torch.testing.assert_close(losses[name], torch.tensor(0.0), atol=2e-7, rtol=0)


def test_uv_does_not_choose_an_independent_orientation_or_mask_other_geometry():
    target = curve_targets(1)
    xyz = target["edge_xyz"][:, None].repeat(1, 2, 1, 1)
    uv = target["edge_uv"].flip(2).clone().requires_grad_()
    assert SupervisedCurveLoss()(xyz, uv, target)["curve_uv"] > 0
    target["edge_uv_valid"][:] = False
    target["edge_uv"][:] = float("nan")
    losses = SupervisedCurveLoss()(xyz + 0.3, uv, target)
    assert losses["curve_uv"] == 0 and losses["curve_uv_valid_fraction"] == 0
    assert losses["curve_xyz_l1"] > 0 and losses["curve_xyz_relative"] > 0
    assert all(torch.isfinite(value) for value in losses.values())
    losses["curve_uv"].backward()
    assert torch.equal(uv.grad, torch.zeros_like(uv))


def test_uv_mask_counts_coordinates_and_has_no_invalid_point_gradient():
    predicted = torch.zeros(1, 2, 16, 2, requires_grad=True)
    target = torch.full_like(predicted, float("nan"))
    valid = torch.zeros(1, 2, 16, dtype=torch.bool)
    valid[0, 1, 3] = True
    target[0, 1, 3] = torch.tensor([0.2, 0.8])
    losses = MaskedCurveUVLoss()(predicted, target, valid)
    torch.testing.assert_close(losses["curve_uv"], torch.tensor(0.5))
    torch.testing.assert_close(losses["curve_uv_valid_fraction"], torch.tensor(1 / 32))
    losses["curve_uv"].backward()
    assert predicted.grad[valid].abs().sum() > 0
    assert torch.count_nonzero(predicted.grad[~valid]) == 0


def test_relative_xyz_uses_gt_max_radius_and_caps_short_edge_weight():
    t = torch.linspace(-1, 1, 16)
    target = torch.stack((t, 0 * t, 0 * t), -1)[None].repeat(2, 1, 1)
    target[0] *= 0.002
    target[1] *= 2.0
    prediction = target[:, None].repeat(1, 2, 1, 1)
    prediction[..., 1] += 0.03
    loss = RelativeCurveXYZLoss(scale_floor=0.01)(prediction, target)
    # Per-edge L1 is .01; only GT radii (.002 and 2) determine the divisor.
    torch.testing.assert_close(loss, torch.tensor((0.01 / 0.01 + 0.01 / 2) / 2))


def test_tangent_masks_only_gt_degeneracy_and_penalizes_collapsed_predictions():
    target = torch.tensor([[[0., 0., 0.], [0., 0., 0.], [1e-7, 0., 0.], [1., 0., 0.]]])
    prediction = torch.zeros(1, 2, 4, 3, requires_grad=True)
    losses = CurveTangentLoss()(prediction, target)
    torch.testing.assert_close(losses["curve_tangent"], torch.tensor(1.))
    torch.testing.assert_close(losses["curve_tangent_valid_fraction"], torch.tensor(1 / 3))
    losses["curve_tangent"].backward()
    assert torch.isfinite(prediction.grad).all() and prediction.grad.abs().sum() > 0
    assert torch.count_nonzero(prediction.grad[:, :, :2]) == 0


def test_all_degenerate_gt_tangents_return_differentiable_zero():
    prediction = torch.randn(2, 2, 16, 3, requires_grad=True)
    losses = CurveTangentLoss()(prediction, torch.zeros(2, 16, 3))
    assert losses["curve_tangent"] == losses["curve_tangent_valid_fraction"] == 0
    losses["curve_tangent"].backward()
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))


def test_old_mse_metric_is_unchanged_and_is_only_a_v2_diagnostic():
    target = curve_targets()
    generator = torch.Generator().manual_seed(9)
    xyz = torch.randn(2, 2, 16, 3, generator=generator, requires_grad=True)
    losses = SupervisedCurveLoss()(xyz, target["edge_uv"], target)
    expected = CurveGeometryLoss()(xyz, target["edge_xyz"])["curve_xyz"]
    torch.testing.assert_close(losses["curve_xyz"], expected)
    assert not losses["curve_xyz"].requires_grad
    assert losses["curve_xyz_l1"].requires_grad


def test_l1_controls_training_orientation_even_when_mse_prefers_reversal():
    target = curve_targets(1)
    target["edge_xyz"][..., 1:] = 0
    xyz = target["edge_xyz"][:, None].repeat(1, 2, 1, 1)
    xyz[:, :, -1, 0] = -100  # One outlier changes MSE's preferred direction.
    gt = target["edge_xyz"][:, None]
    assert (xyz - gt).abs().mean() < (xyz - gt.flip(2)).abs().mean()
    assert (xyz - gt).square().mean() > (xyz - gt.flip(2)).square().mean()
    losses = SupervisedCurveLoss()(xyz, target["edge_uv"], target)
    assert losses["curve_uv"] == 0  # UV follows the L1 choice, not diagnostic MSE.
    torch.testing.assert_close(losses["curve_xyz_l1"], (xyz - gt).abs().mean())
    torch.testing.assert_close(losses["curve_xyz"], (xyz - gt.flip(2)).square().mean())


def test_v2_weighted_total_and_positive_gt_supervision_ignore_probability_gate():
    prediction, target = decoder_case()
    criterion = UVDecoderLoss(supervision_version="geometry_uv_v2")
    losses = criterion(prediction, target)
    weights = {
        "face_local": 1, "face_center_scale": 1, "face_xyz": 1,
        "curve_xyz_l1": 1, "curve_uv": 0.1, "curve_tangent": 0.01,
        "curve_xyz_relative": 0.05, "curve_incidence": 0.1, "edge_existence": 0.1,
    }
    assert {name.removeprefix("weighted_") for name in losses if name.startswith("weighted_")} == set(weights)
    for name, weight in weights.items():
        torch.testing.assert_close(losses[f"weighted_{name}"], losses[name] * weight)
    expected = sum(losses[name] * weight for name, weight in weights.items())
    torch.testing.assert_close(losses["total"], expected)
    assert not losses["face_normal"].requires_grad
    assert "weighted_face_normal" not in losses and "weighted_curve_xyz" not in losses
    losses["total"].backward()
    assert prediction["edge_xyz"].grad[0].abs().sum() > 0
    assert prediction["edge_uv"].grad[0].abs().sum() > 0
    assert torch.count_nonzero(prediction["edge_xyz"].grad[1:]) == 0
    assert torch.count_nonzero(prediction["edge_uv"].grad[1:]) == 0
    assert prediction["edge_logits"].grad[0] < 0
    assert all(torch.isfinite(value.grad).all() for value in prediction.values() if value.requires_grad)
    assert not criterion.state_dict() and not tuple(criterion.parameters())


def test_empty_v2_edges_preserve_finite_losses_and_gradients():
    prediction, target = decoder_case(num_faces=1, true_pairs=())
    losses = UVDecoderLoss(supervision_version="geometry_uv_v2")(prediction, target)
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert prediction["face_local"].grad.abs().sum() > 0
    assert prediction["edge_xyz"].grad is not None and prediction["edge_uv"].grad is not None


def test_v2_missing_uv_labels_is_an_error_and_legacy_remains_explicit():
    prediction, target = decoder_case()
    del target["edge_uv"]
    with pytest.raises(KeyError, match="edge_uv"):
        UVDecoderLoss(supervision_version="geometry_uv_v2")(prediction, target)
    losses = UVDecoderLoss()(prediction, target)
    assert "curve_xyz_l1" not in losses and losses["curve_xyz"].requires_grad
    torch.testing.assert_close(losses["weighted_curve_xyz"], losses["curve_xyz"])
    with pytest.raises(ValueError, match="supervision version"):
        UVDecoderLoss(supervision_version="typo")
