"""Auxiliary UV supervision contracts; not a reconstruction-quality study."""

from copy import deepcopy

import pytest
import numpy as np
import torch

from src.brepnet.models.blocks import UVPathHead, enumerate_candidate_pairs
from src.brepnet.models.vae import AutoEncoder_intersection, IntersectionLoss, _map_directed_targets
from src.brepnet.test.test_intersection_losses import leaf, make_case
from src.brepnet.test.test_intersection_vae import curve_config, encoder_batch


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def uv_config(**changes):
    config = curve_config()
    config.update({
        "uv_boundary": {"enabled": True, "hidden_dim": 16, "num_layers": 2,
                        "nhead": 4, "dim_feedforward": 32, "dropout": 0.,
                        "candidate_chunk_size": 2},
        "init_uv_boundary_checkpoint": None,
        **changes,
    })
    return config


def add_targets(batch):
    rows = len(batch["edge_face_connectivity"])
    uv = torch.rand(rows, 2, 16, 2)
    batch["edge_uv_targets"] = {
        "uv": uv, "valid": torch.ones(rows, 2, 16, dtype=torch.bool),
        "residual": torch.zeros(rows, 2, 16),
        "method": torch.ones(rows, 2, 16, dtype=torch.long),
    }
    return batch


def uv_case(*, connected=True, face_counts=(2,)):
    case = make_case(face_counts=face_counts, connected=connected, curve_length=16)
    output, batch = case[:2]
    add_targets(batch)
    # make_case's grids have X varying down rows; thus v=(x+1)/2, u=1/2.
    ids = batch["edge_face_connectivity"][:, 0]
    uv = batch["edge_uv_targets"]["uv"]
    uv[..., 0] = .5
    uv[..., 1] = (batch["edge_points"][ids, None, :, 0] + 1) / 2
    output.edge_uv = leaf(torch.zeros(len(output.pairs.indices), 2, 16, 2))
    targets = _map_directed_targets(output, batch)
    with torch.no_grad():
        output.edge_uv[targets.candidate_rows] = uv
    return case


def uv_losses(case):
    return IntersectionLoss({}, uv_boundary_enabled=True)(*case, torch.tensor(0.), True)


def test_head_shapes_shared_feature_gradients_and_chunk_equivalence():
    torch.manual_seed(47)
    head = UVPathHead(32, 64, candidate_chunk_size=2)
    faces = torch.randn(5, 32, requires_grad=True)
    pairs = enumerate_candidate_pairs(torch.tensor([2, 3]))
    edges = torch.randn(8, 64, requires_grad=True)
    result = head(edges, faces, pairs.indices)
    assert result.shape == (8, 2, 16, 2)
    whole = deepcopy(head)
    whole.chunk_size = 99
    torch.testing.assert_close(whole(edges, faces, pairs.indices), result, atol=1e-6, rtol=1e-5)
    result.square().mean().backward()
    for tensor in (faces, edges, head.context[0].weight, head.position[0].weight):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0
    # Learned four-channel output is unconstrained; no hidden sigmoid/clamp.
    with torch.no_grad():
        head.output[-1].weight.zero_()
        head.output[-1].bias.fill_(2.)
    assert (head(edges, faces, pairs.indices) == 2.).all()


def test_empty_head_connects_all_parameters_for_ddp():
    head = UVPathHead(32, 64, hidden_dim=16, nhead=4, dim_feedforward=32)
    output = head(torch.empty(0, 64), torch.randn(2, 32), torch.empty(0, 2, dtype=torch.long))
    assert output.shape == (0, 2, 16, 2)
    output.sum().backward()
    assert all(parameter.grad is not None for parameter in head.parameters())


def test_fixed_uv_anchor_detects_in_plane_drift_and_reaches_both_geometries():
    case = uv_case()
    output = case[0]
    initial = uv_losses(case)
    assert initial["uv"] == 0 and initial["uv_anchor"] == 0
    with torch.no_grad():
        output.refined.edge_geometry.points[..., 0] += .05
    losses = uv_losses(case)
    # Both support planes contain this translated curve, so nearest projection misses it.
    assert losses["incidence"] < 1e-6
    torch.testing.assert_close(losses["uv_anchor"], torch.tensor(.05 / 3))
    losses["uv_anchor"].backward()
    for tensor in (output.refined.edge_geometry.points, output.refined.face_geometry.points):
        assert torch.isfinite(tensor.grad).all() and tensor.grad.abs().sum() > 0
    assert output.edge_uv.grad is None  # Predicted UV cannot move the fixed anchor.


def test_gt_interpolation_residual_cancels_even_when_gt_curve_is_off_grid():
    case = uv_case()
    output, batch = case[:2]
    batch["edge_points"][..., 2] += .013
    with torch.no_grad():
        output.refined.edge_geometry.points[..., 2] += .013
    assert uv_losses(case)["uv_anchor"] == 0


def test_connectivity_row_labels_are_not_indexed_by_edge_id():
    case = uv_case()
    output, batch = case[:2]
    # Preserve row/pair/UV order but change physical edge storage order.
    for name in ("edge_points", "edge_norm", "edge_bbox"):
        batch[name] = batch[name].flip(0)
    batch["edge_face_connectivity"][:, 0] = torch.tensor([1, 0])
    targets = _map_directed_targets(output, batch)
    losses = IntersectionLoss({}, True)._uv_boundary(output, batch, targets)
    assert losses["uv"] == 0 and losses["uv_anchor"] == 0
    assert losses["uv_valid_fraction"] == 1


@pytest.mark.parametrize("face_counts,connected", [((2,), True), ((1,), False), ((3,), False)])
def test_empty_masks_return_finite_graph_connected_zero(face_counts, connected):
    case = uv_case(face_counts=face_counts, connected=connected)
    case[1]["edge_uv_targets"]["valid"].zero_()
    case[1]["edge_uv_targets"]["uv"].fill_(float("nan"))
    losses = uv_losses(case)
    assert losses["uv"] == 0 and losses["uv_anchor"] == 0
    assert losses["uv_valid_fraction"] == 0
    (losses["uv"] + losses["uv_anchor"]).backward()
    assert case[0].edge_uv.grad is not None


def test_disabled_keeps_state_rng_losses_and_inference_exactly_unchanged():
    torch.manual_seed(52)
    old = AutoEncoder_intersection(curve_config()).eval()
    old_rng = torch.rand(3)
    torch.manual_seed(52)
    disabled = AutoEncoder_intersection(curve_config(uv_boundary={"enabled": False})).eval()
    torch.testing.assert_close(torch.rand(3), old_rng, rtol=0, atol=0)
    assert old.state_dict().keys() == disabled.state_dict().keys()
    for key, value in old.state_dict().items():
        torch.testing.assert_close(disabled.state_dict()[key], value, rtol=0, atol=0)
    with torch.no_grad():
        old_loss, _ = old(encoder_batch(), v_test=True)
        new_loss, _ = disabled(encoder_batch(), v_test=True)
    assert old_loss.keys() == new_loss.keys()
    for key in old_loss:
        torch.testing.assert_close(new_loss[key], old_loss[key], rtol=0, atol=0)


def test_strict_migration_training_and_full_reload_preserve_all_old_geometry(tmp_path):
    torch.manual_seed(53)
    old = AutoEncoder_intersection(curve_config()).eval()
    source = tmp_path / "conditioned.ckpt"
    torch.save({"state_dict": {"model." + key: value for key, value in old.state_dict().items()},
                "epoch": 157, "global_step": 57354}, source)
    new = AutoEncoder_intersection(uv_config(init_uv_boundary_checkpoint=str(source), freeze_encoder=True)).eval()
    for key, value in old.state_dict().items():
        torch.testing.assert_close(new.state_dict()[key], value, rtol=0, atol=0)
    assert all(key.startswith("decoder.uv_boundary.") for key in set(new.state_dict()) - set(old.state_dict()))
    batch = encoder_batch()
    with torch.no_grad():
        encoded = old.encode(batch)
        z = old.sample(encoded["face_features"], True)[0]
        before, after = (model.decode_state(z, batch["face_counts"]) for model in (old, new))
        for a, b in ((before.coarse, after.coarse), (before.refined, after.refined)):
            for x, y in ((a.face_geometry.points, b.face_geometry.points),
                         (a.edge_geometry.points, b.edge_geometry.points), (a.edge_logits, b.edge_logits),
                         (a.face_features, b.face_features), (a.edge_features, b.edge_features)):
                torch.testing.assert_close(x, y, rtol=0, atol=0)
        assert after.edge_uv.shape == (8, 2, 16, 2)
        # Pure inference accepts latents and requires no sidecar/GT.
        exported = new.decode_latents(z[:2])
        assert exported["pred_edge_uv"].shape == (len(exported["pred_edge"]), 2, 16, 2)
        batch_with_uv = add_targets(batch)
        exported_batch = new.export_training_items(after, z, encoded["face_features"], batch_with_uv)
        for item, expected_rows in zip(exported_batch["items"], (slice(0, 2), slice(2, 4))):
            np.testing.assert_array_equal(item["gt_edge_uv"], batch_with_uv["edge_uv_targets"]["uv"][expected_rows])
            np.testing.assert_array_equal(item["gt_edge_uv_valid"], batch_with_uv["edge_uv_targets"]["valid"][expected_rows])
    new.train()
    losses, _ = new(add_targets(batch))
    assert all(torch.isfinite(value).all() for value in losses.values())
    torch.testing.assert_close(losses["weighted_uv"], .01 * losses["uv"])
    torch.testing.assert_close(losses["weighted_uv_anchor"], .1 * losses["uv_anchor"])
    torch.testing.assert_close(losses["total_loss"], torch.stack([v for k, v in losses.items() if k.startswith("weighted_")]).sum())
    losses["total_loss"].backward()
    assert all(parameter.grad is None for parameter in new.encoder.parameters())
    for parameter in (new.decoder.uv_boundary.context[0].weight, new.decoder.refiner.edge_delta.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    saved = tmp_path / "uv.ckpt"
    torch.save(new.state_dict(), saved)
    loaded = AutoEncoder_intersection(uv_config(checkpoint=str(saved))).eval()
    resumed = AutoEncoder_intersection(uv_config(restoring_checkpoint=True)).eval()
    resumed.load_state_dict(torch.load(saved, weights_only=False), strict=True)
    for key, value in new.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
        torch.testing.assert_close(resumed.state_dict()[key], value, rtol=0, atol=0)
    broken = new.state_dict()
    broken.pop(next(key for key in broken if key.startswith("decoder.uv_boundary.")))
    torch.save(broken, saved)
    with pytest.raises(RuntimeError, match="Missing key"):
        AutoEncoder_intersection(uv_config(checkpoint=str(saved)))


@pytest.mark.parametrize("damage", ["missing", "shape", "extra", "duplicate", "already_uv"])
def test_migration_rejects_incompatible_or_partial_sources(tmp_path, damage):
    source = tmp_path / "source.ckpt"
    model = AutoEncoder_intersection(uv_config() if damage == "already_uv" else curve_config())
    state = {"model." + key: value for key, value in model.state_dict().items()}
    if damage == "missing":
        state.pop(next(key for key in state if "conditioning" in key))
    elif damage == "shape":
        state["model.decoder.classifier.bias"] = torch.zeros(2)
    elif damage == "extra":
        state["model.decoder.curve_point_feedback.weight"] = torch.zeros(1)
    elif damage == "duplicate":
        state["decoder.classifier.bias"] = state["model.decoder.classifier.bias"]
    torch.save({"state_dict": state}, source)
    with pytest.raises((ValueError, RuntimeError), match="UV boundary|Ambiguous|size mismatch"):
        AutoEncoder_intersection(uv_config(init_uv_boundary_checkpoint=str(source)))


def test_resume_does_not_repeat_component_initialization():
    with pytest.raises(ValueError, match="Resume"):
        AutoEncoder_intersection(uv_config(
            init_uv_boundary_checkpoint="source.ckpt", restoring_checkpoint=True,
        ))
