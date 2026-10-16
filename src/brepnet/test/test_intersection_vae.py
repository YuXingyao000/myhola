"""Migration and pipeline contracts; these do not establish research quality."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.brepnet.dataset import denormalize_coord1112, normalize_coord1112
from src.brepnet.models.vae import AutoEncoder_intersection, AutoEncoder_light


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def tiny_config(**changes):
    return {
        **AutoEncoder_light.DEFAULTS,
        "norm": "layer", "in_channels": 6, "hidden_channels": 16,
        "latent_channels": 2, "bottleneck_dim": 16, "nhead": 2,
        "num_encoder_layers": 1, "num_decoder_layers": 1, "num_gat_layers": 1,
        "intersection_dim": 16, "intersection_layers": 1,
        "face_bbox_decoder_blocks": 1, "edge_bbox_decoder_blocks": 1,
        "gaussian_weights": 1e-6, "freeze_encoder": False,
        "init_legacy_checkpoint": None, "edge_threshold": 0.5, "trainable_scope": "all",
        "refinement": {
            "enabled": True, "hidden_dim": 16, "geometry_dim": 8,
            "geometry_chunk_size": 4, "candidate_chunk_size": 4,
            "checkpoint_candidates": True,
        },
        "loss_weights": {
            "coarse_weight": 0.25, "classification_weight": 0.1,
            "feature_weight": 1., "encoder_edge_weight": 1., "xyz_weight": 1.,
            "incidence_weight": .1, "twin_weight": .1,
            "geometry_tolerance": 1e-5, "geometry_chunk_size": 4,
        },
        **changes,
    }


def encoder_batch():
    generator = torch.Generator().manual_seed(61)
    faces = torch.randn(5, 16, 16, 6, generator=generator) * .2
    first = torch.randn(16, 6, generator=generator) * .2
    second = torch.randn(16, 6, generator=generator) * .2
    edges = torch.stack((first, first.flip(0), second, second.flip(0)))
    counts = torch.tensor([2, 3])
    object_ids = torch.arange(2).repeat_interleave(counts)
    f_xyz, f_dir, f_center, f_scale = normalize_coord1112(faces)
    e_xyz, e_dir, e_center, e_scale = normalize_coord1112(edges)
    return {
        "v_prefix": ("first", "second"), "face_counts": counts,
        "face_points": faces, "edge_points": edges,
        "face_norm": torch.cat((f_xyz, f_dir), -1), "edge_norm": torch.cat((e_xyz, e_dir), -1),
        "face_bbox": torch.cat((f_center, f_scale), -1), "edge_bbox": torch.cat((e_center, e_scale), -1),
        "edge_face_connectivity": torch.tensor([[0, 0, 1], [1, 1, 0], [2, 2, 3], [3, 3, 2]]),
        "attn_mask": object_ids[:, None] != object_ids[None, :],
        "zero_positions": torch.tensor([[2, 4], [4, 2]]),
    }


def test_strict_legacy_migration_preserves_encoder_and_initial_geometry(tmp_path):
    legacy = AutoEncoder_light(tiny_config()).eval()
    checkpoint = tmp_path / "legacy.ckpt"
    torch.save({"state_dict": {f"model.{k}": v for k, v in legacy.state_dict().items()}}, checkpoint)
    model = AutoEncoder_intersection(tiny_config(init_legacy_checkpoint=str(checkpoint), freeze_encoder=True)).eval()
    batch = encoder_batch()
    with torch.no_grad():
        old_encoding = legacy.encode(batch, True)
        new_encoding = model.encode(batch, True)
        for name in ("face_features", "edge_features"):
            torch.testing.assert_close(new_encoding[name], old_encoding[name], rtol=0, atol=0)
        old_z, old_kl, old_stats = legacy.sample(old_encoding["face_features"], True)
        new_z, new_kl, new_stats = model.sample(new_encoding["face_features"], True)
        for actual, expected in ((new_z, old_z), (new_kl, old_kl), (new_stats, old_stats)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        old = legacy.decode({**old_encoding, "face_latents": old_z}, batch)
        prediction = model.decode_state(new_z, batch["face_counts"])
        for stage in (prediction.coarse, prediction.refined):
            torch.testing.assert_close(stage.face_geometry.local_points, old["face_points_local"])
            torch.testing.assert_close(stage.face_geometry.center_scale, old["face_center_scale"])
            global_old = denormalize_coord1112(old["face_points_local"], old["face_center_scale"])[..., :3]
            torch.testing.assert_close(stage.face_geometry.points, global_old)
        for row, (_, i, j) in enumerate(batch["edge_face_connectivity"]):
            index = ((prediction.pairs.indices[:, 0] == i) & (prediction.pairs.indices[:, 1] == j)).nonzero()[0, 0]
            torch.testing.assert_close(prediction.coarse.edge_geometry.local_points[index], old["edge_points_local"][row])
        torch.testing.assert_close(model.decoder.classifier.weight, legacy.classifier.weight, rtol=0, atol=0)


def test_frozen_training_gradients_and_strict_new_checkpoint(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save(AutoEncoder_light(tiny_config()).state_dict(), path)
    model = AutoEncoder_intersection(tiny_config(init_legacy_checkpoint=str(path), freeze_encoder=True)).train()
    assert not model.encoder.training and not model.posterior.training
    assert model.decoder.training
    batch = encoder_batch()
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    losses, _ = model(batch)
    assert all(value.isfinite().all() for value in losses.values())
    losses["total_loss"].backward()
    assert all(p.grad is None for p in model.encoder.parameters())
    assert model.decoder.refiner.edge_delta.weight.grad.abs().sum() > 0
    assert model.decoder.refiner.face_delta.weight.grad.abs().sum() > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model(batch)[0]["total_loss"].backward()
    assert model.decoder.refiner.geometry_read.point_encoder[0].weight.grad.abs().sum() > 0
    assert model.decoder.refiner.message_encoder[0].weight.grad.abs().sum() > 0

    new_path = tmp_path / "refinement.pt"
    torch.save(model.state_dict(), new_path)
    restored = AutoEncoder_intersection(tiny_config(checkpoint=str(new_path), freeze_encoder=True)).eval()
    model.eval()
    with torch.no_grad():
        loss_a, data_a = model(batch, v_test=True)
        loss_b, data_b = restored(batch, v_test=True)
    torch.testing.assert_close(loss_a["total_loss"], loss_b["total_loss"], rtol=0, atol=0)
    for a, b, count in zip(data_a["items"], data_b["items"], batch["face_counts"]):
        assert a["representation"].item() == "legacy_half_edges"
        assert len(a["pred_face"]) == int(count)
        assert a["pred_face_adj_prob"].shape == (int(count), int(count))
        np.testing.assert_array_equal(a["pred_face_adj_prob"], a["pred_face_adj_prob"].T)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        if len(a["pred_edge_face_connectivity"]):
            assert a["pred_edge_face_connectivity"][:, 1:].max() < int(count)


def test_decoder_input_has_no_gt_dependency_and_validates_batch():
    model = AutoEncoder_intersection(tiny_config()).eval()
    batch = encoder_batch()
    with torch.no_grad():
        z, _, _ = model.sample(model.encode(batch)["face_features"], True)
        before = model.decode_latents(z[:2])
        batch["zero_positions"] = torch.zeros((100, 2), dtype=torch.long)
        batch["edge_face_connectivity"] = batch["edge_face_connectivity"].flip(0)
        after = model.decode_latents(z[:2])
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])
    batch["attn_mask"].fill_(False)
    with pytest.raises(ValueError, match="attn_mask"):
        model.encode(batch)
    with pytest.raises(ValueError, match="random weights"):
        AutoEncoder_intersection(tiny_config(freeze_encoder=True))


def test_unfrozen_encoder_and_posterior_receive_gradients():
    model = AutoEncoder_intersection(tiny_config())
    loss, _ = model(encoder_batch())
    loss["total_loss"].backward()
    assert model.encoder.face_coords[0].weight.grad.abs().sum() > 0
    assert model.posterior.projection[0].weight.grad.abs().sum() > 0


def save_curve_baseline(path):
    model = AutoEncoder_intersection(tiny_config()).eval()
    with torch.no_grad():
        for head in (model.decoder.refiner.face_delta, model.decoder.refiner.edge_delta):
            head.weight.normal_(std=.02)
            head.bias.normal_(std=.01)
    torch.save({
        "epoch": 10, "global_step": 3993,
        "state_dict": {"model." + key: value for key, value in model.state_dict().items()},
    }, path)
    return model


def curve_config(**changes):
    return tiny_config(curve_conditioning={"enabled": True, "hidden_dim": 128}, **changes)


def test_curve_initialization_preserves_every_baseline_weight_output_and_loss(tmp_path):
    torch.manual_seed(85)
    path = tmp_path / "refinement.ckpt"
    old = save_curve_baseline(path)
    new = AutoEncoder_intersection(curve_config(init_refinement_checkpoint=str(path))).eval()
    retained = old.state_dict()
    added = set(new.state_dict()) - set(retained)
    assert added and all(key.startswith("decoder.edge_geometry.conditioning.") for key in added)
    for key, value in retained.items():
        torch.testing.assert_close(new.state_dict()[key], value, rtol=0, atol=0)
    old_names = list(dict(old.named_parameters()))
    new_names = [key for key, _ in new.named_parameters() if key not in added]
    assert old_names == new_names
    assert new.initialization_report["source_global_step"] == 3993
    with torch.no_grad():
        first_loss, first = old(encoder_batch(), v_test=True)
        second_loss, second = new(encoder_batch(), v_test=True)
    for key in first_loss:
        torch.testing.assert_close(second_loss[key], first_loss[key], rtol=0, atol=0)
    for a, b in zip(first["items"], second["items"]):
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])


@pytest.mark.parametrize("corruption", ["missing_head", "missing_inter", "extra", "shape", "already_conditioned"])
def test_curve_initialization_rejects_incomplete_or_modified_baseline(tmp_path, corruption):
    path = tmp_path / "refinement.ckpt"
    save_curve_baseline(path)
    checkpoint = torch.load(path, weights_only=False)
    state = checkpoint["state_dict"]
    if corruption == "missing_head":
        state.pop("model.decoder.classifier.bias")
    elif corruption == "missing_inter":
        state.pop(next(key for key in state if key.startswith("model.decoder.inter.")))
    elif corruption == "extra":
        state["model.extra.weight"] = torch.ones(1)
    elif corruption == "shape":
        state["model.decoder.classifier.bias"] = torch.ones(2)
    else:
        conditioned = AutoEncoder_intersection(curve_config())
        checkpoint["state_dict"] = {"model." + key: value for key, value in conditioned.state_dict().items()}
    torch.save(checkpoint, path)
    with pytest.raises((ValueError, RuntimeError)):
        AutoEncoder_intersection(curve_config(init_refinement_checkpoint=str(path)))


def test_curve_checkpoint_restores_trained_conditioning_and_optimizer(tmp_path):
    torch.manual_seed(86)
    model = AutoEncoder_intersection(curve_config())
    # A full checkpoint must restore learned values, not zero them again.
    with torch.no_grad():
        condition = model.decoder.edge_geometry.conditioning
        condition.point_affine.weight.normal_(std=.02)
        condition.bbox_affine.weight.normal_(std=.02)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model(encoder_batch())[0]["total_loss"].backward()
    optimizer.step()
    path = tmp_path / "conditioned.ckpt"
    torch.save({
        "state_dict": {"model." + key: value for key, value in model.state_dict().items()},
        "optimizer_states": [optimizer.state_dict()], "epoch": 3, "global_step": 123,
    }, path)
    loaded = torch.load(path, weights_only=False)
    strict = AutoEncoder_intersection(curve_config(checkpoint=str(path))).eval()
    resumed = AutoEncoder_intersection(curve_config(restoring_checkpoint=True)).eval()
    resumed.load_state_dict({key.removeprefix("model."): value for key, value in loaded["state_dict"].items()}, strict=True)
    restored_optimizer = torch.optim.Adam(resumed.parameters(), lr=1e-4)
    restored_optimizer.load_state_dict(loaded["optimizer_states"][0])
    model.eval()
    with torch.no_grad():
        expected = model(encoder_batch(), v_test=True)[0]["total_loss"]
        for candidate in (strict, resumed):
            actual = candidate(encoder_batch(), v_test=True)[0]["total_loss"]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for first, second in zip(optimizer.state.values(), restored_optimizer.state.values()):
        for key in first:
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    # Full-load routes cannot accept an old checkpoint by silently adding zeros.
    damaged = loaded["state_dict"]
    damaged.pop(next(key for key in damaged if key.startswith("model.decoder.edge_geometry.conditioning.")))
    torch.save(loaded, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        AutoEncoder_intersection(curve_config(checkpoint=str(path)))


def test_curve_frozen_encoder_and_zero_head_gradient_startup(tmp_path):
    path = tmp_path / "refinement.ckpt"
    save_curve_baseline(path)
    model = AutoEncoder_intersection(curve_config(init_refinement_checkpoint=str(path), freeze_encoder=True)).train()
    assert not model.encoder.training and not model.posterior.training
    loss, _ = model(encoder_batch())
    loss["total_loss"].backward()
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    condition = model.decoder.edge_geometry.conditioning
    for head in (condition.point_affine, condition.bbox_affine):
        assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all()
        assert head.weight.grad.abs().sum() > 0
    # At exact zero initialization, hidden projection learns after the affine head moves.
    assert condition.projection[0].weight.grad is not None
    assert torch.count_nonzero(condition.projection[0].weight.grad) == 0


def point_feedback_config(**changes):
    return curve_config(
        curve_point_feedback={"enabled": True, "hidden_dim": 16, "feature_dim": 8},
        **changes,
    )


def save_point_feedback_baseline(path):
    model = AutoEncoder_intersection(curve_config()).eval()
    with torch.no_grad():
        for head in (
            model.decoder.refiner.face_delta,
            model.decoder.refiner.edge_delta,
            model.decoder.edge_geometry.conditioning.point_affine,
            model.decoder.edge_geometry.conditioning.bbox_affine,
        ):
            head.weight.normal_(std=.02)
            head.bias.normal_(std=.01)
    torch.save({
        "epoch": 157, "global_step": 57354,
        "state_dict": {"model." + key: value for key, value in model.state_dict().items()},
    }, path)
    return model


def test_point_feedback_migration_preserves_every_trained_weight_prediction_and_loss(tmp_path):
    torch.manual_seed(95)
    path = tmp_path / "conditioned.ckpt"
    old = save_point_feedback_baseline(path)
    new = AutoEncoder_intersection(point_feedback_config(init_curve_feedback_checkpoint=str(path))).eval()
    retained = old.state_dict()
    added = set(new.state_dict()) - set(retained)
    assert added and all(key.startswith("decoder.curve_point_feedback.") for key in added)
    for key, value in retained.items():
        torch.testing.assert_close(new.state_dict()[key], value, rtol=0, atol=0)
    assert [key for key, _ in new.named_parameters() if key not in added] == list(dict(old.named_parameters()))
    report = new.initialization_report
    assert report["source_global_step"] == 57354
    assert report["initialized_curve_point_feedback_keys"] == sorted(added)
    with torch.no_grad():
        first_loss, first = old(encoder_batch(), v_test=True)
        second_loss, second = new(encoder_batch(), v_test=True)
    for key in first_loss:
        torch.testing.assert_close(second_loss[key], first_loss[key], rtol=0, atol=0)
    for a, b in zip(first["items"], second["items"]):
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])


@pytest.mark.parametrize("corruption", [
    "missing_classifier", "missing_conditioning", "missing_refiner", "missing_inter",
    "extra", "shape", "already_feedback", "duplicate_normalized_key", "non_tensor",
])
def test_point_feedback_migration_rejects_partial_or_incompatible_sources(tmp_path, corruption):
    path = tmp_path / "conditioned.ckpt"
    save_point_feedback_baseline(path)
    checkpoint = torch.load(path, weights_only=False)
    state = checkpoint["state_dict"]
    if corruption.startswith("missing_"):
        prefixes = {
            "missing_classifier": "model.decoder.classifier.",
            "missing_conditioning": "model.decoder.edge_geometry.conditioning.",
            "missing_refiner": "model.decoder.refiner.",
            "missing_inter": "model.decoder.inter.",
        }
        for key in list(state):
            if key.startswith(prefixes[corruption]):
                del state[key]
    elif corruption == "extra":
        state["model.extra.weight"] = torch.ones(1)
    elif corruption == "shape":
        state["model.decoder.classifier.bias"] = torch.ones(2)
    elif corruption == "non_tensor":
        state["model.decoder.classifier.bias"] = "not a tensor"
    elif corruption == "duplicate_normalized_key":
        state["decoder.classifier.bias"] = state["model.decoder.classifier.bias"]
    else:
        model = AutoEncoder_intersection(point_feedback_config())
        checkpoint["state_dict"] = {"model." + key: value for key, value in model.state_dict().items()}
    torch.save(checkpoint, path)
    with pytest.raises((ValueError, RuntimeError)):
        AutoEncoder_intersection(point_feedback_config(init_curve_feedback_checkpoint=str(path)))


def test_point_feedback_frozen_training_reaches_output_then_hidden_layers(tmp_path):
    torch.manual_seed(96)
    path = tmp_path / "conditioned.ckpt"
    save_point_feedback_baseline(path)
    model = AutoEncoder_intersection(point_feedback_config(
        init_curve_feedback_checkpoint=str(path), freeze_encoder=True,
    )).train()
    assert not model.encoder.training and not model.posterior.training
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    loss, _ = model(encoder_batch())
    assert all(value.isfinite().all() for value in loss.values())
    loss["total_loss"].backward()
    block = model.decoder.curve_point_feedback
    assert block.delta.weight.grad is not None and block.delta.weight.grad.abs().sum() > 0
    assert block.point_encoder[0].weight.grad is not None
    assert torch.count_nonzero(block.point_encoder[0].weight.grad) == 0
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    assert all(parameter.grad is None for parameter in model.posterior.parameters())
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model(encoder_batch())[0]["total_loss"].backward()
    assert block.point_encoder[0].weight.grad.abs().sum() > 0
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)


def test_point_feedback_full_checkpoint_and_optimizer_restore_are_strict(tmp_path):
    torch.manual_seed(97)
    model = AutoEncoder_intersection(point_feedback_config())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model(encoder_batch())[0]["total_loss"].backward()
    optimizer.step()
    path = tmp_path / "feedback.ckpt"
    torch.save({
        "state_dict": {"model." + key: value for key, value in model.state_dict().items()},
        "optimizer_states": [optimizer.state_dict()], "epoch": 1, "global_step": 1,
    }, path)
    loaded = torch.load(path, weights_only=False)
    strict = AutoEncoder_intersection(point_feedback_config(checkpoint=str(path))).eval()
    resumed = AutoEncoder_intersection(point_feedback_config(restoring_checkpoint=True)).eval()
    resumed.load_state_dict({key.removeprefix("model."): value for key, value in loaded["state_dict"].items()}, strict=True)
    restored_optimizer = torch.optim.Adam(resumed.parameters(), lr=1e-4)
    restored_optimizer.load_state_dict(loaded["optimizer_states"][0])
    model.eval()
    with torch.no_grad():
        expected = model(encoder_batch(), v_test=True)[0]["total_loss"]
        for candidate in (strict, resumed):
            actual = candidate(encoder_batch(), v_test=True)[0]["total_loss"]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for first, second in zip(optimizer.state.values(), restored_optimizer.state.values()):
        for key in first:
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    loaded["state_dict"].pop(next(key for key in loaded["state_dict"] if key.startswith("model.decoder.curve_point_feedback.")))
    torch.save(loaded, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        AutoEncoder_intersection(point_feedback_config(checkpoint=str(path)))
