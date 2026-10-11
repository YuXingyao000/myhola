"""Migration and pipeline contracts; these do not establish research quality."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.brepnet.dataset import denormalize_coord1112, normalize_coord1112
from src.brepnet.models.intersection_vae import AutoEncoder_intersection
from src.brepnet.models.vae import AutoEncoder_light


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
