"""Decoder widening preserves initialization, not a claim of improved CAD quality."""
from copy import deepcopy

import pytest
import torch

from src.models.vae import AutoEncoder_intersection
from tests.test_curve_head_roles import model_config
from tests.test_intersection_decoder import activate_residuals
from tests.test_intersection_vae import encoder_batch
from tests.test_uv_boundary_model import add_targets
from tools.migrate_checkpoint import migrate


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def source_model(kind="conditional_attention"):
    torch.manual_seed(691)
    cfg = model_config(fp32=True)
    cfg["intersection"] = {"kind": kind, "candidate_chunk_size": 4,
                           "checkpoint_candidates": False}
    cfg["loss_weights"]["feature_weight"] = 0.
    model = AutoEncoder_intersection(cfg).eval()
    activate_residuals(model.decoder)
    if kind == "conditional_attention":
        torch.nn.init.normal_(model.decoder.inter.spatial_memory.local_projection.weight, std=.02)
    return model, cfg


def widened(model, cfg):
    checkpoint = {"state_dict": {"model." + k: v for k, v in model.state_dict().items()},
                  "hyper_parameters": {"model": deepcopy(cfg)}, "optimizer_states": [{}],
                  "loops": {}, "epoch": 18, "global_step": 7885}
    # Real training checkpoints retain the original run's initialization path.
    checkpoint["hyper_parameters"]["model"]["checkpoint"] = "old-initializer.ckpt"
    width = model.face_feature_dim * 4
    migrate(checkpoint, decoder_face_dim=width)
    candidate = AutoEncoder_intersection(checkpoint["hyper_parameters"]["model"]).eval()
    candidate.load_state_dict({k.removeprefix("model."): v for k, v in checkpoint["state_dict"].items()}, strict=True)
    return candidate, checkpoint


def test_default_width_preserves_state_and_rng():
    _, cfg = source_model()
    torch.manual_seed(692)
    implicit = AutoEncoder_intersection(cfg)
    rng = torch.get_rng_state()
    torch.manual_seed(692)
    explicit = AutoEncoder_intersection({**cfg, "decoder_face_dim": 8})
    assert torch.equal(torch.get_rng_state(), rng)
    for key, value in implicit.state_dict().items():
        torch.testing.assert_close(explicit.state_dict()[key], value, atol=0, rtol=0)


@pytest.mark.parametrize("kind", ["legacy", "conditional_attention"])
def test_widening_preserves_geometry_latent_and_export(kind):
    source, cfg = source_model(kind)
    candidate, checkpoint = widened(source, cfg)
    assert candidate.face_feature_dim == source.face_feature_dim == 8
    assert candidate.decoder.face_dim == 32
    assert candidate.decoder.edge_dim == 16
    assert not candidate.freeze_encoder
    assert "optimizer_states" not in checkpoint and "loops" not in checkpoint
    assert checkpoint["hyper_parameters"]["model"]["checkpoint"] is None
    assert checkpoint["initialization_only"]["source_initializers"]["checkpoint"] == "old-initializer.ckpt"
    assert "already applied" in migrate(checkpoint, decoder_face_dim=32)[-1]
    with pytest.raises(ValueError, match="already migrated"):
        migrate(checkpoint, decoder_face_dim=64)
    changed = set(checkpoint["initialization_only"]["changed_tensors"])
    assert len(changed) == (14 if kind == "conditional_attention" else 13)
    for key, value in source.state_dict().items():
        if "model." + key not in changed:
            torch.testing.assert_close(candidate.state_dict()[key], value, atol=0, rtol=0)
    batch = add_targets(encoder_batch())
    with torch.no_grad():
        for key, value in source.encode(batch).items():
            torch.testing.assert_close(candidate.encode(batch)[key], value, atol=0, rtol=0)
        latent = torch.randn(5, 8)
        first = source.decode(latent, batch["face_counts"])
        second = candidate.decode(latent, batch["face_counts"])
        for a, b in ((first.coarse, second.coarse), (first.refined, second.refined)):
            for head in ("face_geometry", "edge_geometry"):
                for field in ("points", "local_points", "center_scale"):
                    torch.testing.assert_close(getattr(getattr(a, head), field),
                                               getattr(getattr(b, head), field), atol=3e-6, rtol=3e-5)
            torch.testing.assert_close(a.edge_logits, b.edge_logits, atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(first.edge_uv, second.edge_uv, atol=3e-6, rtol=3e-5)
        old_losses, old_items = source(batch, v_test=True)
        new_losses, new_items = candidate(batch, v_test=True)
        for key in old_losses:
            torch.testing.assert_close(old_losses[key], new_losses[key], atol=3e-6, rtol=3e-5)
        assert [x["pred_face"].shape for x in old_items["items"]] == [x["pred_face"].shape for x in new_items["items"]]
        empty = candidate.decode(torch.randn(2, 8), torch.tensor([1, 1]))
        assert empty.refined.edge_geometry.points.shape == (0, 16, 3)
        assert empty.edge_uv.shape == (0, 2, 16, 2)


def test_adam_separates_replicated_channels_and_uv_detach_survives():
    source, cfg = source_model()
    candidate, _ = widened(source, cfg)
    z, counts = torch.randn(3, 8), torch.tensor([3])
    target = torch.randn(3, 16, 16, 3)
    optimizer = torch.optim.Adam(candidate.parameters(), lr=2e-5)
    projection = candidate.decoder.face_features.output_projection
    torch.testing.assert_close(projection.weight[:8], projection.weight[8:16], atol=0, rtol=0)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        output = candidate.decode(z, counts)
        (output.refined.face_geometry.points - target).square().mean().backward()
        assert projection.weight.grad[8:].abs().sum() > 0
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in candidate.parameters())
        optimizer.step()
    assert (projection.weight[:8] - projection.weight[8:16]).abs().max() > 1e-7
    optimizer.zero_grad(set_to_none=True)
    candidate.decode(z, counts).edge_uv.square().mean().backward()
    assert all(p.grad is None for p in candidate.decoder.edge_geometry.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in candidate.decoder.shared_curve.parameters())


def test_optional_point_feedback_input_is_migrated_and_local_source_is_rejected():
    _, cfg = source_model("legacy")
    cfg["shared_curve"]["enabled"] = False
    cfg["curve_point_feedback"] = {"enabled": True, "hidden_dim": 16, "feature_dim": 8}
    source = AutoEncoder_intersection(cfg).eval()
    candidate, _ = widened(source, cfg)
    z, counts = torch.randn(3, 8), torch.tensor([3])
    with torch.no_grad():
        torch.testing.assert_close(source.decode(z, counts).refined.edge_geometry.points,
                                   candidate.decode(z, counts).refined.edge_geometry.points,
                                   atol=3e-6, rtol=3e-5)
    cfg["intersection"]["kind"] = "local_attention"
    with pytest.raises(ValueError, match="local attention needs its own mapping"):
        widened(source, cfg)


def test_migration_rejects_width_that_removes_residual_projection():
    source, cfg = source_model()
    checkpoint = {"state_dict": {"model." + k: v for k, v in source.state_dict().items()},
                  "hyper_parameters": {"model": cfg}}
    with pytest.raises(ValueError, match="shortcut structure"):
        migrate(checkpoint, decoder_face_dim=16)
