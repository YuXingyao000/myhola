"""Warm widening preserves mean decoding and opens trainable latent capacity."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn

from src.brepnet.models.vae import AutoEncoder_intersection
from src.brepnet.test.test_intersection_vae import encoder_batch
from src.brepnet.test.test_shared_curve_uv import config as shared_config
from src.brepnet.test.test_uv_boundary_model import add_targets


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def config(**changes):
    # Keep residual projections present on both sides, as in the full model.
    # hidden_channels=16 would coincide with old E / new F and change key sets.
    return shared_config(hidden_channels=64, **changes)


@pytest.fixture
def trained_source(tmp_path):
    torch.manual_seed(211)
    model = AutoEncoder_intersection(config()).eval()
    # Exercise learned feedback/fusion, rather than only their zero residuals.
    with torch.no_grad():
        for head in (
            model.decoder.refiner.face_delta,
            model.decoder.refiner.edge_delta,
            model.decoder.shared_curve.residual[-1],
            model.decoder.edge_geometry.conditioning.point_affine,
            model.decoder.edge_geometry.conditioning.bbox_affine,
        ):
            head.weight.normal_(std=.02)
            head.bias.normal_(std=.01)
    path = tmp_path / "shared.ckpt"
    torch.save({
        "epoch": 116, "global_step": 42471,
        "state_dict": {f"model.{key}": value for key, value in model.state_dict().items()},
    }, path)
    return model, path


def expanded(path, **changes):
    return AutoEncoder_intersection(config(
        latent_channels=4, gaussian_weights=5e-7,
        init_latent_expansion_checkpoint=str(path), **changes,
    ))


def assert_decoding_close(actual, expected):
    torch.testing.assert_close(actual.pairs.indices, expected.pairs.indices)
    for stage_name in ("coarse", "refined"):
        new_stage, old_stage = getattr(actual, stage_name), getattr(expected, stage_name)
        torch.testing.assert_close(new_stage.edge_logits, old_stage.edge_logits, atol=2e-5, rtol=2e-4)
        for primitive in ("face_geometry", "edge_geometry"):
            new_geometry, old_geometry = getattr(new_stage, primitive), getattr(old_stage, primitive)
            for field in ("local_points", "center_scale", "points"):
                torch.testing.assert_close(
                    getattr(new_geometry, field), getattr(old_geometry, field), atol=2e-5, rtol=2e-4,
                )
    torch.testing.assert_close(actual.edge_uv, expected.edge_uv, atol=2e-5, rtol=2e-4)


def test_widening_preserves_mean_geometry_uv_and_unchanged_backbones(trained_source):
    old, path = trained_source
    new = expanded(path).eval()
    assert new.face_feature_dim == 2 * old.face_feature_dim
    assert new.decoder.edge_dim == 2 * old.decoder.edge_dim
    old_state, new_state = old.state_dict(), new.state_dict()
    assert old_state.keys() == new_state.keys()
    for key, value in old_state.items():
        if value.shape == new_state[key].shape:
            torch.testing.assert_close(new_state[key], value, rtol=0, atol=0)
    assert new.initialization_report["source_global_step"] == 42471

    batch = encoder_batch()
    with torch.no_grad():
        old_encoding, new_encoding = old.encode(batch), new.encode(batch)
        for key in ("face_features", "edge_features"):
            torch.testing.assert_close(new_encoding[key], old_encoding[key].repeat(1, 2), atol=2e-5, rtol=2e-4)
        old_mean, old_kl, old_stats = old.sample(old_encoding["face_features"], True)
        new_mean, new_kl, new_stats = new.sample(new_encoding["face_features"], True)
        torch.testing.assert_close(new_mean, old_mean.repeat(1, 2), atol=2e-5, rtol=2e-4)
        torch.testing.assert_close(new_kl, old_kl)
        old_mu, old_std = old_stats.chunk(2, dim=-1)
        torch.testing.assert_close(
            new_stats, torch.cat((old_mu.repeat(1, 2), old_std.repeat(1, 2)), dim=-1),
            atol=2e-5, rtol=2e-4,
        )
        before = old.decode_state(old_mean, batch["face_counts"])
        after = new.decode_state(new_mean, batch["face_counts"])
    assert_decoding_close(after, before)


def test_sampled_training_updates_encoder_and_separates_copied_channels(trained_source):
    _, path = trained_source
    model = expanded(path).train()
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert model.encoder.training and model.posterior.training and model.decoder.training
    torch.manual_seed(223)
    losses, _ = model(add_targets(encoder_batch()))
    assert all(torch.isfinite(value).all() for value in losses.values())
    losses["total_loss"].backward()
    for parameter in (
        model.encoder.face_coords[0].weight,
        model.encoder.edge_coords[0].weight,
        model.posterior.projection[-1].weight,
        model.decoder.inter.attn_proj_in.weight,
        model.decoder.shared_curve.residual[-1].weight,
        model.decoder.uv_boundary.context[0].weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    # Independent posterior noise and asymmetric read weights let the copied
    # channels learn separately, including the CNN before the posterior.
    posterior = model.posterior.projection[-1].weight
    first, second = posterior.grad.chunk(2, dim=0)
    assert not torch.allclose(first, second, atol=1e-9, rtol=1e-5)
    face_projection = [layer for layer in model.encoder.face_coords if isinstance(layer, nn.Conv2d)][-1]
    first, second = face_projection.weight.grad.chunk(2, dim=0)
    assert not torch.allclose(first, second, atol=1e-9, rtol=1e-5)


def test_expanded_full_checkpoint_and_optimizer_resume_are_strict(trained_source, tmp_path):
    _, source = trained_source
    model = expanded(source).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    model(add_targets(encoder_batch()))[0]["total_loss"].backward()
    optimizer.step()
    path = tmp_path / "expanded.ckpt"
    checkpoint = {
        "state_dict": {f"model.{key}": value for key, value in model.state_dict().items()},
        "optimizer_states": [optimizer.state_dict()], "epoch": 1, "global_step": 1,
    }
    torch.save(checkpoint, path)
    loaded = AutoEncoder_intersection(config(
        latent_channels=4, gaussian_weights=5e-7, checkpoint=str(path),
    )).eval()
    resumed = AutoEncoder_intersection(config(
        latent_channels=4, gaussian_weights=5e-7, restoring_checkpoint=True,
    )).eval()
    resumed.load_state_dict(model.state_dict(), strict=True)
    resumed_optimizer = torch.optim.Adam(resumed.parameters(), lr=1.)
    resumed_optimizer.load_state_dict(checkpoint["optimizer_states"][0])
    assert resumed_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    for old_state, new_state in zip(optimizer.state.values(), resumed_optimizer.state.values()):
        for key in old_state:
            torch.testing.assert_close(new_state[key], old_state[key], rtol=0, atol=0)
    z = torch.randn(3, model.face_feature_dim)
    with torch.no_grad():
        reference = model.eval().decode_state(z)
        assert_decoding_close(loaded.decode_state(z), reference)
        assert_decoding_close(resumed.decode_state(z), reference)

    damaged = deepcopy(checkpoint)
    damaged["state_dict"].pop("model.decoder.classifier.bias")
    torch.save(damaged, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        AutoEncoder_intersection(config(latent_channels=4, checkpoint=str(path)))


@pytest.mark.parametrize("corruption", ["missing_key", "unrelated_width"])
def test_widening_rejects_incomplete_source_or_unrelated_architecture_changes(trained_source, corruption):
    _, path = trained_source
    if corruption == "missing_key":
        checkpoint = torch.load(path, weights_only=False)
        checkpoint["state_dict"].pop("model.decoder.inter.layers.0.linear1.weight")
        torch.save(checkpoint, path)
        changes = {}
    else:
        changes = {"bottleneck_dim": 32}
    with pytest.raises((ValueError, RuntimeError)):
        expanded(path, **changes)


def compose_expansion(name, overrides=()):
    config_dir = str(Path(__file__).resolve().parents[3] / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        return compose(config_name=name, overrides=list(overrides))


def test_control_retains_same_architecture_and_training_budget():
    candidate = compose_expansion("train_vae_latent_expansion")
    control = compose_expansion("train_vae_latent_expansion_control")
    assert candidate.model.latent_channels == 16
    assert control.model.latent_channels == 8
    assert not candidate.model.freeze_encoder and not control.model.freeze_encoder
    candidate_model = OmegaConf.to_container(candidate.model, resolve=False)
    control_model = OmegaConf.to_container(control.model, resolve=False)
    assert candidate.model.gaussian_weights * 2 == control.model.gaussian_weights
    for name in ("latent_channels", "gaussian_weights", "init_latent_expansion_checkpoint", "checkpoint"):
        candidate_model.pop(name, None)
        control_model.pop(name, None)
    assert candidate_model == control_model
    for name in ("devices", "batch_size", "learning_rate", "max_steps", "max_epochs", "precision", "seed"):
        assert candidate.trainer[name] == control.trainer[name]
    assert candidate.trainer.wandb.enabled and not candidate.trainer.wandb.offline
    assert OmegaConf.to_container(candidate.dataset) == OmegaConf.to_container(control.dataset)


def test_trainer_resume_clears_expansion_source_without_reading_it(tmp_path, monkeypatch):
    from src.brepnet import train

    cfg = compose_expansion("train_vae_latent_expansion", [
        "dataset.raw_data_root=unused", "dataset.train_dataset=unused",
        "dataset.val_dataset=unused", "dataset.test_dataset=unused", "dataset.edge_uv_root=unused",
        f"trainer.output_dir={tmp_path}", "trainer.resume_from_checkpoint=expanded.ckpt",
        "model.init_latent_expansion_checkpoint=missing-source.ckpt",
    ])
    captured = {}

    def build_model(model_cfg, condition):
        captured.update(dict(model_cfg))
        return nn.Linear(2, 2)

    monkeypatch.setattr(train, "build_model", build_model)
    monkeypatch.setattr(train, "_load_dataset_class", lambda config: object)
    train.UnifiedTrainer(cfg)
    assert captured["restoring_checkpoint"] is True
    assert captured["init_latent_expansion_checkpoint"] is None
    assert captured["latent_channels"] == 16
