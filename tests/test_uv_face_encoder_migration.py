"""UV encoder migration/resume contracts; these are not quality experiments."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from src.models.blocks import UVFaceEncoder
from src.models.vae import AutoEncoder_intersection
from tests.test_intersection_vae import curve_config, encoder_batch


FACE_PREFIX = "encoder.face_coords."


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def uv_config(**changes):
    config = curve_config()
    config.update({
        "face_encoder": {
            "kind": "uv_attention", "hidden_dim": 16, "embedding_dim": 8,
            "num_layers": 1, "nhead": 2, "dim_feedforward": 32, "dropout": 0.,
        },
        "init_face_encoder_checkpoint": None,
        **changes,
    })
    return config


def save_conditioned_source(path):
    model = AutoEncoder_intersection(curve_config()).eval()
    # Migration must preserve learned values, including originally zero heads.
    with torch.no_grad():
        model.decoder.refiner.face_delta.weight.normal_(std=.01)
        model.decoder.refiner.edge_delta.weight.normal_(std=.01)
        for parameter in model.decoder.edge_geometry.conditioning.parameters():
            parameter.normal_(std=.01)
    torch.save({
        "epoch": 157, "global_step": 57354,
        "state_dict": {"model." + key: value for key, value in model.state_dict().items()},
    }, path)
    return model


def test_default_and_explicit_cnn_keep_weights_rng_and_predictions_identical():
    torch.manual_seed(103)
    default = AutoEncoder_intersection(curve_config()).eval()
    default_random = torch.rand(9)
    torch.manual_seed(103)
    explicit = AutoEncoder_intersection(curve_config(face_encoder={"kind": "cnn"})).eval()
    explicit_random = torch.rand(9)
    assert isinstance(default.encoder.face_coords, nn.Sequential)
    assert list(default.state_dict()) == list(explicit.state_dict())
    for key, expected in default.state_dict().items():
        torch.testing.assert_close(explicit.state_dict()[key], expected, rtol=0, atol=0)
    torch.testing.assert_close(explicit_random, default_random, rtol=0, atol=0)
    with torch.no_grad():
        before = default.encode(encoder_batch())
        after = explicit.encode(encoder_batch())
    for key in before:
        torch.testing.assert_close(after[key], before[key], rtol=0, atol=0)


def test_uv_construction_preserves_nonface_parameters_and_rng_sequence():
    torch.manual_seed(107)
    cnn = AutoEncoder_intersection(curve_config())
    cnn_random = torch.rand(9)
    torch.manual_seed(107)
    uv = AutoEncoder_intersection(uv_config())
    uv_random = torch.rand(9)
    assert isinstance(uv.encoder.face_coords, UVFaceEncoder)
    for key, expected in cnn.state_dict().items():
        if not key.startswith(FACE_PREFIX):
            torch.testing.assert_close(uv.state_dict()[key], expected, rtol=0, atol=0)
    torch.testing.assert_close(uv_random, cnn_random, rtol=0, atol=0)


@pytest.mark.parametrize("prefixed", [True, False])
def test_migration_replaces_only_face_cnn_and_retains_every_other_tensor(tmp_path, prefixed):
    torch.manual_seed(109)
    path = tmp_path / "conditioned.ckpt"
    old = save_conditioned_source(path)
    if not prefixed:
        source = torch.load(path, weights_only=False)
        source["state_dict"] = {key.removeprefix("model."): value
                                for key, value in source["state_dict"].items()}
        torch.save(source, path)
    new = AutoEncoder_intersection(uv_config(init_face_encoder_checkpoint=str(path))).eval()
    before, after = old.state_dict(), new.state_dict()
    retained = {key for key in before if not key.startswith(FACE_PREFIX)}
    assert retained == {key for key in after if not key.startswith(FACE_PREFIX)}
    for key in retained:
        torch.testing.assert_close(after[key], before[key], rtol=0, atol=0)
    assert "encoder.face_coords.0.weight" not in after
    assert "encoder.face_coords.geometry_embedding.0.weight" in after
    report = new.initialization_report
    assert report["source_epoch"] == 157 and report["source_global_step"] == 57354
    assert report["retained_state_keys"] == len(retained)
    assert report["discarded_face_encoder_state_keys"] == len(before) - len(retained)
    assert report["discarded_face_encoder_parameters"] == sum(
        value.numel() for key, value in before.items() if key.startswith(FACE_PREFIX)
    )
    assert report["initialized_face_encoder_state_keys"] == sorted(set(after) - retained)
    assert report["initialized_face_encoder_parameters"] == sum(
        parameter.numel() for parameter in new.encoder.face_coords.parameters()
    )
    # Edge encoding has no dependency on the newly initialized face encoder.
    with torch.no_grad():
        torch.testing.assert_close(
            new.encode(encoder_batch())["edge_features"],
            old.encode(encoder_batch())["edge_features"], rtol=0, atol=0,
        )


@pytest.mark.parametrize("corruption", [
    "missing_cnn", "missing_encoder",
    "missing_posterior", "missing_decoder", "missing_conditioning", "retained_shape",
    "extra", "duplicate_prefix", "point_feedback", "edge_value", "local_inter",
    "already_uv",
])
def test_migration_rejects_incomplete_wrong_or_previously_modified_sources(tmp_path, corruption):
    path = tmp_path / "source.ckpt"
    save_conditioned_source(path)
    checkpoint = torch.load(path, weights_only=False)
    state = checkpoint["state_dict"]
    if corruption == "missing_cnn":
        state.pop("model.encoder.face_coords.0.weight")
    elif corruption.startswith("missing_"):
        prefix = {
            "missing_encoder": "model.encoder.edge_coords.",
            "missing_posterior": "model.posterior.",
            "missing_decoder": "model.decoder.classifier.",
            "missing_conditioning": "model.decoder.edge_geometry.conditioning.",
        }[corruption]
        state.pop(next(key for key in state if key.startswith(prefix)))
    elif corruption == "retained_shape":
        state["model.decoder.classifier.bias"] = torch.zeros(2)
    elif corruption == "duplicate_prefix":
        state["decoder.classifier.bias"] = state["model.decoder.classifier.bias"]
    elif corruption == "already_uv":
        checkpoint["state_dict"] = AutoEncoder_intersection(uv_config()).state_dict()
    else:
        extra = {
            "extra": "model.unknown.weight",
            "point_feedback": "model.decoder.curve_point_feedback.delta.weight",
            "edge_value": "model.encoder.graph_face_edge.0.value_net.0.weight",
            "local_inter": "model.decoder.inter.query_projection.weight",
        }[corruption]
        state[extra] = torch.zeros(1)
    torch.save(checkpoint, path)
    with pytest.raises((ValueError, RuntimeError), match="Face encoder|Ambiguous|size mismatch"):
        AutoEncoder_intersection(uv_config(init_face_encoder_checkpoint=str(path)))


def test_full_checkpoint_and_adam_resume_preserve_trained_uv_and_inference(tmp_path):
    source = tmp_path / "conditioned.ckpt"
    save_conditioned_source(source)
    model = AutoEncoder_intersection(uv_config(init_face_encoder_checkpoint=str(source))).train()
    assert model.encoder.training and model.posterior.training and model.decoder.training
    assert all(parameter.requires_grad for parameter in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    losses, _ = model(encoder_batch())
    assert all(torch.isfinite(value).all() for value in losses.values())
    losses["total_loss"].backward()
    for parameter in (
        model.encoder.face_coords.geometry_embedding[0].weight,
        model.encoder.face_coords.uv_embedding[0].weight,
        model.encoder.face_coords.query,
        model.encoder.edge_coords[0].weight,
        model.posterior.projection[0].weight,
    ):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    optimizer.step()
    path = tmp_path / "uv.ckpt"
    torch.save({
        "state_dict": {"model." + key: value for key, value in model.state_dict().items()},
        "optimizer_states": [optimizer.state_dict()], "epoch": 1, "global_step": 1,
    }, path)
    loaded = torch.load(path, weights_only=False)
    strict = AutoEncoder_intersection(uv_config(checkpoint=str(path))).eval()
    resumed = AutoEncoder_intersection(uv_config(restoring_checkpoint=True)).eval()
    resumed.load_state_dict({key.removeprefix("model."): value
                             for key, value in loaded["state_dict"].items()}, strict=True)
    restored_optimizer = torch.optim.Adam(resumed.parameters(), lr=1e-4)
    restored_optimizer.load_state_dict(loaded["optimizer_states"][0])
    model.eval()
    with torch.no_grad():
        expected_loss, expected = model(encoder_batch(), v_test=True)
        for candidate in (strict, resumed):
            for key, value in model.state_dict().items():
                torch.testing.assert_close(candidate.state_dict()[key], value, rtol=0, atol=0)
            actual_loss, actual = candidate(encoder_batch(), v_test=True)
            torch.testing.assert_close(actual_loss["total_loss"], expected_loss["total_loss"], rtol=0, atol=0)
            for before, after in zip(expected["items"], actual["items"]):
                for key in before:
                    np.testing.assert_array_equal(after[key], before[key])
    assert len(optimizer.state) == len(restored_optimizer.state) > 0
    for first, second in zip(optimizer.state.values(), restored_optimizer.state.values()):
        assert first.keys() == second.keys()
        for key in first:
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    assert optimizer.param_groups[0]["lr"] == restored_optimizer.param_groups[0]["lr"]

    for corruption in ("missing_uv", "old_cnn", "old_source"):
        damaged = deepcopy(loaded)
        if corruption == "missing_uv":
            damaged["state_dict"].pop("model.encoder.face_coords.query")
        elif corruption == "old_cnn":
            damaged["state_dict"]["model.encoder.face_coords.0.weight"] = torch.zeros(1)
        else:
            damaged = torch.load(source, weights_only=False)
        torch.save(damaged, path)
        with pytest.raises(RuntimeError, match="Missing key|Unexpected key"):
            AutoEncoder_intersection(uv_config(checkpoint=str(path)))


@pytest.mark.parametrize("changes,match", [
    ({"checkpoint": "full.ckpt", "init_face_encoder_checkpoint": "unused.ckpt"}, "only one"),
    ({"restoring_checkpoint": True, "init_face_encoder_checkpoint": "unused.ckpt"}, "Resume"),
    ({"face_encoder": {"kind": "unknown"}}, "Unknown face_encoder.kind"),
])
def test_uv_configuration_selects_unambiguous_model_and_weights(changes, match):
    with pytest.raises(ValueError, match=match):
        AutoEncoder_intersection(uv_config(**changes))


def test_uv_frozen_inference_is_allowed(tmp_path):
    model = AutoEncoder_intersection(uv_config())
    path = tmp_path / "uv.pt"
    torch.save(model.state_dict(), path)
    frozen = AutoEncoder_intersection(uv_config(
        checkpoint=str(path), mode="frozen_inference", freeze_encoder=True,
    ))
    assert not frozen.training and not frozen.encoder.training
    assert all(not parameter.requires_grad for parameter in frozen.parameters())


def test_trainer_resume_clears_uv_migration_without_opening_source(tmp_path, monkeypatch):
    from hydra import compose, initialize_config_dir
    from src import train

    with initialize_config_dir(version_base="1.3", config_dir=str(Path(__file__).resolve().parents[1] / "configs")):
        cfg = compose(config_name="train_vae_uv_encoder", overrides=[
            "dataset.raw_data_root=unused", "dataset.train_dataset=unused",
            "dataset.val_dataset=unused", "dataset.test_dataset=unused",
            f"trainer.output_dir={tmp_path}", "trainer.resume_from_checkpoint=full.ckpt",
            "model.init_face_encoder_checkpoint=missing-source.ckpt",
        ])
    captured = {}

    def build_model(model_cfg, condition):
        captured.update(dict(model_cfg))
        return nn.Linear(2, 2)

    monkeypatch.setattr(train, "build_model", build_model)
    monkeypatch.setattr(train, "_load_dataset_class", lambda config: object)
    train.UnifiedTrainer(cfg)
    assert captured["restoring_checkpoint"] is True
    assert captured["face_encoder"]["kind"] == "uv_attention"
    for key in ("checkpoint", "init_legacy_checkpoint", "init_refinement_checkpoint",
                "init_curve_feedback_checkpoint", "init_encoder_edge_value_checkpoint",
                "init_face_encoder_checkpoint"):
        assert captured[key] is None
