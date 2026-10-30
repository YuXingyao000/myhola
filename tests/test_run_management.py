"""Run isolation, resumption and per-object scientific export contracts."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.common.prediction_export import save_item_predictions, vae_prediction_items
from src.common.run_management import prepare_run, resolve_run_directory


@pytest.fixture
def isolated_launch(monkeypatch):
    for key in ("BREPNET_ACTIVE_RUN_DIR", "BREPNET_INVOCATION_ID", "RANK", "LOCAL_RANK"):
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_resume_run_selection_and_explicit_override(tmp_path, isolated_launch):
    checkpoint = tmp_path / "original" / "weights" / "last.ckpt"
    assert resolve_run_directory(tmp_path, "new_name", None, checkpoint) == str(tmp_path / "original")
    assert resolve_run_directory(tmp_path, "new_name", tmp_path / "explicit", checkpoint) == str(tmp_path / "explicit")
    legacy = tmp_path / "legacy" / "checkpoints" / "epoch=4-val" / "loss=.1.ckpt"
    assert resolve_run_directory(tmp_path, "unused", None, legacy) == str(tmp_path / "legacy")
    assert resolve_run_directory(tmp_path, "unused", None, tmp_path / "legacy/checkpoints/last.ckpt") == str(tmp_path / "legacy")
    with pytest.raises(ValueError, match="trainer.run_dir"):
        resolve_run_directory(tmp_path, "unused", None, tmp_path / "unknown.ckpt")


def test_resume_rejects_unrecognized_paths_beneath_global_checkpoint_root(tmp_path, isolated_launch):
    output_root = tmp_path / "checkpoints"
    run = output_root / "experiment" / "run-timestamp"
    for checkpoint in (
        output_root / "unknown.ckpt",
        run / "other" / "unknown.ckpt",
        run / "weights" / "other" / "unknown.ckpt",
        run / "checkpoints" / "other" / "unknown.ckpt",
        run / "checkpoints" / "epoch=4-val" / "unknown.ckpt",
    ):
        with pytest.raises(ValueError, match="trainer.run_dir"):
            resolve_run_directory(output_root, "unused", None, checkpoint)
    assert resolve_run_directory(output_root, "unused", None, run / "weights/last.ckpt") == str(run)
    assert resolve_run_directory(output_root, "unused", None, run / "checkpoints/epoch=4-val/loss=.1.ckpt") == str(run)
    assert resolve_run_directory(output_root, "unused", run, run / "other/unknown.ckpt") == str(run)


def test_new_launches_are_isolated_and_resume_keeps_metadata(tmp_path, isolated_launch):
    monkeypatch = isolated_launch
    monkeypatch.setenv("BREPNET_INVOCATION_ID", "first")
    first = Path(resolve_run_directory(tmp_path, "joint_boundary"))
    cfg = OmegaConf.create({
        "trainer": {"seed": 7, "exp_name": "joint_boundary", "resume_from_checkpoint": None, "evaluate": False},
        "eval": {"enabled": False},
    })
    record = prepare_run(cfg, first)
    assert record["wandb_id"]
    assert (first / "metadata/invocations/first/config.yaml").is_file()
    # Lightning children inherit the same launch id and may prepare the same run.
    assert prepare_run(cfg, first)["wandb_id"] == record["wandb_id"]
    monkeypatch.delenv("BREPNET_ACTIVE_RUN_DIR")
    monkeypatch.setenv("BREPNET_INVOCATION_ID", "second")
    assert Path(resolve_run_directory(tmp_path, "joint_boundary")) != first
    with pytest.raises(ValueError, match="already exists"):
        prepare_run(cfg, first)
    cfg.trainer.resume_from_checkpoint = str(first / "weights/last.ckpt")
    resumed = prepare_run(cfg, first)
    assert resumed["wandb_id"] == record["wandb_id"]
    assert (first / "metadata/invocations/first/config.yaml").is_file()
    assert (first / "metadata/invocations/second/config.yaml").is_file()
    command = json.loads((first / "metadata/invocations/second/command.json").read_text())
    assert command["seed"] == 7


def test_joint_exports_preserve_variable_objects_and_representation(tmp_path):
    prefixes = ["one", "two"]
    items = [
        {"pred_face": np.ones((1, 16, 16, 3)), "pred_edge": np.zeros((0, 16, 3)), "representation": "shared_boundary_v1"},
        {"pred_face": torch.full((3, 16, 16, 3), 2.0), "pred_edge": np.ones((2, 16, 3)), "representation": "shared_boundary_v1"},
    ]
    exported = vae_prediction_items(prefixes, {"items": items}, {"face_coords": torch.tensor(3.)})
    save_item_predictions(prefixes, exported, tmp_path)
    with np.load(tmp_path / "one/data.npz", allow_pickle=False) as first, np.load(tmp_path / "two/data.npz", allow_pickle=False) as second:
        assert first["pred_face"].shape[0] == 1
        assert second["pred_face"].shape[0] == 3
        assert first["pred_edge"].shape[0] == 0
        assert second["pred_edge"].shape[0] == 2
        assert second["representation"].item() == "shared_boundary_v1"
        assert "face_coords_loss" not in first  # A batch mean is not a per-item score.


def test_legacy_batch_export_never_repeats_a_concatenated_batch(tmp_path):
    with pytest.raises(ValueError, match="batch_size=1"):
        vae_prediction_items(["one", "two"], {"pred_face": np.ones((4, 16, 16, 3))}, {})
    with pytest.raises(ValueError, match="Prediction count"):
        save_item_predictions(["one", "two"], [{}], tmp_path)
    with pytest.raises(ValueError, match="identifier"):
        save_item_predictions(["../outside"], [{}], tmp_path)


def test_checkpoint_names_do_not_create_metric_subdirectories(tmp_path):
    from pytorch_lightning.callbacks import ModelCheckpoint

    callback = ModelCheckpoint(
        dirpath=tmp_path,
        filename="epoch={epoch:04d}-step={step:08d}-val_loss={val/loss:.6f}",
        auto_insert_metric_name=False,
    )
    path = Path(callback.format_checkpoint_name({"epoch": 4, "step": 100, "val/loss": torch.tensor(.125)}))
    assert path.parent == tmp_path
    assert path.name == "epoch=0004-step=00000100-val_loss=0.125000.ckpt"


def test_vae_test_metrics_and_exports_use_the_same_posterior_mean_pass(tmp_path):
    from types import SimpleNamespace

    from src.train import UnifiedTrainer

    calls, logged, exports = [], [], []

    def model(batch, v_test):
        calls.append(v_test)
        # Distinct values expose an accidental separate stochastic loss pass.
        prediction = torch.tensor(2.0 if v_test else 9.0)
        return {"total_loss": prediction}, {"items": [{"prediction": prediction}]}

    trainer = SimpleNamespace(
        stage="vae", log_root=tmp_path,
        cfg=OmegaConf.create({"eval": {"output_dir": str(tmp_path / "predictions")}}),
        model=model, log=lambda *args, **kwargs: logged.append(args),
        _save_vae_predictions=lambda *args: exports.append(args),
    )
    result = UnifiedTrainer.test_step(trainer, {"v_prefix": ["one"]}, 0)
    assert calls == [True]
    assert result.item() == 2.0
    assert logged[0][1].item() == exports[0][1]["items"][0]["prediction"].item()
