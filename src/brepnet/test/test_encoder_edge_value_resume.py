"""Resume launcher provenance/budget checks; never start a training process."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

from omegaconf import OmegaConf
import pytest


@pytest.fixture
def launcher(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[3] / "checkpoints/encoder_edge_value/run_pair.py"
    spec = importlib.util.spec_from_file_location("encoder_edge_resume_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    experiment = tmp_path / "checkpoints/encoder_edge_value"
    monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(module, "EXPERIMENT", experiment)
    monkeypatch.setattr(module, "METADATA", experiment / "metadata")
    monkeypatch.setattr(module, "ACTIVE", experiment / "metadata/active_pair.json")
    monkeypatch.setattr(module, "DATA", tmp_path / "data")
    run = experiment / "candidate/test-run"
    (run / "weights").mkdir(parents=True)
    (run / "logs").mkdir()
    source = tmp_path / "source.py"
    source.write_text("original\n")
    config = {
        "trainer": {"devices": 8, "batch_size": 16, "learning_rate": 2e-5,
                    "precision": "bf16-mixed", "seed": 42, "gradient_clip_val": 1.,
                    "run_dir": str(run), "exp_name": "encoder_edge_value/candidate",
                    "evaluate": False, "wandb": {"enabled": True}},
        "model": {"freeze_encoder": False, "encoder_edge_value": {"enabled": True}},
        "dataset": {"raw_data_root": str(module.DATA)}, "condition": {}, "eval": {"enabled": False},
    }
    splits = {}
    for name in ("train", "val", "test"):
        path = tmp_path / f"{name}.txt"
        path.write_text("one\n")
        config["dataset"][name + "_dataset"] = str(path)
        splits[name] = {"path": str(path), "sha256": module.sha256(path), "count": 1}
    folder = run / "metadata/invocations/original"
    folder.mkdir(parents=True)
    OmegaConf.save(OmegaConf.create(config), folder / "config.yaml")
    (run / "metadata/run.json").write_text(json.dumps({"run_dir": str(run),
        "experiment": "encoder_edge_value/candidate", "initial_invocation_id": "original", "wandb_id": "same-run-id"}))
    selected, last = run / "weights/best.ckpt", run / "weights/last.ckpt"
    selected.write_bytes(b"evaluated-best")
    last.write_bytes(b"evaluated-last")
    arm = {"run_dir": str(run), "selected": {"path": str(selected), "sha256": module.sha256(selected)},
           "last_saved": {"path": str(last), "sha256": module.sha256(last), "global_step": 14520, "epoch": 39},
           "source_file_hashes": {"source.py": module.sha256(source)}}
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"arms": {"candidate": arm}}))
    module.METADATA.mkdir(parents=True)
    (module.METADATA / "test-run.json").write_text(json.dumps({"splits": splits}))
    state = {"global_step": 14520, "epoch": 39, "optimizer_count": 1, "optimizer_state_count": 8,
             "fit_loop_present": True, "hyper_parameters": deepcopy(config)}
    monkeypatch.setattr(module, "resume_checkpoint_record", lambda path: deepcopy(state))
    monkeypatch.setattr(module, "input_record", lambda: {"splits": splits, "git": {"commit": "new"}})
    module.test_audit, module.test_state, module.test_run = audit, state, run
    return module


def test_readonly_preflight_preserves_config_identity_and_cumulative_budget(launcher):
    plan = launcher.prepare_resume("candidate", 29040, 80, launcher.test_audit)
    assert plan["additional_steps"] == 14520 and plan["wandb_id"] == "same-run-id"
    assert plan["optimizer_state_count"] == 8
    assert not (launcher.test_run / "weights/evaluated_snapshots").exists()
    mappings = launcher.preserve_resume_sources(plan)
    command = launcher.build_resume_command(plan, mappings, "resume-id")
    assert command[1:4] == ["-m", "src.brepnet.train", "--config-path"]
    assert command[4] == str(Path(plan["config_path"]).parent)
    assert "trainer.max_steps=29040" in command and "trainer.max_epochs=80" in command
    assert not any("learning_rate=" in part or "freeze_encoder=" in part or "wandb.id=" in part for part in command)
    assert any(part.startswith("trainer.resume_from_checkpoint=") and "evaluated_snapshots" in part for part in command)
    assert "model.init_encoder_edge_value_checkpoint=null" in command
    assert any("metadata/hydra_resume/resume-id" in part for part in command)


@pytest.mark.parametrize("max_steps,max_epochs", [(14520, 80), (0, 80), (29040, 40)])
def test_resume_rejects_finished_or_nonextended_budgets(launcher, max_steps, max_epochs):
    with pytest.raises(ValueError, match="limits must both extend"):
        launcher.prepare_resume("candidate", max_steps, max_epochs, launcher.test_audit)


@pytest.mark.parametrize("corruption", ["last_bytes", "model", "optimizer", "run_path", "source", "split"])
def test_resume_rejects_unverified_checkpoint_config_path_and_sources(launcher, corruption):
    if corruption == "last_bytes":
        (launcher.test_run / "weights/last.ckpt").write_bytes(b"changed")
    elif corruption == "model":
        launcher.test_state["hyper_parameters"]["model"]["freeze_encoder"] = True
    elif corruption == "optimizer":
        launcher.test_state["optimizer_state_count"] = 0
    elif corruption == "source":
        (launcher.REPO / "source.py").write_text("changed")
    elif corruption == "split":
        previous = launcher.METADATA / "test-run.json"
        data = json.loads(previous.read_text())
        data["splits"]["train"]["sha256"] = "different-original-hash"
        previous.write_text(json.dumps(data))
    else:
        data = json.loads(launcher.test_audit.read_text())
        data["arms"]["candidate"]["run_dir"] = str(launcher.EXPERIMENT / "control/test-run")
        launcher.test_audit.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        launcher.prepare_resume("candidate", 29040, 80, launcher.test_audit)


def test_archive_is_a_verified_independent_copy_and_existing_bad_archive_fails(launcher):
    plan = launcher.prepare_resume("candidate", 29040, 80, launcher.test_audit)
    mappings = launcher.preserve_resume_sources(plan)
    assert len(mappings) == 2
    original = Path(mappings[-1]["source_path"])
    snapshot = Path(mappings[-1]["archive_path"])
    original.write_bytes(b"later training checkpoint")
    assert snapshot.read_bytes() == b"evaluated-last"
    original.write_bytes(b"evaluated-last")
    snapshot.write_bytes(b"damaged archive")
    with pytest.raises(ValueError, match="snapshot has the wrong"):
        launcher.preserve_resume_sources(plan)


def test_existing_stop_interface_targets_new_resume_identity_only(launcher, monkeypatch):
    launcher.METADATA.mkdir(parents=True, exist_ok=True)
    marker = launcher.METADATA / "resume.stop.json"
    record = {"launch_id": "resume", "launch_type": "single_arm_resume",
              "runner": {"pid": 77}, "child": None, "stop_marker": str(marker)}
    launcher.ACTIVE.write_text(json.dumps(record))
    monkeypatch.setattr(launcher, "is_alive", lambda identity: identity == {"pid": 77})
    sent = []
    monkeypatch.setattr(launcher.os, "kill", lambda pid, signal: sent.append((pid, signal)))
    assert launcher.stop() == 0
    assert sent == [(77, launcher.signal.SIGINT)]
    assert json.loads(marker.read_text())["launch_id"] == "resume"
