"""Local run layout and provenance, shared by Hydra training entry points."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import OmegaConf


def invocation_id() -> str:
    """Keep the launch identifier stable in Lightning's child processes."""
    return os.environ.setdefault(
        "IMG2BREP_INVOCATION_ID", datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    )


def _checkpoint_path(value) -> Path | None:
    if value is None or str(value).lower() in {"", "none", "null"}:
        return None
    return Path(str(value)).expanduser().resolve()


def resolve_run_directory(output_root, experiment, explicit_run=None, resume_checkpoint=None) -> str:
    """An explicit run wins; otherwise resume in the documented checkpoint layout."""
    if explicit_run:
        return str(Path(str(explicit_run)).expanduser().resolve())
    resume = _checkpoint_path(resume_checkpoint)
    if resume is not None:
        # Match layouts adjacent to the file only. Walking all ancestors would
        # mistake the global ./checkpoints output root for a legacy run folder.
        checkpoint_dir = resume.parent
        output_root_path = Path(str(output_root)).expanduser().resolve()
        if checkpoint_dir.name == "weights":
            return str(checkpoint_dir.parent)
        if checkpoint_dir.name == "checkpoints" and checkpoint_dir != output_root_path:
            return str(checkpoint_dir.parent)
        # Legacy Lightning auto_insert_metric_name split val/loss into one
        # extra directory: run/checkpoints/epoch=0004-val/loss=0.1234.ckpt.
        legacy_dir = checkpoint_dir.parent
        if (
            legacy_dir.name == "checkpoints"
            and legacy_dir != output_root_path
            and re.fullmatch(r"epoch=\d+-val", checkpoint_dir.name)
            and re.fullmatch(r"loss=.+\.ckpt", resume.name)
        ):
            return str(legacy_dir.parent)
        raise ValueError(
            "Cannot infer the original run from this checkpoint path; set trainer.run_dir explicitly."
        )
    inherited = os.environ.get("IMG2BREP_ACTIVE_RUN_DIR")
    if inherited:
        return inherited
    experiment_path = Path(str(experiment))
    if experiment_path.is_absolute() or ".." in experiment_path.parts:
        raise ValueError("trainer.exp_name must be a relative name beneath trainer.output_dir.")
    return str((Path(str(output_root)).expanduser() / experiment_path / invocation_id()).resolve())


def hydra_log_file(run_dir) -> str:
    logs = Path(str(run_dir)) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    return str(logs / f"train-{invocation_id()}-rank{rank}.log")


def register_run_resolvers() -> None:
    if not OmegaConf.has_resolver("brep_run_dir"):
        OmegaConf.register_new_resolver("brep_run_dir", resolve_run_directory, use_cache=True)
    if not OmegaConf.has_resolver("brep_log_file"):
        OmegaConf.register_new_resolver("brep_log_file", hydra_log_file, use_cache=True)


def _git_output(args: list[str], repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=repo, stderr=subprocess.DEVNULL, text=True, timeout=10
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def prepare_run(cfg, run_dir: Path) -> dict:
    """Create only the selected run; retain one provenance snapshot per invocation."""
    run_dir = run_dir.resolve()
    os.environ["IMG2BREP_ACTIVE_RUN_DIR"] = str(run_dir)
    for name in ("weights", "logs", "wandb", "raw_predictions", "visualizations", "metadata"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    run_file = run_dir / "metadata" / "run.json"
    if run_file.exists():
        run_record = json.loads(run_file.read_text())
        if (
            run_record.get("initial_invocation_id") != invocation_id()
            and not _checkpoint_path(cfg.trainer.resume_from_checkpoint)
            and not cfg.trainer.evaluate and not cfg.eval.enabled
        ):
            raise ValueError(
                f"Run already exists: {run_dir}. Resume with trainer.resume_from_checkpoint "
                "or choose a new trainer.run_dir; existing runs are never overwritten."
            )
    else:
        run_record = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": str(cfg.trainer.exp_name),
            "run_dir": str(run_dir),
            "wandb_id": uuid.uuid4().hex[:12],
            "initial_invocation_id": invocation_id(),
        }
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if rank != 0:
        return run_record
    run_file.write_text(json.dumps(run_record, indent=2) + "\n")
    snapshot = run_dir / "metadata" / "invocations" / invocation_id()
    snapshot.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, snapshot / "config.yaml", resolve=True)
    repo = Path(__file__).resolve().parents[2]
    git_record = {
        "commit": _git_output(["rev-parse", "HEAD"], repo),
        "branch": _git_output(["branch", "--show-current"], repo),
        "status": _git_output(["status", "--short"], repo),
    }
    (snapshot / "git.json").write_text(json.dumps(git_record, indent=2) + "\n")
    (snapshot / "git.diff").write_text(_git_output(["diff", "--no-ext-diff", "HEAD"], repo) or "")
    (snapshot / "command.json").write_text(json.dumps({
        "argv": sys.argv,
        "seed": int(cfg.trainer.seed),
        "resume_from_checkpoint": cfg.trainer.resume_from_checkpoint,
        "evaluate": bool(cfg.trainer.evaluate or cfg.eval.enabled),
        "invocation_id": invocation_id(),
    }, indent=2) + "\n")
    return {**run_record, "git": git_record}


def predictions_directory(cfg, run_dir: Path) -> Path:
    explicit = cfg.eval.output_dir or cfg.trainer.test_output_dir
    return Path(str(explicit)).expanduser().resolve() if explicit else run_dir / "raw_predictions" / invocation_id()
