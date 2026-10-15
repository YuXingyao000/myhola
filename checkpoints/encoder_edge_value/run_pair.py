#!/usr/bin/env python3
"""Sequential launcher for the fixed encoder-edge-value candidate/control pair."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from datetime import datetime, timezone


REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = REPO / "checkpoints/encoder_edge_value"
METADATA = EXPERIMENT / "metadata"
ACTIVE = METADATA / "active_pair.json"
SOURCE = Path(
    "/mnt/d/python/checkpoints/intersection_refinement/curve_conditioning/"
    "20261013T083729.740439Z/weights/epoch=0157-step=00057354-val_loss=0.038204.ckpt"
)
SOURCE_SHA256 = "4c7910b8169206c74fcf38c358cf171359149e0429c8debbc227d391aa56613f"
DATA = Path("/mnt/d/data/deepcad_v7")
SPLITS = {
    name: REPO / f"checkpoints/joint_boundary/validation/{name}_contract_audit/accepted.txt"
    for name in ("train", "val", "test")
}
SPLIT_COUNTS = {"train": 46480, "val": 3495, "test": 2417}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def process_identity(pid: int) -> dict | None:
    """Linux PID plus boot/start time prevents signals to a recycled PID."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {
            "pid": pid,
            "start_time_ticks": fields[19],
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "process_group_id": int(fields[2]),
        }
    except (OSError, IndexError):
        return None


def is_alive(identity: dict | None) -> bool:
    return bool(identity and process_identity(identity["pid"]) == identity)


def signal_child(identity: dict | None) -> bool:
    if not is_alive(identity):
        return False
    try:
        pid = identity["pid"]
        if os.getpgid(pid) != pid:
            raise RuntimeError("Child no longer owns its recorded process group; refusing signal")
        os.killpg(pid, signal.SIGINT)
        return True
    except ProcessLookupError:
        return False


def hydra_string(key: str, value: str | Path) -> str:
    # This is Hydra grammar inside one argv item, never shell interpolation.
    return f"{key}={json.dumps(str(value))}"


def build_command(arm: str, run_dir: Path) -> list[str]:
    command = [
        sys.executable, "-m", "src.brepnet.train", "--config-name", "train_vae_encoder_edge_value",
        hydra_string("dataset.raw_data_root", DATA),
        *[hydra_string(f"dataset.{name}_dataset", path) for name, path in SPLITS.items()],
        hydra_string("trainer.output_dir", REPO / "checkpoints"),
        hydra_string("trainer.run_dir", run_dir),
        hydra_string("trainer.exp_name", f"encoder_edge_value/{arm}"),
        hydra_string("trainer.wandb.name", f"encoder_edge_value_{arm}_{run_dir.name}"),
        "trainer.devices=8", "trainer.batch_size=16", "trainer.max_epochs=40",
        "trainer.max_steps=14520", "trainer.max_time=null", "trainer.learning_rate=2e-5",
        "trainer.gradient_clip_val=1.0", "trainer.precision=bf16-mixed", "trainer.seed=42",
        "trainer.check_val_every_n_epoch=1", "trainer.limit_train_batches=1.0",
        "trainer.limit_val_batches=1.0", "trainer.resume_from_checkpoint=null",
        "model.freeze_encoder=false", "model.curve_conditioning.enabled=true",
        "model.curve_point_feedback.enabled=false", "dataset.is_aug=1",
        "dataset.is_overfit=false", "dataset.epoch_scale_factor=1",
    ]
    if arm == "candidate":
        command.extend([
            "model.encoder_edge_value.enabled=true", "model.checkpoint=null",
            hydra_string("model.init_encoder_edge_value_checkpoint", SOURCE),
        ])
    elif arm == "control":
        command.extend([
            "model.encoder_edge_value.enabled=false", "model.init_encoder_edge_value_checkpoint=null",
            hydra_string("model.checkpoint", SOURCE),
        ])
    else:
        raise ValueError(f"Unknown arm: {arm}")
    return command


def input_record() -> dict:
    if not DATA.is_dir():
        raise FileNotFoundError(DATA)
    source_hash = sha256(SOURCE)
    if source_hash != SOURCE_SHA256:
        raise ValueError("Source checkpoint SHA-256 differs from PLAN.md")
    splits = {}
    for name, path in SPLITS.items():
        entries = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        if len(entries) != SPLIT_COUNTS[name] or len(set(entries)) != len(entries):
            raise ValueError(f"Unexpected count or duplicate IDs in {name} split")
        splits[name] = {"path": str(path), "sha256": sha256(path), "count": len(entries)}
    git = {}
    for key, arguments in {
        "commit": ["rev-parse", "HEAD"],
        "branch": ["branch", "--show-current"],
        "status": ["status", "--short"],
    }.items():
        git[key] = subprocess.check_output(["git", *arguments], cwd=REPO, text=True).strip()
    return {
        "source_checkpoint": str(SOURCE), "source_sha256": source_hash,
        "raw_data_root": str(DATA), "splits": splits, "git": git,
    }


def read_active() -> dict | None:
    return json.loads(ACTIVE.read_text()) if ACTIVE.exists() else None


def checkpoint_progress(run_dir: Path) -> dict:
    """Read only saved completion counters via the selected torch environment."""
    checkpoint = run_dir / "weights/last.ckpt"
    code = (
        "import json,sys,torch; "
        "c=torch.load(sys.argv[1],map_location='cpu',weights_only=False,mmap=True); "
        "print(json.dumps({'global_step':int(c['global_step']),'epoch':int(c['epoch'])}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(checkpoint)], cwd=REPO,
        capture_output=True, text=True, check=True, timeout=60,
    )
    return {"checkpoint": str(checkpoint), **json.loads(result.stdout)}


def start() -> int:
    METADATA.mkdir(parents=True, exist_ok=True)
    with (METADATA / "pair.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("A pair launcher is already running") from None
        previous = read_active()
        if previous and (is_alive(previous.get("runner")) or is_alive(previous.get("child"))):
            raise RuntimeError("The previous runner or training child is still alive; refusing another pair")
        provenance = input_record()
        launch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        record = {
            "launch_id": launch_id, "created_at_utc": utc_now(), "state": "starting",
            "runner": process_identity(os.getpid()), "child": None, "current_arm": None,
            "cwd": str(REPO), "python": sys.executable, **provenance,
            "stop_marker": str(METADATA / f"{launch_id}.stop.json"), "arms": {},
        }
        for arm in ("candidate", "control"):
            run_dir = EXPERIMENT / arm / launch_id
            record["arms"][arm] = {
                "state": "pending", "run_dir": str(run_dir),
                "command": build_command(arm, run_dir), "exit_code": None,
                "console_log": str(run_dir / "logs/console.log"),
            }
        archive = METADATA / f"{launch_id}.json"
        stop_marker = Path(record["stop_marker"])
        stop_requested = False
        child = None

        def request_stop(signum, frame):
            nonlocal stop_requested
            stop_requested = True

        def save():
            record["updated_at_utc"] = utc_now()
            atomic_json(archive, record)
            atomic_json(ACTIVE, record)

        old_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        save()
        print(f"Pair {launch_id}; candidate then control. Metadata: {ACTIVE}", flush=True)
        try:
            for arm, info in record["arms"].items():
                if stop_requested or stop_marker.exists():
                    record["state"] = "interrupted"
                    break
                run_dir = Path(info["run_dir"])
                record["current_arm"] = arm
                if input_record() != provenance:
                    raise RuntimeError("Git, source checkpoint or splits changed since pair launch; refusing mismatched training")
                if stop_requested or stop_marker.exists():
                    record["state"] = "interrupted"
                    break
                run_dir.mkdir(parents=True, exist_ok=False)
                (run_dir / "logs").mkdir()
                info.update(state="starting", started_at_utc=utc_now())
                record.update(state="running", current_arm=arm)
                save()
                # Do not let one arm inherit the other run's Hydra/W&B identity.
                environment = os.environ.copy()
                for key in ("BREPNET_INVOCATION_ID", "BREPNET_ACTIVE_RUN_DIR", "WANDB_RUN_ID"):
                    environment.pop(key, None)
                with Path(info["console_log"]).open("x") as output:
                    child = subprocess.Popen(
                        info["command"], cwd=REPO, env=environment, stdout=output,
                        stderr=subprocess.STDOUT, start_new_session=True,
                    )
                    record["child"] = process_identity(child.pid)
                    info["child"] = record["child"]
                    info["state"] = "running"
                    save()
                    interrupt_sent = False
                    while True:
                        if (stop_requested or stop_marker.exists()) and not interrupt_sent:
                            stop_requested = True
                            signal_child(record["child"])
                            interrupt_sent = True
                            record["state"] = "stopping"
                            save()
                        try:
                            exit_code = child.wait(timeout=1)
                            break
                        except subprocess.TimeoutExpired:
                            pass
                interrupted = stop_requested or stop_marker.exists()
                completed = False
                if exit_code == 0 and not interrupted:
                    info["checkpoint_progress"] = checkpoint_progress(run_dir)
                    completed = info["checkpoint_progress"]["global_step"] == 14520
                info.update(
                    state="interrupted" if interrupted else (
                        "completed" if completed else "incomplete" if exit_code == 0 else "failed"
                    ),
                    exit_code=exit_code, finished_at_utc=utc_now(),
                )
                record["child"] = None
                child = None
                save()
                print(f"{arm}: {info['state']} (exit {exit_code}); {run_dir}", flush=True)
                if interrupted or not completed:
                    record["state"] = info["state"]
                    break
            else:
                record["state"] = "completed"
        except BaseException as error:
            record.update(state="failed", error=f"{type(error).__name__}: {error}")
            if record["current_arm"] is not None:
                record["arms"][record["current_arm"]].update(state="failed", finished_at_utc=utc_now())
            if child is not None:
                signal_child(record["child"])
                record["arms"][record["current_arm"]]["exit_code"] = child.wait()
                record["child"] = None
            raise
        finally:
            for info in record["arms"].values():
                if info["state"] == "pending":
                    info["state"] = "cancelled"
            record["finished_at_utc"] = utc_now()
            save()
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
        return 0 if record["state"] == "completed" else 130 if record["state"] == "interrupted" else 1


def status() -> int:
    record = read_active()
    if record is None:
        print("No pair has been launched.")
        return 0
    summary = {
        "launch_id": record["launch_id"], "state": record["state"],
        "runner_alive": is_alive(record.get("runner")),
        "child_alive": is_alive(record.get("child")),
        "current_arm": record.get("current_arm"), "arms": record["arms"],
        "stop_requested": Path(record["stop_marker"]).exists(),
        "metadata": str(ACTIVE),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def stop() -> int:
    record = read_active()
    if record is None:
        print("No pair has been launched.")
        return 0
    atomic_json(Path(record["stop_marker"]), {
        "launch_id": record["launch_id"], "requested_at_utc": utc_now(),
    })
    runner = record.get("runner")
    if is_alive(runner):
        try:
            os.kill(runner["pid"], signal.SIGINT)
        except ProcessLookupError:
            pass
        print("Stop requested: interrupt the active arm and cancel the pending arm.")
    elif signal_child(record.get("child")):
        print("Runner already exited; interrupted its verified surviving training process group.")
    else:
        print("No matching live process remains; no signal sent. Pending work stays cancelled.")
    print("last.ckpt is only guaranteed to reflect the latest saved epoch, not the interruption step.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "status", "stop"))
    args = parser.parse_args()
    try:
        return {"start": start, "status": status, "stop": stop}[args.action]()
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
