"""Run a fixed validation ablation, reusing SVR and evaluating only the new DPO.

The three stages are generation/postprocessing, observation audit and a paired
cross-run report. ``--check-only`` performs read-only CPU preflight checks.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
from importlib.metadata import version
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from uuid import uuid4

from src.preference.io import read_ids
from src.eval.preference_compare import build_protocol, validate_reuse_protocol
from src.eval.preference_ablation_report import compare_protocols, load_protocol
from src.common.io import atomic_json, fingerprint, read_json, sha256_file


REPO = Path(__file__).resolve().parents[2]
VERSION = "svr_observation_dpo_evaluation_workflow_v1"
STAGES = ("comparison", "observation", "report")


def independent_output(output: Path, sources: list[Path]) -> None:
    for source in sources:
        source = source.resolve()
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError(f"Output overlaps an immutable source: {source}")


def checkpoint_metadata(path: Path, original_sha: str, identities: list[str], step: int) -> dict:
    # mmap reads metadata without materializing the 4.5 GB tensor state or using CUDA.
    import torch
    saved = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    hp = saved["hyper_parameters"]
    if saved.get("global_step") != step or hp["model"].get("name") != "PreferenceDiffusion":
        raise ValueError(f"Expected a PreferenceDiffusion checkpoint at step {step}")
    provenance = hp["dataset"]["provenance"]
    if provenance.get("source_checkpoint_sha256") != original_sha:
        raise ValueError("DPO training source differs from original SVR")
    training = Path(hp["dataset"]["train_dataset"])
    training = training if training.is_absolute() else REPO / training
    if sha256_file(training) != provenance["train_list_sha256"]:
        raise ValueError("Training identity list changed since checkpoint creation")
    overlap = set(read_ids(training)) & set(identities)
    if overlap:
        raise ValueError(f"Training/validation overlap: {sorted(overlap)[:10]}")
    return {"global_step": saved["global_step"], "epoch": saved["epoch"],
            "training_list": str(training.resolve()), "dataset_provenance": dict(provenance)}


def preflight(config: dict) -> dict:
    """No output creation, model construction, sampling or metric recomputation."""
    output = Path(config["output_dir"])
    independent_output(output, [Path(config[key]) for key in (
        "old_run", "old_audit", "raw_data_root", "condition_data_root",
        "original_checkpoint", "preference_checkpoint", "split_list")])
    if config["font"] and not Path(config["font"]).is_file():
        raise ValueError(f"Missing font: {config['font']}")
    identities = read_ids(Path(config["split_list"]))
    if len(identities) != config["expected_count"] or len(set(identities)) != len(identities):
        raise ValueError("Validation count or uniqueness differs from the frozen population")
    comparison_config = {key: config[key] for key in (
        "original_checkpoint", "preference_checkpoint", "split_list", "expected_count",
        "raw_data_root", "condition_data_root", "gpus", "cpu_workers", "seed")}
    comparison_config.update(run_dir=str(output / "comparison"), reuse_run_dir=config["old_run"],
                             reuse_methods=["baseline"])
    comparison = build_protocol(comparison_config, identities)
    old, _, _ = validate_reuse_protocol(comparison)
    compare_protocols(old, comparison)
    if old["config"]["cpu_workers"] != config["cpu_workers"]:
        raise ValueError("Keep the original CPU worker count for the bounded postprocessing budget")
    if old["checkpoint_sha256"]["preference_checkpoint"] == comparison["checkpoint_sha256"]["preference_checkpoint"]:
        raise ValueError("New DPO checkpoint is identical to the old DPO")
    audit_path = Path(config["old_audit"]) / "protocol.json"
    audit = load_protocol(audit_path, audit=True)
    old_protocol_path = Path(config["old_run"]) / "metadata/protocol.json"
    if audit["comparison_protocol_sha256"] != sha256_file(old_protocol_path):
        raise ValueError("Old audit belongs to a different comparison")
    audit_config = audit["config"]
    if (Path(audit_config["comparison_run"]).resolve() != Path(config["old_run"]) or
            set(audit_config["methods"]) != {"baseline", "dpo_5000"}):
        raise ValueError("Old audit must cover both methods in the old run")
    audit_status = read_json(audit_path.with_name("status.json"))
    total = 2 * len(identities)
    if (audit_status.get("status") != "complete" or audit_status.get("completed") != total or
            audit_status.get("total") != total):
        raise ValueError("Old observation audit is incomplete")
    if config["audit_workers"] != audit_config["workers"]:
        raise ValueError("Keep the old observation worker count for the bounded measurement budget")
    for name, digest in audit["source_sha256"].items():
        if sha256_file(Path(name)) != digest:
            raise ValueError(f"Observation measurement source changed: {name}")
    import OCC
    runtime = {"python": sys.version, "pythonocc": OCC.VERSION,
               "packages": {name: version(name) for name in ("numpy", "scipy", "torch", "matplotlib")}}
    if runtime != audit["runtime"]:
        raise ValueError("Observation runtime differs from the old audit")
    metadata = checkpoint_metadata(Path(config["preference_checkpoint"]),
        comparison["checkpoint_sha256"]["original_checkpoint"], identities, config["expected_step"])
    source_paths = [old_protocol_path, audit_path, audit_path.with_name("status.json"),
                    old_protocol_path.parents[1] / "status.json", Path(__file__),
                    Path(__file__).with_name("preference_ablation_report.py")]
    if config["font"]:
        source_paths.append(Path(config["font"]))
    result = {"version": VERSION, "config": config, "comparison": comparison,
              "old_audit": audit, "checkpoint_metadata": metadata,
              "source_sha256": {str(path): sha256_file(path) for path in source_paths},
              "python": sys.executable}
    result["fingerprint"] = fingerprint(result)
    return result


def expected_audit(protocol: dict) -> dict:
    old = protocol["old_audit"]
    root = Path(protocol["config"]["output_dir"])
    config = {key: value for key, value in old["config"].items() if key != "fingerprint"}
    config.update(comparison_run=str(root / "comparison"), output_dir=str(root / "observation"),
                  methods=["dpo_5000"])
    value = {**old, "config": config,
             "comparison_protocol_sha256": sha256_file(root / "comparison/metadata/protocol.json")}
    value.pop("fingerprint", None)
    value["fingerprint"] = fingerprint(value)
    # Match the existing audit-v1 format; its config alias is part of the file contract.
    config["fingerprint"] = value["fingerprint"]
    return value


def commands(protocol: dict) -> dict[str, list[str]]:
    config = protocol["config"]
    root = Path(config["output_dir"])
    prefix = [sys.executable, "-u", "-m"]
    compare = prefix + ["src.eval.preference_compare"]
    for key, value in protocol["comparison"]["config"].items():
        compare.append("--" + key.replace("_", "-"))
        compare.extend(map(str, value if isinstance(value, list) else [value]))
    audit = prefix + ["src.eval.diffusion_audit", "--comparison-run", str(root / "comparison"),
        "--output-dir", str(root / "observation"), "--methods", "dpo_5000"]
    for key in ("workers", "timeout", "distance_threshold", "resolution", "mesh_deflection", "mask_threshold"):
        audit.extend(["--" + key.replace("_", "-"), str(protocol["old_audit"]["config"][key])])
    if config["font"]:
        audit.extend(["--font", config["font"]])
    report = prefix + ["src.eval.preference_ablation_report", "--old-run", config["old_run"],
        "--new-run", str(root / "comparison"), "--old-audit", config["old_audit"],
        "--new-audit", str(root / "observation"), "--output-dir", str(root / "report"),
        "--expected-count", str(config["expected_count"]), "--bootstrap-repeats", str(config["bootstrap_repeats"])]
    return dict(zip(STAGES, (compare, audit, report)))


def validate_stage(stage: str, protocol: dict) -> dict[str, str | None]:
    root = Path(protocol["config"]["output_dir"])
    directory = root / stage
    status = read_json(directory / "status.json")
    if status.get("status") != "complete":
        raise ValueError(f"Stage did not complete: {stage}")
    count = protocol["config"]["expected_count"]
    if stage == "comparison":
        from src.eval.preference_compare import verify_completed_comparison
        artifacts = verify_completed_comparison(protocol["comparison"])
    elif stage == "observation":
        if (load_protocol(directory / "protocol.json", audit=True) != expected_audit(protocol) or
                status.get("completed") != count or status.get("total") != count):
            raise ValueError("New observation protocol or denominator differs")
        cases = read_json(directory / "manifest.json").get("cases", [])
        if [(case["method"], case["image_id"]) for case in cases] != [
                ("dpo_5000", identity) for identity in protocol["comparison"]["identities"]]:
            raise ValueError("New observation manifest differs")
        artifacts = {}
    else:
        provenance = load_protocol(directory / "provenance.json")
        if status.get("intended_count") != count or status.get("fingerprint") != provenance["fingerprint"]:
            raise ValueError("Report denominator or fingerprint differs")
        expected_sources = {"old_run": protocol["config"]["old_run"],
                            "new_run": str(root / "comparison"),
                            "old_audit": protocol["config"]["old_audit"],
                            "new_audit": str(root / "observation")}
        if (provenance["sources"] != expected_sources or
                provenance["expected_count"] != count or
                provenance["bootstrap_repeats"] != protocol["config"]["bootstrap_repeats"] or
                provenance["comparison_fingerprints"][1] != protocol["comparison"]["fingerprint"]):
            raise ValueError("Report belongs to a different comparison or statistical protocol")
        artifacts = dict(provenance["input_sha256"])
    # Freeze outputs as well as their declared inputs; omit transient logs and locks.
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix in (".json", ".csv", ".md", ".png", ".svg", ".txt"):
            if "logs" not in path.relative_to(directory).parts:
                artifacts[str(path.resolve())] = sha256_file(path)
    return artifacts


def verify_receipt(receipt: dict) -> None:
    if receipt.get("fingerprint") != fingerprint({key: value for key, value in receipt.items() if key != "fingerprint"}):
        raise ValueError("Stage receipt fingerprint differs")
    for name, expected in receipt["artifacts"].items():
        path = Path(name)
        actual = sha256_file(path) if path.is_file() else None
        if actual != expected:
            raise ValueError(f"Completed stage artifact changed: {name}")


@contextmanager
def workflow_lock(root: Path):
    with (root / ".workflow.lock").open("a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Evaluation output is already in use: {root}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def run_child(command: list[str], log: Path) -> None:
    with log.open("ab", buffering=0) as stream:
        child = subprocess.Popen(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            code = child.wait()
            if code:
                raise RuntimeError(f"Stage exited with code {code}; see {log}")
        except BaseException:
            # Include spawned GPU/CPU grandchildren, even if the coordinator exited.
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
            finally:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
            raise


def stage_needs_resume(directory: Path, stage: str) -> bool:
    """Preserve and reinitialize only an empty, interrupted stage creation."""
    protocol_name = {"comparison": "metadata/protocol.json", "observation": "protocol.json",
                     "report": "provenance.json"}[stage]
    if (directory / protocol_name).is_file():
        return True
    locks = {"comparison": ".comparison.lock", "observation": None, "report": ".lock"}
    lock_name = locks[stage]
    lock = (directory / lock_name).open("a") if lock_name else None
    try:
        if lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory / protocol_name).is_file():
            return True
        for path in directory.rglob("*"):
            if path.is_symlink() or (path.is_file() and path != directory / str(lock_name)):
                raise ValueError(f"Stage has artifacts but no frozen protocol: {directory}")
        archive = directory.parent / "interrupted_initializations"
        archive.mkdir(exist_ok=True)
        directory.rename(archive / f"{stage}-{uuid4().hex}")
    finally:
        if lock:
            lock.close()
    return False


def run_workflow(protocol: dict, *, resume: bool = False) -> None:
    root = Path(protocol["config"]["output_dir"])
    if not resume:
        root.mkdir(parents=True, exist_ok=False)
    elif not root.is_dir():
        raise ValueError("Resume output does not exist")
    with workflow_lock(root):
        frozen = root / "protocol.json"
        if resume:
            if read_json(frozen) != protocol:
                raise ValueError("Resume requires the same inputs, code, runtime and configuration")
        else:
            atomic_json(frozen, protocol)
            atomic_json(root / "launch.json", {
                "created_utc": datetime.now(timezone.utc).isoformat(), "command": [sys.executable, *sys.argv],
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                "commands": commands(protocol)})
            (root / "git.diff").write_bytes(subprocess.check_output(["git", "diff", "HEAD"], cwd=REPO))
        (root / "logs").mkdir(exist_ok=True)
        (root / "receipts").mkdir(exist_ok=True)
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Received signal {signum}")
        handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
        stage = None
        try:
            for stage, command in commands(protocol).items():
                directory = root / stage
                if directory.is_symlink():
                    raise ValueError(f"Workflow stage must be a local directory: {directory}")
                receipt_path = root / "receipts" / f"{stage}.json"
                if receipt_path.is_file():
                    receipt = read_json(receipt_path)
                    if receipt.get("stage") != stage or receipt.get("workflow_fingerprint") != protocol["fingerprint"]:
                        raise ValueError("Stage receipt belongs to a different workflow")
                    verify_receipt(receipt)
                    print(f"Verified completed stage: {stage}", flush=True)
                    continue
                if read_json(directory / "status.json").get("status") != "complete":
                    if directory.exists():
                        if not resume:
                            raise ValueError(f"Stage output already exists: {directory}")
                        if stage_needs_resume(directory, stage):
                            command.append("--resume")
                    atomic_json(root / "status.json", {"status": "running", "stage": stage})
                    print(f"Starting {stage}; log: {root / 'logs' / (stage + '.log')}", flush=True)
                    run_child(command, root / "logs" / f"{stage}.log")
                artifacts = validate_stage(stage, protocol)
                receipt = {"stage": stage, "workflow_fingerprint": protocol["fingerprint"], "artifacts": artifacts}
                receipt["fingerprint"] = fingerprint(receipt)
                atomic_json(receipt_path, receipt)
            atomic_json(root / "status.json", {"status": "complete", "report": str(root / "report/README.md")})
            print(f"Completed; report: {root / 'report/README.md'}", flush=True)
        except BaseException as error:
            atomic_json(root / "status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                "stage": stage, "error": f"{type(error).__name__}: {error}"})
            raise
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("old-run", "old-audit", "original-checkpoint", "preference-checkpoint", "split-list",
                 "raw-data-root", "condition-data-root", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    for name, default in (("cpu-workers", 16), ("audit-workers", 16), ("expected-count", 3505),
                          ("bootstrap-repeats", 10000)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--expected-step", type=int, choices=[5000], default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--font", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0:
        parser.error("Provide distinct nonnegative GPU indices")
    if any(getattr(args, key) < 1 for key in ("cpu_workers", "audit_workers", "expected_count", "expected_step", "bootstrap_repeats")):
        parser.error("Worker counts, budgets and expected counts must be positive")
    # Only child inference workers explicitly select GPUs. Everything here is CPU-only.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    config = {key: str(value.resolve()) if isinstance(value, Path) else value
              for key, value in vars(args).items() if key not in ("resume", "check_only")}
    print("Checking frozen split, checkpoints, source protocols and measurement runtime (CPU only)...", flush=True)
    protocol = preflight(config)
    print(json.dumps({"preflight": "passed", "count": config["expected_count"],
                      "checkpoint_step": protocol["checkpoint_metadata"]["global_step"],
                      "checkpoint_sha256": protocol["comparison"]["checkpoint_sha256"],
                      "fingerprint": protocol["fingerprint"]}), flush=True)
    if not args.check_only:
        run_workflow(protocol, resume=args.resume)


if __name__ == "__main__":
    main()
