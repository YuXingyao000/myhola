"""Extend the frozen photo comparison to full validation, retaining audited results."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import csv
import io
import json
import multiprocessing as mp
from pathlib import Path
import signal
import subprocess
import sys
import time

from src.brepnet.eval.photo_baseline import METHODS, evaluate_one, gpu_worker, initialize_cpu
from src.brepnet.eval.preference_compare import _verify_evaluation, comparison_lock
from tools.vae_compare_post import atomic_json, fingerprint, read_json, sha256_file


def frozen_protocol(path: Path) -> dict:
    value = read_json(path)
    if not value or value.get("fingerprint") != fingerprint({k: v for k, v in value.items() if k != "fingerprint"}):
        raise ValueError(f"Protocol fingerprint changed: {path}")
    return value


def verify_generation(protocol: dict, method: str, mid: str, directory: Path) -> dict:
    from src.brepnet.data.preference_generate import candidate_seed

    row = read_json(directory / "inference.json")
    if (row.get("protocol_fingerprint") != protocol["fingerprint"] or
            row.get("method") != method or row.get("image_id") != mid or
            row.get("seed") != candidate_seed(42, mid, 0) or
            row.get("input_sha256") != protocol["input_sha256"][mid] or
            row.get("status") not in {"ok", "generation_error", "interrupted"}):
        raise ValueError(f"Generation identity/input/protocol changed: {directory}")
    hashes = row.get("output_sha256", {})
    if row["status"] == "ok" and set(hashes) != {"latent.npy", "data.npz"}:
        raise ValueError("Successful generation lacks output hashes")
    for name, digest in hashes.items():
        if Path(name).name != name or sha256_file(directory / name) != digest:
            raise ValueError(f"Generated artifact changed: {directory}/{name}")
    return row


def audit_reuse(source: Path, current: dict) -> tuple[dict, dict]:
    """Verify all original outcomes, including failed geometry, before linking."""
    previous = frozen_protocol(source / "metadata/protocol.json")
    if read_json(source / "status.json").get("status") != "complete":
        raise ValueError("Only complete prior comparisons can be reused")
    for key in ("checkpoint_sha256", "sampling", "post", "checkpoint_selection"):
        if previous[key] != current[key]:
            raise ValueError(f"Reuse protocol differs: {key}")
    old_ids = previous["identities"]
    if len(old_ids) != len(set(old_ids)) or not set(old_ids).issubset(current["identities"]):
        raise ValueError("Previous population is not a unique subset")
    if (source / "validation.txt").read_text().split() != old_ids:
        raise ValueError("Previous frozen list changed")
    for mid in old_ids:
        if previous["input_sha256"][mid] != current["input_sha256"][mid]:
            raise ValueError(f"Reused input changed: {mid}")
    # The old generator/decoder and every metric/post source remain identical.
    for name, digest in previous["source_sha256"].items():
        if sha256_file(Path(name)) != digest:
            raise ValueError(f"Reused computational source changed: {name}")
    from src.brepnet.eval.photo_report import load_rows
    _, published_rows = load_rows(source)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(published_rows[METHODS[0]][0]), lineterminator="\n")
    writer.writeheader()
    for method in METHODS:
        writer.writerows(published_rows[method])
    if buffer.getvalue() != (source / "per_object.csv").read_text():
        raise ValueError("Previous measurements differ from the published per-object table")
    artifacts = {str(source / "per_object.csv"): sha256_file(source / "per_object.csv")}
    config = {"raw_data_root": current["config"]["gt_root"], "seed": 42}
    for method in METHODS:
        for mid in old_ids:
            raw = source / "raw_predictions" / method / mid
            row = verify_generation(previous, method, mid, raw)
            artifacts[str(raw / "inference.json")] = sha256_file(raw / "inference.json")
            if row["status"] == "ok":
                artifacts.update(_verify_evaluation(config, mid, source / "evaluations" / method / mid, raw, row))
    return previous, artifacts


def prepare(config: dict) -> dict:
    import torch
    from importlib.metadata import version
    import OCC
    from src.brepnet.data.photo_dataset import photo_cache_provenance

    run, source = Path(config["run_dir"]), Path(config["reuse_run"])
    if run == source or run.is_relative_to(source) or source.is_relative_to(run):
        raise ValueError("Full run must not overlap the immutable source run")
    ids = Path(config["split_list"]).read_text().split()
    if len(ids) != config["expected_count"] or len(set(ids)) != len(ids):
        raise ValueError("Validation population count/uniqueness differs")
    canonical = Path(config["canonical_validation_list"])
    if ids != canonical.read_text().split():
        raise ValueError("Require the complete canonical validation list in its original order")
    data = photo_cache_provenance({"cache_root": config["cache_root"]})
    cache = read_json(Path(config["cache_root"]) / "manifest.json")
    if set(ids) & set(cache["splits"]["training"]["ids"]):
        raise ValueError("Training/validation overlap")
    previous = frozen_protocol(source / "metadata/protocol.json")
    for method in METHODS:
        if sha256_file(Path(config[method])) != previous["checkpoint_sha256"][method]:
            raise ValueError(f"Checkpoint differs from original comparison: {method}")
    for method, step in (("photo_val_best", 4553), ("photo_latest", 310000)):
        saved = torch.load(config[method], map_location="cpu", mmap=True, weights_only=False)
        if saved["global_step"] != step or saved["hyper_parameters"]["dataset"]["provenance"] != data:
            raise ValueError("Photo checkpoint source or selected step changed")
    inputs = {}
    for mid in ids:
        paths = {"image": Path(config["condition_root"]) / mid / "real_photo.npz",
                 "gt_npz": Path(config["gt_root"]) / mid / "data.npz",
                 "gt_step": Path(config["gt_root"]) / mid / "normalized_shape.step"}
        inputs[mid] = {key: sha256_file(path) if path.is_file() else None for key, path in paths.items()}
    old_runtime = read_json(source / "metadata/runtime.json")
    runtime = {"python": sys.version, "pythonocc": OCC.VERSION,
               "packages": {name: version(name) for name in old_runtime["packages"]}}
    if any(runtime[key] != old_runtime[key] for key in runtime):
        raise ValueError("Runtime changed since the reusable comparison")
    repo = Path(__file__).resolve().parents[3]
    extra_sources = [Path(__file__), Path(__file__).with_name("photo_report.py"),
                     repo / "src/brepnet/data/preference_generate.py",
                     repo / "src/brepnet/eval/preference_compare.py"]
    sources = {**previous["source_sha256"], **{str(p): sha256_file(p) for p in extra_sources}}
    protocol = {"version": "photo_full_validation_v1", "config": config,
                "identities": ids, "input_sha256": inputs, "source_sha256": sources,
                "split_sha256": sha256_file(canonical), "runtime": runtime,
                **{key: previous[key] for key in ("checkpoint_sha256", "sampling", "post", "checkpoint_selection")},
                "reused_ids": previous["identities"], "reuse_fingerprint": previous["fingerprint"],
                "validation_not_test": True,
                "scheduling": "8 GPU batch1 workers; 32 CPU workers consume finished generations concurrently"}
    checked_previous, artifacts = audit_reuse(source, protocol)
    protocol["reused_artifacts_sha256"] = artifacts
    protocol["fingerprint"] = fingerprint(protocol)
    path = run / "metadata/protocol.json"
    if path.exists() and read_json(path) != protocol:
        raise ValueError("Refusing to resume a changed full protocol")
    atomic_json(path, protocol)
    (run / "validation.txt").write_text("\n".join(ids) + "\n")
    for method in METHODS:
        for mid in checked_previous["identities"]:
            for stage in ("raw_predictions", "evaluations"):
                origin = source / stage / method / mid
                if not origin.exists():
                    continue
                dest = run / stage / method / mid
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.is_symlink() and dest.resolve() == origin.resolve():
                    continue
                if dest.exists() or dest.is_symlink():
                    raise ValueError(f"Reuse destination is not an untouched reference: {dest}")
                dest.symlink_to(origin.resolve(), target_is_directory=True)
    for name in sources:
        p = Path(name)
        dest = run / "metadata/source_snapshot" / p.relative_to(repo)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(p.read_bytes())
    atomic_json(run / "REUSE_AUDIT.json", {"source_run": str(source), "reused_per_method": len(previous["identities"]),
                "verified_artifact_count": len(artifacts), "source_protocol": previous["fingerprint"],
                "all_original_results_preserved": True, "published_csv_agrees_with_records": True})
    return protocol


def run_evaluation(protocol: dict) -> None:
    config = protocol["config"]
    run = Path(config["run_dir"])
    ids = protocol["identities"]
    reused = set(protocol["reused_ids"])
    pending = [mid for mid in ids if mid not in reused]
    context = mp.get_context("spawn")
    workers = [context.Process(target=gpu_worker, args=(protocol, gpu, pending[i::len(config["gpus"])]))
               for i, gpu in enumerate(config["gpus"])]
    completed = set()
    futures = {}
    began = time.monotonic()
    pool = ProcessPoolExecutor(max_workers=config["cpu_workers"], mp_context=context, initializer=initialize_cpu)
    for worker in workers:
        worker.start()
    try:
        while True:
            for method in METHODS:
                for mid in pending:
                    key = method, mid
                    if key in completed or key in futures:
                        continue
                    directory = run / "raw_predictions" / method / mid
                    if not (directory / "inference.json").exists():
                        continue
                    row = verify_generation(protocol, method, mid, directory)
                    if row["status"] != "ok":
                        completed.add(key)
                    else:
                        futures[key] = pool.submit(evaluate_one, config, method, mid)
            for key, future in list(futures.items()):
                if future.done():
                    future.result()
                    completed.add(key)
                    del futures[key]
            alive = any(worker.is_alive() for worker in workers)
            failures = [worker.exitcode for worker in workers if worker.exitcode not in (None, 0)]
            if failures:
                raise RuntimeError(f"GPU workers failed: {failures}")
            reused_count = len(reused) * len(METHODS)
            status = {"status": "running", "intended_per_method": len(ids), "total": len(ids)*len(METHODS),
                      "generated_or_terminal": reused_count + len(completed) + len(futures),
                      "evaluated_or_failed": reused_count + len(completed), "reused": reused_count,
                      "pending_cpu": len(futures), "gpu_workers_alive": sum(w.is_alive() for w in workers),
                      "elapsed_seconds": time.monotonic()-began,
                      "updated_at_utc": datetime.now(timezone.utc).isoformat()}
            atomic_json(run / "status.json", status)
            if not alive and not futures:
                if len(completed) != len(pending) * len(METHODS):
                    # A worker can finish its final atomic record during this scan.
                    remaining = [(m, mid) for m in METHODS for mid in pending if (m, mid) not in completed]
                    if all((run / "raw_predictions" / m / mid / "inference.json").is_file()
                           for m, mid in remaining):
                        continue
                    raise ValueError("Missing terminal records after generation completed")
                break
            time.sleep(5)
        from src.brepnet.eval.photo_report import load_rows
        load_rows(run)  # Full denominator and per-object terminal checks before success.
        evaluation_config = {"raw_data_root": config["gt_root"], "seed": 42}
        previous = frozen_protocol(Path(config["reuse_run"]) / "metadata/protocol.json")
        for method in METHODS:
            for mid in ids:
                raw = run / "raw_predictions" / method / mid
                inference = verify_generation(previous if mid in reused else protocol, method, mid, raw)
                if inference["status"] == "ok":
                    _verify_evaluation(evaluation_config, mid, run / "evaluations" / method / mid, raw, inference)
        for name, digest in protocol["source_sha256"].items():
            if sha256_file(Path(name)) != digest:
                raise ValueError(f"Computational source changed during evaluation: {name}")
        for name, digest in protocol["reused_artifacts_sha256"].items():
            if sha256_file(Path(name)) != digest:
                raise ValueError(f"Reused artifact changed during evaluation: {name}")
        atomic_json(run / "VALIDATION.json", {"intended_per_method": len(ids),
                    "terminal_records": len(ids) * len(METHODS), "all_artifact_hashes_verified": True,
                    "reused_artifacts_unchanged": True, "computational_sources_unchanged": True})
        atomic_json(run / "status.json", {**status, "status": "evaluated", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        cmd = [sys.executable, "-m", "src.brepnet.eval.photo_report", "--run-dir", str(run),
               "--training-run", config["training_run"]]
        with (run / "logs/report.log").open("a") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        atomic_json(run / "status.json", {**status, "status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(10)
        pool.shutdown(wait=True, cancel_futures=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--reuse-run", required=True)
    parser.add_argument("--expected-count", type=int, default=3505)
    parser.add_argument("--canonical-validation-list", default="src/brepnet/data/list/deduplicated_deepcad_validation_7_30.txt")
    parser.add_argument("--training-run", default="checkpoints/diffusion_photo_baseline/20261025T125340Z")
    parser.add_argument("--gpus", nargs="+", type=int, default=list(range(8)))
    parser.add_argument("--cpu-workers", type=int, default=32)
    options = vars(parser.parse_args())
    if options["cpu_workers"] != 32 or len(set(options["gpus"])) != len(options["gpus"]) or not options["gpus"]:
        raise ValueError("Require 32 CPU workers and distinct GPU IDs")
    for key in ("run_dir", "reuse_run", "canonical_validation_list", "training_run"):
        options[key] = str(Path(options[key]).resolve())
    original = frozen_protocol(Path(options["reuse_run"]) / "metadata/protocol.json")
    config = {**original["config"], **options, "split_list": options["canonical_validation_list"]}
    run = Path(config["run_dir"])
    (run / "logs").mkdir(parents=True, exist_ok=True)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    with comparison_lock(run):
        protocol = prepare(config)
        run_evaluation(protocol)


if __name__ == "__main__":
    main()
