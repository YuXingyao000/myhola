"""Compare fixed SVR/DPO checkpoints on the same images and seeds."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import multiprocessing as mp
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from src.brepnet.data.preference_io import read_ids
from tools.vae_compare_post import atomic_json, fingerprint, read_json, sha256_file


METHODS = ("baseline", "dpo_5000")
ORCHESTRATION_SOURCES = {
    "src/brepnet/eval/preference_compare.py", "src/brepnet/eval/preference_report.py",
}


def _checked_protocol(path: Path, fingerprint_key: str = "fingerprint") -> dict:
    protocol = read_json(path)
    if not protocol or protocol.get(fingerprint_key) != fingerprint(
        {key: value for key, value in protocol.items() if key != fingerprint_key}
    ):
        raise ValueError(f"Protocol fingerprint mismatch: {path}")
    return protocol


def validate_reuse_protocol(protocol: dict) -> tuple[dict, list[str], list[str]]:
    """Read-only protocol validation, shared by reuse and preflight checks."""
    config = protocol["config"]
    source = Path(config["reuse_run_dir"])
    previous = _checked_protocol(source / "metadata/protocol.json")
    if read_json(source / "status.json").get("status") != "complete":
        raise ValueError("Only a completed comparison can be reused.")
    methods = config.get("reuse_methods") or list(METHODS)
    if len(methods) != len(set(methods)) or set(methods) - set(METHODS):
        raise ValueError("Reuse methods must be distinct known methods")
    for name in ("version", "sampling", "post"):
        if previous[name] != protocol[name]:
            raise ValueError(f"Reuse protocol differs: {name}")
    required_checkpoints = ["original_checkpoint"]
    if "baseline" in methods and config.get("baseline_preference_checkpoint"):
        required_checkpoints.append("baseline_preference_checkpoint")
    if "dpo_5000" in methods:
        required_checkpoints.append("preference_checkpoint")
    for name in required_checkpoints:
        if previous["checkpoint_sha256"][name] != protocol["checkpoint_sha256"][name]:
            raise ValueError(f"Reuse checkpoint differs: {name}")
    for name in ("raw_data_root", "condition_data_root", "seed"):
        if previous["config"][name] != config[name]:
            raise ValueError(f"Reuse configuration differs: {name}")
    old_sources, new_sources = previous["source_sha256"], protocol["source_sha256"]
    changed = {name for name in old_sources.keys() | new_sources.keys()
               if old_sources.get(name) != new_sources.get(name)}
    if changed - ORCHESTRATION_SOURCES:
        raise ValueError(f"Generation or evaluation source changed: {sorted(changed)}")
    identities = previous["identities"]
    if not identities or len(identities) != len(set(identities)):
        raise ValueError("Reused identities must be nonempty and unique")
    frozen = source / "validation.txt"
    if not frozen.is_file():
        frozen = source / "validation100.txt"
    if read_ids(frozen) != identities or sha256_file(frozen) != previous["split_sha256"]:
        raise ValueError("Reused frozen validation list changed")
    if not set(identities).issubset(protocol["identities"]):
        raise ValueError("Reused identities must be a subset of the new frozen list.")
    return previous, methods, sorted(changed)


def _hash_checked(path: Path, expected: str, artifacts: dict) -> None:
    if not expected or not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"Saved input or output SHA256 mismatch: {path}")
    artifacts[str(path.resolve())] = expected


def verify_inference(config: dict, method: str, image_id: str, directory: Path) -> tuple[dict, dict]:
    """Check saved generation without importing torch or loading a GPU model."""
    from src.brepnet.data.preference_generate import candidate_seed
    from src.brepnet.eval.preference_report import INFERENCE_TERMINAL

    path = directory / "inference.json"
    row = read_json(path)
    if (row.get("image_id") != image_id or row.get("method") != method or
            row.get("status") not in INFERENCE_TERMINAL or
            row.get("seed") != candidate_seed(config["seed"], image_id, 0)):
        raise ValueError(f"Invalid terminal inference identity, status or seed: {path}")
    artifacts = {str(path.resolve()): sha256_file(path)}
    inputs = {
        "image": Path(config["condition_data_root"]) / image_id / "imgs.npz",
        "gt_npz": Path(config["raw_data_root"]) / image_id / "data.npz",
        "gt_step": Path(config["raw_data_root"]) / image_id / "normalized_shape.step",
    }
    input_hashes, output_hashes = row.get("input_sha256", {}), row.get("output_sha256", {})
    if (not isinstance(input_hashes, dict) or not isinstance(output_hashes, dict) or
            set(input_hashes) - set(inputs)):
        raise ValueError(f"Invalid inference hash records: {path}")
    if row["status"] == "ok" and (
        set(input_hashes) != set(inputs) or not {"latent.npy", "data.npz"}.issubset(output_hashes)
    ):
        raise ValueError(f"Successful inference lacks source/output hashes: {path}")
    for name, digest in input_hashes.items():
        _hash_checked(inputs[name], digest, artifacts)
    for name, digest in output_hashes.items():
        if Path(name).name != name or name in ("", ".", ".."):
            raise ValueError(f"Unsafe generated artifact name: {path}: {name}")
        _hash_checked(directory / name, digest, artifacts)
    return row, artifacts


def _verify_evaluation(config: dict, image_id: str, directory: Path,
                       raw_directory: Path, inference: dict) -> dict:
    from src.brepnet.eval.preference_report import EVALUATION_TERMINAL, VALIDITIES

    path = directory / "metrics.json"
    row = read_json(path)
    protocol = _checked_protocol(directory / "protocol.json", "protocol_fingerprint")
    if (row.get("image_id") != image_id or row.get("candidate_id") != "c00" or
            row.get("status") not in EVALUATION_TERMINAL or row.get("validity") not in VALIDITIES or
            row.get("protocol_fingerprint") != protocol["protocol_fingerprint"]):
        raise ValueError(f"Invalid terminal evaluation or protocol: {path}")
    expected_inputs = {
        "raw_path": inference["output_sha256"]["data.npz"],
        "gt_npz": inference["input_sha256"]["gt_npz"],
        "gt_step": inference["input_sha256"]["gt_step"],
    }
    if protocol.get("input_sha256") != expected_inputs:
        raise ValueError(f"Evaluation input hashes differ from generation: {path}")
    expected = {"image_id": image_id, "candidate_id": "c00", "seed": config["seed"],
                "timeout_seconds": 180, "max_optimize_iter": 200, "repeats": 3}
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Evaluation budget or identity differs: {path}")
    sources = {
        "raw_path": raw_directory / "data.npz", "output_dir": directory,
        "gt_npz": Path(config["raw_data_root"]) / image_id / "data.npz",
        "gt_step": Path(config["raw_data_root"]) / image_id / "normalized_shape.step",
    }
    if any(Path(protocol.get(key, "")).resolve() != target.resolve() for key, target in sources.items()):
        raise ValueError(f"Evaluation source paths differ: {path}")
    artifacts = {str(item.resolve()): sha256_file(item) for item in
                 (path, directory / "protocol.json")}
    step = directory / "post/c00/recon_brep.step"
    if row.get("step_path"):
        if Path(row["step_path"]).resolve() != step.resolve():
            raise ValueError(f"Evaluation STEP path differs: {path}")
        _hash_checked(step, row.get("step_sha256"), artifacts)
    elif step.is_file():
        # A failed native read can leave a STEP without a completed STEP hash.
        artifacts[str(step.resolve())] = sha256_file(step)
    diagnostics = directory / "post/c00/post_diagnostics.json"
    if diagnostics.is_file():
        artifacts[str(diagnostics.resolve())] = sha256_file(diagnostics)
    return artifacts


def verify_completed_comparison(protocol: dict) -> dict[str, str]:
    """Validate a completed stage for resume without writes or GPU initialization."""
    from src.brepnet.eval.preference_report import load_rows

    config = protocol["config"]
    run = Path(config["run_dir"])
    if read_json(run / "status.json").get("status") != "complete":
        raise ValueError(f"Comparison is not complete: {run}")
    if _checked_protocol(run / "metadata/protocol.json") != protocol:
        raise ValueError(f"Completed comparison protocol differs: {run}")
    identities, _ = load_rows(run, expected_count=len(protocol["identities"]))
    frozen = run / "validation.txt"
    if not frozen.is_file():
        frozen = run / "validation100.txt"
    artifacts = {str(path.resolve()): sha256_file(path) for path in
                 (run / "metadata/protocol.json", run / "status.json")}
    _hash_checked(frozen, protocol["split_sha256"], artifacts)
    reuse_record = run / "metadata/reuse.json"
    if reuse_record.is_file():
        artifacts[str(reuse_record.resolve())] = sha256_file(reuse_record)
    for method in METHODS:
        for image_id in identities:
            raw = run / "raw_predictions" / method / image_id
            inference, hashes = verify_inference(config, method, image_id, raw)
            artifacts.update(hashes)
            if inference["status"] == "ok":
                artifacts.update(_verify_evaluation(
                    config, image_id, run / "evaluations" / method / image_id, raw, inference))
    return artifacts


def reuse_completed_run(protocol: dict) -> dict[str, set[str]]:
    """Reference selected terminal methods after CPU-only integrity checks."""
    config = protocol["config"]
    reused = {method: set() for method in METHODS}
    if config["reuse_run_dir"] is None:
        return reused
    source, run = Path(config["reuse_run_dir"]).resolve(), Path(config["run_dir"]).resolve()
    if run == source or run.is_relative_to(source):
        raise ValueError("Reuse output must be outside the immutable source run")
    previous, methods, changed = validate_reuse_protocol(protocol)

    from src.brepnet.eval.preference_report import load_rows
    identities, _ = load_rows(source, expected_count=len(previous["identities"]))
    artifacts, verified = {}, {}
    links = []
    for method in methods:
        for image_id in identities:
            raw = source / "raw_predictions" / method / image_id
            inference, hashes = verify_inference(config, method, image_id, raw)
            verified.update(hashes)
            if inference["status"] == "ok":
                verified.update(_verify_evaluation(
                    config, image_id, source / "evaluations" / method / image_id, raw, inference))
            for stage, record in (("raw_predictions", "inference.json"), ("evaluations", "metrics.json")):
                relative = Path(stage) / method / image_id
                origin = source / relative
                if not origin.is_dir():
                    continue  # Generation failures have no evaluation directory.
                destination = run / relative
                if destination.is_symlink() and destination.resolve() == origin.resolve():
                    pass  # Resume keeps the original reference.
                elif destination.exists() or destination.is_symlink():
                    raise ValueError(f"Reuse destination already contains different results: {destination}")
                else:
                    links.append((destination, origin.resolve()))
                artifacts[str(relative / record)] = sha256_file(origin / record)
            reused[method].add(image_id)
    record = {
        "source_run": str(source), "source_protocol_fingerprint": previous["fingerprint"],
        "source_protocol_sha256": sha256_file(source / "metadata/protocol.json"),
        "source_status_sha256": sha256_file(source / "status.json"),
        "identities": identities, "methods": methods,
        "reused_by_method": {method: len(ids) for method, ids in reused.items()},
        "remaining_by_method": {method: len(protocol["identities"]) - len(ids)
                                for method, ids in reused.items()},
        "changed_orchestration_sources": changed, "record_sha256": artifacts,
        "verified_artifact_sha256": verified,
    }
    if len(methods) == len(METHODS):
        record.update(reused_per_model=len(identities),
                      remaining_per_model=len(protocol["identities"]) - len(identities))
    saved = read_json(run / "metadata/reuse.json")
    if saved and saved != record:
        raise ValueError("Previously reused source records or artifacts changed")
    for destination, origin in links:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(origin, target_is_directory=True)
    atomic_json(run / "metadata/reuse.json", record)
    return reused


def load_comparison_model(config: dict, method: str, **options):
    """The baseline slot can explicitly hold the continued-DPO control."""
    from src.brepnet.diffusion.legacy import load_legacy_svr, load_preference_svr

    if method not in METHODS:
        raise ValueError(f"Unknown comparison method: {method}")
    checkpoint = (config.get("baseline_preference_checkpoint") if method == "baseline"
                  else config["preference_checkpoint"])
    if checkpoint is None:
        return load_legacy_svr(config["original_checkpoint"], **options)
    model, provenance = load_preference_svr(config["original_checkpoint"], checkpoint, **options)
    if provenance["preference"]["global_step"] != 5000:
        raise ValueError("The fixed comparison requires a 5,000-step checkpoint for each DPO arm.")
    return model, provenance


def _gpu_worker(config: dict, method: str, gpu: int, identities: list[str]) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["OMP_NUM_THREADS"] = "2"
    run = Path(config["run_dir"])
    with (run / "logs" / f"{method}-gpu{gpu}.log").open("a", buffering=1) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        import numpy as np
        import torch
        from torchvision import transforms
        from src.brepnet.data.preference_generate import candidate_seed
        from src.brepnet.diffusion.image_loading import load_image_view
        from src.brepnet.diffusion.legacy import sample_candidates

        torch.set_num_threads(2)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        options = {"device": "cuda:0", "source_cache": run / "metadata" / f"source-gpu{gpu}"}
        model, provenance = load_comparison_model(config, method, **options)
        atomic_json(run / "metadata" / f"{method}-gpu{gpu}.json", provenance)
        transform = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])
        for image_id in identities:
            directory = run / "raw_predictions" / method / image_id
            record_path = directory / "inference.json"
            previous = read_json(record_path)
            if previous:
                verify_inference(config, method, image_id, directory)
                continue
            seed = candidate_seed(config["seed"], image_id, 0)
            row = {"image_id": image_id, "method": method, "seed": seed, "seconds": 0.0}
            if directory.exists():
                atomic_json(record_path, {**row, "status": "generation_error",
                                          "error": "Interrupted incomplete generation; no reroll"})
                continue
            directory.mkdir(parents=True)
            started = time.monotonic()
            try:
                image_path = Path(config["condition_data_root"]) / image_id / "imgs.npz"
                image = transform(load_image_view(
                    image_path, "svr_imgs", 0, read_single_view=True))[None]
                row["input_sha256"] = {
                    "image": sha256_file(image_path),
                    "gt_npz": sha256_file(Path(config["raw_data_root"]) / image_id / "data.npz"),
                    "gt_step": sha256_file(Path(config["raw_data_root"]) / image_id / "normalized_shape.step"),
                }
                result = sample_candidates(model, image, [seed])[0]
                np.save(directory / "latent.npy", result["latent"].numpy(), allow_pickle=False)
                np.savez_compressed(directory / "data.npz", **result["raw"])
                row.update(status="ok", output_sha256={
                    name: sha256_file(directory / name) for name in ("latent.npy", "data.npz")})
            except Exception as exc:
                traceback.print_exc()
                row.update(status="generation_error", error=f"{type(exc).__name__}: {exc}")
                torch.cuda.empty_cache()
            row["seconds"] = time.monotonic() - started
            atomic_json(record_path, row)
            print(f"{method} {image_id}: {row['status']}", flush=True)


def _initialize_cpu() -> None:
    # Only CPU parents fork isolated OCC children; CUDA has never been initialized.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from tools.vae_compare_post import preload_cpu_dependencies
    preload_cpu_dependencies()


def _evaluate(config: dict, method: str, image_id: str) -> dict:
    from src.brepnet.eval.preference import evaluate_candidate
    run = Path(config["run_dir"])
    gt = Path(config["raw_data_root"]) / image_id
    row = evaluate_candidate(
        run / "raw_predictions" / method / image_id / "data.npz",
        gt / "data.npz", gt / "normalized_shape.step",
        run / "evaluations" / method / image_id, image_id,
        candidate_id="c00", seed=config["seed"], repeats=3,
        timeout_seconds=180, max_optimize_iter=200,
        gt_cache_root=run / "gt_metric_samples", start_method="fork")
    # A later geometry timeout cannot erase a completed native validity check.
    native = row.get("native", {})
    if row["validity"] == "unknown" and "native_valid_solid" in native:
        row["validity"] = "valid" if native["native_valid_solid"] else "invalid"
        row["validity_basis"] = "completed native check before later evaluation failure"
        atomic_json(run / "evaluations" / method / image_id / "metrics.json", row)
    return {"method": method, "image_id": image_id,
            "status": row["status"], "validity": row["validity"]}


def generation_assignments(gpus: list[int], identities: list[str],
                           reused_ids: dict[str, set[str]]) -> list[tuple[str, int, list[str]]]:
    """Give all cards to the only pending method, keeping the old split otherwise."""
    pending = {method: [item for item in identities if item not in reused_ids[method]] for method in METHODS}
    active = [method for method in METHODS if pending[method]]
    if len(gpus) < len(active):
        raise ValueError(f"Require at least {len(active)} GPUs for pending methods {active}")
    assignments = []
    for index, method in enumerate(active):
        cards = gpus[index::len(active)]
        for shard, gpu in enumerate(cards):
            ids = pending[method][shard::len(cards)]
            if ids:
                assignments.append((method, gpu, ids))
    return assignments


def evaluation_jobs(config: dict, identities: list[str], reused_ids: dict[str, set[str]]) -> list[tuple[str, str]]:
    """Preserve same-condition checks while skipping only reused method/CAD pairs."""
    run = Path(config["run_dir"])
    jobs = []
    for image_id in identities:
        records = [read_json(run / "raw_predictions" / method / image_id / "inference.json")
                   for method in METHODS]
        if all(row.get("status") == "ok" for row in records):
            if records[0]["seed"] != records[1]["seed"] or records[0]["input_sha256"] != records[1]["input_sha256"]:
                raise ValueError(f"Models did not receive identical inputs/seeds: {image_id}")
        for method, row in zip(METHODS, records):
            if not row:
                raise ValueError(f"Missing terminal inference: {method}/{image_id}")
            if image_id not in reused_ids[method] and row["status"] == "ok":
                jobs.append((method, image_id))
    return jobs


def _reuse_counts(identities: list[str], reused_ids: dict[str, set[str]]) -> dict:
    reused = {method: len(reused_ids[method]) for method in METHODS}
    result = {"intended_per_model": len(identities), "reused_by_method": reused,
              "new_by_method": {method: len(identities) - count for method, count in reused.items()}}
    # Preserve the legacy scalar fields only when they describe both methods.
    if len(set(reused.values())) == 1:
        count = next(iter(reused.values()))
        result.update(reused_per_model=count, new_per_model=len(identities) - count)
    return result


def run_comparison(config: dict, identities: list[str], reused_ids: dict[str, set[str]]) -> None:
    run = Path(config["run_dir"])
    started = time.monotonic()
    context = mp.get_context("spawn")
    assignments = generation_assignments(config["gpus"], identities, reused_ids)
    workers = []
    counts = _reuse_counts(identities, reused_ids)
    atomic_json(run / "status.json", {"status": "generating", **counts})
    try:
        for method, gpu, pending_ids in assignments:
            worker = context.Process(target=_gpu_worker, args=(config, method, gpu, pending_ids))
            worker.start()
            workers.append(worker)
        for worker in workers:
            worker.join()
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
        for worker in workers:
            worker.join(5)
            if worker.is_alive():
                worker.kill()
                worker.join()
    if any(worker.exitcode != 0 for worker in workers):
        raise RuntimeError(f"GPU worker failed: {[worker.exitcode for worker in workers]}")

    jobs = evaluation_jobs(config, identities, reused_ids)
    atomic_json(run / "status.json", {"status": "evaluating", "jobs": len(jobs), "completed": 0})
    pool = ProcessPoolExecutor(max_workers=config["cpu_workers"], mp_context=context,
                               initializer=_initialize_cpu)
    try:
        futures = [pool.submit(_evaluate, config, method, image_id) for method, image_id in jobs]
        for completed, future in enumerate(as_completed(futures), 1):
            print(future.result(), flush=True)
            atomic_json(run / "status.json", {"status": "evaluating", "jobs": len(jobs), "completed": completed})
    finally:
        # Do not run thousands of queued jobs after an interrupt. Already active
        # OCC children retain their existing 180-second bound and finish safely.
        pool.shutdown(wait=True, cancel_futures=True)
    from src.brepnet.eval.preference_report import write_report
    write_report(run, expected_count=len(identities))
    atomic_json(run / "status.json", {"status": "complete", **counts,
                                     "elapsed_seconds": time.monotonic() - started})


@contextmanager
def comparison_lock(run: Path):
    """Reject concurrent coordinators without modifying the immutable source."""
    with (run / ".comparison.lock").open("a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Comparison output is already in use: {run}") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def build_protocol(config: dict, identities: list[str]) -> dict:
    """Build the exact CLI protocol for read-only orchestration preflight.

    Paths in ``config`` are resolved strings; its keys are the CLI arguments
    except ``resume``, including canonical ``reuse_methods`` in METHODS order.
    """
    repo = Path(__file__).resolve().parents[3]
    sources = [Path(__file__), repo / "src/brepnet/diffusion/legacy.py",
               repo / "src/brepnet/diffusion/image_loading.py",
               repo / "src/brepnet/data/preference_generate.py",
               repo / "src/brepnet/eval/preference.py", repo / "src/brepnet/eval/preference_report.py",
               repo / "src/brepnet/eval/metrics/validity.py",
               repo / "shared/occ_utils.py", repo / "shared/common_utils.py",
               repo / "tools/vae_compare_post.py", *sorted((repo / "src/brepnet/post").rglob("*.py"))]
    protocol = {
        "version": "svr_dpo_fixed_validation_v1", "config": config,
        "identities": identities, "split_sha256": sha256_file(Path(config["split_list"])),
        "checkpoint_sha256": {name: sha256_file(Path(config[name]))
                              for name in ("original_checkpoint", "preference_checkpoint")},
        "source_sha256": {str(path.relative_to(repo)): sha256_file(path) for path in sources},
        "sampling": "native DDPM1000 / FP32 / TF32 off / batch1 / c00 / no retries",
        "post": "recover_v1 / drop0 / 200(adaptive600) / CPU / 180s / repeats3",
    }
    if config.get("baseline_preference_checkpoint"):
        # A distinct version prevents treating the continued policy as original SVR.
        protocol["version"] = "svr_dpo_geometry_pair_validation_v1"
        protocol["checkpoint_sha256"]["baseline_preference_checkpoint"] = sha256_file(
            Path(config["baseline_preference_checkpoint"]))
        protocol["method_labels"] = {"baseline": "DPO control", "dpo_5000": "DPO + geometry"}
        protocol["method_labels_zh"] = {"baseline": "继续 DPO", "dpo_5000": "DPO + 几何项"}
        protocol["comparison"] = "DPO + geometry minus continued DPO; both old DPO5000 + 5000 additional steps"
        path = repo / "src/brepnet/diffusion/preference_geometry.py"
        protocol["source_sha256"][str(path.relative_to(repo))] = sha256_file(path)
    protocol["fingerprint"] = fingerprint(protocol)
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--original-checkpoint", type=Path, required=True)
    parser.add_argument("--preference-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-preference-checkpoint", type=Path,
                        help="Continued DPO step5000 control, instead of original SVR in the baseline slot")
    parser.add_argument("--split-list", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=100)
    parser.add_argument("--reuse-run-dir", type=Path, help="Read-only reuse of a completed subset comparison")
    parser.add_argument("--reuse-methods", nargs="+", choices=METHODS,
                        help="Methods to reuse; defaults to both when --reuse-run-dir is set")
    parser.add_argument("--raw-data-root", type=Path, required=True)
    parser.add_argument("--condition-data-root", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--cpu-workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0:
        parser.error("Provide distinct nonnegative GPU indices.")
    if args.cpu_workers < 1:
        parser.error("--cpu-workers must be positive")
    if args.reuse_methods and args.reuse_run_dir is None:
        parser.error("--reuse-methods requires --reuse-run-dir")
    if args.reuse_methods and len(args.reuse_methods) != len(set(args.reuse_methods)):
        parser.error("--reuse-methods must be distinct")
    args.reuse_methods = ([method for method in METHODS if method in (args.reuse_methods or METHODS)]
                          if args.reuse_run_dir else [])
    config = {key: str(value.resolve()) if isinstance(value, Path) else value
              for key, value in vars(args).items() if key != "resume"}
    if config["baseline_preference_checkpoint"] is None:
        del config["baseline_preference_checkpoint"]  # Preserve the original CLI protocol.
    identities = read_ids(args.split_list)
    if len(identities) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} identities, received {len(identities)}.")
    run = Path(config["run_dir"])
    if args.reuse_run_dir:
        source = args.reuse_run_dir.resolve()
        if run == source or run.is_relative_to(source):
            raise ValueError("Reuse output must be outside the immutable source run")
    protocol = build_protocol(config, identities)
    if not args.resume:
        run.mkdir(parents=True, exist_ok=False)
    with comparison_lock(run):
        if args.resume:
            if read_json(run / "metadata/protocol.json") != protocol:
                raise ValueError("Resume inputs, source, or configuration differ.")
        else:
            (run / "metadata").mkdir()
            (run / "logs").mkdir()
            atomic_json(run / "metadata/protocol.json", protocol)
            (run / "validation.txt").write_bytes(args.split_list.read_bytes())
            atomic_json(run / "metadata/run.json", {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "command": [sys.executable, *sys.argv],
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
            })
            (run / "metadata/git.diff").write_bytes(subprocess.check_output(["git", "diff", "HEAD"]))

        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Received signal {signum}")

        old_handler = signal.signal(signal.SIGTERM, interrupted)
        try:
            reused_ids = reuse_completed_run(protocol)
            run_comparison(config, identities, reused_ids)
        except BaseException as error:
            atomic_json(run / "status.json", {
                **read_json(run / "status.json"),
                "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                "error": f"{type(error).__name__}: {error}",
            })
            raise
        finally:
            signal.signal(signal.SIGTERM, old_handler)


if __name__ == "__main__":
    main()
