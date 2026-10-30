"""HoLa 灰模/照片偏好数据：GPU 采样 → 有界 CPU 后处理队列 → 同图选对。

运行示例见 configs/datagen/preference.yaml；正式训练由独立入口启动。
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import time
import traceback

from src.preference.io import append_json, prepare_run
from src.preference.selection import SelectionConfig, select_pair_with_report
from src.preference.source import image_path, is_photo, load_condition, load_model, retain_native_validity
from tools.vae_compare_post import atomic_json, read_json, sha256_file


def candidate_seed(seed: int, image_id: str, index: int) -> int:
    from src.eval.preference import metric_seed
    return metric_seed(seed, image_id, f"diffusion:{index}")


def _worker_log(run: Path, name: str):
    path = run / "logs" / f"{name}.log"
    path.parent.mkdir(exist_ok=True)
    stream = path.open("a", buffering=1)
    os.dup2(stream.fileno(), 1)
    os.dup2(stream.fileno(), 2)
    return stream


def _unknown(image_id: str, candidate_id: str, status: str, error: str) -> dict:
    return {"image_id": image_id, "candidate_id": candidate_id,
            "validity": "unknown", "status": status, "error": error, "raw": {}, "post": []}


def _check_files(directory: Path, hashes: dict, *, allow_missing: bool = False) -> None:
    for name, digest in hashes.items():
        if allow_missing and digest is None:
            if (directory / name).exists():
                raise ValueError(f"Previously missing input changed: {directory / name}")
            continue
        if sha256_file(directory / name) != digest:
            raise ValueError(f"Generated artifact changed: {directory / name}")


def prepare_image(model, config: dict, run: Path, split: str, image_id: str):
    """Validate saved inputs, load the selected RGB, and encode the same-AE GT."""
    import numpy as np
    from src.models.diffusion.legacy import encode_original_gt

    root = run / "candidates" / split / image_id
    root.mkdir(parents=True, exist_ok=True)
    gt = Path(config["raw_data_root"]) / image_id
    paths = (image_path(config, image_id), gt / "data.npz", gt / "normalized_shape.step")
    sources = {str(path): sha256_file(path) if path.is_file() else None for path in paths}
    previous = read_json(root / "sources.json")
    if previous and previous != sources:
        raise ValueError(f"Input assets changed: {image_id}")
    atomic_json(root / "sources.json", sources)
    if (root / "input_failure.json").exists():
        return None  # Preserve the first terminal input failure, without retrying.

    cache = run / "gt_cache" / f"{image_id}_0"
    record = read_json(cache / "source.json")
    if record:
        if record["gt_npz_sha256"] != sources[str(gt / "data.npz")]:
            raise ValueError(f"GT cache input changed: {image_id}")
        _check_files(cache, record["output_sha256"])
    try:
        if any(value is None for value in sources.values()):
            raise FileNotFoundError(f"Missing input assets: {[name for name, value in sources.items() if value is None]}")
        image = load_condition(config, image_id)[None]
        if not record:
            stats = encode_original_gt(model, gt / "data.npz")
            cache.mkdir(parents=True, exist_ok=True)
            temporary, target = cache / "features.tmp", cache / "features.npy"
            with temporary.open("wb") as stream:
                np.save(stream, stats, allow_pickle=False)
            os.replace(temporary, target)
            atomic_json(cache / "source.json", {
                "gt_npz_sha256": sources[str(gt / "data.npz")],
                "output_sha256": {"features.npy": sha256_file(target)},
                "precision": "float32", "layout": "mean32,std32"})
        return image
    except Exception as exc:
        if not is_photo(config):
            raise
        traceback.print_exc()
        atomic_json(root / "input_failure.json", {
            "image_id": image_id, "status": "input_error", "validity": "unknown",
            "error": f"{type(exc).__name__}: {exc}", "sources": sources})
        return None


def input_failure_records(run: Path, split: str, image_id: str, config: dict) -> list[dict]:
    """Count all K requested candidates when an image cannot be prepared."""
    root = run / "candidates" / split / image_id
    failure = read_json(root / "input_failure.json")
    records = []
    for index in range(config["candidates"]):
        candidate_id = f"c{index:02d}"
        path = root / candidate_id / "inference.json"
        row = _unknown(image_id, candidate_id, "input_error", failure["error"])
        row["seed"] = candidate_seed(config["seed"], image_id, index)
        previous = read_json(path)
        if previous and previous != row:
            raise ValueError(f"Input failure conflicts with an existing candidate: {path}")
        if not previous:
            atomic_json(path, row)
        records.append(row)
    return records


def _gpu_worker(gpu: int, worker_index: int, identities: list[str], split: str, config: dict,
                run_string: str, jobs, events) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["OMP_NUM_THREADS"] = str(config["gpu_cpu_threads"])
    run = Path(run_string)
    worker_name = f"{split}-gpu{gpu}-worker{worker_index}"
    log = _worker_log(run, worker_name)
    try:
        import numpy as np
        import torch
        from src.models.diffusion.legacy import sample_candidates

        torch.set_num_threads(config["gpu_cpu_threads"])
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model, provenance = load_model(
            config, run / f"metadata/legacy_source_gpu{gpu}_worker{worker_index}", "cuda:0")
        atomic_json(run / f"metadata/{worker_name}.json", provenance)
        processed_images = 0
        for image_id in identities:
            # Finish all K candidates of the current image before a requested pause.
            if (run / "PAUSE").exists():
                break
            image_root = run / "candidates" / split / image_id
            image = prepare_image(model, config, run, split, image_id)
            if image is None:
                for row in input_failure_records(run, split, image_id, config):
                    events.put(("candidate", row))
                processed_images += 1
                continue
            for index in range(config["candidates"]):
                candidate_id = f"c{index:02d}"
                directory = image_root / candidate_id
                generated = read_json(directory / "inference.json")
                if generated:
                    _check_files(directory, generated.get("output_sha256", {}))
                elif directory.exists():
                    generated = _unknown(image_id, candidate_id, "interrupted", "Incomplete previous generation")
                    atomic_json(directory / "inference.json", generated)
                else:
                    directory.mkdir()
                    started = time.monotonic()
                    seed = candidate_seed(config["seed"], image_id, index)
                    try:
                        result = sample_candidates(model, image, [seed])[0]
                        np.save(directory / "latent.npy", result["latent"].numpy(), allow_pickle=False)
                        np.savez_compressed(directory / "data.npz", **result["raw"])
                        np.savez_compressed(directory / "mappings.npz", **{
                            name: result[name].numpy() for name in
                            ("latent_keep_indices", "slot_to_face", "geometry_keep_indices", "face_to_geometry")})
                        generated = {
                            "image_id": image_id, "candidate_id": candidate_id, "status": "ok", "seed": seed,
                            "seconds": time.monotonic() - started,
                            "output_sha256": {name: sha256_file(directory / name)
                                              for name in ("latent.npy", "data.npz", "mappings.npz")}}
                    except Exception as exc:
                        traceback.print_exc()
                        generated = _unknown(image_id, candidate_id, "generation_error",
                                             f"{type(exc).__name__}: {exc}")
                        generated["seed"] = seed
                        torch.cuda.empty_cache()
                    atomic_json(directory / "inference.json", generated)
                if generated["status"] == "ok":
                    jobs.put((image_id, candidate_id, str(directory)))
                else:
                    events.put(("candidate", generated))
            processed_images += 1
        events.put(("gpu_done", {"worker": worker_name, "images": processed_images}))
    except BaseException:
        events.put(("fatal", f"GPU {gpu}: {traceback.format_exc()}"))
        raise
    finally:
        log.close()


def _cpu_worker(index: int, config: dict, run_string: str, jobs, events) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    run = Path(run_string)
    log = _worker_log(run, f"cpu{index}")
    # Termination unwinds evaluate_candidate's finally and kills its OCC child.
    def terminate(*args):
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, terminate)
    try:
        from tools.vae_compare_post import preload_cpu_dependencies
        preload_cpu_dependencies()
        from src.eval.preference import evaluate_candidate
        while True:
            item = jobs.get()
            if item is None:
                return
            image_id, candidate_id, directory = item
            gt_root = Path(config["raw_data_root"]) / image_id
            record = evaluate_candidate(
                Path(directory) / "data.npz", gt_root / "data.npz", gt_root / "normalized_shape.step",
                Path(directory) / "evaluation", image_id, candidate_id=candidate_id,
                seed=config["seed"], repeats=config["selection"]["post_repeats"],
                timeout_seconds=config["post"]["timeout_seconds"],
                max_optimize_iter=config["post"]["max_optimize_iter"],
                gt_cache_root=run / "gt_metric_samples", start_method="fork")
            if is_photo(config):
                normalized = retain_native_validity(record)
                if normalized != record:
                    atomic_json(Path(directory) / "evaluation/metrics.json", normalized)
                    record = normalized
            events.put(("candidate", record))
    except Exception:
        events.put(("fatal", f"CPU {index}: {traceback.format_exc()}"))
        raise
    finally:
        log.close()


def finish_image(run: Path, split: str, image_id: str, records: list[dict], config: dict) -> tuple[dict | None, dict]:
    pair, report = select_pair_with_report(
        records, seed=config["seed"], config=SelectionConfig(**config["selection"]))
    if pair:
        prefix = Path("candidates") / split / image_id
        pair.update({
            "split": split, "rule_version": config["selection"]["version"],
            "winner_path": str(prefix / pair["winner_id"] / "latent.npy"),
            "loser_path": str(prefix / pair["loser_id"] / "latent.npy"),
            # Full OCC path diagnostics remain in metrics.json, not duplicated
            # into every training manifest row.
            "metrics_evidence": {
                row["candidate_id"]: {
                    **{key: row[key] for key in ("validity", "status", "raw", "post", "native",
                                                "protocol_fingerprint", "error") if key in row},
                    "metrics_path": str(prefix / row["candidate_id"] / "evaluation/metrics.json"),
                }
                for row in records if row["candidate_id"] in (pair["winner_id"], pair["loser_id"])},
        })
    path = run / "metadata/selection" / split / f"{image_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, {"image_id": image_id, "split": split, "pair": pair, "report": report})
    return pair, report


def worker_assignments(gpus: list[int], workers_per_gpu: int, identities: list[str]) -> list[tuple]:
    """Disjoint image shards; use every GPU before adding a second worker."""
    slots = [(gpu, slot) for slot in range(workers_per_gpu) for gpu in gpus][:len(identities)]
    return [(gpu, slot, identities[index::len(slots)]) for index, (gpu, slot) in enumerate(slots)]


def _completed_records(run: Path, split: str, image_id: str, config: dict) -> list[dict] | None:
    """Reuse a complete image after the same input/output hash checks as before."""
    root = run / "candidates" / split / image_id
    rows, generated = [], []
    for index in range(config["candidates"]):
        directory = root / f"c{index:02d}"
        inference = read_json(directory / "inference.json")
        if not inference:
            return None
        row = read_json(directory / "evaluation/metrics.json") if inference["status"] == "ok" else inference
        if not row or row["status"] == "running":
            return None
        rows.append(row)
        generated.append((directory, inference))
    sources = read_json(root / "sources.json")
    if not sources:
        raise ValueError(f"Missing source record: {root}")
    _check_files(Path("/"), sources, allow_missing=is_photo(config))
    if (root / "input_failure.json").exists():
        if any(row.get("status") != "input_error" or row.get("validity") != "unknown" for row in rows):
            raise ValueError(f"Input failure contains inconsistent candidate records: {root}")
        return rows
    cache = run / "gt_cache" / f"{image_id}_0"
    _check_files(cache, read_json(cache / "source.json")["output_sha256"])
    gt = Path(config["raw_data_root"]) / image_id
    for row, (directory, inference) in zip(rows, generated):
        _check_files(directory, inference.get("output_sha256", {}))
        if inference["status"] == "ok":
            protocol = read_json(directory / "evaluation/protocol.json")
            if (row["protocol_fingerprint"] != protocol["protocol_fingerprint"] or
                    protocol["input_sha256"] != {
                        "raw_path": inference["output_sha256"]["data.npz"],
                        "gt_npz": sources[str(gt / "data.npz")],
                        "gt_step": sources[str(gt / "normalized_shape.step")]}):
                raise ValueError(f"Recorded evaluation inputs differ: {directory}")
    return rows


def run_phase(split: str, identities: list[str], config: dict, run: Path) -> dict:
    if not identities:
        return {"images": 0, "pairs": 0}
    manifest_name = "pairs" if split == "train" else "validation_pairs"
    partial = run / f"{manifest_name}.in_progress.jsonl"
    counts = Counter(images=0, candidates=0, pairs=0)
    pending: dict[str, dict[str, dict]] = {}

    def accept(record: dict, stream) -> None:
        rows = pending.setdefault(record["image_id"], {})
        if record["candidate_id"] in rows:
            raise ValueError(f"Duplicate candidate event: {record['image_id']}/{record['candidate_id']}")
        rows[record["candidate_id"]] = record
        counts["candidates"] += 1
        counts[f"validity_{record['validity']}"] += 1
        counts[f"status_{record['status']}"] += 1
        if len(rows) == config["candidates"]:
            pair, _ = finish_image(run, split, record["image_id"], list(rows.values()), config)
            counts["images"] += 1
            if pair:
                append_json(stream, pair)
                counts["pairs"] += 1
                counts[f"pairs_{pair['pair_type']}"] += 1
            pending.pop(record["image_id"])

    previous = read_json(run / "status.json")
    old_summary = read_json(run / f"metadata/{split}_summary.json")
    elapsed_before = (previous.get("elapsed_seconds", 0) if previous.get("phase") == split
                      else old_summary.get("elapsed_seconds", 0))
    remaining = []
    # Restore counters and pairs without loading a GPU model or replaying OCC.
    with partial.open("w") as stream:
        for index, image_id in enumerate(identities):
            rows = _completed_records(run, split, image_id, config)
            if rows is None:
                remaining.append(image_id)
            else:
                for row in rows:
                    accept(row, stream)
            if (index + 1) % 1000 == 0:
                print(f"{split}: checked {index + 1}, restored {counts['images']} images", flush=True)
    if not remaining:
        os.replace(partial, run / f"{manifest_name}.jsonl")
        summary = {**counts, "elapsed_seconds": elapsed_before}
        atomic_json(run / f"metadata/{split}_summary.json", summary)
        return summary

    context = mp.get_context("spawn")
    jobs = context.Queue(maxsize=config["queue_size"])
    events = context.Queue()
    assignments = worker_assignments(config["gpus"], config.get("workers_per_gpu", 1), remaining)
    workers = [context.Process(target=_cpu_worker, args=(i, config, str(run), jobs, events))
               for i in range(config["cpu_workers"])]
    workers += [context.Process(target=_gpu_worker,
                                args=(gpu, slot, ids, split, config, str(run), jobs, events))
                for gpu, slot, ids in assignments]
    restored_candidates = counts["candidates"]
    done_workers, produced_images = 0, 0
    started = time.monotonic()
    for worker in workers:
        worker.start()
    try:
        with partial.open("a") as stream:
            while counts["candidates"] < len(identities) * config["candidates"]:
                if (done_workers == len(assignments) and counts["candidates"] ==
                        restored_candidates + produced_images * config["candidates"]):
                    break  # A PAUSE request stopped producers at image boundaries.
                try:
                    kind, value = events.get(timeout=10)
                except queue.Empty:
                    failed = [worker for worker in workers if worker.exitcode not in (None, 0)]
                    if failed:
                        raise RuntimeError(f"Worker exited unexpectedly: {[worker.exitcode for worker in failed]}")
                    continue
                if kind == "fatal":
                    raise RuntimeError(value)
                if kind == "gpu_done":
                    done_workers += 1
                    produced_images += value["images"]
                    continue
                accept(value, stream)
                current_elapsed = time.monotonic() - started
                status = {"status": "running", "phase": split, "requested_images": len(identities),
                          **counts, "elapsed_seconds": elapsed_before + current_elapsed,
                          "resume_elapsed_seconds": current_elapsed,
                          "resume_candidates": counts["candidates"] - restored_candidates}
                atomic_json(run / "status.json", status)
                if counts["candidates"] % 50 == 0:
                    print(status, flush=True)
        complete = counts["images"] == len(identities)
        if complete:
            os.replace(partial, run / f"{manifest_name}.jsonl")
        for _ in range(config["cpu_workers"]):
            jobs.put(None)
        for worker in workers:
            worker.join(timeout=30)
            if worker.exitcode != 0:
                raise RuntimeError(f"Worker did not exit normally: {worker.pid}, {worker.exitcode}")
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
        for worker in workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.kill()
        jobs.close()
        events.close()
    summary = {**counts, "elapsed_seconds": elapsed_before + time.monotonic() - started}
    if complete:
        atomic_json(run / f"metadata/{split}_summary.json", summary)
    else:
        summary.update(status="paused", phase=split, requested_images=len(identities))
        atomic_json(run / "status.json", summary)
    return summary


def main() -> None:
    from omegaconf import OmegaConf
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/datagen/preference.yaml")
    parser.add_argument("--run-dir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--accept-scheduling-update", action="store_true",
                        help="Record an audited scheduling-only source migration; keep the original data protocol")
    parser.add_argument("overrides", nargs="*", help="Explicit YAML key=value overrides")
    args = parser.parse_args()
    config = OmegaConf.to_container(OmegaConf.merge(
        OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides)), resolve=True)
    config["selection"] = asdict(SelectionConfig(**config["selection"]))
    if (not config["gpus"] or len(config["gpus"]) != len(set(config["gpus"])) or
            min(config["cpu_workers"], config["queue_size"], config["candidates"],
                config.get("workers_per_gpu", 1)) < 1):
        raise ValueError("Require distinct GPUs, workers, queue capacity and positive K")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = Path(args.run_dir or Path(config["output_root"]) / timestamp).resolve()
    if not args.resume:
        protocol = prepare_run(config, run, False)
    # One producer per run; the lock does not disappear after an abrupt exit.
    with (run / "metadata/run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume:
            protocol = prepare_run(config, run, True, args.accept_scheduling_update)
            (run / "PAUSE").unlink(missing_ok=True)
        print(f"run={run} gpus={config['gpus']} workers_per_gpu={config.get('workers_per_gpu', 1)} "
              f"protocol={protocol['fingerprint']}", flush=True)
        try:
            summaries = {}
            for split in ("validation", "train"):
                summaries[split] = run_phase(split, protocol["splits"][split], config, run)
                if summaries[split].get("status") == "paused":
                    return
                if (split == "validation" and summaries[split]["images"] and not summaries[split]["pairs"]
                        and config.get("stop_if_no_validation_pairs", True)):
                    atomic_json(run / "status.json", {"status": "stopped_no_validation_pairs", **summaries})
                    return
            atomic_json(run / "status.json", {"status": "complete", **summaries})
        except BaseException as exc:
            atomic_json(run / "status.json", {
                **(read_json(run / "status.json") or {}), "status": "failed",
                "error": f"{type(exc).__name__}: {exc}"})
            raise


if __name__ == "__main__":
    main()
