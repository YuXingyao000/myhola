"""原 HoLa SVR 偏好数据：GPU 采样 → 有界 CPU 后处理队列 → 同图选对。

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

from src.brepnet.data.preference_io import append_json, prepare_run
from src.brepnet.data.preference_selection import SelectionConfig, select_pair_with_report
from tools.vae_compare_post import atomic_json, read_json, sha256_file


def candidate_seed(seed: int, image_id: str, index: int) -> int:
    from src.brepnet.eval.preference import metric_seed
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


def _check_files(directory: Path, hashes: dict) -> None:
    for name, digest in hashes.items():
        if sha256_file(directory / name) != digest:
            raise ValueError(f"Generated artifact changed: {directory / name}")


def _gpu_worker(gpu: int, identities: list[str], split: str, config: dict,
                run_string: str, jobs, events) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["OMP_NUM_THREADS"] = str(config["gpu_cpu_threads"])
    run = Path(run_string)
    log = _worker_log(run, f"{split}-gpu{gpu}")
    try:
        import numpy as np
        import torch
        from torchvision import transforms
        from src.brepnet.diffusion.image_loading import load_image_view
        from src.brepnet.diffusion.legacy import encode_original_gt, load_legacy_svr, sample_candidates

        torch.set_num_threads(config["gpu_cpu_threads"])
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model, provenance = load_legacy_svr(
            config["checkpoint"], "cuda:0", dino_repo=config["dino_repo"],
            source_cache=run / f"metadata/legacy_source_gpu{gpu}")
        atomic_json(run / f"metadata/{split}-gpu{gpu}.json", provenance)
        transform = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])
        for image_id in identities:
            image_root = run / "candidates" / split / image_id
            image_root.mkdir(parents=True, exist_ok=True)
            image_path = Path(config["condition_data_root"]) / image_id / "imgs.npz"
            gt_root = Path(config["raw_data_root"]) / image_id
            sources = {str(path): sha256_file(path)
                       for path in (image_path, gt_root / "data.npz", gt_root / "normalized_shape.step")}
            previous = read_json(image_root / "sources.json")
            if previous and previous != sources:
                raise ValueError(f"Input assets changed: {image_id}")
            atomic_json(image_root / "sources.json", sources)
            # GT and sampled candidates use the same frozen original AE.
            cache = run / "gt_cache" / f"{image_id}_0"
            cache.mkdir(parents=True, exist_ok=True)
            stats_path = cache / "features.npy"
            cache_record = read_json(cache / "source.json")
            if cache_record:
                _check_files(cache, cache_record["output_sha256"])
            else:
                stats = encode_original_gt(model, gt_root / "data.npz")
                temporary = cache / "features.tmp"
                with temporary.open("wb") as stream:
                    np.save(stream, stats, allow_pickle=False)
                os.replace(temporary, stats_path)
                atomic_json(cache / "source.json", {
                    "gt_npz_sha256": sources[str(gt_root / "data.npz")],
                    "output_sha256": {"features.npy": sha256_file(stats_path)},
                    "precision": "float32", "layout": "mean32,std32"})
            image = transform(load_image_view(image_path, "svr_imgs", 0, read_single_view=True))[None]
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
        events.put(("gpu_done", gpu))
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
        from src.brepnet.eval.preference import evaluate_candidate
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


def run_phase(split: str, identities: list[str], config: dict, run: Path) -> dict:
    if not identities:
        return {"images": 0, "pairs": 0}
    context = mp.get_context("spawn")
    jobs = context.Queue(maxsize=config["queue_size"])
    events = context.Queue()
    gpu_ids = config["gpus"][:len(identities)]
    workers = [context.Process(target=_cpu_worker, args=(i, config, str(run), jobs, events))
               for i in range(config["cpu_workers"])]
    workers += [context.Process(target=_gpu_worker,
                                args=(gpu, identities[i::len(gpu_ids)], split, config, str(run), jobs, events))
                for i, gpu in enumerate(gpu_ids)]
    manifest_name = "pairs" if split == "train" else "validation_pairs"
    partial = run / f"{manifest_name}.in_progress.jsonl"
    counts = Counter(images=0, candidates=0, pairs=0)
    pending: dict[str, list[dict]] = {}
    started = time.monotonic()
    for worker in workers:
        worker.start()
    try:
        with partial.open("w") as stream:
            while counts["candidates"] < len(identities) * config["candidates"]:
                try:
                    kind, value = events.get(timeout=10)
                except queue.Empty:
                    failed = [worker for worker in workers if worker.exitcode not in (None, 0)]
                    if failed:
                        raise RuntimeError(f"Worker exited unexpectedly: {[worker.exitcode for worker in failed]}")
                    continue
                if kind == "fatal":
                    raise RuntimeError(value)
                if kind != "candidate":
                    continue
                counts["candidates"] += 1
                counts[f"validity_{value['validity']}"] += 1
                counts[f"status_{value['status']}"] += 1
                rows = pending.setdefault(value["image_id"], [])
                rows.append(value)
                if len(rows) == config["candidates"]:
                    pair, report = finish_image(run, split, value["image_id"], rows, config)
                    counts["images"] += 1
                    if pair:
                        append_json(stream, pair)
                        counts["pairs"] += 1
                        counts[f"pairs_{pair['pair_type']}"] += 1
                    pending.pop(value["image_id"])
                status = {"status": "running", "phase": split, "requested_images": len(identities),
                          **counts, "elapsed_seconds": time.monotonic() - started}
                atomic_json(run / "status.json", status)
                if counts["candidates"] % 50 == 0:
                    print(status, flush=True)
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
    summary = {**counts, "elapsed_seconds": time.monotonic() - started}
    atomic_json(run / f"metadata/{split}_summary.json", summary)
    return summary


def main() -> None:
    from omegaconf import OmegaConf
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/datagen/preference.yaml")
    parser.add_argument("--run-dir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("overrides", nargs="*", help="Explicit YAML key=value overrides")
    args = parser.parse_args()
    config = OmegaConf.to_container(OmegaConf.merge(
        OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides)), resolve=True)
    config["selection"] = asdict(SelectionConfig(**config["selection"]))
    if (not config["gpus"] or len(config["gpus"]) != len(set(config["gpus"])) or
            min(config["cpu_workers"], config["queue_size"], config["candidates"]) < 1):
        raise ValueError("Require distinct GPUs, workers, queue capacity and positive K")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = Path(args.run_dir or Path(config["output_root"]) / timestamp).resolve()
    protocol = prepare_run(config, run, args.resume)
    # One producer per run; the lock does not disappear after an abrupt exit.
    with (run / "metadata/run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(f"run={run} gpus={config['gpus']} protocol={protocol['fingerprint']}", flush=True)
        try:
            summaries = {}
            for split in ("validation", "train"):
                summaries[split] = run_phase(split, protocol["splits"][split], config, run)
                if split == "validation" and summaries[split]["images"] and not summaries[split]["pairs"]:
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
