"""Compare unchanged batch-one DDPM sampling with 1/2/3 processes on one GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import subprocess
import time


def digest(array):
    return hashlib.sha256(str((array.dtype, array.shape)).encode() + array.tobytes()).hexdigest()


def worker(rank, gpu, count, cpu_threads, directory, barrier, results):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    log = (directory / f"workers{count}-rank{rank}.log").open("w", buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    try:
        import torch
        from torchvision import transforms
        from src.brepnet.diffusion.image_loading import load_image_view
        from src.brepnet.diffusion.legacy import load_legacy_svr, sample_candidates

        torch.set_num_threads(cpu_threads)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model, _ = load_legacy_svr(
            "/mnt/d/ckpt/HoLa/Diffusion_svr_sq30_1500k.ckpt", "cuda:0",
            source_cache=directory / f"source-{count}-{rank}",
            dino_repo="/root/.cache/torch/hub/facebookresearch_dinov2_main")
        transform = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])
        image = transform(load_image_view(
            Path("/mnt/d/data/deepcad_v7_cond/00938923/imgs.npz"),
            "svr_imgs", 0, read_single_view=True))[None]
        sample_candidates(model, image, [920001 + rank])
        torch.cuda.synchronize()
        barrier.wait(timeout=300)
        started = time.monotonic()
        records = []
        for index in range(3):
            seed = 931000 + rank * 3 + index
            then = time.monotonic()
            result = sample_candidates(model, image, [seed])[0]
            torch.cuda.synchronize()
            records.append({
                "seed": seed, "seconds": time.monotonic() - then,
                "latent_sha256": digest(result["latent"].numpy()),
                "raw_sha256": {key: digest(value) for key, value in result["raw"].items()},
            })
        results.put({"rank": rank, "started": started, "finished": time.monotonic(),
                     "samples": records, "cuda_max_memory_allocated": torch.cuda.max_memory_allocated(),
                     "cuda_max_memory_reserved": torch.cuda.max_memory_reserved()})
    except BaseException:
        import traceback
        results.put({"rank": rank, "error": traceback.format_exc()})
        raise


def run(count, gpu, cpu_threads, directory):
    context = mp.get_context("spawn")
    barrier = context.Barrier(count)
    results = context.Queue()
    processes = [context.Process(target=worker, args=(i, gpu, count, cpu_threads, directory, barrier, results))
                 for i in range(count)]
    for process in processes:
        process.start()
    observed = []
    records = []
    while len(records) < count:
        try:
            records.append(results.get(timeout=.5))
        except queue.Empty:
            if not any(process.is_alive() for process in processes):
                raise RuntimeError("Benchmark workers exited without all results")
        query = subprocess.check_output([
            "nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits"], text=True).strip()
        memory, utilization = map(int, query.split(","))
        observed.append({"time": time.monotonic(), "memory_mib": memory, "utilization_percent": utilization})
    for process in processes:
        process.join()
    errors = [record for record in records if "error" in record]
    if errors:
        raise RuntimeError(errors)
    started = min(record["started"] for record in records)
    finished = max(record["finished"] for record in records)
    measured = [sample for sample in observed if started <= sample["time"] <= finished]
    return {
        "workers_per_gpu": count, "sample_count": count * 3,
        "seconds": finished - started, "samples_per_second": count * 3 / (finished - started),
        "peak_memory_mib": max(sample["memory_mib"] for sample in measured),
        "mean_gpu_utilization_percent": sum(sample["utilization_percent"] for sample in measured) / len(measured),
        "workers": sorted(records, key=lambda item: item["rank"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"gpu": args.gpu, "image_id": "00938923", "precision": "float32",
              "batch_size": 1, "ddpm_steps": 1000, "warmup_per_worker": 1,
              "cpu_threads_per_worker": args.cpu_threads, "results": []}
    for count in args.workers:
        result = run(count, args.gpu, args.cpu_threads, args.output)
        report["results"].append(result)
        baseline = report["results"][0]["workers"][0]["samples"]
        current = result["workers"][0]["samples"]
        result["rank_zero_matches_single_worker_bitwise"] = all(
            old["latent_sha256"] == new["latent_sha256"] and old["raw_sha256"] == new["raw_sha256"]
            for old, new in zip(baseline, current))
        (args.output / "gpu_workers.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: value for key, value in result.items() if key != "workers"}), flush=True)


if __name__ == "__main__":
    main()
