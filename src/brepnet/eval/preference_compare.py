"""Compare original SVR and a fixed DPO checkpoint on the same images and seeds."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from src.brepnet.data.preference_io import read_ids
from tools.vae_compare_post import atomic_json, fingerprint, read_json, sha256_file


METHODS = ("baseline", "dpo_5000")


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
        from src.brepnet.diffusion.legacy import load_legacy_svr, load_preference_svr, sample_candidates

        torch.set_num_threads(2)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        options = {"device": "cuda:0", "source_cache": run / "metadata" / f"source-gpu{gpu}"}
        if method == "baseline":
            model, provenance = load_legacy_svr(config["original_checkpoint"], **options)
        else:
            model, provenance = load_preference_svr(
                config["original_checkpoint"], config["preference_checkpoint"], **options)
            if provenance["preference"]["global_step"] != 5000:
                raise ValueError("The dpo_5000 comparison requires a 5,000-step checkpoint.")
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
                for name, digest in previous.get("output_sha256", {}).items():
                    if sha256_file(directory / name) != digest:
                        raise ValueError(f"Changed generated artifact: {directory / name}")
                sources = {
                    "image": Path(config["condition_data_root"]) / image_id / "imgs.npz",
                    "gt_npz": Path(config["raw_data_root"]) / image_id / "data.npz",
                    "gt_step": Path(config["raw_data_root"]) / image_id / "normalized_shape.step",
                }
                for name, digest in previous.get("input_sha256", {}).items():
                    if sha256_file(sources[name]) != digest:
                        raise ValueError(f"Input changed since generation: {sources[name]}")
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


def run_comparison(config: dict, identities: list[str]) -> None:
    run = Path(config["run_dir"])
    started = time.monotonic()
    context = mp.get_context("spawn")
    gpus = config["gpus"]
    assignments = [(method, gpu) for method, cards in zip(METHODS, (gpus[::2], gpus[1::2]))
                   for gpu in cards]
    workers = []
    for method in METHODS:
        cards = [gpu for name, gpu in assignments if name == method]
        for index, gpu in enumerate(cards):
            worker = context.Process(target=_gpu_worker, args=(config, method, gpu, identities[index::len(cards)]))
            worker.start()
            workers.append(worker)
    atomic_json(run / "status.json", {"status": "generating", "intended_per_model": len(identities)})
    for worker in workers:
        worker.join()
    if any(worker.exitcode != 0 for worker in workers):
        raise RuntimeError(f"GPU worker failed: {[worker.exitcode for worker in workers]}")

    jobs = []
    for image_id in identities:
        records = [read_json(run / "raw_predictions" / method / image_id / "inference.json")
                   for method in METHODS]
        if all(row["status"] == "ok" for row in records):
            if records[0]["seed"] != records[1]["seed"] or records[0]["input_sha256"] != records[1]["input_sha256"]:
                raise ValueError(f"Models did not receive identical inputs/seeds: {image_id}")
        jobs.extend((method, image_id) for method, row in zip(METHODS, records) if row["status"] == "ok")
    atomic_json(run / "status.json", {"status": "evaluating", "jobs": len(jobs), "completed": 0})
    with ProcessPoolExecutor(max_workers=config["cpu_workers"], mp_context=context,
                             initializer=_initialize_cpu) as pool:
        futures = [pool.submit(_evaluate, config, method, image_id) for method, image_id in jobs]
        for completed, future in enumerate(as_completed(futures), 1):
            print(future.result(), flush=True)
            atomic_json(run / "status.json", {"status": "evaluating", "jobs": len(jobs), "completed": completed})
    from src.brepnet.eval.preference_report import write_report
    write_report(run)
    atomic_json(run / "status.json", {"status": "complete", "intended_per_model": len(identities),
                                     "elapsed_seconds": time.monotonic() - started})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--original-checkpoint", type=Path, required=True)
    parser.add_argument("--preference-checkpoint", type=Path, required=True)
    parser.add_argument("--split-list", type=Path, required=True)
    parser.add_argument("--raw-data-root", type=Path, required=True)
    parser.add_argument("--condition-data-root", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--cpu-workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if len(args.gpus) < 2 or len(set(args.gpus)) != len(args.gpus):
        parser.error("Provide at least two distinct GPU indices.")
    config = {key: str(value.resolve()) if isinstance(value, Path) else value
              for key, value in vars(args).items() if key != "resume"}
    identities = read_ids(args.split_list)
    if len(identities) != 100:
        raise ValueError("This comparison requires the frozen 100-image validation list.")
    run = Path(config["run_dir"])
    repo = Path(__file__).resolve().parents[3]
    sources = [Path(__file__), repo / "src/brepnet/diffusion/legacy.py",
               repo / "src/brepnet/diffusion/image_loading.py",
               repo / "src/brepnet/data/preference_generate.py",
               repo / "src/brepnet/eval/preference.py", repo / "src/brepnet/eval/preference_report.py",
               repo / "src/brepnet/eval/metrics/validity.py",
               repo / "shared/occ_utils.py", repo / "shared/common_utils.py",
               repo / "tools/vae_compare_post.py", *sorted((repo / "src/brepnet/post").rglob("*.py"))]
    protocol = {
        "version": "svr_dpo_validation100_v1", "config": config,
        "identities": identities, "split_sha256": sha256_file(args.split_list),
        "checkpoint_sha256": {name: sha256_file(Path(config[name]))
                              for name in ("original_checkpoint", "preference_checkpoint")},
        "source_sha256": {str(path.relative_to(repo)): sha256_file(path) for path in sources},
        "sampling": "native DDPM1000 / FP32 / TF32 off / batch1 / c00 / no retries",
        "post": "recover_v1 / drop0 / 200(adaptive600) / CPU / 180s / repeats3",
    }
    protocol["fingerprint"] = fingerprint(protocol)
    if args.resume:
        if read_json(run / "metadata/protocol.json") != protocol:
            raise ValueError("Resume inputs, source, or configuration differ.")
    else:
        run.mkdir(parents=True, exist_ok=False)
        (run / "metadata").mkdir()
        (run / "logs").mkdir()
        atomic_json(run / "metadata/protocol.json", protocol)
        (run / "validation100.txt").write_bytes(args.split_list.read_bytes())
        atomic_json(run / "metadata/run.json", {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "command": [sys.executable, *sys.argv],
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
        })
        (run / "metadata/git.diff").write_bytes(subprocess.check_output(["git", "diff", "HEAD"]))
    run_comparison(config, identities)


if __name__ == "__main__":
    main()
