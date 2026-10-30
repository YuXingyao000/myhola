"""固定三组照片输入比较；共享原 sampler、AE 和既有 OCC 计量流程。"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from tools.vae_compare_post import atomic_json, fingerprint, read_json, sha256_file


METHODS = ("original_svr", "photo_val_best", "photo_latest")


def prepare(config: dict) -> dict:
    """启动前冻结输入和权重；不通过运行结果筛选样本。"""
    import torch
    from src.brepnet.data.photo_dataset import photo_cache_provenance

    run = Path(config["run_dir"])
    ids = Path(config["split_list"]).read_text().split()
    if len(ids) != 100 or len(set(ids)) != 100:
        raise ValueError("Expected the fixed 100 distinct validation IDs")
    cache = Path(config["cache_root"])
    data = photo_cache_provenance({"cache_root": str(cache)})
    manifest = read_json(cache / "manifest.json")
    if ids != manifest["splits"]["validation"]["ids"]:
        raise ValueError("Evaluation population differs from fixed validation100")
    if set(ids) & set(manifest["splits"]["training"]["ids"]):
        raise ValueError("Train/validation overlap")
    checkpoints = {method: sha256_file(Path(config[method])) for method in METHODS}
    if checkpoints["original_svr"] != data["source_checkpoint_sha256"]:
        raise ValueError("Original checkpoint differs from the training source")
    for method, step in (("photo_val_best", 4553), ("photo_latest", 310000)):
        saved = torch.load(config[method], map_location="cpu", mmap=True, weights_only=False)
        if saved["global_step"] != step or saved["hyper_parameters"]["dataset"]["provenance"] != data:
            raise ValueError(f"Unexpected checkpoint/provenance for {method}")
    training_inputs = {row["id"]: row for row in map(json.loads, (cache / "inputs.jsonl").read_text().splitlines())}
    inputs = {}
    for mid in ids:
        paths = {"image": Path(config["condition_root"]) / mid / "real_photo.npz",
                 "gt_npz": Path(config["gt_root"]) / mid / "data.npz",
                 "gt_step": Path(config["gt_root"]) / mid / "normalized_shape.step"}
        inputs[mid] = {key: sha256_file(path) for key, path in paths.items()}
        if (inputs[mid]["image"] != training_inputs[mid]["photo_sha256"] or
                inputs[mid]["gt_npz"] != training_inputs[mid]["gt_npz_sha256"]):
            raise ValueError(f"Input changed since training: {mid}")
    repo = Path(__file__).resolve().parents[3]
    sources = [Path(__file__), repo / "src/brepnet/diffusion/photo_baseline.py",
               repo / "src/brepnet/diffusion/legacy.py", repo / "src/brepnet/diffusion/preference.py",
               repo / "src/brepnet/data/photo_dataset.py", repo / "src/brepnet/eval/preference.py",
               repo / "tools/vae_compare_post.py", *sorted((repo / "src/brepnet/post").glob("*.py"))]
    protocol = {"version": "photo_baseline_validation100_v1", "config": config,
                "identities": ids, "checkpoint_sha256": checkpoints, "input_sha256": inputs,
                "source_sha256": {str(p): sha256_file(p) for p in sources},
                "sampling": {"candidates": 1, "ddpm_steps": 1000, "precision": "float32",
                             "tf32": False, "seed": 42, "image_key": "flux", "selection": "none"},
                "post": {"policy": "recover_v1", "max_optimize_iter": 200, "drop_num": 0,
                         "whole_candidate_timeout_seconds": 180, "metric_repeats": 3},
                "checkpoint_selection": "original; minimum monitored validation loss; final complete saved step",
                "validation_not_test": True}
    protocol["fingerprint"] = fingerprint(protocol)
    path = run / "metadata/protocol.json"
    if path.exists() and read_json(path) != protocol:
        raise ValueError("Refusing to resume with changed inputs/protocol/code")
    atomic_json(path, protocol)
    (run / "validation.txt").write_text("\n".join(ids) + "\n")
    for name in ("logs", "raw_predictions", "evaluations"):
        (run / name).mkdir(exist_ok=True)
    # Include uncommitted runtime implementations in the research provenance.
    for p in sources:
        target = run / "metadata/source_snapshot" / p.relative_to(repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    return protocol


def gpu_worker(protocol: dict, gpu: int, ids: list[str]) -> None:
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2")
    config, run = protocol["config"], Path(protocol["config"]["run_dir"])
    with (run / "logs" / f"gpu{gpu}.log").open("a", buffering=1) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        import numpy as np
        import torch
        from src.brepnet.data.photo_dataset import photo_transform
        from src.brepnet.data.preference_generate import candidate_seed
        from src.brepnet.diffusion.legacy import load_legacy_svr, sample_candidates
        from src.brepnet.diffusion.photo_baseline import load_photo_svr

        torch.set_num_threads(2)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        transform = photo_transform()
        for method in METHODS:
            options = {"device": "cuda:0", "source_cache": run / "metadata" / f"legacy-gpu{gpu}"}
            if method == "original_svr":
                model, provenance = load_legacy_svr(config[method], **options)
            else:
                model, provenance = load_photo_svr(config["original_svr"], config[method], **options)
            atomic_json(run / "metadata" / f"{method}-gpu{gpu}.json", provenance)
            for mid in ids:
                directory = run / "raw_predictions" / method / mid
                if (directory / "inference.json").exists():
                    row = read_json(directory / "inference.json")
                    if row["protocol_fingerprint"] != protocol["fingerprint"]:
                        raise ValueError("Existing generation belongs to another protocol")
                    for name, digest in row.get("output_sha256", {}).items():
                        if sha256_file(directory / name) != digest:
                            raise ValueError("Saved generation has changed")
                    continue
                seed = candidate_seed(42, mid, 0)
                row = {"method": method, "image_id": mid, "seed": seed,
                       "protocol_fingerprint": protocol["fingerprint"],
                       "input_sha256": protocol["input_sha256"][mid]}
                if directory.exists():
                    atomic_json(directory / "inference.json", {**row, "status": "interrupted", "seconds": 0.})
                    continue  # Never reroll an interrupted candidate.
                directory.mkdir(parents=True)
                started = time.monotonic()
                try:
                    path = Path(config["condition_root"]) / mid / "real_photo.npz"
                    if sha256_file(path) != row["input_sha256"]["image"]:
                        raise ValueError("Photo changed after protocol freeze")
                    with np.load(path, allow_pickle=False) as data:
                        image = data["flux"]
                    if image.dtype != np.uint8 or image.shape != (512, 512, 3):
                        raise ValueError(f"Unexpected photo shape/dtype: {image.shape}/{image.dtype}")
                    result = sample_candidates(model, transform(image)[None], [seed])[0]
                    np.save(directory / "latent.npy", result["latent"].numpy(), allow_pickle=False)
                    np.savez_compressed(directory / "data.npz", **result["raw"])
                    row.update(status="ok", output_sha256={name: sha256_file(directory / name)
                               for name in ("latent.npy", "data.npz")})
                except Exception as exc:
                    traceback.print_exc()
                    row.update(status="generation_error", error=f"{type(exc).__name__}: {exc}")
                row["seconds"] = time.monotonic() - started
                atomic_json(directory / "inference.json", row)
                print(f"{method} {mid}: {row['status']} {row['seconds']:.2f}s", flush=True)
            del model
            torch.cuda.empty_cache()


def initialize_cpu():
    from tools.vae_compare_post import preload_cpu_dependencies
    preload_cpu_dependencies()


def evaluate_one(config: dict, method: str, mid: str) -> dict:
    from src.brepnet.eval.preference import evaluate_candidate
    run = Path(config["run_dir"])
    gt = Path(config["gt_root"]) / mid
    row = evaluate_candidate(run / "raw_predictions" / method / mid / "data.npz",
                             gt / "data.npz", gt / "normalized_shape.step",
                             run / "evaluations" / method / mid, mid,
                             seed=42, repeats=3, timeout_seconds=180, max_optimize_iter=200,
                             gt_cache_root=run / "gt_metric_samples", start_method="fork")
    if row["validity"] == "unknown" and "native_valid_solid" in row.get("native", {}):
        row["validity"] = "valid" if row["native"]["native_valid_solid"] else "invalid"
        row["validity_basis"] = "completed native check before later evaluation failure"
        atomic_json(run / "evaluations" / method / mid / "metrics.json", row)
    return {"method": method, "image_id": mid, "status": row["status"], "validity": row["validity"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--original-svr", default="/mnt/d/ckpt/HoLa/Diffusion_svr_sq30_1500k.ckpt")
    parser.add_argument("--photo-val-best", required=True)
    parser.add_argument("--photo-latest", required=True)
    parser.add_argument("--split-list", default="src/brepnet/data/list/deepcad_v7_validation100.txt")
    parser.add_argument("--cache-root", default="/mnt/d/hola_runs/diffusion/photo_baseline/cache_v1")
    parser.add_argument("--condition-root", default="/mnt/d/data/deepcad_v7_cond")
    parser.add_argument("--gt-root", default="/mnt/d/data/deepcad_v7")
    parser.add_argument("--gpus", nargs="+", type=int, default=list(range(8)))
    parser.add_argument("--cpu-workers", type=int, default=32)
    args = vars(parser.parse_args())
    for key in args.keys() - {"gpus", "cpu_workers"}:
        args[key] = str(Path(args[key]).resolve())
    run = Path(args["run_dir"])
    run.mkdir(parents=True, exist_ok=True)
    from src.brepnet.eval.preference_compare import comparison_lock
    with comparison_lock(run):
        protocol = prepare(args)
        context = mp.get_context("spawn")
        workers = [context.Process(target=gpu_worker, args=(protocol, gpu, protocol["identities"][i::len(args["gpus"])]))
                   for i, gpu in enumerate(args["gpus"])]
        atomic_json(run / "status.json", {"status": "generating", "intended_per_method": 100})
        for worker in workers:
            worker.start()
        try:
            for worker in workers:
                worker.join()
            if any(worker.exitcode for worker in workers):
                raise RuntimeError(f"Generation workers failed: {[w.exitcode for w in workers]}")
        finally:
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
        jobs = []
        for mid in protocol["identities"]:
            rows = [read_json(run / "raw_predictions" / m / mid / "inference.json") for m in METHODS]
            if any(not row for row in rows) or len({row["seed"] for row in rows}) != 1:
                raise ValueError(f"Incomplete/unpaired generation: {mid}")
            jobs.extend((m, mid) for m, row in zip(METHODS, rows) if row["status"] == "ok")
        atomic_json(run / "status.json", {"status": "evaluating", "completed": 0, "total": len(jobs)})
        with ProcessPoolExecutor(max_workers=args["cpu_workers"], mp_context=context,
                                 initializer=initialize_cpu) as pool:
            futures = [pool.submit(evaluate_one, args, method, mid) for method, mid in jobs]
            for completed, future in enumerate(as_completed(futures), 1):
                print(future.result(), flush=True)
                atomic_json(run / "status.json", {"status": "evaluating", "completed": completed, "total": len(jobs)})
        atomic_json(run / "status.json", {"status": "complete", "intended_per_method": 100,
                                         "completed_at_utc": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    main()
