"""固定完整照片 validation：复用已审计 baseline，评价 DPO 与空间 best/last。"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback

from src.brepnet.eval.photo_baseline import evaluate_one, initialize_cpu
from src.brepnet.eval.preference_compare import _verify_evaluation, comparison_lock
from tools.vae_compare_post import atomic_json, fingerprint, read_json, sha256_file

METHODS = ("photo_baseline", "photo_dpo5000", "spatial_best", "spatial_last")
NEW_METHODS = METHODS[1:]


def checked_protocol(path: Path) -> dict:
    value = read_json(path)
    if not value or value.get("fingerprint") != fingerprint({k: v for k, v in value.items() if k != "fingerprint"}):
        raise ValueError(f"Protocol fingerprint mismatch: {path}")
    return value


def _nodes(path: Path, names: set[str], normalize_dino_option: bool = False) -> str:
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if len(nodes) != len(names):
        raise ValueError(f"Missing computational definitions: {path}")
    if normalize_dino_option:
        for node in nodes:
            if isinstance(node, ast.FunctionDef):
                pairs = [(arg, default) for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults)
                         if arg.arg != "dino_repo"]
                node.args.kwonlyargs, node.args.kw_defaults = [p[0] for p in pairs], [p[1] for p in pairs]
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == "load_legacy_svr":
                    child.keywords = [kw for kw in child.keywords if kw.arg != "dino_repo"]
    return ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False)


def audit_sources(source: Path, previous: dict) -> dict:
    """只允许已审阅的加载选项/训练分支变化；采样、VAE、OCC、metric 不放宽。"""
    repo = Path(__file__).resolve().parents[3]
    allowed = {
        "src/brepnet/diffusion/photo_baseline.py": ({"apply_photo_state", "load_photo_svr"}, True),
        "src/brepnet/diffusion/preference.py": ({"LegacySVRDenoiser", "per_sample_mse"}, False),
        "src/brepnet/data/preference_generate.py": ({"candidate_seed"}, False),
    }
    changes = {}
    for name, digest in previous["source_sha256"].items():
        path = Path(name)
        if sha256_file(path) == digest:
            continue
        relative = str(path.relative_to(repo))
        if relative not in allowed:
            raise ValueError(f"Baseline computational source changed: {name}")
        names, normalize = allowed[relative]
        old = source / "metadata/source_snapshot" / relative
        if sha256_file(old) != digest or _nodes(old, names, normalize) != _nodes(path, names, normalize):
            raise ValueError(f"Active baseline computation changed: {name}")
        changes[relative] = {"unchanged_active_definitions": sorted(names),
                             "dino_repo_default_none_option_only": normalize}
    return changes


def expected_inference(protocol: dict, method: str, mid: str) -> tuple[str, str]:
    if method != "photo_baseline":
        return method, protocol["fingerprint"]
    base = protocol["baseline_protocol"]
    return "photo_latest", (base["reuse_fingerprint"] if mid in base["reused_ids"] else base["fingerprint"])


def verify_generation(protocol: dict, method: str, mid: str) -> dict:
    from src.brepnet.data.preference_generate import candidate_seed
    directory = Path(protocol["config"]["run_dir"]) / "raw_predictions" / method / mid
    row = read_json(directory / "inference.json")
    expected_method, expected_fingerprint = expected_inference(protocol, method, mid)
    if (row.get("image_id") != mid or row.get("method") != expected_method or
            row.get("protocol_fingerprint") != expected_fingerprint or
            row.get("seed") != candidate_seed(42, mid, 0) or
            row.get("input_sha256") != protocol["input_sha256"][mid] or
            row.get("status") not in {"ok", "generation_error", "interrupted"}):
        raise ValueError(f"Generation identity, source or protocol changed: {directory}")
    outputs = row.get("output_sha256", {})
    if row["status"] == "ok" and set(outputs) != {"latent.npy", "data.npz"}:
        raise ValueError(f"Successful generation lacks complete output hashes: {directory}")
    for name, digest in outputs.items():
        if Path(name).name != name or sha256_file(directory / name) != digest:
            raise ValueError(f"Generated artifact changed: {directory}/{name}")
    return row


def prepare(config: dict) -> dict:
    import torch
    import OCC
    from importlib.metadata import version

    run, source = Path(config["run_dir"]), Path(config["baseline_run"])
    old = checked_protocol(source / "metadata/protocol.json")
    if read_json(source / "status.json").get("status") != "complete":
        raise ValueError("Baseline evaluation is not complete")
    ids = Path(config["split_list"]).read_text().split()
    if ids != old["identities"] or len(ids) != 3505 or len(set(ids)) != 3505:
        raise ValueError("Require the same full 3505 validation CADs")
    changes = audit_sources(source, old)
    current_runtime = {"python": sys.version, "pythonocc": OCC.VERSION,
                       "packages": {name: version(name) for name in old["runtime"]["packages"]}}
    if current_runtime != old["runtime"]:
        raise ValueError("Evaluation runtime differs from baseline")
    paths = {name: Path(config[name]) for name in (*NEW_METHODS, "photo_checkpoint", "original_checkpoint")}
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    if (hashes["photo_checkpoint"] != old["checkpoint_sha256"]["photo_latest"] or
            hashes["original_checkpoint"] != old["checkpoint_sha256"]["original_svr"]):
        raise ValueError("Baseline weights changed")
    selections = {}
    for method, step in (("photo_dpo5000", 5000), ("spatial_best", 20930), ("spatial_last", 63700)):
        saved = torch.load(paths[method], map_location="cpu", mmap=True, weights_only=False)
        hp = saved["hyper_parameters"]
        if saved["global_step"] != step:
            raise ValueError(f"Checkpoint selection changed: {method}")
        if method.startswith("spatial"):
            if (hp["model"]["name"] != "SpatialPhotoDiffusion" or
                    hp["model"]["frozen_source_sha256"] != hashes["original_checkpoint"] or
                    hp["dataset"]["provenance"]["validation_list_sha256"] != old["split_sha256"]):
                raise ValueError("Spatial source/split mismatch")
        elif (hp["model"].get("model_source") != "photo" or hp["model"].get("photo_global_step") != 310000 or
              hp["dataset"]["provenance"]["source_checkpoint_sha256"] != hashes["photo_checkpoint"] or
              hp["dataset"]["provenance"]["validation_list_sha256"] != old["split_sha256"]):
            raise ValueError("Photo DPO source/split mismatch")
        selections[method] = {"step": saved["global_step"], "epoch": saved["epoch"], "model": hp["model"]["name"]}
    inputs = {}
    for mid in ids:
        paths = {"image": Path(config["condition_root"]) / mid / "real_photo.npz",
                 "gt_npz": Path(config["gt_root"]) / mid / "data.npz",
                 "gt_step": Path(config["gt_root"]) / mid / "normalized_shape.step"}
        inputs[mid] = {name: sha256_file(path) if path.is_file() else None for name, path in paths.items()}
        if inputs[mid] != old["input_sha256"][mid]:
            raise ValueError(f"Paired input changed since baseline: {mid}")
    repo = Path(__file__).resolve().parents[3]
    extra = [Path(__file__), Path(__file__).with_name("photo_model_loading.py"),
             Path(__file__).with_name("photo_model_report.py"),
             *[repo / "src/brepnet/diffusion" / name for name in
               ("spatial_model.py", "spatial_blocks.py", "spatial_conditioning.py")]]
    source_paths = {Path(name) for name in old["source_sha256"]} | set(extra)
    protocol = {"version": "photo_spatial_dpo_comparison_v1", "config": config, "methods": list(METHODS),
                "identities": ids, "input_sha256": inputs, "checkpoint_sha256": hashes,
                "checkpoint_selection": selections, "sampling": old["sampling"], "post": old["post"],
                "split_sha256": old["split_sha256"], "runtime": current_runtime,
                "source_sha256": {str(p): sha256_file(p) for p in sorted(source_paths)},
                "baseline_protocol": {k: old[k] for k in ("fingerprint", "reuse_fingerprint", "reused_ids")},
                "baseline_source_audit": changes, "validation_not_test": True,
                "spatial_sampling": "signature adapter to identical CPU-initial/GPU-step RNG; unmodified denoiser"}
    protocol["fingerprint"] = fingerprint(protocol)
    dest = run / "metadata/protocol.json"
    if dest.exists() and read_json(dest) != protocol:
        raise ValueError("Refusing to resume a changed comparison protocol")
    atomic_json(dest, protocol)
    (run / "validation.txt").write_text("\n".join(ids) + "\n")
    for name in ("logs", "raw_predictions", "evaluations"):
        (run / name).mkdir(exist_ok=True)
    for path in source_paths:
        target = run / "metadata/source_snapshot" / path.relative_to(repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    artifacts = {}
    evaluation_config = {"raw_data_root": config["gt_root"], "seed": 42}
    for mid in ids:
        for stage in ("raw_predictions", "evaluations"):
            origin = source / stage / "photo_latest" / mid
            target = run / stage / "photo_baseline" / mid
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.symlink_to(origin.resolve(), target_is_directory=True)
            if not target.is_symlink() or target.resolve() != origin.resolve():
                raise ValueError(f"Unexpected baseline reuse destination: {target}")
        raw = run / "raw_predictions/photo_baseline" / mid
        row = verify_generation(protocol, "photo_baseline", mid)
        artifacts[str(raw / "inference.json")] = sha256_file(raw / "inference.json")
        if row["status"] == "ok":
            artifacts.update(_verify_evaluation(evaluation_config, mid,
                run / "evaluations/photo_baseline" / mid, raw, row))
    atomic_json(run / "metadata/reused_artifacts.json", artifacts)
    atomic_json(run / "REUSE_AUDIT.json", {"baseline_count": len(ids), "verified_artifacts": len(artifacts),
                 "source_changes_reviewed": changes, "all_inputs_equal_baseline": True})
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
        from src.brepnet.diffusion.legacy import sample_candidates
        from src.brepnet.eval.photo_model_loading import load_comparison_model

        torch.set_num_threads(2)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        transform = photo_transform()
        for method in NEW_METHODS:
            if sha256_file(Path(config[method])) != protocol["checkpoint_sha256"][method]:
                raise ValueError(f"Selected checkpoint changed: {method}")
            model, provenance = load_comparison_model(method, config, gpu)
            atomic_json(run / "metadata" / f"{method}-gpu{gpu}.json", provenance)
            for mid in ids:
                directory = run / "raw_predictions" / method / mid
                if (directory / "inference.json").exists():
                    verify_generation(protocol, method, mid)
                    continue
                seed = candidate_seed(42, mid, 0)
                row = {"method": method, "image_id": mid, "seed": seed,
                       "protocol_fingerprint": protocol["fingerprint"], "input_sha256": protocol["input_sha256"][mid]}
                if directory.exists():
                    atomic_json(directory / "inference.json", {**row, "status": "interrupted", "seconds": 0.})
                    continue
                directory.mkdir(parents=True)
                began = time.monotonic()
                try:
                    path = Path(config["condition_root"]) / mid / "real_photo.npz"
                    if sha256_file(path) != row["input_sha256"]["image"]:
                        raise ValueError("Photo changed after protocol freeze")
                    with np.load(path, allow_pickle=False) as data:
                        image = data["flux"]
                    if image.dtype != np.uint8 or image.shape != (512, 512, 3):
                        raise ValueError("Expected flux uint8 [512,512,3]")
                    result = sample_candidates(model, transform(image)[None], [seed])[0]
                    if not torch.isfinite(result["latent"]).all():
                        raise ValueError("Nonfinite sampled latent")
                    np.save(directory / "latent.npy", result["latent"].numpy(), allow_pickle=False)
                    np.savez_compressed(directory / "data.npz", **result["raw"])
                    row.update(status="ok", output_sha256={name: sha256_file(directory / name)
                                                           for name in ("latent.npy", "data.npz")})
                except Exception as exc:
                    traceback.print_exc()
                    row.update(status="generation_error", error=f"{type(exc).__name__}: {exc}")
                row["seconds"] = time.monotonic() - began
                atomic_json(directory / "inference.json", row)
                print(f"{method} {mid}: {row['status']} {row['seconds']:.2f}s", flush=True)
            del model
            torch.cuda.empty_cache()


def run_evaluation(protocol: dict, ids: list[str] | None = None) -> None:
    config, run = protocol["config"], Path(protocol["config"]["run_dir"])
    ids = protocol["identities"] if ids is None else ids
    context = mp.get_context("spawn")
    workers = [context.Process(target=gpu_worker, args=(protocol, gpu, ids[i::len(config["gpus"])]))
               for i, gpu in enumerate(config["gpus"]) if ids[i::len(config["gpus"])]]
    completed, futures = set(), {}
    pending = {(m, mid) for m in NEW_METHODS for mid in ids}
    began = time.monotonic()
    pool = ProcessPoolExecutor(max_workers=config["cpu_workers"], mp_context=context, initializer=initialize_cpu)
    for worker in workers:
        worker.start()
    try:
        while pending or futures or any(w.is_alive() for w in workers):
            for key in list(pending):
                method, mid = key
                if not (run / "raw_predictions" / method / mid / "inference.json").is_file():
                    continue
                row = verify_generation(protocol, method, mid)
                pending.remove(key)
                if row["status"] != "ok":
                    completed.add(key)
                else:
                    futures[key] = pool.submit(evaluate_one, config, method, mid)
            for key, future in list(futures.items()):
                if future.done():
                    future.result()
                    completed.add(key)
                    del futures[key]
            if any(w.exitcode not in (None, 0) for w in workers):
                raise RuntimeError(f"GPU worker failed: {[w.exitcode for w in workers]}")
            if pending and not any(w.is_alive() for w in workers):
                if all((run / "raw_predictions" / method / mid / "inference.json").is_file() for method, mid in pending):
                    continue
                raise RuntimeError("Missing terminal records after GPU workers exited")
            status = {"status": "running", "intended_per_method": len(protocol["identities"]),
                      "new_methods": list(NEW_METHODS), "reused_baseline": len(protocol["identities"]),
                      "generated_or_terminal": len(completed) + len(futures), "evaluated_or_failed": len(completed),
                      "total_new": len(ids) * len(NEW_METHODS), "pending_cpu": len(futures),
                      "gpu_workers_alive": sum(w.is_alive() for w in workers),
                      "elapsed_seconds": time.monotonic() - began,
                      "updated_at_utc": datetime.now(timezone.utc).isoformat()}
            atomic_json(run / "status.json", status)
            time.sleep(5)
        for name, digest in protocol["source_sha256"].items():
            if sha256_file(Path(name)) != digest:
                raise ValueError(f"Computational source changed during evaluation: {name}")
        if ids == protocol["identities"]:
            from src.brepnet.eval.photo_model_report import write_report
            write_report(run)
            final = "complete"
        else:
            final = "preflight_complete_not_full_evaluation"
        atomic_json(run / "status.json", {**status, "status": final,
                                         "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(10)
        pool.shutdown(wait=True, cancel_futures=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--photo-checkpoint", required=True)
    parser.add_argument("--original-checkpoint", required=True)
    for method in NEW_METHODS:
        parser.add_argument('--' + method.replace('_', '-'), required=True)
    parser.add_argument("--split-list", default="src/brepnet/data/list/deduplicated_deepcad_validation_7_30.txt")
    parser.add_argument("--condition-root", default="/mnt/d/data/deepcad_v7_cond")
    parser.add_argument("--gt-root", default="/mnt/d/data/deepcad_v7")
    parser.add_argument("--gpus", nargs="+", type=int, default=list(range(8)))
    parser.add_argument("--cpu-workers", type=int, default=32)
    parser.add_argument("--preflight", action="store_true")
    config = vars(parser.parse_args())
    preflight = config.pop("preflight")
    for key in config.keys() - {"gpus", "cpu_workers"}:
        config[key] = str(Path(config[key]).resolve())
    run = Path(config["run_dir"])
    run.mkdir(parents=True, exist_ok=True)
    with comparison_lock(run):
        protocol = prepare(config)
        run_evaluation(protocol, protocol["identities"][:1] if preflight else None)


if __name__ == "__main__":
    main()
