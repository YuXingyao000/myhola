"""Audit saved single-view diffusion comparisons without regenerating or repairing CAD.

Run ``python -m src.eval.diffusion_audit --help`` from the repository root.
The comparison's frozen identities are the denominator, including missing outputs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from importlib.metadata import version as package_version
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from src.common.io import atomic_json, fingerprint, read_json, sha256_file


VERSION = "single_view_diffusion_audit_v1"


def comparison_cases(run: Path, methods: list[str]) -> list[dict]:
    """Adapt the saved comparison layout; never infer the population from outputs."""
    protocol = read_json(run / "metadata/protocol.json")
    identities = protocol["identities"]
    if not identities or len(identities) != len(set(identities)):
        raise ValueError("Comparison identities must be nonempty and unique")
    config = protocol["config"]
    for method in methods:
        if Path(method).name != method or not (run / "raw_predictions" / method).is_dir():
            raise ValueError(f"Unknown comparison method: {method}")
    cases = []
    for image_id in identities:
        for method in methods:
            raw = run / "raw_predictions" / method / image_id
            evaluation = run / "evaluations" / method / image_id
            cases.append({
                "method": method, "image_id": image_id, "candidate_id": "c00",
                "inference_path": str(raw / "inference.json"),
                "raw_path": str(raw / "data.npz"),
                "evaluation_path": str(evaluation / "metrics.json"),
                "diagnostics_path": str(evaluation / "post/c00/post_diagnostics.json"),
                "step_path": str(evaluation / "post/c00/recon_brep.step"),
                "image_path": str(Path(config["condition_data_root"]) / image_id / "imgs.npz"),
                "gt_step_path": str(Path(config["raw_data_root"]) / image_id / "normalized_shape.step"),
                "gt_npz_path": str(Path(config["raw_data_root"]) / image_id / "data.npz"),
            })
    return cases


def _initialize_worker() -> None:
    from src.postprocess.cpu_runtime import preload_cpu_dependencies
    preload_cpu_dependencies()
    # Preload in the CPU-only parent; each bounded child gets separate OCC state.
    from src.eval import failure_analysis, failure_geometry  # noqa: F401
    from src.eval.metrics import conditional_diversity, observation  # noqa: F401


def _audit_child(case: dict, config: dict, destination: Path) -> None:
    from src.eval.failure_analysis import trace_outcome
    from src.eval.failure_geometry import measure_raw_consistency, inspect_saved_step
    from src.eval.metrics.conditional_diversity import geometry_fingerprints
    from src.eval.metrics.observation import evaluate_observation

    inference = read_json(Path(case["inference_path"]))
    evaluation = read_json(Path(case["evaluation_path"]))
    diagnostics = read_json(Path(case["diagnostics_path"]))
    row = {
        "method": case["method"], "image_id": case["image_id"], "candidate_id": case["candidate_id"],
        "audit_fingerprint": config["fingerprint"], "status": "running", "stage": "sources",
        "inference_status": inference.get("status", "missing"),
        "evaluation_status": evaluation.get("status", "missing"),
        "validity": evaluation.get("validity", "unknown"),
        "trace": trace_outcome(inference, evaluation, diagnostics or None),
        "raw_metrics": evaluation.get("raw", {}), "native": evaluation.get("native", {}),
        "sources": {}, "errors": {},
    }
    atomic_json(destination, row)
    expected = {
        "raw_path": inference.get("output_sha256", {}).get("data.npz"),
        "image_path": inference.get("input_sha256", {}).get("image"),
        "gt_step_path": inference.get("input_sha256", {}).get("gt_step"),
        "gt_npz_path": inference.get("input_sha256", {}).get("gt_npz"),
        "step_path": evaluation.get("step_sha256"),
    }
    for name, value in case.items():
        if not name.endswith("_path"):
            continue
        path = Path(value)
        digest = sha256_file(path) if path.is_file() else None
        row["sources"][name] = {"path": str(path.resolve()), "sha256": digest}
        if expected.get(name) and digest != expected[name]:
            row.update(status="source_mismatch", stage=name)
            row["errors"][name] = "Saved output or input differs from the comparison record"
            atomic_json(destination, row)
            return

    def measure(name, function, *args, **kwargs):
        row["stage"] = name
        atomic_json(destination, row)
        try:
            value = function(*args, **kwargs)
            row[name] = {"status": "ok", **value}
        except Exception as exc:
            row[name] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            row["errors"][name] = row[name]["error"]
            traceback.print_exc()
        atomic_json(destination, row)

    raw_path = Path(case["raw_path"])
    if raw_path.is_file():
        measure("raw_consistency", measure_raw_consistency, raw_path,
                distance_threshold=config["distance_threshold"])
        try:
            row["raw_fingerprints"] = geometry_fingerprints(raw_path)
            row["raw_status"] = "ok"
        except Exception as exc:
            row["raw_status"] = "error"
            row["errors"]["fingerprints"] = str(exc)
    else:
        row["raw_status"] = "missing"
        row["raw_consistency"] = {"status": "missing"}
    step = Path(case["step_path"])
    if step.is_file():
        measure("step_check", inspect_saved_step, step)
        measure("observation", evaluate_observation, step, Path(case["gt_step_path"]),
                Path(case["image_path"]), resolution=config["resolution"],
                mesh_deflection=config["mesh_deflection"], mask_threshold=config["mask_threshold"],
                cache_root=Path(config["output_dir"]) / "gt_render_cache")
    else:
        row["step_check"] = {"status": "missing_step"}
        row["observation"] = {"status": "missing_step"}
    row.update(status="partial" if row["errors"] else "ok", stage="done")
    atomic_json(destination, row)


def _audit_bounded(case: dict, config: dict) -> dict:
    output = Path(config["output_dir"]) / "cases" / case["method"] / case["image_id"]
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "audit.json"
    previous = read_json(destination)
    if previous:
        if previous.get("audit_fingerprint") != config["fingerprint"]:
            raise ValueError(f"Existing audit uses a different protocol: {destination}")
        for name, source in previous.get("sources", {}).items():
            path = Path(source["path"])
            if Path(case[name]).resolve() != path:
                raise ValueError(f"Audit source target changed: {case[name]}")
            current_hash = sha256_file(path) if path.is_file() else None
            if current_hash != source["sha256"]:
                raise ValueError(f"Audit source changed: {path}")
        if previous["status"] != "running":
            return previous
        previous.update(status="interrupted")
        atomic_json(destination, previous)
        return previous

    def child():
        with (output / "worker.log").open("a", buffering=1) as log:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            _audit_child(case, config, destination)

    started = time.monotonic()
    process = mp.get_context("fork").Process(target=child)
    process.start()
    process.join(config["timeout"])
    expired = process.is_alive()
    if expired:
        process.terminate()
        process.join(3)
        if process.is_alive():
            process.kill()
            process.join()
    from src.eval.failure_analysis import trace_outcome
    inference = read_json(Path(case["inference_path"]))
    evaluation = read_json(Path(case["evaluation_path"]))
    row = {"validity": evaluation.get("validity", "unknown"),
           "trace": trace_outcome(inference, evaluation),
           **{key: case[key] for key in ("method", "image_id", "candidate_id")},
           **read_json(destination)}
    if expired or process.exitcode != 0 or row.get("status") == "running":
        row["status"] = "timeout" if expired else "crash"
    row.update(audit_fingerprint=config["fingerprint"], worker_exit_code=process.exitcode,
               elapsed_seconds=time.monotonic() - started)
    atomic_json(destination, row)
    return row


def wait_for_comparison(run: Path, timeout: float, output: Path) -> None:
    deadline = time.monotonic() + timeout
    while True:
        status = read_json(run / "status.json")
        if status.get("status") == "complete":
            return
        exit_code = Path(str(run) + ".exit_code.txt")
        if exit_code.is_file() or time.monotonic() >= deadline:
            raise RuntimeError(f"Comparison is not complete: {status}; exit file: {exit_code.exists()}")
        atomic_json(output / "status.json", {"status": "waiting_for_comparison", "comparison": status})
        time.sleep(min(30, max(0, deadline - time.monotonic())))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=["baseline", "dpo_5000"])
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--wait-seconds", type=float, default=0,
                        help="Wait up to this budget for the existing comparison to finish")
    parser.add_argument("--distance-threshold", type=float, default=.01)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--mesh-deflection", type=float, default=.002)
    parser.add_argument("--mask-threshold", type=int, default=245)
    parser.add_argument("--font", type=Path, help="CJK font file for Chinese report figures")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run, output = args.comparison_run.resolve(), args.output_dir.resolve()
    if not args.methods or len(set(args.methods)) != len(args.methods):
        parser.error("Methods must be distinct")
    if args.workers < 1 or args.timeout <= 0 or args.wait_seconds < 0:
        parser.error("Workers/timeout must be positive; wait budget cannot be negative")
    output.mkdir(parents=True, exist_ok=args.resume)
    config = {key: str(value.resolve()) if isinstance(value, Path) else value
              for key, value in vars(args).items() if key not in ("resume", "wait_seconds", "font")}
    sources = [Path(__file__), *Path(__file__).parent.glob("failure_*.py"),
               Path(__file__).with_name("diffusion_audit_report.py"),
               *[Path(__file__).parent / "metrics" / name for name in ("observation.py", "conditional_diversity.py")]]
    repo = Path(__file__).resolve().parents[2]
    sources.extend(repo / name for name in (
        "src/eval/occ_diagnostics.py", "src/common/io.py", "src/common/step_io.py", "src/postprocess/cpu_runtime.py",
        "src/preprocess/extract_visibility.py", "src/models/diffusion/image_loading.py",
        "src/eval/io.py", "src/eval/preference_report.py"))
    import OCC
    protocol = {
        "version": VERSION, "config": config,
        "runtime": {"python": sys.version, "pythonocc": OCC.VERSION, "packages": {
            name: package_version(name) for name in ("numpy", "scipy", "torch", "matplotlib")}},
        "comparison_protocol_sha256": sha256_file(run / "metadata/protocol.json"),
        "source_sha256": {str(path.resolve()): sha256_file(path) for path in sources},
    }
    protocol["fingerprint"] = fingerprint(protocol)
    config["fingerprint"] = protocol["fingerprint"]
    if args.resume:
        if read_json(output / "protocol.json") != protocol:
            raise ValueError("Resume requires the same sources, comparison and configuration")
    else:
        atomic_json(output / "protocol.json", protocol)
        atomic_json(output / "launch.json", {
            "created_utc": datetime.now(timezone.utc).isoformat(), "command": [sys.executable, *sys.argv],
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        })
        (output / "git.diff").write_bytes(subprocess.check_output(["git", "diff", "HEAD"]))
    wait_for_comparison(run, args.wait_seconds, output)
    cases = comparison_cases(run, args.methods)
    atomic_json(output / "manifest.json", {"cases": cases, "intended_count": len(cases)})
    started = time.monotonic()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                             initializer=_initialize_worker) as pool:
        futures = [pool.submit(_audit_bounded, case, config) for case in cases]
        for completed, future in enumerate(as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            atomic_json(output / "status.json", {"status": "auditing", "completed": completed, "total": len(cases)})
            if completed % 100 == 0:
                print(f"Audited {completed}/{len(cases)}", flush=True)
    from src.eval.diffusion_audit_report import write_report
    write_report(rows, output, font=args.font)
    atomic_json(output / "status.json", {"status": "complete", "completed": len(rows), "total": len(cases),
                                         "elapsed_seconds": time.monotonic() - started})


if __name__ == "__main__":
    main()
