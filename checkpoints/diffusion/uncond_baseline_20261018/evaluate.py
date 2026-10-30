"""Fixed-denominator, GT-free post-processing of the 1,000 legacy samples.

Each object gets one isolated CPU process and one 180-second budget. Existing
terminal results are reused only when source, manifest and input hashes match.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib import metadata as package_metadata
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import signal
import sys
import tempfile
import time
import traceback
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from tools.vae_compare_post import (  # noqa: E402
    atomic_json, compatibility_validity, configure_cpu, fingerprint,
    native_validity, preload_cpu_dependencies, read_json, read_step_at_precision,
    sha256_file, terminate_process, validate_raw_prediction,
)


SAMPLE_COUNT = 1000
WORKERS = 16
TIMEOUT_SECONDS = 180.0
SEED_PREFIX = "hola-uncond-20261018-v1:"
TERMINAL = {"ok", "invalid_raw", "post_error", "no_step", "step_read_error",
            "timeout", "crash", "interrupted"}
INFERENCE_TERMINAL = {"ok", "error", "interrupted"}


def sample_ids() -> list[str]:
    return [f"uncond_{index:06d}" for index in range(SAMPLE_COUNT)]


def sample_seed(sample_id: str) -> int:
    return int.from_bytes(hashlib.sha256((SEED_PREFIX + sample_id).encode()).digest()[:4], "little")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_record(path: Path) -> dict:
    """A broken status is visible in the report rather than silently missing."""
    try:
        return read_json(path)
    except (OSError, ValueError) as exc:
        return {"status": "metadata_error", "error": f"{type(exc).__name__}: {exc}"}


def load_raw(path: Path) -> tuple[dict, dict]:
    """Keep all arrays; adapt only exact integer connectivity, never geometry."""
    import numpy as np
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    face, edge = arrays["pred_face"], arrays["pred_edge"]
    conn = arrays["pred_edge_face_connectivity"]
    if arrays.get("representation", np.asarray("legacy_half_edges")).item() != "legacy_half_edges":
        raise ValueError("This frozen baseline accepts legacy_half_edges only")
    if face.ndim != 4 or face.shape[-1] != 3 or not len(face) or min(face.shape[1:3]) < 2:
        raise ValueError("pred_face must be finite nonempty [F,U,V,3], U,V>=2")
    if edge.ndim != 3 or edge.shape[-1] != 3 or not len(edge) or edge.shape[1] < 2:
        raise ValueError("pred_edge must be finite nonempty [E,L,3], L>=2")
    if not np.isfinite(face).all() or not np.isfinite(edge).all():
        raise ValueError("Raw geometry contains NaN/Inf")
    if conn.shape != (len(edge), 3) or not np.issubdtype(conn.dtype, np.number):
        raise ValueError("Connectivity must be numeric [E,3]")
    if np.issubdtype(conn.dtype, np.complexfloating) or not np.isfinite(conn).all():
        raise ValueError("Connectivity must contain finite real integers")
    # Bounds are checked before casting, avoiding int64 overflow on bad exports.
    upper = max(len(face), len(edge))
    if (conn < 0).any() or (conn >= upper).any() or not np.equal(conn, np.floor(conn)).all():
        raise ValueError("Connectivity is fractional, negative, or outside primitive bounds")
    adapted = conn.astype(np.int64)
    if not np.array_equal(adapted.astype(conn.dtype), conn):
        raise ValueError("Connectivity int64 conversion is not exactly reversible")
    if not np.array_equal(np.sort(adapted[:, 0]), np.arange(len(edge))):
        raise ValueError("Edge IDs must cover the raw curve array exactly once")
    if (adapted[:, 1:] >= len(face)).any():
        raise ValueError("Connectivity contains an invalid face ID")
    arrays["pred_edge_face_connectivity"] = adapted
    return arrays, {
        "raw_num_faces": len(face), "raw_num_directed_edges": len(edge),
        "connectivity_original_dtype": str(conn.dtype),
        "connectivity_post_dtype": str(adapted.dtype),
        "connectivity_dtype_changed": conn.dtype != adapted.dtype,
        "connectivity_values_unchanged": True,
    }


def adapt_raw(raw: Path, destination: Path) -> dict:
    import numpy as np
    arrays, details = load_raw(raw)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite post input: {destination}")
    with destination.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    # Reuse the common export validator after the sole dtype adaptation.
    validate_raw_prediction(destination)
    with np.load(raw, allow_pickle=False) as original, np.load(destination, allow_pickle=False) as saved:
        if set(original.files) != set(saved.files):
            raise ValueError("Adapted archive changed its array names")
        for key in original.files:
            if not np.array_equal(original[key], saved[key], equal_nan=False):
                raise ValueError(f"Adaptation altered values in {key}")
            if key != "pred_edge_face_connectivity" and original[key].dtype != saved[key].dtype:
                raise ValueError(f"Adaptation altered dtype in {key}")
    return {**details, "post_input_sha256": sha256_file(destination),
            "raw_sha256": sha256_file(raw), "raw_usable": True}


def post_protocol(root: Path) -> dict:
    manifest_path = root / "metadata" / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("sample_count") != SAMPLE_COUNT or manifest.get("seed_prefix") != SEED_PREFIX:
        raise ValueError("Prepare the frozen 1,000-sample manifest before post-processing")
    sources = set((REPO_ROOT / "src/brepnet/post").glob("*.py"))
    sources.update(REPO_ROOT / name for name in (
        "tools/vae_compare_post.py", "shared/occ_utils.py", "shared/common_utils.py",
        "src/brepnet/eval/metrics/validity.py", "src/brepnet/data/shared_boundaries.py"))
    versions = {}
    for name in ("numpy", "torch", "scipy", "trimesh", "pythonocc-core"):
        try:
            versions[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            versions[name] = None
    import OCC
    versions["OCC"] = getattr(OCC, "VERSION", None)
    return {
        "schema_version": 1, "version": "hola_uncond_post_20261018_v1",
        "manifest_sha256": sha256_file(manifest_path), "intended_count": SAMPLE_COUNT,
        "post_policy": "legacy", "drop_num": 0, "optimize_geometry": True,
        "max_optimize_iter": 200, "adaptive_max_optimize_iter": 600,
        "workers": WORKERS, "timeout_seconds": TIMEOUT_SECONDS,
        "timeout_scope": "whole per-object subprocess, including adaptation and OCC checks",
        "use_cuda": False, "threads_per_worker": 1, "algorithm_retries": 0,
        "seed": "same per-sample seed as inference; torch/NumPy/Python reset in worker",
        "native_validity": "fresh read mode=0; TopAbs_SOLID and BRepCheck before any mutation",
        "compatibility_validity": "separate read mode=1, precision=.1; SetTolerance(.1); TopAbs_SOLID and BRepCheck",
        "post_source_sha256": {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in sorted(sources)},
        "runner_sha256": sha256_file(Path(__file__)),
        "environment": {"python": sys.version, "executable": sys.executable, "packages": versions},
    }


def execute_job(job: dict, status_path: Path) -> dict:
    started = time.monotonic()
    root, sample_id = Path(job["root"]), job["sample_id"]
    raw = root / "raw_predictions" / sample_id / "data.npz"
    output = root / "post" / sample_id
    row = {"schema_version": 1, "sample_id": sample_id, "seed": sample_seed(sample_id),
           "protocol_fingerprint": job["protocol_fingerprint"],
           "inference_record_sha256": job["inference_record_sha256"],
           "raw_sha256": job["raw_sha256"], "status": "running", "stage": "adapt_raw",
           "started_at_utc": utc_now(), "process_pid": os.getpid(), "raw_usable": False,
           "step_exists": False, "step_readable": False, "native_valid_solid": None,
           "compat_valid_solid": None, "post_status": "not_run", "error": ""}

    def record(**values: Any) -> None:
        row.update(values)
        row["elapsed_seconds"] = time.monotonic() - started
        atomic_json(status_path, row)

    record()
    try:
        if sha256_file(raw) != job["raw_sha256"]:
            raise ValueError("Raw file changed after scheduling")
        record(**adapt_raw(raw, root / "post_inputs" / sample_id / "data.npz"))
    except Exception as exc:
        record(status="invalid_raw", error=f"{type(exc).__name__}: {exc}")
        return row
    try:
        import numpy as np
        import torch
        from src.brepnet.post.construct_brep import construct_brep_for_sample
        random.seed(row["seed"])
        np.random.seed(row["seed"])
        torch.manual_seed(row["seed"])
        if output.exists():
            raise FileExistsError(f"Refusing to erase previous post output: {output}")
        record(stage="post", post_status="running")
        post_started = time.monotonic()
        construct_brep_for_sample(
            root / "post_inputs", root / "post", sample_id, v_drop_num=0,
            is_ray=False, is_log=False, is_optimize_geom=True, v_max_optimize_iter=200,
            isdebug=False, use_cuda=False, from_scratch=True, is_save_data=False,
            post_policy="legacy",
        )
        record(post_status="returned", post_elapsed_seconds=time.monotonic() - post_started)
    except Exception as exc:
        traceback.print_exc()
        record(status="post_error", post_status="error", error=f"{type(exc).__name__}: {exc}")
        return row
    step = output / "recon_brep.step"
    record(step_exists=step.is_file(), success_marker=(output / "success.txt").is_file())
    if not step.is_file():
        record(status="no_step", stage="done", error="Post returned without a STEP")
        return row
    try:
        record(stage="native_validity", step_sha256=sha256_file(step))
        record(**native_validity(read_step_at_precision(step, mode=0)))
    except Exception as exc:
        record(status="step_read_error", error=f"{type(exc).__name__}: {exc}")
        return row
    try:
        record(stage="compatibility_validity")
        record(compat_valid_solid=compatibility_validity(step))
    except Exception as exc:
        record(compatibility_error=f"{type(exc).__name__}: {exc}")
    record(status="ok", stage="done")
    return row


def job_process(job: dict, status_path: Path, log_path: Path) -> None:
    os.setsid()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("xb", buffering=0) as handle:
        os.dup2(handle.fileno(), 1)
        os.dup2(handle.fileno(), 2)
        execute_job(job, status_path)


def run_jobs(jobs: list[dict], metadata: Path, *, workers: int = WORKERS,
             timeout: float = TIMEOUT_SECONDS, target: Callable = job_process) -> None:
    """Kill the complete process group; never give algorithm failures a retry."""
    context = mp.get_context("fork")
    pending, active = iter(jobs), {}
    exhausted, completed = False, 0
    metadata.mkdir(parents=True, exist_ok=True)

    def finish(process, launched: float, job: dict, status_path: Path, forced: str | None = None) -> None:
        row = read_record(status_path)
        row.setdefault("sample_id", job["sample_id"])
        for key in ("protocol_fingerprint", "raw_sha256", "inference_record_sha256"):
            row.setdefault(key, job[key])
        if forced:
            row.update(status=forced, error=f"Whole-process {forced}; budget={timeout:g}s")
        elif process.exitcode != 0 or row.get("status") not in TERMINAL:
            row.update(status="crash", error=f"Worker exit={process.exitcode}, last stage={row.get('stage', 'startup')}")
        step = Path(job["root"]) / "post" / job["sample_id"] / "recon_brep.step"
        row.update(worker_exit_code=process.exitcode, wall_seconds=time.monotonic() - launched,
                   finished_at_utc=utc_now(), step_exists=step.is_file())
        atomic_json(status_path, row)
        with (metadata / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")

    try:
        while active or not exhausted:
            while len(active) < workers and not exhausted:
                try:
                    job = next(pending)
                except StopIteration:
                    exhausted = True
                    break
                status_path = metadata / "samples" / f"{job['sample_id']}.json"
                log_path = metadata / "logs" / f"{job['sample_id']}.log"
                process = context.Process(target=target, args=(job, status_path, log_path))
                launched = time.monotonic()
                process.start()
                active[process.pid] = (process, launched, job, status_path)
            for pid, (process, launched, job, status_path) in list(active.items()):
                expired = process.is_alive() and time.monotonic() - launched >= timeout
                if process.is_alive() and not expired:
                    continue
                if expired:
                    terminate_process(process)
                else:
                    process.join()
                finish(process, launched, job, status_path, "timeout" if expired else None)
                del active[pid]
                completed += 1
                if completed % 20 == 0 or completed == len(jobs):
                    print(f"Post completed {completed}/{len(jobs)}", flush=True)
            if active:
                time.sleep(0.02)
    finally:
        for process, launched, job, status_path in active.values():
            terminate_process(process)
            finish(process, launched, job, status_path, "interrupted")


def schedule_jobs(root: Path, protocol: dict, limit: int) -> list[dict]:
    digest = fingerprint(protocol)
    inference_protocol = read_json(root / "metadata/inference/protocol.json")
    jobs = []
    for sample_id in sample_ids()[:limit]:
        inference_path = root / "metadata/inference/samples" / f"{sample_id}.json"
        inference = read_record(inference_path)
        old_path = root / "metadata/post/samples" / f"{sample_id}.json"
        old = read_record(old_path)
        raw = root / "raw_predictions" / sample_id / "data.npz"
        if old:
            if old.get("protocol_fingerprint") != digest:
                raise ValueError(f"Post protocol changed for {sample_id}; preserve this run")
            if not raw.is_file() or sha256_file(raw) != old.get("raw_sha256"):
                raise ValueError(f"Raw prediction changed for {sample_id}")
            if not inference_path.is_file() or sha256_file(inference_path) != old.get("inference_record_sha256"):
                raise ValueError(f"Inference provenance changed for {sample_id}")
            if old.get("status") not in TERMINAL:
                old.update(status="interrupted", error="Previous post invocation had no terminal record; no retry")
                atomic_json(old_path, old)
            continue
        if inference.get("status") != "ok":
            continue
        if inference.get("identity", {}).get("manifest_sha256") != protocol["manifest_sha256"]:
            raise ValueError(f"Inference manifest mismatch for {sample_id}")
        if inference.get("identity") != inference_protocol.get("identity"):
            raise ValueError(f"Inference source protocol mismatch for {sample_id}")
        if not raw.is_file() or sha256_file(raw) != inference.get("output_sha256"):
            raise ValueError(f"Inference output hash mismatch for {sample_id}")
        for path in (root / "post" / sample_id, root / "post_inputs" / sample_id,
                     root / "metadata/post/logs" / f"{sample_id}.log"):
            if path.exists():
                raise FileExistsError(f"Untracked previous post artifact; preserve and inspect: {path}")
        jobs.append({"sample_id": sample_id, "root": str(root), "protocol_fingerprint": digest,
                     "raw_sha256": inference["output_sha256"],
                     "inference_record_sha256": sha256_file(inference_path)})
    return jobs


def wilson_interval(count: int, denominator: int) -> list[float]:
    if denominator < 1 or not 0 <= count <= denominator:
        raise ValueError("Wilson interval requires 0<=count<=denominator and denominator>0")
    z = 1.959963984540054
    rate = count / denominator
    scale = 1 + z * z / denominator
    center = (rate + z * z / (2 * denominator)) / scale
    radius = z * math.sqrt(rate * (1 - rate) / denominator + z * z / (4 * denominator ** 2)) / scale
    lower = 0.0 if count == 0 else max(0.0, center - radius)
    upper = 1.0 if count == denominator else min(1.0, center + radius)
    return [lower, upper]


def statistics(values: list) -> dict:
    import numpy as np
    values = [float(value) for value in values if value is not None and math.isfinite(value)]
    names = ("min", "p05", "p25", "median", "p75", "p95", "max")
    quantiles = np.quantile(values, [0, .05, .25, .5, .75, .95, 1]).tolist() if values else [None] * 7
    return {"n": len(values), "mean": float(np.mean(values)) if values else None, **dict(zip(names, quantiles))}


def collect_rows(root: Path) -> list[dict]:
    rows = []
    manifest_path = root / "metadata/manifest.json"
    manifest_sha = sha256_file(manifest_path) if manifest_path.is_file() else None
    protocol = read_record(root / "metadata/post/protocol.json")
    protocol_digest = fingerprint(protocol) if protocol else None
    inference_protocol = read_record(root / "metadata/inference/protocol.json")
    for sample_id in sample_ids():
        inference = read_record(root / "metadata/inference/samples" / f"{sample_id}.json")
        post = read_record(root / "metadata/post/samples" / f"{sample_id}.json")
        raw = root / "raw_predictions" / sample_id / "data.npz"
        inference_status = inference.get("status", "pending")
        post_status = post.get("status", "pending" if inference_status not in {"error", "interrupted"} else "not_applicable")
        provenance_errors = []
        if inference and manifest_sha and inference.get("identity", {}).get("manifest_sha256") != manifest_sha:
            provenance_errors.append("Inference manifest hash mismatch")
        if inference and inference_protocol and inference.get("identity") != inference_protocol.get("identity"):
            provenance_errors.append("Inference source protocol mismatch")
        if raw.is_file() and inference.get("output_sha256") and sha256_file(raw) != inference["output_sha256"]:
            provenance_errors.append("Raw prediction hash mismatch")
        if inference_status == "ok" and not raw.is_file():
            provenance_errors.append("Successful inference has no raw prediction")
        if post:
            inference_path = root / "metadata/inference/samples" / f"{sample_id}.json"
            if post.get("protocol_fingerprint") != protocol_digest:
                provenance_errors.append("Post protocol fingerprint mismatch")
            if not inference_path.is_file() or sha256_file(inference_path) != post.get("inference_record_sha256"):
                provenance_errors.append("Post inference record hash mismatch")
            if not raw.is_file() or sha256_file(raw) != post.get("raw_sha256"):
                provenance_errors.append("Post raw hash mismatch")
            adapted = root / "post_inputs" / sample_id / "data.npz"
            if post.get("post_input_sha256") and (not adapted.is_file() or sha256_file(adapted) != post["post_input_sha256"]):
                provenance_errors.append("Post input hash mismatch")
            step = root / "post" / sample_id / "recon_brep.step"
            if post.get("step_sha256") and (not step.is_file() or sha256_file(step) != post["step_sha256"]):
                provenance_errors.append("STEP hash mismatch")
            if post.get("step_readable") and not step.is_file():
                provenance_errors.append("Previously read STEP is missing")
        counts = {"raw_num_faces": inference.get("raw_face_count"),
                  "raw_num_directed_edges": inference.get("raw_directed_edge_count")}
        usable, raw_error = post.get("raw_usable"), ""
        if raw.is_file() and usable is None:
            try:
                _, details = load_raw(raw)
                counts.update({key: details[key] for key in counts})
                usable = True
            except Exception as exc:
                usable, raw_error = False, f"{type(exc).__name__}: {exc}"
        for key in counts:
            if post.get(key) is not None:
                counts[key] = post[key]
        step = root / "post" / sample_id / "recon_brep.step"
        rows.append({
            "sample_id": sample_id, "seed": sample_seed(sample_id),
            "inference_status": inference_status, "post_status": post_status,
            "stage": post.get("stage", inference.get("stage", "not_started")),
            "raw_exists": raw.is_file(), "raw_usable": usable, **counts,
            "step_exists": step.is_file(), "step_readable": post.get("step_readable", False),
            "native_valid_solid": post.get("native_valid_solid"),
            "native_is_solid_type": post.get("native_is_solid_type"),
            "native_brepcheck_valid": post.get("native_brepcheck_valid"),
            "compat_valid_solid": post.get("compat_valid_solid"),
            "step_num_faces": post.get("num_faces"), "step_num_edges": post.get("num_edges"),
            "step_num_vertices": post.get("num_vertices"),
            "inference_seconds": inference.get("inference_seconds"),
            "post_wall_seconds": post.get("wall_seconds"),
            "post_algorithm_seconds": post.get("post_elapsed_seconds"),
            "raw_sha256": post.get("raw_sha256", inference.get("output_sha256")),
            "post_input_sha256": post.get("post_input_sha256"), "step_sha256": post.get("step_sha256"),
            "error": " | ".join(str(value) for value in (inference.get("error"), post.get("error"), raw_error) if value),
            "compatibility_error": post.get("compatibility_error", ""),
            "provenance_error": " | ".join(provenance_errors),
        })
    return rows


def summarize(root: Path) -> dict:
    rows = collect_rows(root)
    counts = {key: sum(row[key] is True for row in rows) for key in (
        "raw_exists", "raw_usable", "step_exists", "step_readable", "native_valid_solid", "compat_valid_solid")}
    counts.update(inference_ok=sum(row["inference_status"] == "ok" for row in rows),
                  inference_terminal=sum(row["inference_status"] in INFERENCE_TERMINAL for row in rows),
                  post_terminal=sum(row["post_status"] in TERMINAL for row in rows),
                  timeout=sum(row["post_status"] == "timeout" for row in rows))
    complete = all(not row["provenance_error"] and row["inference_status"] in INFERENCE_TERMINAL and (
        row["inference_status"] != "ok" or row["post_status"] in TERMINAL) for row in rows)
    counts["pending"] = sum(row["inference_status"] not in INFERENCE_TERMINAL or (
        row["inference_status"] == "ok" and row["post_status"] not in TERMINAL) for row in rows)
    counts["provenance_errors"] = sum(bool(row["provenance_error"]) for row in rows)
    # Do not retain stale success flags after an artifact/provenance mismatch.
    for name in ("step_readable", "native_valid_solid", "compat_valid_solid"):
        counts[name] = sum(row[name] is True and not row["provenance_error"] for row in rows)
    rate_names = ("inference_ok", "raw_exists", "raw_usable", "step_exists", "step_readable", "native_valid_solid", "compat_valid_solid")
    stat_names = ("raw_num_faces", "raw_num_directed_edges", "step_num_faces", "step_num_edges",
                  "step_num_vertices", "inference_seconds", "post_wall_seconds", "post_algorithm_seconds")
    face_counts = [row["raw_num_faces"] for row in rows if row["raw_num_faces"] is not None]
    bands = {"lt7": sum(value < 7 for value in face_counts),
             "7_to_30": sum(7 <= value <= 30 for value in face_counts),
             "gt30": sum(value > 30 for value in face_counts)}
    summary = {
        "schema_version": 1, "created_at_utc": utc_now(), "root": str(root),
        "intended_count": SAMPLE_COUNT, "complete": complete,
        "status": "complete" if complete else "incomplete",
        "interval_interpretation": "Wilson 95%; unfinished samples remain in the 1000 denominator; partial rates are not final performance",
        "counts": counts,
        "rates": {name: {"count": counts[name], "denominator": SAMPLE_COUNT,
                          "rate": counts[name] / SAMPLE_COUNT,
                          "wilson95": wilson_interval(counts[name], SAMPLE_COUNT)} for name in rate_names},
        "inference_status_counts": dict(Counter(row["inference_status"] for row in rows)),
        "post_status_counts": dict(Counter(row["post_status"] for row in rows)),
        "error_counts": dict(Counter(row["error"] for row in rows if row["error"])),
        "statistics": {name: statistics([row[name] for row in rows]) for name in stat_names},
        "raw_face_bands": {"measurable_count": len(face_counts), "missing_count": SAMPLE_COUNT - len(face_counts),
                           "counts": bands, "rates_over_1000": {name: value / SAMPLE_COUNT for name, value in bands.items()}},
        "metrics_not_computed": ["COV", "MMD", "LFD", "GT reconstruction CD", "GT face-count error"],
        "post_protocol": read_record(root / "metadata/post/protocol.json"),
    }
    report = root / "report"
    report.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=report, prefix="per_sample.", suffix=".tmp", delete=False, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, report / "per_sample.csv")
    atomic_json(report / "summary.json", summary)
    return summary


def _selftest_timeout(job: dict, status_path: Path, log_path: Path) -> None:
    os.setsid()
    atomic_json(status_path, {"sample_id": job["sample_id"], "status": "running",
                              "stage": "compatibility_validity", "native_valid_solid": True})
    time.sleep(10)


def self_test() -> dict:
    """Small CPU checks; no model checkpoint or research samples are read."""
    import numpy as np
    checks = []
    with tempfile.TemporaryDirectory(prefix="hola-uncond-evaluate-") as temporary:
        root = Path(temporary)
        raw = root / "raw_predictions/uncond_000000/data.npz"
        raw.parent.mkdir(parents=True)
        arrays = {"pred_face": np.zeros((2, 2, 2, 3), dtype=np.float32),
                  "pred_edge": np.zeros((2, 2, 3), dtype=np.float32),
                  "pred_edge_face_connectivity": np.asarray([[0, 0, 1], [1, 1, 0]], dtype=np.float32),
                  "extra": np.asarray([1.25], dtype=np.float32)}
        np.savez_compressed(raw, **arrays)
        old_hash = sha256_file(raw)
        details = adapt_raw(raw, root / "post_inputs/uncond_000000/data.npz")
        assert details["connectivity_values_unchanged"] and details["connectivity_dtype_changed"]
        assert sha256_file(raw) == old_hash
        checks.append("lossless_float_connectivity_adaptation_and_raw_preservation")
        arrays["pred_edge_face_connectivity"][0, 0] = .5
        bad = root / "fractional.npz"
        np.savez_compressed(bad, **arrays)
        try:
            load_raw(bad)
        except ValueError:
            checks.append("fractional_connectivity_rejected")
        else:
            raise AssertionError("Fractional connectivity accepted")
        summary = summarize(root)
        with (root / "report/per_sample.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == SAMPLE_COUNT and summary["counts"]["raw_exists"] == 1
        assert summary["counts"]["pending"] == SAMPLE_COUNT and not summary["complete"]
        assert summary["statistics"]["step_num_faces"]["n"] == 0
        assert summary["statistics"]["step_num_faces"]["mean"] is None
        assert "NaN" not in json.dumps(summary, allow_nan=False)
        checks.append("fixed_1000_denominator_pending_rows_empty_metrics")
        assert wilson_interval(0, 1000)[0] == 0
        assert math.isclose(wilson_interval(1000, 1000)[1], 1)
        assert wilson_interval(500, 1000)[0] < .5 < wilson_interval(500, 1000)[1]
        checks.append("wilson_boundary_and_interior_intervals")
        job = {"sample_id": "uncond_000001", "root": str(root), "raw_sha256": "test",
               "inference_record_sha256": "test", "protocol_fingerprint": "test"}
        run_jobs([job], root / "metadata/post", workers=1, timeout=.15, target=_selftest_timeout)
        row = read_json(root / "metadata/post/samples/uncond_000001.json")
        assert row["status"] == "timeout" and row["native_valid_solid"] is True
        assert row["wall_seconds"] < 3
        checks.append("whole_process_timeout_preserves_completed_native_check")
        manifest = {"sample_count": SAMPLE_COUNT, "seed_prefix": SEED_PREFIX}
        manifest_path = root / "metadata/manifest.json"
        atomic_json(manifest_path, manifest)
        identity = {"manifest_sha256": sha256_file(manifest_path)}
        inference_path = root / "metadata/inference/samples/uncond_000000.json"
        inference = {"status": "ok", "identity": identity, "output_sha256": old_hash}
        atomic_json(inference_path, inference)
        atomic_json(root / "metadata/inference/protocol.json", {"identity": identity})
        protocol = post_protocol(root)
        atomic_json(root / "metadata/post/protocol.json", protocol)
        terminal_path = root / "metadata/post/samples/uncond_000000.json"
        terminal = {"sample_id": "uncond_000000", "status": "no_step", "raw_sha256": old_hash,
                    "inference_record_sha256": sha256_file(inference_path),
                    "protocol_fingerprint": fingerprint(protocol)}
        atomic_json(terminal_path, terminal)
        assert schedule_jobs(root, protocol, 1) == []
        terminal["status"] = "running"
        atomic_json(terminal_path, terminal)
        assert schedule_jobs(root, protocol, 1) == []
        assert read_json(terminal_path)["status"] == "interrupted"
        checks.append("terminal_and_interrupted_jobs_are_not_retried")
        inference["output_sha256"] = "changed"
        atomic_json(inference_path, inference)
        try:
            schedule_jobs(root, protocol, 1)
        except ValueError:
            checks.append("resume_rejects_changed_inference_provenance")
        else:
            raise AssertionError("Changed inference record was reused")
        assert summarize(root)["counts"]["provenance_errors"] > 0
        checks.append("summary_exposes_hash_mismatches")
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
        from OCC.Extend.DataExchange import write_step_file
        step = root / "box.step"
        write_step_file(BRepPrimAPI_MakeBox(1, 2, 3).Shape(), str(step))
        native = native_validity(read_step_at_precision(step, mode=0))
        assert native["native_valid_solid"] and native["num_faces"] == 6 and native["num_edges"] == 12
        assert compatibility_validity(step)
        checks.append("native_and_compat_occ_checks_without_gt")
    return {"status": "passed", "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("post", "summarize", "all"), default="all")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--limit", type=int, default=SAMPLE_COUNT,
                        help="Consider the first N fixed IDs for post; reports always contain all 1000")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    configure_cpu()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return 0
    if not 1 <= args.limit <= SAMPLE_COUNT:
        parser.error("--limit must be in [1,1000]")
    root = args.root.resolve()
    if args.phase in {"post", "all"}:
        metadata = root / "metadata/post"
        metadata.mkdir(parents=True, exist_ok=True)
        with (metadata / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            protocol = post_protocol(root)
            previous = read_json(metadata / "protocol.json")
            if previous and previous != protocol:
                raise ValueError("Post code or frozen protocol changed; refusing to mix results")
            atomic_json(metadata / "protocol.json", protocol)
            jobs = schedule_jobs(root, protocol, args.limit)
            invocation = {"argv": sys.argv, "time_utc": utc_now(), "limit": args.limit,
                          "pending_jobs": len(jobs), "protocol_fingerprint": fingerprint(protocol)}
            atomic_json(metadata / "invocations" / f"{time.time_ns()}.json", invocation)
            print(f"Post: {len(jobs)} ready; prefix={args.limit}; workers=16; whole-process budget=180s", flush=True)
            if jobs:
                preload_cpu_dependencies()
                run_jobs(jobs, metadata)
    summary = summarize(root)
    print(json.dumps({"status": summary["status"], "intended_count": SAMPLE_COUNT, "counts": summary["counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
