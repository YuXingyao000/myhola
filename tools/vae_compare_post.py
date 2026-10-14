"""Isolated, fixed-denominator OCC comparison for posterior-mean VAE exports.

Run from the repository root with ``python -m tools.vae_compare_post``. This
runner does not change the existing post algorithms. A hard timeout covers the
entire per-object subprocess, including geometry evaluation. Completed stages
remain recorded if a later stage times out. Reinvocation resumes matching
terminal records; algorithm failures/timeouts are not automatically retried.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
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
from typing import Any


METHODS = ("legacy", "joint")
CANDIDATE_METHODS = ("joint", "intersection", "local", "conditioned")
BASELINE_METHODS = ("legacy", "intersection")


def comparison_methods(candidate_method="joint", baseline_method="legacy"):
    if candidate_method not in CANDIDATE_METHODS:
        raise ValueError(f"Unsupported candidate method: {candidate_method}")
    if baseline_method not in BASELINE_METHODS or baseline_method == candidate_method:
        raise ValueError(f"Unsupported baseline/candidate pair: {baseline_method}/{candidate_method}")
    return (baseline_method, candidate_method)

PROTOCOL_VERSION = "vae_compare_post_v1"
SAMPLING = {
    "version": "native_step_per_primitive_v1",
    "face_points_per_face": 1024,
    "face_distribution": "triangle-area-weighted within each face; equal count per face",
    "mesh_linear_deflection": 0.01,
    "mesh_angular_deflection": 0.1,
    "edge_points_per_edge": 128,
    "edge_distribution": "uniform native curve parameter including endpoints",
    "degenerate_edges": "repeat a real topological vertex; never use an origin fallback",
    "coordinates": "as stored; no alignment, scaling or rotation",
    "cd": "mean(pred-to-gt squared nearest distance) + mean(gt-to-pred squared nearest distance)",
    "read_precision_mode": 0,
    "seed": 42,
}
TERMINAL = {
    "ok", "missing_raw", "invalid_raw", "post_error", "no_step", "step_read_error",
    "geometry_error", "timeout", "crash", "interrupted",
}


def stable_seed(sample_id: str, stage: str) -> int:
    return int.from_bytes(hashlib.sha256(f"42:{sample_id}:{stage}".encode()).digest()[:4], "little")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def configure_cpu() -> None:
    # Set before importing numerical libraries, and before any child is forked.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def preload_cpu_dependencies() -> None:
    configure_cpu()
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    # Import once in the single-threaded parent: every job still gets its own
    # process and OCC state, without paying thousands of heavyweight imports.
    from src.brepnet.post.construct_brep import construct_brep_for_sample  # noqa: F401
    from scipy.spatial import cKDTree  # noqa: F401
    import trimesh  # noqa: F401


def squared_chamfer(predicted, target) -> dict[str, float]:
    import numpy as np
    from scipy.spatial import cKDTree
    predicted, target = np.asarray(predicted, dtype=np.float64), np.asarray(target, dtype=np.float64)
    for name, points in (("predicted", predicted), ("target", target)):
        if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
            raise ValueError(f"{name} point cloud must be finite nonempty [N,3]")
    forward = cKDTree(target).query(predicted, workers=1)[0]
    backward = cKDTree(predicted).query(target, workers=1)[0]
    accuracy = float(np.mean(forward ** 2))
    completeness = float(np.mean(backward ** 2))
    return {"accuracy": accuracy, "completeness": completeness, "cd": accuracy + completeness}


def read_step_at_precision(path: Path, mode: int = 0, precision: float = 0.1):
    """Do not let validity.py's import-time STEP settings alter native reads."""
    from OCC.Core.Interface import Interface_Static
    from OCC.Extend.DataExchange import read_step_file
    if not path.is_file():
        raise FileNotFoundError(path)
    old_mode = Interface_Static.IVal("read.precision.mode")
    old_precision = Interface_Static.RVal("read.precision.val")
    try:
        Interface_Static.SetIVal("read.precision.mode", mode)
        Interface_Static.SetRVal("read.precision.val", precision)
        shape = read_step_file(str(path), as_compound=False, verbosity=False)
        if shape is None or shape.IsNull():
            raise ValueError(f"No readable shape: {path}")
        return shape
    finally:
        Interface_Static.SetIVal("read.precision.mode", old_mode)
        Interface_Static.SetRVal("read.precision.val", old_precision)


def native_validity(shape) -> dict[str, Any]:
    """Analyze the as-read shape before any ShapeFix mutation or triangulation."""
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
    from shared.occ_utils import get_primitives
    solid_type = shape.ShapeType() == TopAbs_SOLID
    checked = bool(BRepCheck_Analyzer(shape).IsValid())
    return {
        "step_readable": True,
        "shape_type": int(shape.ShapeType()),
        "native_is_solid_type": solid_type,
        "native_brepcheck_valid": checked,
        "native_valid_solid": bool(solid_type and checked),
        "num_faces": len(get_primitives(shape, TopAbs_FACE, v_remove_half=True)),
        "num_edges": len(get_primitives(shape, TopAbs_EDGE, v_remove_half=True)),
        "num_vertices": len(get_primitives(shape, TopAbs_VERTEX, v_remove_half=True)),
    }


def compatibility_validity(path: Path) -> bool:
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.ShapeFix import ShapeFix_ShapeTolerance
    from OCC.Core.TopAbs import TopAbs_SOLID
    shape = read_step_at_precision(path, mode=1, precision=0.1)
    if shape.ShapeType() != TopAbs_SOLID:
        return False
    ShapeFix_ShapeTolerance().SetTolerance(shape, 0.1)
    return bool(BRepCheck_Analyzer(shape).IsValid())


def sample_native_geometry(shape, seed: int) -> dict[str, Any]:
    """Deterministic samples of native STEP faces/edges, including locations.

    No face is silently skipped. A failed primitive makes this object's geometry
    metric unavailable, independently of its already recorded solid validity.
    """
    import numpy as np
    import trimesh
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
    from OCC.Core.TopLoc import TopLoc_Location
    from shared.occ_utils import get_primitives
    faces = get_primitives(shape, TopAbs_FACE, v_remove_half=True)
    edges = get_primitives(shape, TopAbs_EDGE, v_remove_half=True)
    if not faces or not edges:
        raise ValueError("STEP geometry comparison requires nonempty native faces and edges")
    face_points, edge_points = [], []
    for index, face in enumerate(faces):
        BRepMesh_IncrementalMesh(face, SAMPLING["mesh_linear_deflection"], False, SAMPLING["mesh_angular_deflection"])
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, location)
        if triangulation is None or triangulation.NbTriangles() == 0:
            raise ValueError(f"Native face {index} has no triangulation")
        transform = location.Transformation()
        vertices = [triangulation.Node(i).Transformed(transform).Coord() for i in range(1, triangulation.NbNodes() + 1)]
        triangles = [triangulation.Triangle(i).Get() for i in range(1, triangulation.NbTriangles() + 1)]
        mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(triangles) - 1, process=False)
        if not np.isfinite(mesh.area) or mesh.area <= 0:
            raise ValueError(f"Native face {index} has no finite positive mesh area")
        points, _ = trimesh.sample.sample_surface(mesh, SAMPLING["face_points_per_face"], seed=(seed + index) % (2 ** 32))
        face_points.append(points)
    degenerate_count = 0
    for index, edge in enumerate(edges):
        if BRep_Tool.Degenerated(edge):
            vertices = get_primitives(edge, TopAbs_VERTEX, v_remove_half=True)
            if not vertices:
                raise ValueError(f"Degenerate native edge {index} has no real vertex")
            points = np.repeat(np.asarray(BRep_Tool.Pnt(vertices[0]).Coord())[None], SAMPLING["edge_points_per_edge"], axis=0)
            degenerate_count += 1
        else:
            curve = BRepAdaptor_Curve(edge)
            start, end = curve.FirstParameter(), curve.LastParameter()
            if not np.isfinite([start, end]).all():
                raise ValueError(f"Native edge {index} has an unbounded parameter interval")
            points = np.asarray([curve.Value(float(u)).Coord() for u in np.linspace(start, end, SAMPLING["edge_points_per_edge"])])
        edge_points.append(points)
    face_points, edge_points = np.concatenate(face_points), np.concatenate(edge_points)
    if not np.isfinite(face_points).all() or not np.isfinite(edge_points).all():
        raise ValueError("STEP sampling produced NaN/Inf")
    return {"face_points": face_points, "edge_points": edge_points,
            "num_faces": len(faces), "num_edges": len(edges), "num_degenerate_edges": degenerate_count}


def load_gt_samples(path: Path, cache_root: Path, sample_id: str) -> dict[str, Any]:
    import numpy as np
    seed = stable_seed(sample_id, "geometry")
    key = fingerprint({"step_sha256": sha256_file(path), "sampling": SAMPLING, "seed": seed})
    cache = cache_root / f"{sample_id}-{key}.npz"
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as data:
            return {name: data[name] for name in data.files}
    data = sample_native_geometry(read_step_at_precision(path), seed)
    cache_root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache_root, prefix=sample_id + ".", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **data)
    os.replace(temporary, cache)
    return data


def validate_raw_prediction(path: Path) -> dict[str, Any]:
    import numpy as np
    with np.load(path, allow_pickle=False) as data:
        face = data["pred_face"]
        edge = data["pred_edge"]
        conn = data["pred_edge_face_connectivity"]
        representation = data["representation"].item() if "representation" in data else "legacy_half_edges"
        if face.ndim != 4 or face.shape[-1] != 3 or not len(face) or min(face.shape[1:3]) < 2:
            raise ValueError("pred_face must be nonempty [F,U,V,3] with U,V>=2")
        if edge.ndim != 3 or edge.shape[-1] != 3 or edge.shape[1] < 2:
            raise ValueError("pred_edge must be [E,L,3] with L>=2")
        if not np.isfinite(face).all() or not np.isfinite(edge).all():
            raise ValueError("Raw geometry contains NaN/Inf")
        if conn.shape != (len(edge), 3) or not np.issubdtype(conn.dtype, np.integer):
            raise ValueError("Raw connectivity must be integer [E,3]")
        if not np.array_equal(np.sort(conn[:, 0]), np.arange(len(edge))):
            raise ValueError("Raw edge IDs must cover the curve array exactly once")
        if representation == "shared_boundary_v1":
            from src.brepnet.data.shared_boundaries import shared_prediction_diagnostics
            # Invalid/self face assignments remain raw diagnostics; the existing
            # explicit shared adapter handles them using predictions only.
            shared_prediction_diagnostics(len(face), edge, conn, data["pred_edge_confidence"])
        elif representation == "legacy_half_edges":
            if (conn[:, 1:] < 0).any() or (conn[:, 1:] >= len(face)).any():
                raise ValueError("Legacy raw connectivity has an invalid face ID")
        else:
            raise ValueError(f"Unknown prediction representation {representation!r}")
    return {"representation": representation, "raw_num_faces": len(face), "raw_num_edges": len(edge)}


def execute_job(job: dict, status_path: Path) -> dict:
    import numpy as np
    started = time.monotonic()
    row = {"sample_id": job["sample_id"], "method": job["method"], "mode": job["mode"],
           "protocol_fingerprint": job["protocol_fingerprint"], "raw_sha256": job["raw_sha256"],
           "status": "running", "stage": "validate_raw", "process_pid": os.getpid(),
           "seed": stable_seed(job["sample_id"], "post"), "post_status": "not_run",
           "step_exists": False, "step_readable": False, "success_marker": False,
           "native_valid_solid": False, "compat_valid_solid": False,
           "geometry_status": "not_run", "error": ""}

    def record(**values):
        row.update(values)
        row["elapsed_seconds"] = time.monotonic() - started
        atomic_json(status_path, row)

    raw = Path(job["raw_root"]) / job["sample_id"] / "data.npz"
    output = Path(job["post_root"]) / job["sample_id"]
    record()
    if not raw.is_file():
        record(status="missing_raw", error=f"Missing prediction {raw}")
        return row
    try:
        row.update(validate_raw_prediction(raw))
    except Exception as exc:
        record(status="invalid_raw", error=f"{type(exc).__name__}: {exc}")
        return row
    random.seed(row["seed"])
    np.random.seed(row["seed"])
    try:
        from src.brepnet.post.construct_brep import construct_brep_for_sample
        record(stage="post", post_status="running")
        post_started = time.monotonic()
        construct_brep_for_sample(
            job["raw_root"], job["post_root"], job["sample_id"], v_drop_num=0,
            is_ray=False, is_log=False, is_optimize_geom=job["mode"] == "optimized",
            v_max_optimize_iter=job["max_optimize_iter"], isdebug=False,
            use_cuda=False, from_scratch=True, is_save_data=False,
        )
        record(post_status="returned", post_elapsed_seconds=time.monotonic() - post_started,
               step_exists=(output / "recon_brep.step").is_file(), success_marker=(output / "success.txt").is_file())
    except Exception as exc:
        traceback.print_exc()
        record(status="post_error", post_status="error", error=f"{type(exc).__name__}: {exc}")
        return row
    step = output / "recon_brep.step"
    if not step.is_file():
        record(status="no_step", stage="done", error="Post returned without a STEP")
        return row
    try:
        record(stage="native_validity")
        shape = read_step_at_precision(step, mode=0)
        record(**native_validity(shape))
    except Exception as exc:
        record(status="step_read_error", error=f"{type(exc).__name__}: {exc}")
        return row
    try:
        record(stage="compatibility_validity")
        record(compat_valid_solid=compatibility_validity(step))
    except Exception as exc:
        record(compatibility_error=f"{type(exc).__name__}: {exc}")
    try:
        record(stage="geometry", geometry_status="running")
        geometry_started = time.monotonic()
        gt = load_gt_samples(Path(job["data_root"]) / job["sample_id"] / "normalized_shape.step",
                             Path(job["cache_root"]), job["sample_id"])
        predicted = sample_native_geometry(shape, stable_seed(job["sample_id"], "geometry"))
        values = {}
        for name in ("face", "edge"):
            metrics = squared_chamfer(predicted[name + "_points"], gt[name + "_points"])
            values.update({name + "_" + key: value for key, value in metrics.items()})
        record(**values, geometry_status="ok", geometry_elapsed_seconds=time.monotonic() - geometry_started,
               gt_num_faces=int(gt["num_faces"]), gt_num_edges=int(gt["num_edges"]),
               sampled_degenerate_edges=int(predicted["num_degenerate_edges"]),
               status="ok", stage="done")
    except Exception as exc:
        traceback.print_exc()
        record(status="geometry_error", geometry_status="error", error=f"{type(exc).__name__}: {exc}")
    return row


def job_process(job: dict, status_path: Path, log_path: Path) -> None:
    os.setsid()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as handle:
        os.dup2(handle.fileno(), 1)
        os.dup2(handle.fileno(), 2)
        try:
            execute_job(job, status_path)
        except BaseException:
            traceback.print_exc()
            raise


def terminate_process(process) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        if process.is_alive():
            process.kill()
    process.join(timeout=5)


def run_isolated_jobs(jobs: list[dict], metadata: Path, workers: int, timeout: float,
                      process_target=job_process) -> None:
    """Schedule only active processes; queued jobs never spend their timeout."""
    context = mp.get_context("fork")
    pending = iter(jobs)
    active = {}
    exhausted = False
    complete = 0
    stream_path = metadata / "events.jsonl"
    metadata.mkdir(parents=True, exist_ok=True)
    try:
        with stream_path.open("a", buffering=1) as stream:
            while active or not exhausted:
                while len(active) < workers and not exhausted:
                    try:
                        job = next(pending)
                    except StopIteration:
                        exhausted = True
                        break
                    status_path = metadata / "samples" / job["method"] / (job["sample_id"] + ".json")
                    log_path = metadata / "logs" / job["method"] / (job["sample_id"] + ".log")
                    process = context.Process(target=process_target, args=(job, status_path, log_path))
                    started = time.monotonic()
                    process.start()
                    active[process.pid] = (process, started, job, status_path)
                for pid, (process, started, job, status_path) in list(active.items()):
                    elapsed = time.monotonic() - started
                    expired = process.is_alive() and elapsed >= timeout
                    if not expired and process.is_alive():
                        continue
                    if expired:
                        terminate_process(process)
                    else:
                        process.join()
                    row = read_json(status_path)
                    row.setdefault("sample_id", job["sample_id"])
                    row.setdefault("method", job["method"])
                    row.setdefault("mode", job["mode"])
                    row.setdefault("protocol_fingerprint", job["protocol_fingerprint"])
                    row.setdefault("raw_sha256", job["raw_sha256"])
                    if expired:
                        row.update(status="timeout", error=f"Hard subprocess timeout after {timeout:g}s", timeout_seconds=timeout)
                    elif row.get("status") not in TERMINAL or process.exitcode != 0:
                        row.update(status="crash", error=f"Worker exit code {process.exitcode}; last stage {row.get('stage', 'startup')}")
                    row.update(worker_exit_code=process.exitcode, wall_seconds=elapsed)
                    atomic_json(status_path, row)
                    stream.write(json.dumps({"time_utc": datetime.now(timezone.utc).isoformat(), **row}, allow_nan=False) + "\n")
                    complete += 1
                    del active[pid]
                    if complete % 20 == 0 or complete == len(jobs):
                        print(f"Post {job['mode']}: {complete}/{len(jobs)} finished; last={job['method']}/{job['sample_id']} {row['status']}", flush=True)
                if active:
                    time.sleep(0.05)
    finally:
        for process, _, _, _ in active.values():
            terminate_process(process)


def statistics(values: list[float]) -> dict[str, Any]:
    import numpy as np
    values = [float(value) for value in values if value is not None and math.isfinite(value)]
    return {"count": len(values), "mean": float(np.mean(values)) if values else None,
            "median": float(np.median(values)) if values else None,
            "p90": float(np.quantile(values, 0.9)) if values else None}


def summarize(rows: list[dict], intended: int, original_total: int, candidate_method: str = "joint", baseline_method: str = "legacy") -> dict[str, Any]:
    methods = comparison_methods(candidate_method, baseline_method)
    summary = {"baseline_method": baseline_method, "candidate_method": candidate_method, "intended_count_per_method": intended, "original_test_count": original_total,
               "accepted_coverage_of_original": intended / original_total, "methods": {}}
    metric_names = ("face_cd", "face_accuracy", "face_completeness", "edge_cd", "edge_accuracy", "edge_completeness")
    maps = {}
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        maps[method] = {row["sample_id"]: row for row in selected}
        counts = {key: sum(bool(row.get(key)) for row in selected) for key in
                  ("step_exists", "step_readable", "success_marker", "native_valid_solid", "compat_valid_solid")}
        counts["geometry_computable"] = sum(row.get("geometry_status") == "ok" for row in selected)
        counts["terminal_records"] = sum(row.get("status") in TERMINAL for row in selected)
        summary["methods"][method] = {
            "status_counts": dict(Counter(row.get("status", "not_run") for row in selected)),
            "counts": counts,
            "rates_over_intended": {key: value / intended for key, value in counts.items()},
            "rates_over_original_test": {key: value / original_total for key, value in counts.items()},
            "geometry_all_computable": {name: statistics([row.get(name) for row in selected if row.get("geometry_status") == "ok"]) for name in metric_names},
            "latency_seconds": {name: statistics([row.get(name) for row in selected]) for name in
                                ("wall_seconds", "post_elapsed_seconds", "geometry_elapsed_seconds")},
        }
    common = sorted(set(maps[baseline_method]) & set(maps[candidate_method]))
    common = [sample_id for sample_id in common if all(maps[method][sample_id].get("geometry_status") == "ok" for method in methods)]
    summary["paired_common_geometry"] = {
        "count": len(common), "coverage_over_intended": len(common) / intended,
        "coverage_over_original_test": len(common) / original_total,
        "metrics": {name: {
            **{method: statistics([maps[method][sample_id][name] for sample_id in common]) for method in methods},
            f"{candidate_method}_minus_{baseline_method}": statistics([maps[candidate_method][sample_id][name] - maps[baseline_method][sample_id][name] for sample_id in common]),
        } for name in metric_names},
    }
    transitions = {"both_valid": [], f"{baseline_method}_failed_to_{candidate_method}_valid": [],
                   f"{baseline_method}_valid_to_{candidate_method}_failed": [], "both_failed": [], "unknown": []}
    for sample_id in sorted(set(maps[baseline_method]) | set(maps[candidate_method])):
        old, new = maps[baseline_method].get(sample_id, {}), maps[candidate_method].get(sample_id, {})
        if old.get("status") not in TERMINAL or new.get("status") not in TERMINAL:
            category = "unknown"
        elif bool(old.get("native_valid_solid")):
            category = "both_valid" if bool(new.get("native_valid_solid")) else f"{baseline_method}_valid_to_{candidate_method}_failed"
        else:
            category = f"{baseline_method}_failed_to_{candidate_method}_valid" if bool(new.get("native_valid_solid")) else "both_failed"
        transitions[category].append(sample_id)
    summary["native_validity_transitions"] = {
        "definition": "Failed means no native valid solid in a terminal record, including missing/error/timeout; pending is unknown.",
        "counts": {name: len(ids) for name, ids in transitions.items()}, "ids": transitions,
    }
    return summary


def write_summary(comparison_root: Path, mode: str, metadata: Path, ids: list[str], protocol: dict,
                  candidate_method: str = "joint", reference_rows: list[dict] | None = None,
                  baseline_method: str = "legacy") -> dict:
    rows = []
    references = {row["sample_id"]: row for row in reference_rows or []}
    for method in comparison_methods(candidate_method, baseline_method):
        for sample_id in ids:
            row = references[sample_id] if method == baseline_method and reference_rows is not None else read_json(metadata / "samples" / method / (sample_id + ".json"))
            row = dict(row)
            row.setdefault("result_origin", "current_run" if row else "not_run")
            rows.append(row if "sample_id" in row else {"sample_id": sample_id, "method": method, "mode": mode, "status": "not_run"})
    output = comparison_root / "post" / mode
    output.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with tempfile.NamedTemporaryFile(mode="w", newline="", dir=output, suffix=".csv.tmp", delete=False) as handle:
        temp = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, output / "samples.csv")
    summary = summarize(rows, len(ids), protocol["original_test_count"], candidate_method=candidate_method, baseline_method=baseline_method)
    summary.update(protocol=protocol, csv_path=str(output / "samples.csv"),
                   reference_provenance=protocol.get("reference_baseline", protocol.get("reference_legacy")),
                   latency_note="Reference baseline latency is from its historical run; reference rows were not executed in this invocation." if reference_rows is not None else "All recorded jobs belong to this comparison root.")
    atomic_json(output / "summary.json", summary)
    return summary


REFERENCE_CONTRACT_FIELDS = (
    "source", "mode", "data_root", "split_sha256", "intended_count", "original_test_count",
    "timeout_seconds", "timeout_scope", "max_optimize_iter", "adaptive_max_optimize_iter",
    "drop_num", "use_cuda", "threads_per_process", "sampling", "native_validity",
    "compatibility_validity", "post_algorithm_files",
)


def load_reference_rows(reference_root: Path, comparison_root: Path, mode: str,
                        ids: list[str], protocol: dict, baseline_method: str = "legacy") -> tuple[list[dict], dict]:
    """Validate and read historical baseline outcomes without rewriting any asset.

    Scheduler/runner hashes may differ. Every recorded post algorithm, geometry
    sampling setting and per-ID raw hash must match. Historical fingerprints and
    timing values remain unchanged in the combined output.
    """
    reference_root, comparison_root = reference_root.resolve(), comparison_root.resolve()
    if reference_root == comparison_root:
        raise ValueError("Reference root must differ from the current comparison root")
    source_path = reference_root / "metadata" / "post" / mode / "protocol.json"
    source = read_json(source_path)
    if not source:
        raise ValueError(f"Missing reference protocol: {source_path}")
    differences = [name for name in REFERENCE_CONTRACT_FIELDS
                   if name not in source or name not in protocol or source[name] != protocol[name]]
    if differences:
        raise ValueError(f"Reference post contract differs: {', '.join(differences)}")
    if not source["post_algorithm_files"]:
        raise ValueError("Reference protocol has no post algorithm hashes")
    source_key = fingerprint(source)
    source_metadata = reference_root / "metadata" / "post" / mode / "samples" / baseline_method
    if {path.stem for path in source_metadata.glob("*.json")} != set(ids):
        raise ValueError("Reference baseline record IDs differ from the intended split")
    rows, record_hashes, step_hashes = [], {}, {}
    for sample_id in ids:
        path = source_metadata / (sample_id + ".json")
        row = read_json(path)
        if row.get("sample_id") != sample_id or row.get("method") != baseline_method or row.get("mode") != mode:
            raise ValueError(f"Reference record identity differs: {sample_id}")
        if row.get("status") not in TERMINAL or row.get("protocol_fingerprint") != source_key:
            raise ValueError(f"Reference record is not a matching terminal result: {sample_id}")
        current_raw = comparison_root / "raw_predictions" / "posterior_mean" / baseline_method / sample_id / "data.npz"
        source_raw = reference_root / "raw_predictions" / "posterior_mean" / baseline_method / sample_id / "data.npz"
        if not current_raw.is_file() or not source_raw.is_file():
            raise ValueError(f"Cannot verify reference raw input: {sample_id}")
        current_hash = sha256_file(current_raw)
        if current_hash != row.get("raw_sha256") or sha256_file(source_raw) != current_hash:
            raise ValueError(f"Reference raw SHA256 differs: {sample_id}")
        step_path = reference_root / "post" / mode / baseline_method / sample_id / "recon_brep.step"
        if row.get("step_exists") and not step_path.is_file():
            raise ValueError(f"Reference record's saved STEP is missing: {sample_id}")
        if step_path.is_file():
            step_hashes[sample_id] = sha256_file(step_path)
        record_hashes[sample_id] = sha256_file(path)
        rows.append({**row, "result_origin": "reference_reused", "source_record_path": str(path),
                     "source_record_sha256": record_hashes[sample_id], "source_method": baseline_method,
                     "source_step_path": str(step_path) if step_path.is_file() else None,
                     "source_step_sha256": step_hashes.get(sample_id)})
    provenance = {
        "method": baseline_method, "result_origin": "reference_reused", "reference_root": str(reference_root),
        "source_protocol_path": str(source_path), "source_protocol_sha256": sha256_file(source_path),
        "source_protocol_fingerprint": source_key, "source_protocol": source,
        "record_count": len(rows), "record_hash_manifest_sha256": fingerprint(record_hashes),
        "step_hash_manifest_sha256": fingerprint(step_hashes), "step_hash_count": len(step_hashes),
        "step_hash_note": "STEP hashes are recorded at reuse time; no historical STEP hash was available for retrospective comparison.",
        "verified_contract_fields": list(REFERENCE_CONTRACT_FIELDS),
        "note": "Read-only historical terminal records; original fingerprints and historical latency preserved. Runner hash may differ; all post algorithm hashes must match.",
    }
    return rows, provenance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-list", type=Path, required=True)
    parser.add_argument("--mode", choices=("fit_only", "optimized"), required=True)
    parser.add_argument("--baseline-method", choices=BASELINE_METHODS, default="legacy")
    parser.add_argument("--candidate-method", choices=CANDIDATE_METHODS, default="joint")
    parser.add_argument("--reference-root", type=Path, help="Read-only compatible historical baseline results; never schedule baseline jobs")
    parser.add_argument("--methods", "--method", nargs="+", choices=("legacy", *CANDIDATE_METHODS))
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-optimize-iter", type=int, default=200)
    parser.add_argument("--original-test-count", type=int, default=2424)
    parser.add_argument("--summary-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    methods = comparison_methods(args.candidate_method, args.baseline_method)
    if args.methods is None:
        args.methods = [args.candidate_method] if args.reference_root is not None else list(methods)
    if len(args.methods) != len(set(args.methods)) or set(args.methods) - set(methods):
        raise ValueError("--methods must be unique members of --baseline-method and --candidate-method")
    if args.reference_root is not None and args.baseline_method in args.methods:
        raise ValueError("--reference-root is read-only reuse; do not also schedule baseline")
    if args.workers < 1 or not math.isfinite(args.timeout) or args.timeout <= 0 or args.max_optimize_iter < 1:
        raise ValueError("workers, timeout and max-optimize-iter must be positive")
    ids = [line.strip() for line in args.split_list.read_text().splitlines() if line.strip()]
    if not ids or len(ids) != len(set(ids)) or any(Path(item).name != item or item in {".", ".."} for item in ids):
        raise ValueError("Split must contain unique nonempty individual model IDs")
    if args.original_test_count < len(ids):
        raise ValueError("original-test-count cannot be smaller than intended split")
    args.comparison_root = args.comparison_root.resolve()
    args.data_root = args.data_root.resolve()
    metadata = args.comparison_root / "metadata" / "post" / args.mode
    metadata.mkdir(parents=True, exist_ok=True)
    protocol = {
        "version": PROTOCOL_VERSION, "source": "posterior_mean", "mode": args.mode,
        "comparison_root": str(args.comparison_root), "data_root": str(args.data_root),
        "split_sha256": sha256_file(args.split_list), "intended_count": len(ids),
        "original_test_count": args.original_test_count, "timeout_seconds": args.timeout,
        "timeout_scope": "whole subprocess from launch, including native/compat validity and geometry metrics",
        "max_optimize_iter": args.max_optimize_iter,
        "adaptive_max_optimize_iter": args.max_optimize_iter * 3 if args.mode == "optimized" else 0,
        "drop_num": 0, "use_cuda": False, "threads_per_process": 1,
        "sampling": SAMPLING, "native_validity": "read.precision.mode=0; as-read TopAbs_SOLID and BRepCheck before mutation",
        "compatibility_validity": "fresh read mode=1, precision=.1; SetTolerance(.1); TopAbs_SOLID and BRepCheck",
        "post_algorithm_files": {str(path): sha256_file(path) for path in sorted(Path("src/brepnet/post").glob("*.py"))},
        "runner_sha256": sha256_file(Path(__file__)),
    }
    protocol["candidate_method"] = args.candidate_method
    protocol["baseline_method"] = args.baseline_method
    reference_rows = None
    if args.reference_root is not None:
        reference_rows, provenance = load_reference_rows(args.reference_root, args.comparison_root, args.mode, ids, protocol, baseline_method=args.baseline_method)
        protocol["reference_legacy" if args.baseline_method == "legacy" else "reference_baseline"] = provenance
    digest = fingerprint(protocol)
    with (metadata / "protocol.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        existing = read_json(metadata / "protocol.json")
        if existing and existing != protocol:
            raise ValueError("Post protocol/config/code fingerprint changed; choose a new comparison root or restore the original contract")
        atomic_json(metadata / "protocol.json", protocol)
    locks = []
    try:
        for method in sorted(set(args.methods)):
            lock = (metadata / f"{method}.lock").open("a")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(lock)
        if args.summary_only:
            write_summary(args.comparison_root, args.mode, metadata, ids, protocol,
                          candidate_method=args.candidate_method, reference_rows=reference_rows, baseline_method=args.baseline_method)
            return 0
        jobs = []
        for sample_id in ids:
            for method in args.methods:
                raw_root = args.comparison_root / "raw_predictions" / "posterior_mean" / method
                raw = raw_root / sample_id / "data.npz"
                raw_sha = sha256_file(raw) if raw.is_file() else None
                status_path = metadata / "samples" / method / (sample_id + ".json")
                old = read_json(status_path)
                if old:
                    if old.get("protocol_fingerprint") != digest or old.get("raw_sha256") != raw_sha:
                        # A previous missing input has not consumed a post attempt.
                        if not (old.get("status") == "missing_raw" and old.get("raw_sha256") is None and raw_sha is not None and old.get("protocol_fingerprint") == digest):
                            raise ValueError(f"Input or protocol changed for {method}/{sample_id}; refusing to mix results")
                    elif old.get("status") in TERMINAL:
                        continue
                    else:
                        old.update(status="interrupted", error="Previous invocation ended before this worker recorded a terminal result; no automatic retry")
                        atomic_json(status_path, old)
                        continue
                jobs.append({"sample_id": sample_id, "method": method, "mode": args.mode,
                             "raw_root": str(raw_root), "raw_sha256": raw_sha,
                             "post_root": str(args.comparison_root / "post" / args.mode / method),
                             "data_root": str(args.data_root), "cache_root": str(args.comparison_root / "metadata" / "gt_step_samples"),
                             "max_optimize_iter": args.max_optimize_iter, "protocol_fingerprint": digest})
        print(f"{args.mode}: {len(jobs)} jobs pending; {len(ids)} intended IDs per method; workers={args.workers}; timeout={args.timeout:g}s", flush=True)
        atomic_json(metadata / "invocation.json", {"argv": sys.argv, "methods": args.methods, "workers": args.workers,
                                                   "pending_jobs": len(jobs), "time_utc": datetime.now(timezone.utc).isoformat()})
        if jobs:
            preload_cpu_dependencies()
            run_isolated_jobs(jobs, metadata, args.workers, args.timeout)
        summary = write_summary(args.comparison_root, args.mode, metadata, ids, protocol,
                          candidate_method=args.candidate_method, reference_rows=reference_rows, baseline_method=args.baseline_method)
        print(json.dumps({method: value["counts"] for method, value in summary["methods"].items()}, indent=2), flush=True)
        return 0
    finally:
        for lock in locks:
            lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
