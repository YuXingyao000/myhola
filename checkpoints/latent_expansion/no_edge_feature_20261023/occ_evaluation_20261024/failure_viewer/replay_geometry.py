"""Capture a checked diagnostic replay of the frozen final recovery path.

Only diagnostic outputs are written. Production code, historic outputs and
validity decisions are unchanged; the solid-building stage is deliberately
skipped. See SPEC.md and SCHEMA.md beside this file.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import sys
import time
import traceback

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

HERE = Path(__file__).resolve().parent
EVALUATION = HERE.parent
REPO = EVALUATION.parents[3]
sys.path.insert(0, str(REPO))

PATH_LABEL = "fit_score_v2_roundtrip_optimized"
FACE_GRID = 24
EDGE_POINTS = 64
CHECK_FIELDS = ("faces", "selected_edge_count", "periodic_edge_replacements")
PROTOCOL = {
    "version": "failure_viewer_replay_v1",
    "path": PATH_LABEL,
    "optimized": True,
    "fitting_policy": "fit_score_v2",
    "max_optimize_iterations": 200,
    "existing_adaptive_max_iterations": 600,
    "cpu_threads": 1,
    "face_display_grid": [FACE_GRID, FACE_GRID],
    "edge_sampling": "64 inclusive uniform native parameter points of post-periodic-replacement topology edges",
    "owner_distance": "OCC BRepExtrema_DistShapeShape(point vertex, untrimmed natural bounded analytic owner face)",
    "units": "original normalized XYZ length; no alignment or scaling",
    "solid_evaluation": "skipped; historic validity retained",
    "endpoint_proxy": "same-owner nearest endpoint on a different edge; geometric closure uses fitted topology IsClosed or endpoint distance <= 1e-7",
    "closure_absolute_epsilon": 1e-7,
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_gzip(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    with temporary.open("wb") as stream:
        with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as compressed:
            compressed.write(encoded)
    os.replace(temporary, path)


def finite(value) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def summarize(values) -> dict:
    all_values = list(values)
    present = np.asarray([value for value in all_values if value is not None and np.isfinite(value)], dtype=np.float64)
    return {"count": len(present), "unknown_count": len(all_values) - len(present),
            "mean": float(present.mean()) if len(present) else None,
            "median": float(np.median(present)) if len(present) else None,
            "p95": float(np.quantile(present, .95)) if len(present) else None,
            "max": float(present.max()) if len(present) else None}


def source_paths(sid: str) -> tuple[Path, Path]:
    return (EVALUATION / "raw_predictions/posterior_mean/no_edge_feature64" / sid / "data.npz",
            EVALUATION / "post/optimized/no_edge_feature64" / sid / "post_diagnostics.json")


def failed_ids() -> list[str]:
    with (EVALUATION / "post/optimized/samples.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = [row["sample_id"] for row in rows
              if row["method"] == "no_edge_feature64" and row["native_valid_solid"].lower() != "true"]
    if len(result) != 316 or len(set(result)) != 316:
        raise ValueError("Expected the fixed 316 distinct historical failures")
    return result


def code_hashes() -> dict[str, str]:
    current = {str(path.relative_to(REPO)): sha(path)
               for path in sorted((REPO / "src/brepnet/post").glob("*.py"))}
    expected = json.loads((EVALUATION / "validation/post_preflight.json").read_text())["post_algorithm_files"]
    if current != expected:
        raise ValueError("Frozen production post code differs")
    return current


def load_dependencies() -> None:
    import torch
    torch.set_num_threads(1)
    from src.brepnet.post import construct_brep  # noqa: F401
    from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape  # noqa: F401


def projection_sanity() -> dict:
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeVertex
    from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
    from OCC.Core.Geom import Geom_Plane
    from OCC.Core.gp import gp_Dir, gp_Pln, gp_Pnt
    plane = Geom_Plane(gp_Pln(gp_Pnt(0., 0., 0.), gp_Dir(0., 0., 1.)))
    face = BRepBuilderAPI_MakeFace(plane, 0., 1., 0., 1., 1e-6).Face()
    rows = []
    for xyz, expected, expected_closest in (((.25, .75, 2.), 2., (.25, .75, 0.)),
                                            ((2., .5, 1.), np.sqrt(2), (1., .5, 0.))):
        query = BRepExtrema_DistShapeShape(BRepBuilderAPI_MakeVertex(gp_Pnt(*xyz)).Vertex(), face)
        if not query.IsDone() or not query.NbSolution():
            raise AssertionError("Analytic bounded-face sanity projection failed")
        closest = query.PointOnShape2(1).Coord()
        actual = query.Value()
        if abs(actual - expected) > 1e-9 or not np.allclose(closest, expected_closest, atol=1e-9, rtol=0):
            raise AssertionError("Analytic bounded-face projection gave incorrect distance")
        rows.append({"point": xyz, "closest": closest, "distance": actual, "expected": float(expected)})
    return {"passed": True, "examples": rows}


def sample_geometry(capture: dict) -> dict:
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeVertex
    from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
    from OCC.Core.gp import gp_Pnt

    surfaces, curves = capture["surfaces"], capture["curves"]
    faces, edges, edge_closed, bounds = [], [], [], []
    natural_faces = []
    for surface in surfaces:
        domain = tuple(map(float, surface.Bounds()))
        if not np.isfinite(domain).all() or not domain[0] < domain[1] or not domain[2] < domain[3]:
            raise ValueError(f"Invalid finite surface parameter bounds: {domain}")
        umin, umax, vmin, vmax = domain
        points = np.asarray([[surface.Value(float(u), float(v)).Coord()
                              for v in np.linspace(vmin, vmax, FACE_GRID)]
                             for u in np.linspace(umin, umax, FACE_GRID)], dtype=np.float64)
        if not np.isfinite(points).all():
            raise ValueError("Nonfinite fitted surface display samples")
        faces.append(points.tolist())
        natural_faces.append(BRepBuilderAPI_MakeFace(surface, umin, umax, vmin, vmax, 1e-6).Face())
        bounds.append(list(domain))
    for curve in curves:
        adaptor = BRepAdaptor_Curve(curve)
        first, last = float(adaptor.FirstParameter()), float(adaptor.LastParameter())
        if not np.isfinite([first, last]).all():
            raise ValueError("Nonfinite fitted edge parameter limits")
        points = np.asarray([adaptor.Value(float(t)).Coord() for t in np.linspace(first, last, EDGE_POINTS)], dtype=np.float64)
        if not np.isfinite(points).all():
            raise ValueError("Nonfinite fitted curve samples")
        edges.append(points.tolist())
        edge_closed.append(bool(adaptor.IsClosed()) or float(np.linalg.norm(points[0] - points[-1])) <= 1e-7)

    connectivity = capture["connectivity"].tolist()
    owner_records, failures = [], []
    for edge, first, second in connectivity:
        for owner in (first, second):
            distances, closest = [], []
            for point_index, point in enumerate(edges[edge]):
                try:
                    vertex = BRepBuilderAPI_MakeVertex(gp_Pnt(*point)).Vertex()
                    query = BRepExtrema_DistShapeShape(vertex, natural_faces[owner])
                    if not query.IsDone() or query.NbSolution() < 1:
                        raise ValueError("OCC distance solver returned no solution")
                    distance = finite(query.Value())
                    coordinate = np.asarray(query.PointOnShape2(1).Coord(), dtype=np.float64)
                    if distance is None or not np.isfinite(coordinate).all():
                        raise ValueError("OCC distance solver returned nonfinite values")
                    connector_length = float(np.linalg.norm(np.asarray(point) - coordinate))
                    if abs(connector_length - distance) > 1e-7:
                        raise ValueError("Distance/closest-point connector length mismatch")
                    distances.append(distance)
                    closest.append(coordinate.tolist())
                except Exception as exc:
                    distances.append(None)
                    closest.append(None)
                    failures.append({"edge": edge, "owner": owner, "point": point_index,
                                     "error": f"{type(exc).__name__}: {exc}"})
            valid = [value for value in distances if value is not None]
            # Partial projection measurements remain explicit; summary values
            # are not silently treated as a complete 64-point measurement.
            complete = len(valid) == EDGE_POINTS
            owner_records.append({"edge": edge, "owner": owner,
                "mean": float(np.mean(valid)) if complete else None,
                "p95": float(np.quantile(valid, .95)) if complete else None,
                "max": float(np.max(valid)) if complete else None,
                "distances": distances, "closest": closest,
                "valid_points": len(valid), "unknown_points": EDGE_POINTS - len(valid),
                "method": "bounded_analytic_face_brepextrema"})

    incident = [[] for _ in faces]
    for edge, first, second in connectivity:
        incident[first].append(edge)
        incident[second].append(edge)
    endpoint_records = []
    for owner, edge_ids in enumerate(incident):
        for edge in edge_ids:
            for endpoint in (0, -1):
                if edge_closed[edge]:
                    neighbor_edge, neighbor_endpoint = edge, -1 if endpoint == 0 else 0
                    distance = float(np.linalg.norm(np.asarray(edges[edge][0]) - np.asarray(edges[edge][-1])))
                    basis = "geometric_closed_curve_self_seam; topology_IsClosed_or_endpoint_gap_le_1e-7"
                else:
                    candidates = [(float(np.linalg.norm(np.asarray(edges[edge][endpoint]) - np.asarray(edges[other][tip]))), other, tip)
                                  for other in edge_ids if other != edge for tip in (0, -1)]
                    if candidates:
                        distance, neighbor_edge, neighbor_endpoint = min(candidates)
                        basis = "same_owner_nearest_endpoint_on_different_edge; not_global_wire_matching"
                    else:
                        distance, neighbor_edge, neighbor_endpoint = None, None, None
                        basis = "unknown_open_curve_without_other_edge_on_owner"
                endpoint_records.append({"owner": owner, "edge": edge, "endpoint": endpoint,
                    "neighbor_edge": neighbor_edge, "neighbor_endpoint": neighbor_endpoint,
                    "distance": distance, "basis": basis})
    return {"label": "拟合后：日志核验的诊断重放", "faces": faces, "edges": edges,
            "connectivity": connectivity, "owner_records": owner_records,
            "endpoint_records": endpoint_records, "face_parameter_bounds": bounds,
            "projection_method": PROTOCOL["owner_distance"], "projection_errors": failures,
            "optimized_face_points": capture["optimized_faces"].tolist(),
            "optimized_edge_points": capture["optimized_edges"].tolist(),
            "periodic_edge_replacements": capture["periodic_edge_replacements"]}


def replay_one(sid: str) -> dict:
    from src.brepnet.post import construct_brep as production
    from tools.vae_compare_post import stable_seed
    started = time.monotonic()
    raw, diagnostics = source_paths(sid)
    payload = {"id": sid, "status": "running", "matches_history": False,
               "mismatch_fields": [], "note": "Independent diagnostic replay; solid construction skipped; historic validity unchanged.",
               "sources": {"raw": {"path": str(raw), "sha256": sha(raw)},
                           "diagnostics": {"path": str(diagnostics), "sha256": sha(diagnostics)}},
               "protocol": PROTOCOL}
    capture = {}
    original_fit, original_solid = production._fit_occ_geometry, production._try_save_complete_solid
    output = HERE / "replay" / "intermediate" / sid

    def capture_fit(shape, fitting_policy="legacy"):
        result = original_fit(shape, fitting_policy=fitting_policy)
        capture.update(surfaces=result[0], curves=result[3],
                       connectivity=shape.edge_face_connectivity.copy(),
                       optimized_faces=shape.recon_face_points.copy(),
                       optimized_edges=shape.recon_edge_points.copy(),
                       periodic_edge_replacements=list(map(int, shape.replace_edge_idx)))
        return result

    def skip_solid(trimmed_faces, output, report, flush, deadline):
        report["solid_evaluation_skipped"] = True
        return False

    try:
        historic = json.loads(diagnostics.read_text())
        history = historic["paths"][-1]
        if history["label"] != PATH_LABEL:
            raise ValueError("Historical final path differs")
        random.seed(stable_seed(sid, "post"))
        np.random.seed(stable_seed(sid, "post"))
        production._fit_occ_geometry = capture_fit
        production._try_save_complete_solid = skip_solid
        report = {"label": PATH_LABEL, "implementation": "complete", "optimized": True,
                  "fitting_policy": "fit_score_v2", "native_valid_solid": False, "boundary_complete": None}
        production._run_complete_post_path(raw.parent.parent, output, sid, optimized=True,
            fitting_policy="fit_score_v2", max_iterations=200, use_cuda=False,
            report=report, flush=lambda: None, deadline=None)
        payload["report"] = report
        payload["trim_replay_seconds"] = time.monotonic() - started
        payload["mismatch_fields"] = [key for key in CHECK_FIELDS if report.get(key) != history.get(key)]
        payload["matches_history"] = not payload["mismatch_fields"]
        payload["view"] = sample_geometry(capture)
        records = payload["view"]["owner_records"]
        means, p95s, maxima = ([record[key] for record in records] for key in ("mean", "p95", "max"))
        all_projected = bool(records) and all(value is not None for value in means)
        payload["metrics"] = {
            "owner_mean": float(np.mean(means)) if all_projected else None,
            "owner_p95_max": max(p95s) if all_projected else None,
            "owner_max": max(maxima) if all_projected else None,
            "endpoint_mean": summarize(record["distance"] for record in payload["view"]["endpoint_records"])["mean"],
            "endpoint_max": summarize(record["distance"] for record in payload["view"]["endpoint_records"])["max"],
            "owner_relation_count": len(records),
            "unknown_projection_points": sum(record["unknown_points"] for record in records),
            "unknown_endpoint_count": sum(record["distance"] is None for record in payload["view"]["endpoint_records"]),
        }
        payload["status"] = "ok" if payload["matches_history"] else "history_mismatch"
        if sha(raw) != payload["sources"]["raw"]["sha256"] or sha(diagnostics) != payload["sources"]["diagnostics"]["sha256"]:
            raise ValueError("Historic inputs changed during replay")
    except BaseException as exc:
        payload["status"] = "error"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["traceback"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        production._fit_occ_geometry, production._try_save_complete_solid = original_fit, original_solid
        payload["elapsed_seconds"] = time.monotonic() - started
        atomic_gzip(HERE / "replay" / f"{sid}.json.gz", payload)
    return {"id": sid, "status": payload["status"], "matches_history": payload["matches_history"],
            "mismatch_fields": payload["mismatch_fields"], "error": payload.get("error"),
            "metrics": payload.get("metrics"), "elapsed_seconds": payload["elapsed_seconds"]}


def logged_replay(sid: str) -> dict:
    log = HERE / "replay" / "logs" / f"{sid}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab", buffering=0) as stream:
        os.dup2(stream.fileno(), 1)
        os.dup2(stream.fileno(), 2)
        return replay_one(sid)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        raise ValueError("Workers must be 1 through 16")
    started = time.monotonic()
    frozen = code_hashes()
    ids = failed_ids()
    if args.ids:
        if not set(args.ids) <= set(ids):
            raise ValueError("Only fixed validation failures may be replayed")
        ids = args.ids
    load_dependencies()
    sanity = projection_sanity()
    pending, results = [], []
    for sid in ids:
        existing = HERE / "replay" / f"{sid}.json.gz"
        if args.resume and existing.is_file():
            with gzip.open(existing, "rt") as stream:
                record = json.load(stream)
            raw, diag = source_paths(sid)
            if (record.get("protocol") != PROTOCOL or record["sources"]["raw"]["sha256"] != sha(raw)
                    or record["sources"]["diagnostics"]["sha256"] != sha(diag)):
                raise ValueError(f"Existing replay identity differs: {sid}")
            results.append({key: record.get(key) for key in
                            ("id", "status", "matches_history", "mismatch_fields", "error", "metrics", "elapsed_seconds")})
        else:
            pending.append(sid)
    print(f"Replay intended={len(ids)} cached={len(results)} pending={len(pending)} workers={args.workers}", flush=True)
    with mp.get_context("fork").Pool(args.workers, maxtasksperchild=1) as pool:
        for result in pool.imap_unordered(logged_replay, pending):
            results.append(result)
            print(f"Replay {len(results)}/{len(ids)} {result['id']} {result['status']} {result['elapsed_seconds']:.2f}s", flush=True)
    results.sort(key=lambda row: row["id"])
    unchanged = code_hashes() == frozen
    summary = {"version": PROTOCOL["version"], "completed_utc": datetime.now(timezone.utc).isoformat(),
        "intended_count": len(ids), "ids": ids, "results": results, "protocol": PROTOCOL,
        "projection_sanity": sanity, "production_code_sha256": frozen, "production_code_unchanged": unchanged,
        "script_sha256": sha(Path(__file__)), "spec_sha256": sha(HERE / "SPEC.md"),
        "schema_sha256": sha(HERE / "SCHEMA.md"), "statuses": dict(Counter(row["status"] for row in results)),
        "history_match_count": sum(row["matches_history"] for row in results),
        "elapsed_seconds": time.monotonic() - started, "workers": args.workers,
        "metrics": {key: summarize((row.get("metrics") or {}).get(key) for row in results)
                    for key in ("owner_mean", "owner_p95_max", "owner_max", "endpoint_mean", "endpoint_max")},
        "unknown_projection_points": sum((row.get("metrics") or {}).get("unknown_projection_points", 0) for row in results),
        "unknown_endpoint_count": sum((row.get("metrics") or {}).get("unknown_endpoint_count", 0) for row in results),
    }
    filename = "replay_summary.json" if len(ids) == 316 else "replay/smoke_summary.json"
    atomic_json(HERE / filename, summary)
    print(json.dumps({key: summary[key] for key in ("intended_count", "statuses", "history_match_count", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
