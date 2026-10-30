"""Extract traceable UV supervision without changing existing geometry or splits.

Run with explicit --data-root, --output-root, and one or more --split lists.
The default is serial, foreground execution; existing matching outputs resume.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any

import numpy as np

from src.preprocess.edge_uv import (
    SCHEMA_VERSION, SAMPLE_COUNT, UV_DOMAIN_TOLERANCE, atomic_save_npz,
    empty_edge_uv_sidecar, load_edge_uv_sidecar, read_source_geometry,
    sha256_file, validate_edge_uv_sidecar,
)

GENERATOR_VERSION = "edge_uv_v1"


def match_face_grids(source: np.ndarray, step: np.ndarray, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    """Match all ordered 256 XYZ samples, never STEP traversal indices alone."""
    if source.shape != step.shape or source.ndim != 4 or source.shape[1:] != (16, 16, 3):
        raise ValueError(f"face_grid_shape_mismatch: source={source.shape}, step={step.shape}")
    error = np.empty((len(source), len(step)), dtype=np.float64)
    for index, points in enumerate(source):
        error[index] = np.linalg.norm(step - points, axis=-1).max(axis=(1, 2))
    candidates = error <= tolerance
    if not (candidates.sum(axis=0) == 1).all() or not (candidates.sum(axis=1) == 1).all():
        raise ValueError("face_match_not_unique: full ordered grids require a unique bijection")
    indices = candidates.argmax(axis=1).astype(np.int64)
    return indices, error[np.arange(len(source)), indices]


def _points(values: list[Any]) -> np.ndarray:
    return np.asarray([value.Coord() for value in values], dtype=np.float64)


def _shapes(shape: Any, kind: Any, cast: Any) -> list[Any]:
    from OCC.Core.TopExp import TopExp_Explorer

    explorer = TopExp_Explorer(shape, kind)
    result = []
    while explorer.More():
        item = cast(explorer.Current())
        # Face occurrences may repeat in assemblies. Keep location and orientation.
        if not any(item.IsEqual(existing) for existing in result):
            result.append(item)
        explorer.Next()
    return result


def _chart(surface: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bounds = np.array([
        surface.FirstUParameter(), surface.LastUParameter(),
        surface.FirstVParameter(), surface.LastVParameter(),
    ], dtype=np.float64)
    if not np.isfinite(bounds).all() or (bounds[[1, 3]] <= bounds[[0, 2]]).any():
        raise ValueError("invalid_surface_parameter_bounds")
    periods = np.array([
        surface.UPeriod() if surface.IsUPeriodic() else 0.0,
        surface.VPeriod() if surface.IsVPeriodic() else 0.0,
    ], dtype=np.float64)
    grid = _points([
        surface.Value(float(u), float(v))
        for v in np.linspace(bounds[2], bounds[3], SAMPLE_COUNT)
        for u in np.linspace(bounds[0], bounds[1], SAMPLE_COUNT)
    ]).reshape(SAMPLE_COUNT, SAMPLE_COUNT, 3)
    return bounds, periods, grid


def continuous_chart_path(
    uv: np.ndarray, bounds: np.ndarray, periods: np.ndarray, *, native: bool,
) -> np.ndarray:
    """Keep a single continuous chart branch; never wrap points independently.

    Native pcurves already name the coedge branch. Projection first unwraps the
    complete ordered path and requires a unique constant integer-period shift.
    A seam-constant projected path with two legal chart branches is rejected.
    """
    result = np.asarray(uv, dtype=np.float64).copy()
    low, high = bounds[[0, 2]], bounds[[1, 3]]
    span = high - low
    if result.shape != (SAMPLE_COUNT, 2) or not np.isfinite(result).all():
        raise ValueError("nonfinite_or_malformed_uv_path")
    for axis in range(2):
        tolerance = UV_DOMAIN_TOLERANCE * span[axis]
        values = result[:, axis]
        period = periods[axis]
        if period > 0:
            if not native:
                delta = (np.diff(values) + period / 2) % period - period / 2
                if np.any(np.abs(np.abs(delta) - period / 2) < 1e-7 * period):
                    raise ValueError("periodic_half_period_direction_ambiguous")
                values = np.concatenate([values[:1], values[0] + np.cumsum(delta)])
            if native and (values >= low[axis] - tolerance).all() and (values <= high[axis] + tolerance).all():
                result[:, axis] = values
                continue
            first = math.ceil((low[axis] - tolerance - float(values.min())) / period)
            last = math.floor((high[axis] + tolerance - float(values.max())) / period)
            if first != last:
                raise ValueError("periodic_chart_branch_ambiguous_or_outside")
            values = values + first * period
        if (values < low[axis] - tolerance).any() or (values > high[axis] + tolerance).any():
            raise ValueError("uv_outside_source_chart")
        # Only remove numerical overshoot at the original chart boundary.
        result[:, axis] = np.clip(values, low[axis], high[axis])
    return result


def _residual(surface: Any, xyz: np.ndarray, uv: np.ndarray) -> np.ndarray:
    back = _points([surface.Value(float(u), float(v)) for u, v in uv])
    return np.linalg.norm(back - xyz, axis=1)


def _native_paths(
    xyz: np.ndarray, faces: list[Any], face_edges: list[list[Any]], surfaces: list[Any],
    bounds: np.ndarray, periods: np.ndarray, pair: tuple[int, int], tolerance: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Curve2d

    candidates = []
    i, j = pair
    for edge in face_edges[i]:
        peers = [other for other in face_edges[j] if edge.IsSame(other)]
        if not peers:
            continue
        try:
            curve = BRepAdaptor_Curve(edge)
            times = np.linspace(curve.FirstParameter(), curve.LastParameter(), SAMPLE_COUNT)
            sampled = _points([curve.Value(float(t)) for t in times])
            orientations = []
            if np.linalg.norm(sampled - xyz, axis=1).max() <= tolerance:
                orientations.append(times)
            if np.linalg.norm(sampled[::-1] - xyz, axis=1).max() <= tolerance:
                orientations.append(times[::-1])
            for ordered_times, peer in itertools.product(orientations, peers):
                paths, errors = [], []
                for face_index, owner_edge in ((i, edge), (j, peer)):
                    pcurve = BRepAdaptor_Curve2d(owner_edge, faces[face_index])
                    uv = _points([pcurve.Value(float(t)) for t in ordered_times])
                    uv = continuous_chart_path(uv, bounds[face_index], periods[face_index], native=True)
                    paths.append(uv)
                    errors.append(_residual(surfaces[face_index], xyz, uv))
                candidate = (np.stack(paths), np.stack(errors))
                if candidate[1].max() <= tolerance and not any(
                    np.allclose(candidate[0], old[0], atol=1e-9, rtol=0) for old in candidates
                ):
                    candidates.append(candidate)
        except (RuntimeError, ValueError):
            # An unavailable pcurve is explicitly reported by the projection method.
            continue
    return candidates


def _project_path(xyz: np.ndarray, face: Any, surface: Any, bounds: np.ndarray, periods: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
    from OCC.Core.gp import gp_Pnt, gp_Vec

    geometry = BRep_Tool.Surface(face)  # This overload applies the face location.
    # The NPZ stores float32 XYZ. An exact boundary sample can therefore project
    # a few ulps outside the face trim. Search a small chart margin, then require
    # and clamp only the documented UV_DOMAIN_TOLERANCE in the original chart.
    margin = UV_DOMAIN_TOLERANCE * (bounds[[1, 3]] - bounds[[0, 2]])
    search_bounds = bounds + np.array([-margin[0], margin[0], -margin[1], margin[1]])
    intrinsic = np.asarray(geometry.Bounds(), dtype=np.float64)
    search_bounds[[0, 2]] = np.maximum(search_bounds[[0, 2]], intrinsic[[0, 2]])
    search_bounds[[1, 3]] = np.minimum(search_bounds[[1, 3]], intrinsic[[1, 3]])
    path = []
    for point in xyz:
        projector = GeomAPI_ProjectPointOnSurf(gp_Pnt(*map(float, point)), geometry, *map(float, search_bounds), 1e-10)
        count = projector.NbPoints()
        if count == 0:
            raise ValueError("projection_has_no_solution")
        distances = np.array([projector.Distance(index) for index in range(1, count + 1)])
        closest = float(distances.min())
        solutions = [np.array(projector.Parameters(int(index) + 1)) for index in np.flatnonzero(distances <= closest + 1e-8)]
        first = solutions[0]
        for other in solutions[1:]:
            difference = other - first
            for axis in range(2):
                if periods[axis] > 0:
                    difference[axis] = (difference[axis] + periods[axis] / 2) % periods[axis] - periods[axis] / 2
            if np.max(np.abs(difference) / (bounds[[1, 3]] - bounds[[0, 2]])) > 1e-7:
                raise ValueError("projection_multiple_nonperiodic_solutions")
        # At a singularity XYZ does not identify the parameter branch.
        position, tangent_u, tangent_v = gp_Pnt(), gp_Vec(), gp_Vec()
        surface.D1(float(first[0]), float(first[1]), position, tangent_u, tangent_v)
        norm_u, norm_v = tangent_u.Magnitude(), tangent_v.Magnitude()
        if norm_u < 1e-12 or norm_v < 1e-12 or tangent_u.Crossed(tangent_v).Magnitude() < 1e-10 * norm_u * norm_v:
            raise ValueError("projection_singular_chart")
        path.append(first)
    uv = continuous_chart_path(np.stack(path), bounds, periods, native=False)
    return uv, _residual(surface, xyz, uv)


def extract_edge_uv(
    source_path: str | Path, step_path: str | Path, *, match_tolerance: float = 1e-5,
    projection_tolerance: float = 1e-3,
) -> dict[str, np.ndarray]:
    """Source file/read errors raise; confirmed-source UV failures yield invalid labels."""
    source_path, step_path = Path(source_path), Path(step_path)
    for path in (source_path, step_path):
        if not path.is_file():
            raise FileNotFoundError(f"required source file is missing: {path}")
    source = read_source_geometry(source_path)
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
    from OCC.Core.TopoDS import topods
    from OCC.Extend.DataExchange import read_step_file

    try:
        shape = read_step_file(str(step_path), verbosity=False)
        if shape.IsNull():
            raise ValueError("STEP reader returned a null shape")
    except Exception as error:
        raise ValueError(f"{step_path}: unreadable STEP source: {type(error).__name__}: {error}") from error
    rows = source["edge_face_connectivity"]
    source_faces = source["sample_points_faces"][..., :3]
    parameters = {"match_tolerance": match_tolerance, "projection_tolerance": projection_tolerance}
    labels = empty_edge_uv_sidecar(
        rows, len(source_faces), source_sha256=sha256_file(source_path),
        step_sha256=sha256_file(step_path), parameters=parameters,
    )
    try:
        step_faces = _shapes(shape, TopAbs_FACE, topods.Face)
        step_surfaces = [BRepAdaptor_Surface(face) for face in step_faces]
        charts = [_chart(surface) for surface in step_surfaces]
        indices, errors = match_face_grids(source_faces, np.stack([chart[2] for chart in charts]), match_tolerance)
        faces = [step_faces[index] for index in indices]
        surfaces = [step_surfaces[index] for index in indices]
        bounds = np.stack([charts[index][0] for index in indices])
        periods = np.stack([charts[index][1] for index in indices])
        labels.update(face_uv_bounds=bounds, face_step_indices=indices, face_match_error=errors, face_periods=periods)
        face_edges = [_shapes(face, TopAbs_EDGE, topods.Edge) for face in faces]
        for row_index, (edge_id, first, second) in enumerate(rows):
            xyz = source["sample_points_lines"][edge_id, :, :3].astype(np.float64)
            if np.linalg.norm(xyz - xyz[0], axis=1).max() <= match_tolerance:
                labels["side_reason"][row_index] = "degenerate_source_edge"
                continue
            pair = (int(first), int(second))
            natives = _native_paths(xyz, faces, face_edges, surfaces, bounds, periods, pair, match_tolerance)
            if len(natives) > 1:
                labels["side_reason"][row_index] = "multiple_native_pcurve_paths"
                continue
            for side, face_index in enumerate(pair):
                try:
                    if natives:
                        uv, residual = natives[0][0][side], natives[0][1][side]
                        method = 1
                    else:
                        uv, residual = _project_path(xyz, faces[face_index], surfaces[face_index], bounds[face_index], periods[face_index])
                        method = 2
                    normal_uv = (uv - bounds[face_index, [0, 2]]) / (bounds[face_index, [1, 3]] - bounds[face_index, [0, 2]])
                    valid = np.isfinite(residual) & (residual <= projection_tolerance)
                    labels["edge_uv"][row_index, side] = normal_uv.astype(np.float32)
                    labels["edge_uv_residual"][row_index, side] = residual.astype(np.float32)
                    labels["edge_uv_valid"][row_index, side] = valid
                    labels["edge_uv_method"][row_index, side] = np.where(valid, method, 0)
                    labels["side_reason"][row_index, side] = (
                        "native_pcurve" if method == 1 else "projected_native_path_unavailable"
                    ) if valid.all() else "projection_residual_exceeded"
                except (RuntimeError, ValueError) as error:
                    labels["side_reason"][row_index, side] = str(error)[:256]
    except Exception as error:
        # Both sources were read successfully. UV labels remain auxiliary when
        # chart correspondence or a subsequent OCCT operation cannot be confirmed.
        reason = f"{type(error).__name__}: {error}"
        labels = empty_edge_uv_sidecar(
            rows, len(source_faces), source_sha256=str(labels["source_sha256"].item()),
            step_sha256=str(labels["step_sha256"].item()), parameters=parameters, reason=reason,
        )
    valid = labels["edge_uv_valid"]
    labels["status"] = np.array("ok" if valid.size and valid.all() else ("partial" if valid.any() else "invalid"))
    labels["reason"] = np.array("; ".join(sorted(set(labels["side_reason"].reshape(-1).tolist()))) if valid.size else "empty_connectivity")
    return validate_edge_uv_sidecar(
        labels, connectivity=rows, num_faces=len(source_faces), source_sha256=str(labels["source_sha256"].item())
    )


def _atomic_text(path: Path, text: str) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_lists(paths: list[Path]) -> tuple[list[str], list[dict[str, Any]], dict[str, list[str]]]:
    model_ids, records, memberships = [], [], {}
    for path in paths:
        seen = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            model_id = line.strip()
            if not model_id:
                continue
            if model_id in {".", ".."} or "/" in model_id or "\\" in model_id or any(c.isspace() for c in model_id):
                raise ValueError(f"unsafe model ID in {path}: {model_id!r}")
            if model_id in seen:
                raise ValueError(f"duplicate model ID in {path}: {model_id}")
            seen.add(model_id)
            if model_id not in memberships:
                model_ids.append(model_id)
                memberships[model_id] = []
            memberships[model_id].append(str(path))
        if not seen:
            raise ValueError(f"empty split list: {path}")
        records.append({"path": str(path), "sha256": sha256_file(path), "count": len(seen)})
    return model_ids, records, memberships


def _under(root: Path, model_id: str, name: str) -> Path:
    path = root / model_id / name
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"path escapes explicit root: {path}")
    return path


def _label_record(model_id: str, labels: dict[str, np.ndarray], memberships: list[str]) -> dict[str, Any]:
    valid = labels["edge_uv_valid"]
    residual = labels["edge_uv_residual"][valid]
    face_errors = labels["face_match_error"]
    matched = face_errors[face_errors >= 0]
    return {
        "model_id": model_id, "source_splits": memberships,
        "status": str(labels["status"].item()), "reason": str(labels["reason"].item()),
        "source_sha256": str(labels["source_sha256"].item()), "step_sha256": str(labels["step_sha256"].item()),
        "points": int(valid.size), "valid_points": int(valid.sum()),
        "native_points": int((labels["edge_uv_method"] == 1).sum()),
        "projected_points": int((labels["edge_uv_method"] == 2).sum()),
        "invalid_points": int((~valid).sum()),
        "valid_residual_sum": float(residual.sum(dtype=np.float64)),
        "valid_residual_max": float(residual.max()) if residual.size else None,
        "face_match_error_max": float(matched.max()) if matched.size else None,
    }


def _write_reports(root: Path, records: dict[str, dict[str, Any]], model_ids: list[str], selected_count: int) -> dict[str, Any]:
    ordered = [records[model_id] for model_id in model_ids if model_id in records]
    counts = Counter(record["status"] for record in ordered)
    summary = {
        "schema_version": SCHEMA_VERSION, "updated_utc": datetime.now(timezone.utc).isoformat(),
        "input_total": len(model_ids), "selected_this_invocation": selected_count,
        "recorded": len(ordered), "pending": len(model_ids) - len(ordered),
        "status_counts": dict(counts),
    }
    for key in ("points", "valid_points", "native_points", "projected_points", "invalid_points"):
        summary[key] = sum(record.get(key, 0) for record in ordered)
    total_residual = sum(record.get("valid_residual_sum", 0) for record in ordered)
    summary["valid_residual_mean"] = total_residual / summary["valid_points"] if summary["valid_points"] else None
    summary["valid_residual_max"] = max((record["valid_residual_max"] for record in ordered if record.get("valid_residual_max") is not None), default=None)
    _atomic_text(root / "manifest.jsonl", "".join(json.dumps(record, sort_keys=True) + "\n" for record in ordered))
    _atomic_text(root / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--step-root", type=Path, help="Defaults to --data-root; contains ID/normalized_shape.step.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, action="append", required=True, help="Original ID list; repeat to union lists with recorded membership.")
    parser.add_argument("--limit", type=int, help="Only process the first N IDs; can later resume the same root without a limit.")
    parser.add_argument("--match-tolerance", type=float, default=1e-5)
    parser.add_argument("--projection-tolerance", type=float, default=1e-3)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if any(not np.isfinite(value) or value <= 0 for value in (args.match_tolerance, args.projection_tolerance)):
        parser.error("tolerances must be finite and positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    data_root = args.data_root.expanduser().resolve()
    step_root = (args.step_root or args.data_root).expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not data_root.is_dir() or not step_root.is_dir():
        raise ValueError("--data-root and --step-root must exist")
    for root in (data_root, step_root):
        if output_root.is_relative_to(root) or root.is_relative_to(output_root):
            raise ValueError("output root must not overlap either source root")
    model_ids, splits, memberships = _load_lists([path.expanduser().resolve() for path in args.split])
    selected = model_ids[:args.limit] if args.limit is not None else model_ids
    # Fail before producing a dataset if the required OCCT runtime is unavailable.
    import OCC

    parameters = {"match_tolerance": args.match_tolerance, "projection_tolerance": args.projection_tolerance}
    contract = {
        "schema_version": SCHEMA_VERSION, "generator_version": GENERATOR_VERSION,
        "generator_sha256": sha256_file(Path(__file__)),
        "schema_sha256": sha256_file(Path(__file__).with_name("edge_uv.py")),
        "occ_version": OCC.VERSION, "data_root": str(data_root), "step_root": str(step_root),
        "splits": splits, "input_total": len(model_ids), "parameters": parameters,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    import fcntl

    with (output_root / ".extract.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        dataset_path = output_root / "dataset.json"
        if dataset_path.exists():
            existing = json.loads(dataset_path.read_text(encoding="utf-8"))
            if existing.get("contract") != contract:
                raise ValueError("existing dataset contract differs: use a new version/output root; no files overwritten")
        else:
            if any(path.name != ".extract.lock" for path in output_root.iterdir()):
                raise ValueError("nonempty output root without dataset.json; refusing unversioned outputs")
            git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True, capture_output=True)
            metadata = {"contract": contract, "created_utc": datetime.now(timezone.utc).isoformat(), "git_commit": git.stdout.strip() if git.returncode == 0 else None}
            _atomic_text(dataset_path, json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        manifest = output_root / "manifest.jsonl"
        records = {}
        if manifest.exists():
            for line in manifest.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                model_id = record["model_id"]
                if model_id not in memberships or model_id in records:
                    raise ValueError("manifest contains unknown or duplicate model IDs")
                records[model_id] = record
        for index, model_id in enumerate(selected, 1):
            started = time.perf_counter()
            source_path = _under(data_root, model_id, "data.npz")
            step_path = _under(step_root, model_id, "normalized_shape.step")
            output_path = _under(output_root, model_id, "edge_uv.npz")
            missing = next((path for path in (source_path, step_path) if not path.is_file()), None)
            if missing is not None:
                error = FileNotFoundError(f"required source file is missing: {missing}")
                records[model_id] = {"model_id": model_id, "source_splits": memberships[model_id], "status": "source_failed", "reason": f"{type(error).__name__}: {error}", "elapsed_seconds": time.perf_counter() - started}
                print(f"[{index}/{len(selected)}] {model_id} source_failed: {error}", flush=True)
                if index % 25 == 0:
                    _write_reports(output_root, records, model_ids, len(selected))
                continue
            # Never turn mismatched/corrupt existing labels into a silent overwrite.
            if output_path.exists():
                source = read_source_geometry(source_path)
                labels = load_edge_uv_sidecar(output_path, source_path=source_path, connectivity=source["edge_face_connectivity"], num_faces=len(source["sample_points_faces"]))
                current_step_hash = sha256_file(step_path)
                if str(labels["step_sha256"].item()) != current_step_hash or json.loads(str(labels["parameters_json"].item())) != parameters:
                    raise ValueError(f"{model_id}: existing STEP digest or generation parameters differ")
                action = "resumed"
            else:
                try:
                    labels = extract_edge_uv(source_path, step_path, **parameters)
                except (OSError, ValueError, KeyError) as error:
                    records[model_id] = {"model_id": model_id, "source_splits": memberships[model_id], "status": "source_failed", "reason": f"{type(error).__name__}: {error}", "elapsed_seconds": time.perf_counter() - started}
                    print(f"[{index}/{len(selected)}] {model_id} source_failed: {error}", flush=True)
                    if index % 25 == 0:
                        _write_reports(output_root, records, model_ids, len(selected))
                    continue
                atomic_save_npz(output_path, labels)
                action = "written"
            record = _label_record(model_id, labels, memberships[model_id])
            elapsed = time.perf_counter() - started
            generation_seconds = elapsed if action == "written" else records.get(model_id, {}).get("generation_seconds")
            record.update(action=action, elapsed_seconds=elapsed, generation_seconds=generation_seconds)
            records[model_id] = record
            print(f"[{index}/{len(selected)}] {model_id} {action} {record['status']} valid={record['valid_points']}/{record['points']} native={record['native_points']} projected={record['projected_points']}", flush=True)
            if index % 25 == 0:
                _write_reports(output_root, records, model_ids, len(selected))
        summary = _write_reports(output_root, records, model_ids, len(selected))
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        if summary["status_counts"].get("source_failed", 0):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
