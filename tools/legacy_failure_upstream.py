"""GT-referenced upstream observations for the frozen legacy failure census.

Distances are lengths in normalized XYZ, not squared Chamfer distances. These
are screening observations, not a classifier of unique causes of invalid CAD.
No geometry, topology, model, or postprocessing decision is changed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.spatial import cKDTree

from tools.vae_compare_post import atomic_json, configure_cpu, fingerprint, read_json, run_isolated_jobs


SCHEMA = "legacy_failure_upstream_v1"
THRESHOLDS = (0.002, 0.006, 0.01, 0.08)
SETTINGS = {
    "thresholds": THRESHOLDS, "distance_units": "normalized XYZ length",
    "curve_samples": 64, "near_coverage": 0.95,
    "curve_distance": "64 uniform arc-length samples to the other's original polyline; symmetric mean, max of directional p95/max",
    "near_curve_rule": "at least 95% of samples in EACH direction lie within tolerance",
    "new_nearness_rule": "prediction near and corresponding GT pair NOT near at the SAME tolerance",
    "raw_cross_pair_rule": "any candidate combination between distinct unordered face pairs; prefer smallest passed threshold, then symmetric p95",
    "face_distance": "nearest 16x16 grid samples; a proximity proxy, NOT distance to the underlying surface or trimmed-face overlap",
    "endpoint_rule": "known GT-closed pair: own endpoint gap; known GT-open pair: other curves' endpoints only; unknown pair: no closure assumption",
    "dedup": "not_applied: frozen decoder call explicitly sets v_deduplicated=False",
    "uncertainty": "Threshold flags are descriptive screening, not validated causes or CAD validity criteria.",
}


def key(value):
    return format(value, "g")


def unknown(reason):
    return {"status": "unknown", "reason": str(reason)}


def stats(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    return {"count": len(values), "mean": float(values.mean()) if len(values) else None,
            "p95": float(np.quantile(values, .95)) if len(values) else None,
            "max": float(values.max()) if len(values) else None}


def digest(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def arc_samples(curve, count=64):
    curve = np.asarray(curve, dtype=np.float64)
    lengths = np.linalg.norm(np.diff(curve, axis=0), axis=-1)
    cumulative = np.r_[0., np.cumsum(lengths)]
    if cumulative[-1] <= 1e-15:
        return np.repeat(curve[:1], count, axis=0)
    keep = np.r_[True, np.diff(cumulative) > 0]
    locations = np.linspace(0, cumulative[-1], count)
    return np.stack([np.interp(locations, cumulative[keep], curve[keep, axis]) for axis in range(3)], -1)


def point_polyline_distances(points, curve):
    start = curve[:-1]
    vector = curve[1:] - start
    delta = points[:, None] - start[None]
    fraction = (delta * vector[None]).sum(-1) / np.maximum((vector * vector).sum(-1), 1e-30)[None]
    difference = delta - np.clip(fraction, 0, 1)[..., None] * vector[None]
    return np.sqrt((difference * difference).sum(-1).min(-1))


def curve_distance(first, second):
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    forward = point_polyline_distances(arc_samples(first), second)
    backward = point_polyline_distances(arc_samples(second), first)
    return distance_record(forward, backward)


def distance_record(forward, backward):
    return {"mean": float((forward.mean() + backward.mean()) / 2),
            "p95": float(max(np.quantile(forward, .95), np.quantile(backward, .95))),
            "max": float(max(forward.max(), backward.max())),
            "near": {key(t): bool((forward <= t).mean() >= .95 and (backward <= t).mean() >= .95)
                     for t in THRESHOLDS}}


def groups_from_connectivity(conn, face_count, edge_count):
    groups = defaultdict(list)
    invalid, same = [], []
    for index, (edge, first, second) in enumerate(conn):
        edge, first, second = int(edge), int(first), int(second)
        if not 0 <= edge < edge_count or not 0 <= first < face_count or not 0 <= second < face_count:
            invalid.append(index)
        elif first == second:
            same.append(index)
        else:
            groups[tuple(sorted((first, second)))].append((edge, first, second))
    return groups, invalid, same


def structure(conn, face_count, edge_count, gt_pairs):
    groups, invalid, same = groups_from_connectivity(conn, face_count, edge_count)
    pairs = set(groups)
    directed = [(int(row[1]), int(row[2])) for row in conn]
    counts = Counter(directed)
    reciprocal = sum(len(rows) == 2 and rows[0][1:] == rows[1][1:][::-1] for rows in groups.values())
    return {"status": "ok", "num_faces": face_count, "num_curve_rows": edge_count,
            "num_unique_pairs": len(pairs), "invalid_connectivity_row_indices": invalid,
            "same_face_row_indices": same, "fp_pairs": sorted(pairs - gt_pairs),
            "fn_pairs": sorted(gt_pairs - pairs), "normal_reciprocal_pair_count": reciprocal,
            "unilateral_pairs": [pair for pair, rows in groups.items() if len(rows) == 1],
            "exact_directed_duplicate_count": sum(max(count - 1, 0) for count in counts.values()),
            "duplicate_directed_pairs": [{"pair": pair, "count": count} for pair, count in counts.items() if count > 1],
            "unsupported_multiplicity_pairs": [pair for pair, rows in groups.items() if len(rows) > 2]}


def reciprocal_diagnostics(curves, conn, face_count, gt_curves, gt_conn):
    groups, _, _ = groups_from_connectivity(conn, face_count, len(curves))
    gt_groups, _, _ = groups_from_connectivity(gt_conn, face_count, len(gt_curves))
    records = []
    for pair, rows in groups.items():
        if len(rows) != 2 or rows[0][1:] != rows[1][1:][::-1]:
            continue
        distance = curve_distance(curves[rows[0][0]], curves[rows[1][0]])
        baseline = gt_groups.get(pair, [])
        truth = curve_distance(gt_curves[baseline[0][0]], gt_curves[baseline[1][0]]) if len(baseline) == 2 else None
        records.append({"face_pair": pair, "edge_ids": [row[0] for row in rows],
                        "distance": distance, "gt_distance": truth,
                        "gt_reference_status": "ok" if truth else "unknown_pair"})
    threshold_counts = {}
    for t in THRESHOLDS:
        name = key(t)
        threshold_counts[name] = {
            "disagreement_pairs": sum(not r["distance"]["near"][name] for r in records),
            "excess_over_gt_pairs": sum(not r["distance"]["near"][name] and r["gt_distance"] is not None
                                         and r["gt_distance"]["near"][name] for r in records),
            "gt_unknown_pairs": sum(r["gt_distance"] is None for r in records),
        }
    return {"status": "ok", "normal_reciprocal_count": len(records), "records": records,
            "distance_p95": stats([r["distance"]["p95"] for r in records]), "thresholds": threshold_counts}


def bbox_gap(first, second):
    return float(np.linalg.norm(np.maximum(np.maximum(first.min(0) - second.max(0),
                                                       second.min(0) - first.max(0)), 0)))


def proximity_counts(records):
    result = {}
    for t in THRESHOLDS:
        name = key(t)
        near = [row for row in records if row["prediction_distance"]["near"][name]]
        result[name] = {"pred_near_count": len(near),
                        "gt_baseline_close_count": sum(row["gt_distance"] is not None and row["gt_distance"]["near"][name] for row in near),
                        "new_nearness_count": sum(row["gt_distance"] is not None and not row["gt_distance"]["near"][name] for row in near),
                        "gt_baseline_unknown_count": sum(row["gt_distance"] is None for row in near)}
    return result


def cross_pair_curves(curves, conn, face_count, gt_map):
    groups, _, _ = groups_from_connectivity(conn, face_count, len(curves))
    pairs = sorted(groups)
    records = []
    exact = []
    comparisons = 0
    for i, pair1 in enumerate(pairs):
        for pair2 in pairs[i + 1:]:
            candidates = []
            for row1 in groups[pair1]:
                for row2 in groups[pair2]:
                    e1, e2 = row1[0], row2[0]
                    first, second = curves[e1], curves[e2]
                    coordinate_equal = np.array_equal(first, second) or np.array_equal(first, second[::-1])
                    if coordinate_equal:
                        exact.append({"edge_ids": [e1, e2], "face_pairs": [pair1, pair2]})
                    if bbox_gap(first, second) > max(THRESHOLDS):
                        continue
                    distance = curve_distance(first, second)
                    comparisons += 1
                    first_pass = next((t for t in THRESHOLDS if distance["near"][key(t)]), float("inf"))
                    candidates.append((first_pass, distance["p95"], e1, e2, distance))
            if not candidates:
                continue
            _, _, e1, e2, distance = min(candidates, key=lambda item: item[:4])
            if not any(distance["near"].values()):
                continue
            truth = curve_distance(gt_map[pair1]["curve"], gt_map[pair2]["curve"]) if pair1 in gt_map and pair2 in gt_map else None
            records.append({"face_pairs": [pair1, pair2], "edge_ids": [e1, e2],
                            "prediction_distance": distance, "gt_distance": truth})
    return {"status": "ok", "unordered_pair_comparisons": len(pairs) * (len(pairs) - 1) // 2,
            "curve_comparisons_after_bbox_gate": comparisons,
            "exact_coordinate_duplicate_records": exact,
            "exact_coordinate_duplicate_count": len(exact),
            "exact_duplicate_note": "Different face pairs with exactly equal/reversed coordinate arrays; not a proof that the B-Rep entities should be merged.",
            "records": records, "thresholds": proximity_counts(records)}


def face_proximity(faces, truth):
    if faces.shape != truth.shape:
        return unknown("Face identity/grid shape differs from GT")
    pred_points = faces.reshape(len(faces), -1, 3)
    gt_points = truth.reshape(len(truth), -1, 3)
    pred_trees, gt_trees = [cKDTree(points) for points in pred_points], [cKDTree(points) for points in gt_points]
    records = []
    for first in range(len(faces)):
        for second in range(first + 1, len(faces)):
            if bbox_gap(pred_points[first], pred_points[second]) > max(THRESHOLDS):
                continue
            distance = distance_record(pred_trees[second].query(pred_points[first], workers=1)[0],
                                       pred_trees[first].query(pred_points[second], workers=1)[0])
            if not any(distance["near"].values()):
                continue
            baseline = distance_record(gt_trees[second].query(gt_points[first], workers=1)[0],
                                       gt_trees[first].query(gt_points[second], workers=1)[0])
            records.append({"face_ids": [first, second], "prediction_distance": distance, "gt_distance": baseline})
    return {"status": "ok", "description": SETTINGS["face_distance"],
            "records": records, "thresholds": proximity_counts(records)}


def endpoint_proxy(curves, conn, face_count, gt_map):
    groups, invalid, same = groups_from_connectivity(conn, face_count, len(curves))
    if invalid or same or any(len(rows) != 1 for rows in groups.values()):
        return unknown("Endpoint proxy requires exactly one selected edge per valid distinct-face pair")
    adjacent = [[] for _ in range(face_count)]
    for pair, rows in groups.items():
        for face in pair:
            adjacent[face].append((rows[0][0], pair))
    records, global_values, edge_spans = [], [], []
    for pair, rows in groups.items():
        edge = rows[0][0]
        pred_gap = float(np.linalg.norm(curves[edge, 0] - curves[edge, -1]))
        truth = gt_map.get(pair)
        gt_gap = float(np.linalg.norm(truth["curve"][0] - truth["curve"][-1])) if truth else None
        edge_spans.append({"edge_id": edge, "face_pair": pair, "gt_closed": truth["closed"] if truth else None,
                           "own_endpoint_gap": pred_gap, "gt_own_endpoint_gap": gt_gap,
                           "arc_length": float(np.linalg.norm(np.diff(curves[edge], axis=0), axis=-1).sum()),
                           "gt_arc_length": float(np.linalg.norm(np.diff(truth["curve"], axis=0), axis=-1).sum()) if truth else None,
                           "new_open_endpoint_collapse": {key(t): bool(not truth["closed"] and pred_gap <= t < gt_gap)
                                                           if truth else None for t in THRESHOLDS}})
    for face, entries in enumerate(adjacent):
        endpoints = [(edge, endpoint, pair, curves[edge, endpoint]) for edge, pair in entries for endpoint in (0, -1)]
        values, unmatchable, unknown_closure = [], 0, 0
        details = []
        for edge, endpoint, pair, position in endpoints:
            other_distances = [float(np.linalg.norm(point - position)) for other, _, _, point in endpoints if other != edge]
            if pair not in gt_map:
                unknown_closure += 1
                residual = None
                basis = "unknown_gt_closure"
            elif gt_map[pair]["closed"]:
                residual = float(np.linalg.norm(curves[edge, 0] - curves[edge, -1]))
                basis = "GT_closed_own_endpoint_gap"
            else:
                others = [point for other, _, _, point in endpoints if other != edge]
                residual = float(np.linalg.norm(np.asarray(others) - position, axis=-1).min()) if others else None
                basis = "other_edge_endpoints" if others else "no_other_edge_endpoint"
                if not others:
                    unmatchable += 1
            if residual is not None:
                values.append(residual)
            details.append({"edge_id": edge, "endpoint": endpoint, "basis": basis, "distance": residual})
            details[-1]["other_endpoint_neighbor_counts"] = {key(t): sum(d <= t for d in other_distances) for t in THRESHOLDS}
        global_values.extend(values)
        records.append({"face_id": face, "edge_ids": [entry[0] for entry in entries],
                        "status": "ok" if entries and not unknown_closure else "unknown",
                        "reason": "no incident edge" if not entries else "some pairs lack GT closure" if unknown_closure else None,
                        "unmatchable_endpoint_count": unmatchable, "unknown_closure_endpoint_count": unknown_closure,
                        "distances": stats(values), "endpoints": details,
                        "above_threshold": {key(t): sum(v > t for v in values) + unmatchable for t in THRESHOLDS}})
    return {"status": "ok", "description": "Nearest endpoint residual only; no ordering, perfect matching, pcurve or valid-loop guarantee.",
            "distances": stats(global_values), "faces": records, "edge_spans": edge_spans,
            "faces_without_incident_edges": [row["face_id"] for row in records if not row["edge_ids"]],
            "unknown_closure_endpoint_count": sum(row["unknown_closure_endpoint_count"] for row in records),
            "thresholds": {key(t): {"faces_with_residual": sum(row["above_threshold"][key(t)] > 0 for row in records),
                                    "endpoints_with_residual": sum(row["above_threshold"][key(t)] for row in records),
                                    "endpoints_with_multiple_near_partners": sum(endpoint["other_endpoint_neighbor_counts"][key(t)] > 1
                                                                                for row in records for endpoint in row["endpoints"]),
                                    "new_open_endpoint_collapse_edges": sum(row["new_open_endpoint_collapse"][key(t)] is True for row in edge_spans)} for t in THRESHOLDS}}


def grid_area(face):
    a, b, c, d = face[:-1, :-1], face[1:, :-1], face[1:, 1:], face[:-1, 1:]
    return float((np.linalg.norm(np.cross(b - a, c - a), axis=-1).sum()
                  + np.linalg.norm(np.cross(c - a, d - a), axis=-1).sum()) / 2)


def gt_geometry(faces, curves, conn, gt_faces, gt_map):
    face_result = unknown("Face identity/grid shape differs from GT")
    if faces.shape == gt_faces.shape:
        records = []
        for index, (pred, truth) in enumerate(zip(faces, gt_faces)):
            area, gt_area = grid_area(pred), grid_area(truth)
            records.append({"face_id": index, "xyz_mae": float(np.abs(pred - truth).mean()),
                            "point_l2": stats(np.linalg.norm(pred - truth, axis=-1)),
                            "grid_triangle_area": area, "gt_grid_triangle_area": gt_area,
                            "area_ratio": area / gt_area if gt_area > 0 else None})
        face_result = {"status": "ok", "xyz_mae": float(np.abs(faces - gt_faces).mean()), "records": records}
    records, unmatched = [], []
    for edge, first, second in conn:
        pair = tuple(sorted((int(first), int(second))))
        if pair not in gt_map:
            unmatched.append(int(edge))
            continue
        distance = curve_distance(curves[int(edge)], gt_map[pair]["curve"])
        records.append({"edge_id": int(edge), "face_pair": pair, "distance": distance})
    return {"status": "ok", "faces": face_result,
            "edges": {"status": "ok", "matched_count": len(records), "unmatched_edge_ids": unmatched,
                      "mean_distance_over_edges": stats([row["distance"]["mean"] for row in records]),
                      "p95_distance_over_edges": stats([row["distance"]["p95"] for row in records]), "records": records}}


def stage_metrics(faces, curves, conn, gt_faces, gt_map, *, raw=False):
    return {"status": "ok", "gt_geometry": gt_geometry(faces, curves, conn, gt_faces, gt_map),
            "endpoint_proxy": unknown("not_applied to redundant raw reciprocal half-edges") if raw else endpoint_proxy(curves, conn, len(faces), gt_map),
            "cross_pair_near_curves": cross_pair_curves(curves, conn, len(faces), gt_map),
            "face_collapse_proxy": face_proximity(faces, gt_faces)}


def stage_deltas(selected, optimized):
    if selected.get("status") != "ok" or optimized.get("status") != "ok":
        return unknown("Selected and optimized stages are both required")
    result = {"status": "ok", "sign": "optimized minus selected; positive distance change is deterioration of that metric"}
    before, after = selected["gt_geometry"], optimized["gt_geometry"]
    if before["faces"]["status"] == after["faces"]["status"] == "ok":
        result["face_xyz_mae_change"] = after["faces"]["xyz_mae"] - before["faces"]["xyz_mae"]
        result["face_changes"] = [{"face_id": first["face_id"], "xyz_mae_change": second["xyz_mae"] - first["xyz_mae"]}
                                  for first, second in zip(before["faces"]["records"], after["faces"]["records"])]
    else:
        result["face_xyz_mae_change"] = None
    mapping = {row["edge_id"]: row for row in before["edges"]["records"]}
    result["edge_changes"] = [{"edge_id": row["edge_id"], "face_pair": row["face_pair"],
                               "mean_distance_change": row["distance"]["mean"] - mapping[row["edge_id"]]["distance"]["mean"],
                               "p95_distance_change": row["distance"]["p95"] - mapping[row["edge_id"]]["distance"]["p95"]}
                              for row in after["edges"]["records"] if row["edge_id"] in mapping]
    result["edge_mean_distance_change"] = stats([row["mean_distance_change"] for row in result["edge_changes"]])
    before_end, after_end = selected["endpoint_proxy"], optimized["endpoint_proxy"]
    result["endpoint_mean_residual_change"] = None
    if before_end.get("status") == after_end.get("status") == "ok":
        first, second = before_end["distances"]["mean"], after_end["distances"]["mean"]
        if first is not None and second is not None:
            result["endpoint_mean_residual_change"] = second - first
        result["endpoint_threshold_changes"] = {
            key(t): after_end["thresholds"][key(t)]["endpoints_with_residual"]
            - before_end["thresholds"][key(t)]["endpoints_with_residual"] for t in THRESHOLDS}
    return result


def mechanism_flags(report):
    raw = report.get("structure", {}).get("raw", {})
    selected = report.get("structure", {}).get("selected", {})
    flags = {
        "raw_extra_gt_pair": bool(raw["fp_pairs"]) if raw.get("status") == "ok" else None,
        "raw_missing_gt_pair": bool(raw["fn_pairs"]) if raw.get("status") == "ok" else None,
        "raw_duplicate_directed_pair": raw["exact_directed_duplicate_count"] > 0 if raw.get("status") == "ok" else None,
        "raw_same_face_pair": bool(raw["same_face_row_indices"]) if raw.get("status") == "ok" else None,
        "selection_removed_gt_pair": bool(selected["removed_gt_pairs"]) if selected.get("status") == "ok" else None,
    }
    by_threshold = {}
    for t in THRESHOLDS:
        name = key(t)
        threshold_flags = {}
        reciprocal = report.get("reciprocal", {})
        if reciprocal.get("status") == "ok":
            counts = reciprocal["thresholds"][name]
            threshold_flags["reciprocal_excess_disagreement"] = True if counts["excess_over_gt_pairs"] else None if counts["gt_unknown_pairs"] else False
        else:
            threshold_flags["reciprocal_excess_disagreement"] = None
        for stage_name in ("raw", "selected", "optimized"):
            stage = report.get("stages", {}).get(stage_name, {})
            for module, suffix in (("cross_pair_near_curves", "new_cross_pair_curve_nearness"), ("face_collapse_proxy", "new_face_grid_nearness")):
                value = stage.get(module, {})
                if value.get("status") == "ok":
                    counts = value["thresholds"][name]
                    threshold_flags[f"{stage_name}_{suffix}"] = True if counts["new_nearness_count"] else None if counts["gt_baseline_unknown_count"] else False
                else:
                    threshold_flags[f"{stage_name}_{suffix}"] = None
            if stage_name != "raw":
                value = stage.get("endpoint_proxy", {})
                count = value.get("thresholds", {}).get(name, {}).get("endpoints_with_residual", 0)
                unresolved = value.get("unknown_closure_endpoint_count", 0) or value.get("faces_without_incident_edges", [])
                threshold_flags[f"{stage_name}_endpoint_gap"] = (True if count else None if unresolved else False) if value.get("status") == "ok" else None
                if value.get("status") == "ok":
                    collapse = value["thresholds"][name]["new_open_endpoint_collapse_edges"]
                    threshold_flags[f"{stage_name}_new_open_edge_endpoint_collapse"] = True if collapse else None if value.get("unknown_closure_endpoint_count", 0) else False
                else:
                    threshold_flags[f"{stage_name}_new_open_edge_endpoint_collapse"] = None
        by_threshold[name] = threshold_flags
    return flags, by_threshold


def input_sources(case, root):
    sid = case["id"]
    paths = {"raw": Path(case["paths"]["raw_legacy_npz"]), "gt": Path(case["paths"]["gt_npz"]),
             "trace": root / "traces" / f"{sid}.json", "stage_geometry": root / "replay" / sid / "trace_geometry.npz"}
    return {name: {"path": str(path.resolve()), "sha256": digest(path)} for name, path in paths.items()}


def analyze_case(case, sources):
    from src.brepnet.data.shared_boundaries import build_shared_boundary_targets
    sid = case["id"]
    with np.load(sources["raw"]["path"], allow_pickle=False) as data:
        if data["comparison_mode"].item() != "posterior_mean":
            raise ValueError("Only fixed-identity posterior_mean comparisons are supported")
        faces, curves, conn = (data[name].copy() for name in ("pred_face", "pred_edge", "pred_edge_face_connectivity"))
    with np.load(sources["gt"]["path"], allow_pickle=False) as data:
        gt_faces, gt_curves, gt_conn = (data[name].copy() for name in ("sample_points_faces", "sample_points_lines", "edge_face_connectivity"))
    target = build_shared_boundary_targets(gt_faces, gt_curves, gt_conn, sample_id=sid,
                                           reciprocal_tolerance=.01, closed_tolerance=1e-5, allow_closed_phase=True)
    if faces.shape != gt_faces[..., :3].shape or curves.ndim != 3 or curves.shape[1:] != (16, 3):
        raise ValueError("Unexpected raw face/curve shape or GT face correspondence")
    if conn.shape != (len(curves), 3) or not np.issubdtype(conn.dtype, np.integer):
        raise ValueError("Invalid raw connectivity shape/type")
    if not np.isfinite(faces).all() or not np.isfinite(curves).all():
        raise ValueError("Nonfinite raw geometry")
    gt_faces, gt_curves = gt_faces[..., :3].astype(np.float64), gt_curves[..., :3].astype(np.float64)
    faces, curves = faces.astype(np.float64), curves.astype(np.float64)
    gt_map = {tuple(map(int, pair)): {"curve": curve.astype(np.float64), "closed": bool(closed)}
              for pair, curve, closed in zip(target["face_pairs"], target["curves"], target["is_closed"])}
    raw_structure = structure(conn, len(faces), len(curves), set(gt_map))
    report = {"status": "ok", "dedup": {"status": "not_applied", "removed_faces": 0, "reason": SETTINGS["dedup"]},
              "face_correspondence": "identity and original order; no correspondence to final STEP face numbering",
              "structure": {"raw": raw_structure, "gt_num_faces": len(gt_faces),
                            "face_count_difference": len(faces) - len(gt_faces), "gt_boundary_count": len(gt_map)},
              "reciprocal": reciprocal_diagnostics(curves, conn, len(faces), gt_curves, gt_conn),
              "stages": {}, "unknowns": []}
    if raw_structure["invalid_connectivity_row_indices"]:
        raise ValueError("Invalid raw connectivity indices; geometric measurements cannot be safely associated")
    report["stages"]["raw"] = stage_metrics(faces, curves, conn, gt_faces, gt_map, raw=True)
    report["structure"]["selected"] = unknown("Trace selection unavailable")
    for name in ("selected", "optimized"):
        report["stages"][name] = unknown("Stage geometry unavailable")
    trace = read_json(Path(sources["trace"]["path"])) if sources["trace"]["sha256"] else {}
    selection = trace.get("selection")
    if selection:
        selected_conn = np.asarray(selection["connectivity"], dtype=np.int64).reshape(-1, 3)
        selected_structure = structure(selected_conn, len(faces), selection["output_edge_count"], set(gt_map))
        raw_pairs = set(gt_map) - set(map(tuple, raw_structure["fn_pairs"])) | set(map(tuple, raw_structure["fp_pairs"]))
        selected_pairs = {tuple(sorted(map(int, row[1:]))) for row in selected_conn}
        selected_structure.update(removed_gt_pairs=sorted((raw_pairs - selected_pairs) & set(gt_map)),
                                  removed_false_pairs=sorted((raw_pairs - selected_pairs) - set(gt_map)),
                                  directed_redundancy_note="Selected curves are shared representatives; unilateral_pair counts here are expected.")
        report["structure"]["selected"] = selected_structure
    if sources["stage_geometry"]["sha256"]:
        with np.load(sources["stage_geometry"]["path"], allow_pickle=False) as stage_data:
            for name in ("selected", "optimized"):
                fields = (name + "_faces", name + "_edges", "selected_connectivity")
                if not all(field in stage_data for field in fields):
                    continue
                sf, se, sc = (stage_data[field].copy() for field in fields)
                if sf.shape != faces.shape or se.ndim != 3 or sc.shape != (len(se), 3):
                    report["stages"][name] = unknown("Invalid stage geometry shapes")
                    continue
                if not np.isfinite(sf).all() or not np.isfinite(se).all():
                    report["stages"][name] = unknown("Nonfinite stage geometry")
                    continue
                if selection and not np.array_equal(sc, selected_conn):
                    report["stages"][name] = unknown("Snapshot connectivity differs from trace selection")
                    continue
                report["stages"][name] = stage_metrics(sf.astype(np.float64), se.astype(np.float64), sc, gt_faces, gt_map)
    for name, stage in report["stages"].items():
        if stage.get("status") != "ok":
            report["unknowns"].append(name + ": " + stage["reason"])
    report["stage_deltas"] = stage_deltas(report["stages"]["selected"], report["stages"]["optimized"])
    report["mechanism_flags"], report["mechanism_flags_by_threshold"] = mechanism_flags(report)
    report["status"] = "partial" if report["unknowns"] else "ok"
    return report


def worker(job, status_path, log_path):
    os.setsid()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as handle:
        os.dup2(handle.fileno(), 1)
        os.dup2(handle.fileno(), 2)
        started = time.monotonic()
        output = {"schema": SCHEMA, "sample_id": job["sample_id"], "protocol_fingerprint": job["protocol_fingerprint"],
                  "sources": job["sources"], "settings": SETTINGS, "status": "running"}
        path = Path(job["output"])
        atomic_json(path, output)
        atomic_json(status_path, {"status": "running", "stage": "upstream"})
        try:
            output.update(analyze_case(job["case"], job["sources"]))
        except Exception as exc:
            traceback.print_exc()
            output.update(status="error", error=f"{type(exc).__name__}: {exc}",
                          mechanism_flags={name: None for name in ("raw_extra_gt_pair", "raw_missing_gt_pair", "raw_duplicate_directed_pair", "raw_same_face_pair", "selection_removed_gt_pair")},
                          mechanism_flags_by_threshold={})
        output["elapsed_seconds"] = time.monotonic() - started
        atomic_json(path, output)
        atomic_json(status_path, {"status": "ok" if output["status"] in ("ok", "partial") else "post_error",
                                  "stage": "done", "elapsed_seconds": output["elapsed_seconds"]})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be 1..8")
    configure_cpu()
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    root = args.root.resolve()
    cases = read_json(args.manifest)["cases"]
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate manifest IDs")
    protocol = {"schema": SCHEMA, "settings": SETTINGS, "manifest_sha256": digest(args.manifest),
                "tool_sha256": digest(__file__), "gt_helper_sha256": digest("src/brepnet/data/shared_boundaries.py"),
                "workers": args.workers, "cpu_threads": 1, "hard_timeout_seconds": 180}
    protocol_hash = fingerprint(protocol)
    protocol_path = root / "metadata" / "upstream" / "protocol.json"
    old_protocol = read_json(protocol_path)
    if old_protocol and (old_protocol != json.loads(json.dumps(protocol)) or not args.resume):
        raise ValueError("Existing root requires --resume and identical protocol")
    atomic_json(protocol_path, protocol)
    jobs = []
    for case in cases:
        sid = case["id"]
        sources = input_sources(case, root)
        output = root / "upstream" / f"{sid}.json"
        previous = read_json(output)
        if previous:
            if not args.resume or previous.get("protocol_fingerprint") != protocol_hash:
                raise ValueError(f"Refusing to overwrite {output}")
            if previous.get("sources") == sources and previous.get("status") in ("ok", "partial", "error", "timeout", "crash"):
                continue
            # A trace/snapshot may become available after a partial run. Preserve
            # that diagnostic record before recomputing against new inputs.
            archive = root / "metadata" / "upstream" / "superseded" / f"{sid}.{time.time_ns()}.json"
            archive.parent.mkdir(parents=True, exist_ok=True)
            output.rename(archive)
        jobs.append({"sample_id": sid, "method": "legacy", "mode": "upstream",
                     "protocol_fingerprint": protocol_hash, "raw_sha256": sources["raw"]["sha256"],
                     "case": case, "sources": sources, "output": str(output)})
    metadata = root / "metadata" / "upstream"
    run_isolated_jobs(jobs, metadata, args.workers, 180, process_target=worker)
    statuses = Counter()
    for case in cases:
        sid = case["id"]
        path = root / "upstream" / f"{sid}.json"
        output = read_json(path)
        status = read_json(metadata / "samples" / "legacy" / f"{sid}.json")
        if status.get("status") in ("timeout", "crash"):
            output.update(status=status["status"], error=status.get("error"))
            atomic_json(path, output)
        statuses[output.get("status", "missing")] += 1
    atomic_json(root / "upstream_summary.json", {"intended": len(cases), "status_counts": dict(statuses), "protocol_fingerprint": protocol_hash})
    print(json.dumps(dict(statuses)), flush=True)


if __name__ == "__main__":
    main()
