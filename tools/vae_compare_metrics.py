"""Deterministic raw metrics for the frozen 2026-10-11 VAE comparison.

No OCC, model execution, GT candidate selection, or point-cloud alignment.
Run from the repository root: python -m tools.vae_compare_metrics --help.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from src.brepnet.data.shared_boundaries import build_shared_boundary_targets


ACCEPTED_COUNT = 2417
ORIGINAL_COUNT = 2424
ACCEPTED_SHA256 = "2864849e8240134c6033cc6cd64efa19337f67fea94b89a493690ae84355b7b3"
METHODS = ("legacy", "joint")
CANDIDATE_METHODS = ("joint", "intersection", "local", "conditioned")
BASELINE_METHODS = ("legacy", "intersection")


def comparison_methods(candidate_method="joint", baseline_method="legacy"):
    if candidate_method not in CANDIDATE_METHODS:
        raise ValueError(f"Unsupported candidate method: {candidate_method}")
    if baseline_method not in BASELINE_METHODS or baseline_method == candidate_method:
        raise ValueError(f"Unsupported baseline/candidate pair: {baseline_method}/{candidate_method}")
    return (baseline_method, candidate_method)

MODES = ("posterior_mean", "archived_latents")
GEOMETRY_METRICS = (
    "face_mae", "face_cd_sq", "face_pred_to_gt_sq", "face_gt_to_pred_sq",
    "edge_cd_sq", "edge_pred_to_gt_sq", "edge_gt_to_pred_sq",
)
TOPOLOGY_METRICS = ("topology_precision", "topology_recall", "topology_f1")
DIAGNOSTIC_METRICS = (
    "pred_face_count", "gt_face_count", "raw_edge_count", "gt_boundary_count",
    "unique_pair_count", "unique_pair_count_error", "same_face_count",
    "invalid_face_index_count", "invalid_edge_index_count", "repeated_edge_reference_count",
    "joint_duplicate_pair_count", "joint_query_count_error", "legacy_directed_duplicate_count",
    "legacy_unilateral_pair_count", "legacy_reciprocal_pair_count",
    "legacy_reciprocal_cd_sq_mean", "legacy_reciprocal_cd_sq_max",
)
METRICS = GEOMETRY_METRICS + TOPOLOGY_METRICS + DIAGNOSTIC_METRICS
PAIRED_METRICS = GEOMETRY_METRICS + TOPOLOGY_METRICS + ("unique_pair_count_error",)
LIMITATIONS = [
    "This compares two trained models with different training histories, not an equal-budget architecture ablation.",
    "Raw geometry uses the original 16x16 face grid and 16-point curves: no area/arc-length weighting or parameter alignment.",
    "Squared CD is the SUM of directional means in normalized global XYZ; it does not penalize coincident duplicates.",
    "Edge CD includes ALL thresholded curves, including duplicate pairs and same-face assignments; no GT selection or post adaptation.",
    "GT boundaries are the audited merged non-seam representatives, not native CAD physical edges or UV trimming loops.",
    "Unordered topology P/R/F1 concerns valid distinct-face pairs only; invalid rows and duplicate entities are reported separately.",
    "Legacy reciprocal half-edges are expected, not duplicates. Raw edge/query counts and representation-specific duplicate counts are not paired differences.",
    "Legacy reciprocal discrepancy uses one direction pair per unordered pair; directed-duplicate groups are excluded from this discrepancy and counted separately.",
    "Geometry means are computable subsets; all missing, malformed, nonfinite and empty cases remain in intended coverage denominators.",
    "Archived latents have no established GT face correspondence: aligned MAE, GT topology P/R/F1 and count-error comparisons are omitted.",
    "Neither raw reconstruction nor archived-latent replay establishes OCC validity or a complete real-photo reconstruction result.",
]


def squared_chamfer(prediction: np.ndarray, truth: np.ndarray) -> tuple[float, float, float]:
    """Return pred->GT, GT->pred, and their sum; both-empty is explicitly zero."""
    prediction, truth = np.asarray(prediction, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    for points in (prediction, truth):
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("CD requires finite [N,3] point clouds")
    if not len(prediction) and not len(truth):
        return 0.0, 0.0, 0.0
    if not len(prediction) or not len(truth):
        raise ValueError("CD is undefined when exactly one point cloud is empty")
    forward = float(np.square(cKDTree(truth).query(prediction, workers=1)[0]).mean())
    backward = float(np.square(cKDTree(prediction).query(truth, workers=1)[0]).mean())
    return forward, backward, forward + backward


def pair_scores(predicted: set, truth: set) -> dict:
    """Empty-denominator convention: P=1 for no predictions, R=1 for no GT."""
    true_positive = len(predicted & truth)
    precision = true_positive / len(predicted) if predicted else 1.0
    recall = true_positive / len(truth) if truth else 1.0
    f1 = 2 * true_positive / (len(predicted) + len(truth)) if predicted or truth else 1.0
    return {
        "topology_tp": true_positive, "topology_fp": len(predicted - truth),
        "topology_fn": len(truth - predicted),
        "topology_precision": precision, "topology_recall": recall, "topology_f1": f1,
    }


def _record_cd(row: dict, prefix: str, predicted: np.ndarray, truth: np.ndarray) -> None:
    forward, backward, total = squared_chamfer(predicted.reshape(-1, 3), truth.reshape(-1, 3))
    row.update({f"{prefix}_pred_to_gt_sq": forward, f"{prefix}_gt_to_pred_sq": backward,
                f"{prefix}_cd_sq": total})


def _geometry_status(points: np.ndarray, entity: str) -> str:
    expected = (16, 16) if entity == "face" else (16,)
    if points.ndim != len(expected) + 2 or points.shape[1:-1] != expected or points.shape[-1] not in (3, 6):
        return "invalid_shape"
    if not np.issubdtype(points.dtype, np.number) or not np.isfinite(points).all():
        return "nonfinite_or_nonnumeric"
    return "ok" if len(points) else "empty_prediction"


def _internal_topology(row: dict, prediction: dict, curves: np.ndarray, face_count: int, method: str) -> set | None:
    connectivity = prediction.get("pred_edge_face_connectivity")
    if connectivity is None:
        row["internal_topology_status"] = "missing_connectivity"
        return None
    conn = np.asarray(connectivity)
    if conn.shape != (len(curves), 3) or not np.issubdtype(conn.dtype, np.integer):
        row["internal_topology_status"] = "invalid_connectivity_shape_or_dtype"
        return None
    valid_edge = (conn[:, 0] >= 0) & (conn[:, 0] < len(curves))
    valid_face = ((conn[:, 1:] >= 0) & (conn[:, 1:] < face_count)).all(axis=1)
    self_pair = valid_face & (conn[:, 1] == conn[:, 2])
    repeated_edges = int(valid_edge.sum() - len(np.unique(conn[valid_edge, 0])))
    row.update({
        "invalid_face_index_count": int((~valid_face).sum()),
        "invalid_edge_index_count": int((~valid_edge).sum()),
        "repeated_edge_reference_count": repeated_edges,
        "same_face_count": int(self_pair.sum()),
        "internal_topology_status": "ok" if valid_edge.all() and valid_face.all() and not repeated_edges else "invalid_indices_or_references",
    })
    valid_conn = conn[valid_edge & valid_face & ~self_pair]
    directed = [tuple(map(int, pair)) for pair in valid_conn[:, 1:]]
    pairs = {tuple(sorted(pair)) for pair in directed}
    row["unique_pair_count"] = len(pairs)
    if method == "joint":
        row["joint_duplicate_pair_count"] = len(valid_conn) - len(pairs)
    else:
        row["legacy_directed_duplicate_count"] = len(directed) - len(set(directed))
        by_pair = defaultdict(list)
        for edge, a, b in valid_conn:
            by_pair[tuple(sorted((int(a), int(b))))].append((int(edge), int(a), int(b)))
        unilateral, reciprocal, differences = 0, 0, []
        for values in by_pair.values():
            directions = {(a, b) for _, a, b in values}
            if len(directions) == 1:
                unilateral += 1
            else:
                reciprocal += 1
                if len(values) == 2 and row["edge_status"] in {"ok", "empty_both", "empty_ground_truth"}:
                    differences.append(squared_chamfer(curves[values[0][0]], curves[values[1][0]])[2])
        row.update(legacy_unilateral_pair_count=unilateral, legacy_reciprocal_pair_count=reciprocal)
        if differences:
            row["legacy_reciprocal_cd_sq_mean"] = float(np.mean(differences))
            row["legacy_reciprocal_cd_sq_max"] = float(np.max(differences))
    return pairs


def evaluate_prediction(prediction: dict, gt_faces: np.ndarray, gt_curves: np.ndarray,
                        gt_pairs: np.ndarray, *, method: str = "joint", mode: str) -> dict:
    """Compute independent stages so an empty/bad edge set cannot erase face metrics."""
    if method not in ("legacy", *CANDIDATE_METHODS):
        raise ValueError(f"Unsupported method: {method}")
    row = {"prediction_status": "ok", "gt_status": "ok", "face_status": "missing_field",
           "edge_status": "missing_field", "face_mae_status": "not_applicable",
           "topology_status": "not_applicable", "internal_topology_status": "not_computable",
           "gt_face_count": len(gt_faces), "gt_boundary_count": len(gt_curves)}
    for key in METRICS:
        row.setdefault(key, None)
    if "input_latent_sha256" in prediction:
        digest = np.asarray(prediction["input_latent_sha256"])
        if digest.ndim == 0:
            row["input_latent_sha256"] = str(digest.item())
    allowed_tags = {"shared_boundary_v1"} if method == "joint" else {"legacy_half_edges", "legacy_directed_v1"}
    tag = np.asarray(prediction.get("representation", "legacy_half_edges" if method == "legacy" else ""))
    if tag.ndim != 0 or str(tag.item()) not in allowed_tags:
        row.update(prediction_status="invalid_representation", status="failed")
        return row
    faces = np.asarray(prediction.get("pred_face", []))
    curves = np.asarray(prediction.get("pred_edge", []))
    if "pred_face" in prediction:
        row["face_status"] = _geometry_status(faces, "face")
    if "pred_edge" in prediction:
        row["edge_status"] = _geometry_status(curves, "edge")
    if faces.ndim == 4:
        row["pred_face_count"] = len(faces)
    if curves.ndim == 3:
        row["raw_edge_count"] = len(curves)
    if row["face_status"] == "ok":
        faces = faces[..., :3]
        _record_cd(row, "face", faces, gt_faces)
        if mode == "posterior_mean":
            if faces.shape == gt_faces.shape:
                row["face_mae"] = float(np.abs(faces.astype(np.float64) - gt_faces).mean())
                row["face_mae_status"] = "ok"
            else:
                row["face_mae_status"] = "face_count_or_grid_mismatch"
    elif mode == "posterior_mean":
        row["face_mae_status"] = row["face_status"]
    if row["edge_status"] in {"ok", "empty_prediction"}:
        curves = curves[..., :3]
        if len(curves) and not len(gt_curves):
            row["edge_status"] = "empty_ground_truth"
        elif not len(curves) and not len(gt_curves):
            row["edge_status"] = "empty_both"
            _record_cd(row, "edge", curves, gt_curves)
        elif len(curves):
            _record_cd(row, "edge", curves, gt_curves)
    if method == "joint":
        confidence = np.asarray(prediction.get("pred_edge_confidence", []))
        row["confidence_status"] = (
            "ok" if curves.ndim == 3 and confidence.shape == (len(curves),)
            and np.issubdtype(confidence.dtype, np.number) and np.isfinite(confidence).all()
            and ((confidence >= 0) & (confidence <= 1)).all() else "invalid_or_missing_confidence"
        )
    if faces.ndim == 4 and curves.ndim == 3:
        pairs = _internal_topology(row, prediction, curves, len(faces), method)
        if mode == "posterior_mean":
            if len(faces) != len(gt_faces):
                row["topology_status"] = "face_count_mismatch"
            elif pairs is not None:
                row.update(pair_scores(pairs, {tuple(map(int, pair)) for pair in gt_pairs}))
                row["topology_status"] = row["internal_topology_status"]
                row["unique_pair_count_error"] = abs(len(pairs) - len(gt_curves))
                if method == "joint":
                    row["joint_query_count_error"] = abs(len(curves) - len(gt_curves))
            else:
                row["topology_status"] = row["internal_topology_status"]
    geometry_ok = row["face_status"] == "ok" and row["edge_status"] in {"ok", "empty_both"}
    topology_ok = row["internal_topology_status"] == "ok"
    correspondence_ok = mode != "posterior_mean" or row["face_mae_status"] == "ok"
    confidence_ok = row.get("confidence_status", "ok") == "ok"
    row["status"] = "ok" if geometry_ok and topology_ok and correspondence_ok and confidence_ok else "partial_or_invalid"
    return row


def _load_prediction(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in (
            "representation", "pred_face", "pred_edge", "pred_edge_face_connectivity",
            "pred_edge_confidence", "input_latent_sha256",
        ) if key in source}


def evaluate_sample(model_id: str, *, data_root: Path, prediction_root: Path, mode: str, candidate_method: str = "joint", baseline_method: str = "legacy") -> list[dict]:
    methods = comparison_methods(candidate_method, baseline_method)
    rows = []
    try:
        with np.load(data_root / model_id / "data.npz", allow_pickle=False) as source:
            gt_faces = source["sample_points_faces"]
            targets = build_shared_boundary_targets(
                gt_faces, source["sample_points_lines"], source["edge_face_connectivity"],
                sample_id=model_id, reciprocal_tolerance=0.01,
                closed_tolerance=1e-5, allow_closed_phase=True,
            )
        gt_faces = gt_faces[..., :3]
        if gt_faces.shape[1:] != (16, 16, 3) or targets["curves"].shape[1:] != (16, 3):
            raise ValueError("GT must preserve the frozen 16x16 face / 16-point curve grid")
    except Exception as exc:
        return [{"model_id": model_id, "method": method, "mode": mode, "status": "failed",
                 "gt_status": "invalid_or_missing_gt", "prediction_status": "not_evaluated",
                 "error": f"{type(exc).__name__}: {exc}", "shared_input_status": "not_verified"}
                for method in methods]
    for method in methods:
        path = prediction_root / method / model_id / "data.npz"
        base = {"model_id": model_id, "method": method, "mode": mode, "prediction_file": str(path),
                "prediction_file_exists": path.is_file(), "error": ""}
        if not path.is_file():
            base.update(status="failed", gt_status="ok", prediction_status="missing_prediction")
        else:
            try:
                arrays = _load_prediction(path)
                base.update(evaluate_prediction(arrays, gt_faces, targets["curves"], targets["face_pairs"],
                                                method=method, mode=mode))
            except Exception as exc:
                base.update(status="failed", gt_status="ok", prediction_status="unreadable_or_invalid_prediction",
                            error=f"{type(exc).__name__}: {exc}")
        rows.append(base)
    digests = [row.get("input_latent_sha256") for row in rows]
    both_loaded = all(row["prediction_status"] == "ok" for row in rows)
    if both_loaded and all(digests):
        shared_status = "verified" if digests[0] == digests[1] else "mismatch"
    else:
        shared_status = "missing_digest" if both_loaded else "not_verified"
    for row in rows:
        row["shared_input_status"] = shared_status
    return rows


def statistics(values: list[float]) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "mean": float(values.mean()) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "p90": float(np.percentile(values, 90)) if len(values) else None}


def _finite(row: dict, key: str) -> bool:
    return row.get(key) is not None and bool(np.isfinite(row[key]))


def _micro_topology(rows: list[dict]) -> dict:
    eligible = [row for row in rows if _finite(row, "topology_f1")]
    if not eligible:
        return {"objects": 0, "precision": None, "recall": None, "f1": None}
    tp, fp, fn = (sum(row[key] for row in eligible) for key in ("topology_tp", "topology_fp", "topology_fn"))
    return {"objects": len(eligible), "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 1.0,
            "recall": tp / (tp + fn) if tp + fn else 1.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0}


def summarize(rows: list[dict], ids: list[str], *, mode: str, original_count: int = ORIGINAL_COUNT,
              candidate_method: str = "joint", baseline_method: str = "legacy") -> dict:
    methods = comparison_methods(candidate_method, baseline_method)
    if len(ids) > original_count:
        raise ValueError("Intended count cannot exceed the original population")
    summary = {"mode": mode, "baseline_method": baseline_method, "candidate_method": candidate_method, "intended_count": len(ids), "original_count": original_count,
               "contract_unsupported_count": original_count - len(ids), "methods": {}, "paired": {},
               "diagnostic_convention": "legacy_* columns describe the directed-half-edge representation used by legacy, intersection, local and conditioned; joint_* columns apply only to shared-boundary queries.",
               "limitations": LIMITATIONS,
               "empty_pair_convention": "both empty: P=R=F1=1; prediction-only empty: P=1,R=0,F1=0; GT-only empty: P=0,R=1,F1=0"}
    lookup = {(row["method"], row["model_id"]): row for row in rows}
    if len(lookup) != len(rows) or set(lookup) != {(method, item) for method in methods for item in ids}:
        raise ValueError("Exactly one metric row per method and intended ID is required")
    for method in methods:
        group = [lookup[method, item] for item in ids]
        counts = {
            "files_present": sum(bool(row.get("prediction_file_exists")) for row in group),
            "raw_schema_and_metrics_ok": sum(row["status"] == "ok" for row in group),
            "face_geometry_computable": sum(_finite(row, "face_cd_sq") for row in group),
            "edge_geometry_computable": sum(_finite(row, "edge_cd_sq") for row in group),
            "aligned_face_mae_computable": sum(_finite(row, "face_mae") for row in group),
            "topology_scores_computable": sum(_finite(row, "topology_f1") for row in group),
        }
        summary["methods"][method] = {
            "coverage": {name: {"count": count, "intended_fraction": count / len(ids),
                                "original_fraction": count / original_count} for name, count in counts.items()},
            "status_counts": dict(Counter(row["status"] for row in group)),
            "stage_status_counts": {key: dict(Counter(row.get(key, "not_computable") for row in group))
                                    for key in ("gt_status", "prediction_status", "face_status", "edge_status",
                                                "face_mae_status", "internal_topology_status", "topology_status")},
            "metrics": {key: statistics([row[key] for row in group if _finite(row, key)]) for key in METRICS},
            "topology_micro": _micro_topology(group),
        }
    shared = Counter(lookup[baseline_method, item]["shared_input_status"] for item in ids)
    summary["shared_input_status_counts"] = dict(shared)
    summary["protocol_valid"] = not (shared["mismatch"] or shared["missing_digest"])
    for key in PAIRED_METRICS:
        matched = [(lookup[baseline_method, item], lookup[candidate_method, item]) for item in ids
                   if lookup[baseline_method, item]["shared_input_status"] == "verified"
                   and _finite(lookup[baseline_method, item], key) and _finite(lookup[candidate_method, item], key)]
        old = [first[key] for first, _ in matched]
        new = [second[key] for _, second in matched]
        differences = np.asarray(new) - np.asarray(old)
        direction = "higher" if key in TOPOLOGY_METRICS else "lower"
        wins = differences > 1e-12 if direction == "higher" else differences < -1e-12
        summary["paired"][key] = {
            "common_count": len(matched), baseline_method: statistics(old), candidate_method: statistics(new),
            f"{candidate_method}_minus_{baseline_method}": statistics(differences), "better_direction": direction,
            f"{candidate_method}_win_fraction": float(wins.mean()) if len(matched) else None,
            "tie_fraction": float((np.abs(differences) <= 1e-12).mean()) if len(matched) else None,
            "tie_absolute_tolerance": 1e-12,
        }
    shared_ids = [item for item in ids if lookup[baseline_method, item]["shared_input_status"] == "verified"
                  and all(_finite(lookup[method, item], "topology_f1") for method in methods)]
    summary["paired_topology_micro"] = {method: _micro_topology([lookup[method, item] for item in shared_ids]) for method in methods}
    return summary


def read_frozen_ids(path: Path) -> list[str]:
    raw = path.read_bytes()
    ids = [line.strip() for line in raw.decode().splitlines() if line.strip()]
    if not ids or len(set(ids)) != len(ids) or any(Path(item).name != item or item in {".", ".."} for item in ids):
        raise ValueError("Split must contain nonempty, unique individual model IDs")
    digest = hashlib.sha256(raw).hexdigest()
    if len(ids) != ACCEPTED_COUNT or digest != ACCEPTED_SHA256:
        raise ValueError(f"Frozen comparison requires the audited {ACCEPTED_COUNT}-ID list with SHA256 {ACCEPTED_SHA256}; got {len(ids)} IDs / {digest}")
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-list", type=Path, required=True)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--baseline-method", choices=BASELINE_METHODS, default="legacy")
    parser.add_argument("--candidate-method", choices=CANDIDATE_METHODS, default="joint")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    ids = read_frozen_ids(args.split_list)
    output = args.comparison_root / "raw_metrics" / args.mode
    if (output / "per_object.csv").exists() or (output / "summary.json").exists():
        parser.error("Metrics already exist; use a fresh comparison root to prevent mixing protocols")
    prediction_root = args.comparison_root / "raw_predictions" / args.mode
    evaluate = partial(evaluate_sample, data_root=args.data_root, prediction_root=prediction_root, mode=args.mode, candidate_method=args.candidate_method, baseline_method=args.baseline_method)
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, pair in enumerate(pool.map(evaluate, ids), start=1):
            rows.extend(pair)
            if index % 250 == 0:
                print(f"Scored {index}/{len(ids)} objects", flush=True)
    summary = summarize(rows, ids, mode=args.mode, candidate_method=args.candidate_method, baseline_method=args.baseline_method)
    summary.update(
        created_at_utc=datetime.now(timezone.utc).isoformat(), workers=args.workers,
        data_root=str(args.data_root.resolve()), prediction_root=str(prediction_root.resolve()),
        split_list=str(args.split_list.resolve()), split_sha256=hashlib.sha256(args.split_list.read_bytes()).hexdigest(),
        protocol="comparison_20261011_v1_raw", reciprocal_tolerance=0.01, closed_tolerance=1e-5,
        allow_closed_phase=True, gt_source="data_root/data.npz via build_shared_boundary_targets, never prediction GT fields",
    )
    # Validate the complete summary before publishing either output artifact.
    summary_json = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    output.mkdir(parents=True, exist_ok=True)
    preferred = ["model_id", "method", "mode", "status", "error"]
    fields = preferred + sorted(set().union(*(row.keys() for row in rows)) - set(preferred))
    with (output / "per_object.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(summary_json, encoding="utf-8")
    print(json.dumps({"output": str(output), "intended_count": len(ids), "original_count": ORIGINAL_COUNT,
                      "protocol_valid": summary["protocol_valid"], "shared_input_status_counts": summary["shared_input_status_counts"]}), flush=True)
    return int(not summary["protocol_valid"])


if __name__ == "__main__":
    raise SystemExit(main())
