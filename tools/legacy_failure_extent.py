"""Sampled AABB span proxies for the frozen failure cases; not wall thickness."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.legacy_failure_geometry import atomic_json, digest


def extent_diagnostics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    truth, prediction = np.asarray(truth)[..., :3], np.asarray(prediction)[..., :3]
    if truth.shape != prediction.shape or truth.ndim != 4 or truth.shape[1:] != (16, 16, 3):
        raise ValueError("Expected corresponding [faces,16,16,3] XYZ grids")
    if not len(truth) or not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Nonempty finite grids required")
    truth, prediction = truth.astype(np.float64), prediction.astype(np.float64)
    low, high = truth.min(axis=(0, 1, 2)), truth.max(axis=(0, 1, 2))
    span = high - low
    error = np.abs(prediction - truth).mean(axis=(0, 1, 2))
    mae = float(error.mean())
    minimum, maximum = float(span.min()), float(span.max())
    return {
        "status": "ok", "face_count": len(truth),
        "gt_sampled_aabb_min": low.tolist(), "gt_sampled_aabb_max": high.tolist(),
        "gt_sampled_aabb_span_xyz": span.tolist(),
        "min_axis_span": minimum, "max_axis_span": maximum,
        "min_over_max_span": minimum / maximum if maximum > 0 else None,
        "raw_abs_error_mean_xyz": error.tolist(),
        "raw_abs_error_mean_xyz_over_span": [float(e / s) if s > 0 else None for e, s in zip(error, span)],
        "raw_xyz_mae": mae,
        "raw_xyz_mae_over_min_span": mae / minimum if minimum > 0 else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.root / "diagnostics_extent.json"
    if output.exists():
        raise FileExistsError(output)
    manifest_path = args.root / "selected_cases.json"
    manifest = json.loads(manifest_path.read_text())
    result = {
        "schema": "legacy_failure_sampled_extent_v1", "manifest_sha256": digest(manifest_path),
        "source_sha256": digest(Path(__file__)),
        "definition": "AABB of GT untrimmed 16x16 sampled surface XYZ, in the saved identity coordinate frame. Axis spans are overall extent proxies, NOT wall thickness. Pointwise errors use the unchanged GT face/grid correspondence; no alignment or rescaling.",
        "zero_span_policy": "Ratios with an exactly zero denominator are null; no epsilon is added.",
        "cases": {},
    }
    for case in manifest["cases"]:
        sid = case["id"]
        truth_path = args.data_root / sid / "data.npz"
        prediction_path = args.comparison_root / "raw_predictions/posterior_mean/legacy" / sid / "data.npz"
        try:
            with np.load(truth_path, allow_pickle=False) as truth, np.load(prediction_path, allow_pickle=False) as prediction:
                report = extent_diagnostics(truth["sample_points_faces"], prediction["pred_face"])
        except Exception as exc:
            report = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        report["sources"] = {"gt": {"path": str(truth_path.resolve()), "sha256": digest(truth_path)},
                             "raw": {"path": str(prediction_path.resolve()), "sha256": digest(prediction_path)}}
        result["cases"][sid] = report
    atomic_json(output, result)
    print(output)


if __name__ == "__main__":
    main()
