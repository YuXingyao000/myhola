"""Frozen, sampled-geometry metrics for the UV decoder evaluation.

All geometry arithmetic is float64. CD is the sum of directional squared
nearest-sample means, not a continuous-curve distance. Empty geometry gives
None, including both-empty edge sets. Topology retains the established
``pair_scores`` empty-denominator convention. No prediction is selected by GT.
"""

from __future__ import annotations

import numpy as np

from tools.vae_compare_metrics import pair_scores, squared_chamfer


QUANTILES = (50, 90, 95, 99)


def _geometry(arrays: dict, name: str, trailing_shape: tuple) -> np.ndarray:
    value = np.asarray(arrays[name])
    if value.ndim != len(trailing_shape) + 1 or value.shape[1:] != trailing_shape:
        raise ValueError(f"{name}: expected [N,{trailing_shape}], got {value.shape}")
    if not np.issubdtype(value.dtype, np.number) or np.iscomplexobj(value):
        raise ValueError(f"{name}: expected real numeric geometry")
    if not np.isfinite(value).all():
        raise ValueError(f"{name}: nonfinite geometry")
    return value.astype(np.float64, copy=False)


def _pairs(arrays: dict, name: str, count: int, faces: int) -> set[tuple[int, int]]:
    value = np.asarray(arrays[name])
    if value.shape != (count, 2) or not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"{name}: expected integer [{count},2], got {value.shape}/{value.dtype}")
    if len(value) and ((value < 0).any() or (value >= faces).any()
                       or (value[:, 0] >= value[:, 1]).any()):
        raise ValueError(f"{name}: expected in-range unordered owners i < j")
    result = {tuple(map(int, row)) for row in value}
    if len(result) != count:
        raise ValueError(f"{name}: duplicate owner pair")
    return result


def _cd(row: dict, prefix: str, predicted: np.ndarray, truth: np.ndarray) -> None:
    keys = (f"{prefix}_pred_to_gt_sq", f"{prefix}_gt_to_pred_sq", f"{prefix}_cd_sq")
    if predicted.size and truth.size:
        values = squared_chamfer(predicted.reshape(-1, 3), truth.reshape(-1, 3))
    else:
        values = (None, None, None)
    row.update(zip(keys, values))


def _distribution(row: dict, prefix: str, values: np.ndarray, *, rms: bool = False) -> None:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    row[f"{prefix}_count"] = len(values)
    names = ("mean", "p50", "p90", "p95", "p99", "max") + (("rms",) if rms else ())
    if not len(values):
        row.update({f"{prefix}_{name}": None for name in names})
        return
    row[f"{prefix}_mean"] = float(values.mean())
    row[f"{prefix}_max"] = float(values.max())
    for quantile in QUANTILES:
        row[f"{prefix}_p{quantile}"] = float(np.percentile(values, quantile))
    if rms:
        row[f"{prefix}_rms"] = float(np.sqrt(np.mean(values * values)))


def _bilateral(row: dict, prefix: str, curves: np.ndarray) -> None:
    """Per-point corresponding gap; per-edge orientation/phase-free sampled CD."""
    gap = np.linalg.norm(curves[:, 0] - curves[:, 1], axis=-1)
    _distribution(row, f"{prefix}_gap", gap, rms=True)
    pair_cd = np.array([squared_chamfer(pair[0], pair[1])[2] for pair in curves])
    _distribution(row, f"{prefix}_pair_cd_sq", pair_cd)


def _gt_diagnostics(row: dict, predicted: np.ndarray, truth: np.ndarray) -> None:
    """GT-pair candidates are supplied unthresholded, in GT owner-pair order."""
    count = len(truth)
    row["gt_pair_status"] = "ok" if count else "empty_ground_truth"
    _bilateral(row, "gt_pair", predicted)
    if count:
        forward = np.square(predicted - truth[:, None]).mean(axis=(1, 2, 3))
        reverse = np.square(predicted - truth[:, None, ::-1]).mean(axis=(1, 2, 3))
        mse = np.minimum(forward, reverse)
    else:
        mse = np.empty(0, dtype=np.float64)
    row["gt_pair_curve_xyz_mse"] = float(mse.mean()) if count else None
    _distribution(row, "gt_pair_curve_xyz_mse", mse)

    # Combined point sets and the two individual sides are deliberately separate.
    pooled, side0, side1 = [], [], []
    for pair, target in zip(predicted, truth):
        pooled.append(squared_chamfer(pair.reshape(-1, 3), target)[2])
        side0.append(squared_chamfer(pair[0], target)[2])
        side1.append(squared_chamfer(pair[1], target)[2])
    side0, side1 = np.asarray(side0), np.asarray(side1)
    _distribution(row, "gt_pair_edge_cd_sq", np.asarray(pooled))
    _distribution(row, "gt_pair_edge_side0_cd_sq", side0)
    _distribution(row, "gt_pair_edge_side1_cd_sq", side1)
    _distribution(row, "gt_pair_edge_side_mean_cd_sq", (side0 + side1) / 2)
    _distribution(row, "gt_pair_edge_side_worst_cd_sq", np.maximum(side0, side1))


def measure(arrays: dict) -> dict:
    """Return JSON-compatible scalars; malformed inputs raise a named ValueError.

    Required geometry: pred_face/gt_face [F,16,16,3], gt_edge [R,16,3],
    pred_edge_xyz_by_face [E,2,16,3], diagnostic_gt_pair_xyz [R,2,16,3].
    Required topology: pred_edge_owner_pairs [E,2], gt_edge_owner_pairs [R,2].
    Face order must correspond. The caller records failures and retains the
    intended CAD denominator. NPZ retains arrays for pooled-edge/point analysis.
    """
    representation = np.asarray(arrays.get("representation", "dual_uv_surface_v1"))
    if representation.ndim != 0 or str(representation.item()) != "dual_uv_surface_v1":
        raise ValueError("representation: expected dual_uv_surface_v1")
    faces = _geometry(arrays, "pred_face", (16, 16, 3))
    gt_faces = _geometry(arrays, "gt_face", (16, 16, 3))
    curves = _geometry(arrays, "pred_edge_xyz_by_face", (2, 16, 3))
    gt_curves = _geometry(arrays, "gt_edge", (16, 3))
    diagnostic = _geometry(arrays, "diagnostic_gt_pair_xyz", (2, 16, 3))
    if len(faces) != len(gt_faces) or not len(faces):
        raise ValueError("pred_face/gt_face: expected matching nonempty face counts")
    if len(diagnostic) != len(gt_curves):
        raise ValueError("diagnostic_gt_pair_xyz: expected one candidate per GT edge")
    pred_pairs = _pairs(arrays, "pred_edge_owner_pairs", len(curves), len(faces))
    gt_pairs = _pairs(arrays, "gt_edge_owner_pairs", len(gt_curves), len(gt_faces))
    row = {
        "face_status": "ok", "face_mae_status": "ok", "topology_status": "ok",
        "pred_face_count": len(faces), "gt_face_count": len(gt_faces),
        "active_pair_count": len(curves), "raw_edge_count": 2 * len(curves),
        "gt_boundary_count": len(gt_curves), "unique_pair_count": len(pred_pairs),
        "unique_pair_count_error": abs(len(pred_pairs) - len(gt_pairs)),
        "face_mae": float(np.abs(faces - gt_faces).mean()),
    }
    row.update(pair_scores(pred_pairs, gt_pairs))
    if len(curves) and len(gt_curves):
        row["edge_status"] = "ok"
    elif len(curves):
        row["edge_status"] = "empty_ground_truth"
    elif len(gt_curves):
        row["edge_status"] = "empty_prediction"
    else:
        row["edge_status"] = "empty_both"
    _cd(row, "face", faces, gt_faces)
    _cd(row, "edge", curves, gt_curves)
    _cd(row, "edge_side0", curves[:, 0], gt_curves)
    _cd(row, "edge_side1", curves[:, 1], gt_curves)
    side0, side1 = row["edge_side0_cd_sq"], row["edge_side1_cd_sq"]
    row["edge_side_mean_cd_sq"] = (side0 + side1) / 2 if side0 is not None else None
    row["edge_side_worst_cd_sq"] = max(side0, side1) if side0 is not None else None
    _bilateral(row, "active", curves)
    _gt_diagnostics(row, diagnostic, gt_curves)
    return row
