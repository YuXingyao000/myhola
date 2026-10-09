"""Visibility-filtered condition metrics for single-view B-Rep generation."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import numpy as np

from src.brepnet.data.rotations import cube24_rotation_matrix
from src.brepnet.eval.io import save_npz_result, write_csv, write_json, write_text
from src.brepnet.eval.metrics import condition
from src.brepnet.eval.protocol import EvalSample, error_path


DEFAULT_VISIBILITY_ROOT = Path("/mnt/d/data/deepcad_v7_visibility")


def visible_result_path(sample: EvalSample) -> Path:
    return sample.pred_dir / "eval_visible_condition.npz"


def load_visible_result(sample_dir: str | Path) -> dict[str, Any] | None:
    path = Path(sample_dir) / "eval_visible_condition.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    if "results" not in data:
        return None
    return data["results"].item()


def load_visible_indices(
    visibility_root: str | Path,
    folder_name: str,
    view_id: int,
    threshold: float,
    num_gt_faces: int,
) -> tuple[np.ndarray, np.ndarray]:
    visibility_path = Path(visibility_root) / folder_name / f"view_{int(view_id):03d}" / "visibility.npz"
    with np.load(visibility_path) as data:
        visible_fraction = np.asarray(data["visible_fraction"], dtype=np.float32)
    if visible_fraction.shape[0] < num_gt_faces:
        raise ValueError(
            f"{visibility_path} has {visible_fraction.shape[0]} faces, expected at least {num_gt_faces}"
        )
    visible_fraction = np.clip(visible_fraction[:num_gt_faces], 0.0, 1.0)
    visible_indices = np.where(visible_fraction >= threshold)[0].astype(np.int64)
    if visible_indices.size == 0:
        raise ValueError(
            f"No visible GT faces for {folder_name} at threshold={threshold}; "
            "lower --visible-threshold or inspect visibility cache."
        )
    return visible_indices, visible_fraction


def evaluate_visible_condition(
    eval_root: str | Path,
    gt_root: str | Path,
    folder_name: str,
    *,
    visibility_root: str | Path = DEFAULT_VISIBILITY_ROOT,
    view_id: int = 0,
    visible_threshold: float = 0.05,
    v_num_per_m: int = 100,
) -> dict[str, Any]:
    eval_data = condition.prepare_eval_data(
        Path(eval_root),
        Path(gt_root),
        folder_name,
        v_num_per_m=v_num_per_m,
    )

    rotation_matrix = cube24_rotation_matrix(0)
    gt_face_points = condition.transform_point_sets(eval_data["gt_face_points"], rotation_matrix)
    recon_face_points = eval_data["recon_face_points"]
    visible_indices, visible_fraction = load_visible_indices(
        visibility_root,
        folder_name,
        view_id,
        visible_threshold,
        len(gt_face_points),
    )
    visible_gt_face_points = [gt_face_points[int(idx)] for idx in visible_indices]

    face_acc_cd, face_com_cd, face_cd = condition.get_chamfer(recon_face_points, visible_gt_face_points)
    id_recon_gt_face, id_gt_recon_face, cost_face = condition.get_match_ids(
        recon_face_points,
        visible_gt_face_points,
    )
    face_fscore, face_pre, face_rec = condition.get_detection(
        id_recon_gt_face,
        id_gt_recon_face,
        cost_face,
    )

    results = {
        "visible_face_cd": face_cd,
        "visible_face_recon_to_gt_cd": face_acc_cd,
        "visible_face_gt_to_recon_cd": face_com_cd,
        "visible_face_fscore": face_fscore,
        "visible_face_precision": face_pre,
        "visible_face_recall": face_rec,
        "visible_num_gt_face": int(visible_indices.size),
        "num_gt_face": int(len(gt_face_points)),
        "num_recon_face": int(len(recon_face_points)),
        "visible_threshold": float(visible_threshold),
        "visibility_view_id": int(view_id),
        "visible_fraction_mean": float(np.mean(visible_fraction)),
        "visible_fraction_selected_mean": float(np.mean(visible_fraction[visible_indices])),
    }
    if not condition.metrics_are_finite(results):
        raise RuntimeError(f"NaN/Inf visible metrics for sample {folder_name}")
    return results


def eval_one(
    eval_root: str | Path,
    gt_root: str | Path,
    folder_name: str,
    *,
    visibility_root: str | Path = DEFAULT_VISIBILITY_ROOT,
    view_id: int = 0,
    visible_threshold: float = 0.05,
    v_num_per_m: int = 100,
) -> dict[str, Any]:
    results = evaluate_visible_condition(
        eval_root,
        gt_root,
        folder_name,
        visibility_root=visibility_root,
        view_id=view_id,
        visible_threshold=visible_threshold,
        v_num_per_m=v_num_per_m,
    )
    save_npz_result(visible_result_path(EvalSample(name=folder_name, pred_dir=Path(eval_root) / folder_name)), results)
    return results


def eval_one_with_try(*args, **kwargs):
    try:
        return eval_one(*args, **kwargs)
    except Exception:
        eval_root, _, folder_name = args[:3]
        write_text(error_path(EvalSample(name=folder_name, pred_dir=Path(eval_root) / folder_name)), traceback.format_exc())
        return None


def compute_statistics(eval_root: str | Path, listfile: str | Path | None = None) -> dict[str, Any]:
    eval_root = Path(eval_root)
    if listfile:
        names = [item.strip() for item in Path(listfile).read_text().splitlines() if item.strip()]
        folders = sorted(set(names) & {path.name for path in eval_root.iterdir() if path.is_dir()})
    else:
        folders = sorted(path.name for path in eval_root.iterdir() if path.is_dir())

    rows = []
    exception_folders = []
    for folder_name in folders:
        item = load_visible_result(eval_root / folder_name)
        if item is None:
            exception_folders.append(folder_name)
            continue
        rows.append((folder_name, item))

    summary: dict[str, Any] = {
        "num_eval": len(rows),
        "num_total": len(folders),
        "exception_folders": exception_folders,
    }
    for _, item in rows:
        for key, value in item.items():
            if isinstance(value, (str, np.ndarray, list, dict)):
                continue
            summary.setdefault(f"{key}_values", []).append(value)

    for key in list(summary.keys()):
        if not key.endswith("_values"):
            continue
        metric_name = key[: -len("_values")]
        values = summary.pop(key)
        if values:
            summary[f"{metric_name}_mean"] = float(np.mean(values))
            summary[f"{metric_name}_median"] = float(np.median(values))

    csv_rows = [{"prefix": name, **item} for name, item in rows]
    write_csv(eval_root / "eval_visible_condition.csv", csv_rows)
    write_json(eval_root / "eval_visible_condition_summary.json", summary)
    return summary
