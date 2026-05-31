"""Conditioned BREP reconstruction metric.

职责：比较预测 BREP 与 GT BREP 的几何和拓扑一致性。该指标会分别评估
face、edge、vertex 的 Chamfer/precision/recall/F-score，并评估 FE
(face-edge) 和 EV (edge-vertex) 拓扑匹配。

默认协议固定为 identity-first cube24 的 0 号旋转，也就是 identity GT。
评估运行时不再搜索或接收旧 64/Euler 旋转编号。
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh
from chamferdist import ChamferDistance
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_VERTEX

from shared.occ_utils import get_curve_length, get_points_along_edge, get_primitives, get_triangulations
from src.brepnet.data.rotations import cube24_rotation_matrix
from src.brepnet.eval.adapters.baseline import fallback_geometry, get_model_normalize, load_baseline_reconstruction
from src.brepnet.eval.io import save_npz_result, write_text
from src.brepnet.eval.metrics.validity import check_step_valid_soild
from src.brepnet.eval.protocol import EvalSample, condition_result_path, error_path


CHAMFER_DEVICE: torch.device | None = None
CHAMFER_DISTANCE: ChamferDistance | None = None


def is_vertex_close(p1: np.ndarray, p2: np.ndarray, tol: float = 1e-3) -> bool:
    return np.linalg.norm(np.array(p1) - np.array(p2)) < tol


def extract_shape_data(v_shape: Any, v_num_per_m: int = 100):
    faces, face_points, edges, edge_points, vertices, vertex_points = [], [], [], [], [], []
    for face in get_primitives(v_shape, TopAbs_FACE, v_remove_half=True):
        try:
            v, f = get_triangulations(face, 0.1, 0.1)
            if len(f) == 0:
                continue
        except Exception:
            continue
        mesh_item = trimesh.Trimesh(vertices=v, faces=f)
        num_samples = min(max(int(v_num_per_m * v_num_per_m * mesh_item.area), 5), 10000)
        pc_item, id_face = trimesh.sample.sample_surface(mesh_item, num_samples)
        normals = mesh_item.face_normals[id_face]
        faces.append(face)
        face_points.append(np.concatenate((pc_item, normals), axis=1))

    for edge in get_primitives(v_shape, TopAbs_EDGE, v_remove_half=True):
        num_samples = min(max(int(v_num_per_m * get_curve_length(edge)), 5), 10000)
        edges.append(edge)
        edge_points.append(get_points_along_edge(edge, num_samples))

    for vertex in get_primitives(v_shape, TopAbs_VERTEX, v_remove_half=True):
        vertices.append(vertex)
        vertex_points.append(np.asarray([BRep_Tool.Pnt(vertex).Coord()]))

    if len(vertex_points) == 0:
        vertex_points = [np.zeros((1, 3), dtype=np.float32)]
    vertex_points = np.stack(vertex_points, axis=0)
    return faces, face_points, edges, edge_points, vertices, vertex_points


def get_data(v_shape: Any, v_num_per_m: int = 100):
    """Backward-compatible name used by older scripts."""
    return extract_shape_data(v_shape, v_num_per_m=v_num_per_m)


def get_chamfer_components() -> tuple[torch.device, ChamferDistance]:
    global CHAMFER_DEVICE, CHAMFER_DISTANCE
    target_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if CHAMFER_DEVICE is None or CHAMFER_DISTANCE is None or CHAMFER_DEVICE != target_device:
        CHAMFER_DEVICE = target_device
        CHAMFER_DISTANCE = ChamferDistance()
    return CHAMFER_DEVICE, CHAMFER_DISTANCE


def get_chamfer(v_recon_points: list[np.ndarray], v_gt_points: list[np.ndarray]) -> tuple[float, float, float]:
    device, chamfer_distance = get_chamfer_components()
    recon_fp = torch.from_numpy(np.concatenate(v_recon_points, axis=0)).float().to(device)[:, :3]
    gt_fp = torch.from_numpy(np.concatenate(v_gt_points, axis=0)).float().to(device)[:, :3]
    fp_acc_cd = chamfer_distance(
        recon_fp.unsqueeze(0), gt_fp.unsqueeze(0), bidirectional=False, point_reduction="mean"
    ).cpu().item()
    fp_com_cd = chamfer_distance(
        gt_fp.unsqueeze(0), recon_fp.unsqueeze(0), bidirectional=False, point_reduction="mean"
    ).cpu().item()
    return fp_acc_cd, fp_com_cd, fp_acc_cd + fp_com_cd


def get_match_ids(v_recon_points: list[np.ndarray], v_gt_points: list[np.ndarray]):
    from scipy.optimize import linear_sum_assignment

    cost = np.zeros([len(v_recon_points), len(v_gt_points)])
    for i in range(cost.shape[0]):
        for j in range(cost.shape[1]):
            _, _, cost[i][j] = get_chamfer(
                [v_recon_points[i][..., :3]],
                [v_gt_points[j][..., :3]],
            )
    recon_indices, recon_to_gt = linear_sum_assignment(cost)
    result_recon2gt = -1 * np.ones(len(v_recon_points), dtype=np.int32)
    result_gt2recon = -1 * np.ones(len(v_gt_points), dtype=np.int32)
    result_recon2gt[recon_indices] = recon_to_gt
    result_gt2recon[recon_to_gt] = recon_indices
    return result_recon2gt, result_gt2recon, cost


def get_detection(id_recon_gt, id_gt_recon, cost_matrix, v_threshold: float = 0.1):
    true_positive = 0
    for i in range(len(id_recon_gt)):
        if id_recon_gt[i] != -1 and cost_matrix[i, id_recon_gt[i]] < v_threshold:
            true_positive += 1
    precision = true_positive / (len(id_recon_gt) + 1e-6)
    recall = true_positive / (len(id_gt_recon) + 1e-6)
    return 2 * precision * recall / (precision + recall + 1e-6), precision, recall


def get_topology(faces, edges, vertices):
    recon_face_edge, recon_edge_vertex = {}, {}
    for i_face, face in enumerate(faces):
        face_edge = []
        for edge in get_primitives(face, TopAbs_EDGE):
            face_edge.append(edges.index(edge) if edge in edges else edges.index(edge.Reversed()))
        recon_face_edge[i_face] = list(set(face_edge))

    for i_edge, edge in enumerate(edges):
        edge_vertex = []
        for vertex in get_primitives(edge, TopAbs_VERTEX):
            edge_vertex.append(vertices.index(vertex) if vertex in vertices else vertices.index(vertex.Reversed()))
        recon_edge_vertex[i_edge] = list(set(edge_vertex))
    return recon_face_edge, recon_edge_vertex


def get_topo_detection(recon_face_edge, gt_face_edge, id_recon_gt_face, id_recon_gt_edge):
    positive = 0
    for i_recon_face, edges in recon_face_edge.items():
        if i_recon_face >= len(id_recon_gt_face):
            continue
        i_gt_face = id_recon_gt_face[i_recon_face]
        if i_gt_face == -1:
            continue
        for i_edge in edges:
            if i_edge < len(id_recon_gt_edge) and id_recon_gt_edge[i_edge] in gt_face_edge[i_gt_face]:
                positive += 1
    precision = positive / (sum(len(edges) for edges in recon_face_edge.values()) + 1e-6)
    recall = positive / (sum(len(edges) for edges in gt_face_edge.values()) + 1e-6)
    return 2 * precision * recall / (precision + recall + 1e-6), precision, recall


def get_model_normalize(points_with_normal: np.ndarray, to_unit_sphere: bool = False):
    del to_unit_sphere
    assert len(points_with_normal.shape) == 2 and points_with_normal.shape[1] == 3
    points = points_with_normal[:, :3]
    center = (points.max(axis=0) + points.min(axis=0)) / 2.0
    scale = (points.max(axis=0) - points.min(axis=0)).max()
    return center, scale


def transform_points(points: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    transformed = np.array(points, copy=True)
    transformed[..., :3] = transformed[..., :3] @ rotation_matrix.T
    if transformed.shape[-1] >= 6:
        transformed[..., 3:6] = transformed[..., 3:6] @ rotation_matrix.T
    return transformed


def transform_point_sets(point_sets: list[np.ndarray], rotation_matrix: np.ndarray) -> list[np.ndarray]:
    return [transform_points(points, rotation_matrix) for points in point_sets]


def prepare_eval_data(
    eval_root: Path,
    gt_root: Path,
    folder_name: str,
    is_point2cad: bool = False,
    is_complexgen: bool = False,
    is_nvdnet: bool = False,
    v_num_per_m: int = 100,
) -> dict[str, Any]:
    assert [is_point2cad, is_complexgen, is_nvdnet].count(True) <= 1

    baseline = None
    if is_point2cad:
        baseline = "point2cad"
    elif is_complexgen:
        baseline = "complexgen"
    elif is_nvdnet:
        baseline = "nvdnet"

    if baseline is not None:
        recon_face_points, recon_edge_points, recon_vertex_points, recon_face_edge, recon_edge_vertex = load_baseline_reconstruction(
            eval_root,
            gt_root,
            folder_name,
            baseline,
            v_num_per_m,
        )
    else:
        try:
            step_path = eval_root / folder_name / "recon_brep.step"
            if not step_path.exists():
                raise FileNotFoundError(f"Missing {step_path}")
            _, recon_shape = check_step_valid_soild(step_path, return_shape=True)
            if recon_shape is None:
                raise ValueError(f"Invalid STEP shape for {folder_name}")
            recon_faces, recon_face_points, recon_edges, recon_edge_points, recon_vertices, recon_vertex_points = extract_shape_data(
                recon_shape, v_num_per_m
            )
            recon_face_edge, recon_edge_vertex = get_topology(recon_faces, recon_edges, recon_vertices)
        except Exception:
            recon_face_points, recon_edge_points, recon_vertex_points, recon_face_edge, recon_edge_vertex = fallback_geometry()

    _, gt_shape = check_step_valid_soild(gt_root / folder_name / "normalized_shape.step", return_shape=True)
    if gt_shape is None:
        raise FileNotFoundError(f"Missing or invalid GT STEP for {folder_name}")
    gt_faces, gt_face_points, gt_edges, gt_edge_points, gt_vertices, gt_vertex_points = extract_shape_data(gt_shape, v_num_per_m)
    gt_face_edge, gt_edge_vertex = get_topology(gt_faces, gt_edges, gt_vertices)

    return {
        "recon_face_points": recon_face_points,
        "recon_edge_points": recon_edge_points,
        "recon_vertex_points": recon_vertex_points,
        "recon_face_edge": recon_face_edge,
        "recon_edge_vertex": recon_edge_vertex,
        "gt_face_points": gt_face_points,
        "gt_edge_points": gt_edge_points,
        "gt_vertex_points": gt_vertex_points,
        "gt_face_edge": gt_face_edge,
        "gt_edge_vertex": gt_edge_vertex,
    }


def evaluate_metrics(
    recon_face_points,
    recon_edge_points,
    recon_vertex_points,
    recon_face_edge,
    recon_edge_vertex,
    gt_face_points,
    gt_edge_points,
    gt_vertex_points,
    gt_face_edge,
    gt_edge_vertex,
) -> dict[str, Any]:
    face_acc_cd, face_com_cd, face_cd = get_chamfer(recon_face_points, gt_face_points)
    edge_acc_cd, edge_com_cd, edge_cd = get_chamfer(recon_edge_points, gt_edge_points)
    vertex_acc_cd, vertex_com_cd, vertex_cd = get_chamfer(recon_vertex_points, gt_vertex_points)

    id_recon_gt_face, id_gt_recon_face, cost_face = get_match_ids(recon_face_points, gt_face_points)
    id_recon_gt_edge, id_gt_recon_edge, cost_edge = get_match_ids(recon_edge_points, gt_edge_points)
    id_recon_gt_vertex, id_gt_recon_vertex, cost_vertices = get_match_ids(recon_vertex_points, gt_vertex_points)

    face_fscore, face_pre, face_rec = get_detection(id_recon_gt_face, id_gt_recon_face, cost_face)
    edge_fscore, edge_pre, edge_rec = get_detection(id_recon_gt_edge, id_gt_recon_edge, cost_edge)
    vertex_fscore, vertex_pre, vertex_rec = get_detection(id_recon_gt_vertex, id_gt_recon_vertex, cost_vertices)
    fe_fscore, fe_pre, fe_rec = get_topo_detection(recon_face_edge, gt_face_edge, id_recon_gt_face, id_recon_gt_edge)
    ev_fscore, ev_pre, ev_rec = get_topo_detection(recon_edge_vertex, gt_edge_vertex, id_recon_gt_edge, id_recon_gt_vertex)

    return {
        "face_cd": face_cd,
        "edge_cd": edge_cd,
        "vertex_cd": vertex_cd,
        "face_fscore": face_fscore,
        "edge_fscore": edge_fscore,
        "vertex_fscore": vertex_fscore,
        "fe_fscore": fe_fscore,
        "ev_fscore": ev_fscore,
        "face_acc_cd": face_acc_cd,
        "edge_acc_cd": edge_acc_cd,
        "vertex_acc_cd": vertex_acc_cd,
        "face_com_cd": face_com_cd,
        "edge_com_cd": edge_com_cd,
        "vertex_com_cd": vertex_com_cd,
        "fe_pre": fe_pre,
        "ev_pre": ev_pre,
        "fe_rec": fe_rec,
        "ev_rec": ev_rec,
        "vertex_pre": vertex_pre,
        "edge_pre": edge_pre,
        "face_pre": face_pre,
        "vertex_rec": vertex_rec,
        "edge_rec": edge_rec,
        "face_rec": face_rec,
        "num_recon_face": len(recon_face_points),
        "num_gt_face": len(gt_face_points),
        "num_recon_edge": len(recon_edge_points),
        "num_gt_edge": len(gt_edge_points),
        "num_recon_vertex": len(recon_vertex_points),
        "num_gt_vertex": len(gt_vertex_points),
    }


def metrics_are_finite(results: dict[str, Any]) -> bool:
    for value in results.values():
        if isinstance(value, np.ndarray) and not np.all(np.isfinite(value)):
            return False
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return False
    return True


def build_rotation_summary(rotation_index: int, rotation_matrix: np.ndarray, result: dict[str, Any] | None = None, error: str = ""):
    summary = {
        "rotation_index": rotation_index,
        "rotation_matrix": rotation_matrix.astype(np.float32),
        "valid": result is not None,
        "error": error,
        "face_fscore": float("-inf"),
        "edge_fscore": float("-inf"),
        "vertex_fscore": float("-inf"),
        "face_cd": float("inf"),
        "edge_cd": float("inf"),
        "vertex_cd": float("inf"),
    }
    if result is not None:
        for key in ("face_fscore", "edge_fscore", "vertex_fscore", "face_cd", "edge_cd", "vertex_cd"):
            summary[key] = float(result[key])
    return summary


def evaluate_condition(
    eval_root: str | Path,
    gt_root: str | Path,
    folder_name: str,
    *,
    is_point2cad: bool = False,
    is_complexgen: bool = False,
    is_nvdnet: bool = False,
    v_num_per_m: int = 100,
) -> dict[str, Any]:
    eval_data = prepare_eval_data(
        Path(eval_root),
        Path(gt_root),
        folder_name,
        is_point2cad=is_point2cad,
        is_complexgen=is_complexgen,
        is_nvdnet=is_nvdnet,
        v_num_per_m=v_num_per_m,
    )

    rotation_id = 0
    rotation_matrix = cube24_rotation_matrix(rotation_id)
    results = evaluate_metrics(
        eval_data["recon_face_points"],
        eval_data["recon_edge_points"],
        eval_data["recon_vertex_points"],
        eval_data["recon_face_edge"],
        eval_data["recon_edge_vertex"],
        transform_point_sets(eval_data["gt_face_points"], rotation_matrix),
        transform_point_sets(eval_data["gt_edge_points"], rotation_matrix),
        transform_points(eval_data["gt_vertex_points"], rotation_matrix),
        eval_data["gt_face_edge"],
        eval_data["gt_edge_vertex"],
    )
    if not metrics_are_finite(results):
        raise RuntimeError(f"NaN/Inf metrics for sample {folder_name}")
    results["best_rotation_index"] = rotation_id
    results["best_rotation_matrix"] = rotation_matrix.astype(np.float32)
    results["rotation_scores"] = np.array([build_rotation_summary(rotation_id, rotation_matrix, result=results)], dtype=object)
    results["rotation_search_status"] = "identity_only"
    return results


def save_eval_results(eval_root: str | Path, folder_name: str, results: dict[str, Any], write_legacy_eval: bool = False) -> None:
    sample = EvalSample(name=folder_name, pred_dir=Path(eval_root) / folder_name)
    save_npz_result(condition_result_path(sample, legacy=False), results)
    if write_legacy_eval:
        save_npz_result(condition_result_path(sample, legacy=True), results)


def write_error_marker(eval_root: str | Path, folder_name: str, message: str, write_legacy_error: bool = True) -> None:
    sample = EvalSample(name=folder_name, pred_dir=Path(eval_root) / folder_name)
    write_text(error_path(sample), message)
    if write_legacy_error:
        write_text(sample.pred_dir / "error.txt", message)


def eval_one(
    eval_root: str | Path,
    gt_root: str | Path,
    folder_name: str,
    is_point2cad: bool = False,
    is_complexgen: bool = False,
    is_nvdnet: bool = False,
    v_num_per_m: int = 100,
    write_legacy_eval: bool = False,
) -> dict[str, Any]:
    results = evaluate_condition(
        eval_root,
        gt_root,
        folder_name,
        is_point2cad=is_point2cad,
        is_complexgen=is_complexgen,
        is_nvdnet=is_nvdnet,
        v_num_per_m=v_num_per_m,
    )
    save_eval_results(eval_root, folder_name, results, write_legacy_eval=write_legacy_eval)
    return results


def eval_one_with_try(*args, **kwargs):
    try:
        return eval_one(*args, **kwargs)
    except Exception:
        eval_root, _, folder_name = args[:3]
        write_error_marker(eval_root, folder_name, traceback.format_exc())
        return None


def load_condition_result(sample_dir: str | Path) -> dict[str, Any] | None:
    sample_dir = Path(sample_dir)
    for filename in ("eval_condition.npz", "eval.npz"):
        path = sample_dir / filename
        if path.exists():
            data = np.load(path, allow_pickle=True)
            if "results" not in data:
                continue
            return data["results"].item()
    return None


def compute_statistics(eval_root: str | Path, v_only_valid: bool = False, listfile: str | Path | None = None) -> dict[str, Any]:
    eval_root = Path(eval_root)
    if listfile:
        valid_names = [item.strip() for item in Path(listfile).read_text().splitlines() if item.strip()]
        all_folders = sorted(set(valid_names) & {path.name for path in eval_root.iterdir() if path.is_dir()})
    else:
        all_folders = sorted(path.name for path in eval_root.iterdir() if path.is_dir())

    rows, exception_folders = [], []
    for folder_name in all_folders:
        item = load_condition_result(eval_root / folder_name)
        if item is None:
            exception_folders.append(folder_name)
            continue
        if v_only_valid and not (eval_root / folder_name / "success.txt").exists():
            continue
        rows.append((folder_name, item))

    if not rows:
        summary = {"num_eval": 0, "num_total": len(all_folders), "exception_folders": exception_folders}
        print("No condition results found.")
        return summary

    results: dict[str, list[Any]] = {"prefix": [name for name, _ in rows]}
    for _, item in rows:
        for key, value in item.items():
            if isinstance(value, (str, np.ndarray, list, dict)):
                continue
            results.setdefault(key, []).append(value)

    summary: dict[str, Any] = {
        "num_eval": len(rows),
        "num_total": len(all_folders),
        "exception_folders": exception_folders,
    }
    for key, values in results.items():
        if key == "prefix" or not values:
            continue
        try:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_median"] = float(np.median(values))
        except Exception:
            pass

    print("Number")
    print(f"Vertices: {summary.get('num_recon_vertex_mean')}/{summary.get('num_gt_vertex_mean')}")
    print(f"Edge: {summary.get('num_recon_edge_mean')}/{summary.get('num_gt_edge_mean')}")
    print(f"Face: {summary.get('num_recon_face_mean')}/{summary.get('num_gt_face_mean')}")
    print("Chamfer")
    print(f"Vertices: {summary.get('vertex_cd_mean')}")
    print(f"Edge: {summary.get('edge_cd_mean')}")
    print(f"Face: {summary.get('face_cd_mean')}")
    print("Detection")
    print(f"Vertices: {summary.get('vertex_fscore_mean')}")
    print(f"Edge: {summary.get('edge_fscore_mean')}")
    print(f"Face: {summary.get('face_fscore_mean')}")
    print("Topology")
    print(f"FE: {summary.get('fe_fscore_mean')}")
    print(f"EV: {summary.get('ev_fscore_mean')}")
    print(f"{len(rows)}/{len(all_folders)} are evaluated")

    try:
        import matplotlib.pyplot as plt

        face_chamfer = [item["face_cd"] for _, item in rows if "face_cd" in item]
        if face_chamfer:
            report_dir = eval_root / "reports"
            report_dir.mkdir(exist_ok=True)
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.hist(face_chamfer, bins=50, range=(0, 0.05), density=True, alpha=0.5, color="b", label="Face")
            ax.set_title("Face Chamfer Distance")
            ax.set_xlabel("Chamfer Distance")
            ax.set_ylabel("Density")
            ax.legend()
            fig.savefig(report_dir / "condition_face_chamfer.png", dpi=600)
            plt.close(fig)
    except Exception:
        pass

    return summary
