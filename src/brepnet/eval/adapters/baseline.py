"""Adapters for legacy baseline reconstruction formats.

这些函数只负责把不同 baseline 的输出读成 condition metric 需要的统一结构：

- face point sets
- edge point sets
- vertex point sets
- face-edge topology
- edge-vertex topology

指标计算本身不放在这里。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from src.brepnet.eval.baseline_preprocess.data_processor import ComplexGenProcessor, NVDNetProcessor


def fallback_geometry():
    return (
        [np.zeros((1, 6), dtype=np.float32)],
        [np.zeros((1, 6), dtype=np.float32)],
        [np.zeros((1, 3), dtype=np.float32)],
        {},
        {},
    )


def get_model_normalize(points_with_normal: np.ndarray, to_unit_sphere: bool = False):
    del to_unit_sphere
    assert len(points_with_normal.shape) == 2 and points_with_normal.shape[1] == 3
    points = points_with_normal[:, :3]
    center = (points.max(axis=0) + points.min(axis=0)) / 2.0
    scale = (points.max(axis=0) - points.min(axis=0)).max()
    return center, scale


def load_point2cad_reconstruction(
    eval_root: str | Path,
    gt_root: str | Path,
    folder_name: str,
    v_num_per_m: int = 100,
):
    del gt_root
    eval_root = Path(eval_root)
    sample_dir = eval_root / folder_name

    mesh_path = sample_dir / "clipped/mesh_transformed.ply"
    if not mesh_path.exists():
        raise FileNotFoundError(f"Missing mesh_transformed for {folder_name}")

    mesh = trimesh.load(mesh_path)
    color = np.stack(
        (
            [item[1] for item in mesh.metadata["_ply_raw"]["face"]["data"]],
            [item[2] for item in mesh.metadata["_ply_raw"]["face"]["data"]],
            [item[3] for item in mesh.metadata["_ply_raw"]["face"]["data"]],
        ),
        axis=1,
    )
    color_file = Path(__file__).resolve().parents[1] / "point2cad_color.txt"
    color_map = [list(map(int, item.strip().split(" "))) for item in color_file.read_text().splitlines()]
    index = np.asarray([color_map.index(item.tolist()) for item in color])

    recon_face_points = [None] * (index.max() + 1)
    for face_id in range(index.max() + 1):
        item_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[index == face_id])
        num_samples = min(max(int(item_mesh.area * v_num_per_m * v_num_per_m), 5), 10000)
        pc_item, id_face = trimesh.sample.sample_surface(item_mesh, num_samples)
        recon_face_points[face_id] = np.concatenate((pc_item, item_mesh.face_normals[id_face]), axis=1)

    curve_file = sample_dir / "clipped/curve_points.xyzc"
    if not curve_file.exists():
        raise FileNotFoundError(f"Missing curve_points for {folder_name}")
    curve_points = np.asarray([list(map(float, item.strip().split(" "))) for item in curve_file.read_text().splitlines()])
    recon_edge_points = [
        curve_points[curve_points[:, 3] == edge_id][:, :3]
        for edge_id in range(int(curve_points.max(axis=0)[3]) + 1)
    ]

    corners = sample_dir / "clipped/remove_duplicates_corners.ply"
    if corners.exists():
        recon_vertex_points = trimesh.load(corners).vertices[:, None]
    else:
        recon_vertex_points = np.asarray((0, 0, 0), dtype=np.float32)[None, None]

    recon_face_edge: dict[int, list[int]] = {}
    recon_edge_vertex: dict[int, list[int]] = {}
    ev_mode = False
    for line in (sample_dir / "topo/topo_fix.txt").read_text().splitlines():
        items = line.strip().split(" ")
        if items[0] == "EV":
            ev_mode = True
            continue
        if len(items) == 1:
            continue
        if ev_mode:
            recon_edge_vertex[int(items[0])] = list(map(int, items[1:]))
        else:
            recon_face_edge[int(items[0])] = list(map(int, items[1:]))

    return recon_face_points, recon_edge_points, recon_vertex_points, recon_face_edge, recon_edge_vertex


def load_complexgen_reconstruction(
    eval_root: str | Path,
    gt_root: str | Path,
    folder_name: str,
    v_num_per_m: int = 100,
):
    try:
        eval_root = Path(eval_root)
        gt_root = Path(gt_root)
        complex_data = ComplexGenProcessor(eval_root / folder_name / f"{folder_name}_geom_refine.json", build_brep=True)
        gt_mesh = trimesh.load(gt_root / folder_name / "mesh.ply")
        center, scale = get_model_normalize(np.array(gt_mesh.vertices))
        _, recon_face_points, _, recon_edge_points, _, recon_vertex_points = complex_data.get_data(v_num_per_m)
        if len(recon_face_points.shape) == 4:
            recon_face_points = recon_face_points.reshape(
                recon_face_points.shape[0],
                -1,
                recon_face_points.shape[-1],
            )
        recon_vertex_points = recon_vertex_points * scale + center
        for points in recon_face_points:
            points[:, :3] = points[:, :3] * scale + center
        for points in recon_edge_points:
            points[:, :3] = points[:, :3] * scale + center
        return recon_face_points, recon_edge_points, recon_vertex_points, complex_data.FaceEdge, complex_data.EdgeVertex
    except Exception:
        return fallback_geometry()


def load_nvdnet_reconstruction(
    eval_root: str | Path,
    gt_root: str | Path,
    folder_name: str,
    v_num_per_m: int = 100,
):
    del gt_root
    try:
        nvdnet_data = NVDNetProcessor(Path(eval_root) / folder_name, build_brep=False)
        _, recon_face_points, _, recon_edge_points, _, recon_vertex_points = nvdnet_data.get_data(v_num_per_m)
        return recon_face_points, recon_edge_points, recon_vertex_points, nvdnet_data.FaceEdge, nvdnet_data.EdgeVertex
    except Exception:
        return fallback_geometry()


def load_baseline_reconstruction(
    eval_root: str | Path,
    gt_root: str | Path,
    folder_name: str,
    baseline: str,
    v_num_per_m: int = 100,
) -> tuple[Any, Any, Any, dict[int, list[int]], dict[int, list[int]]]:
    if baseline == "point2cad":
        return load_point2cad_reconstruction(eval_root, gt_root, folder_name, v_num_per_m)
    if baseline == "complexgen":
        return load_complexgen_reconstruction(eval_root, gt_root, folder_name, v_num_per_m)
    if baseline == "nvdnet":
        return load_nvdnet_reconstruction(eval_root, gt_root, folder_name, v_num_per_m)
    raise ValueError(f"Unknown baseline adapter: {baseline}")
