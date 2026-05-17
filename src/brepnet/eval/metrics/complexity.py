"""Complexity metric.

职责：只描述生成 solid 自身的复杂度，不比较 GT。当前统计 face/edge/vertex
数量、边图环复杂度和曲面平均曲率。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_VERTEX
from OCC.Core.TopExp import TopExp_Explorer

from shared.occ_utils import get_primitives
from src.brepnet.eval.metrics.validity import check_step_valid_solid
from src.brepnet.eval.protocol import EvalSample, first_existing_file


def remove_outliers_zscore(data: list[float], threshold: float = 10) -> list[float]:
    if len(data) == 0 or sum(data) == 0:
        return data
    mean = np.mean(data)
    std_dev = np.std(data)
    if std_dev == 0:
        return data
    return [x for x in data if abs((x - mean) / std_dev) <= threshold]


def extract_edges_and_vertices(shape: Any) -> tuple[dict[tuple[float, float, float], int], list[tuple[int, int]]]:
    explorer_edges = TopExp_Explorer(shape, TopAbs_EDGE)
    vertex_map: dict[tuple[float, float, float], int] = {}
    edges: list[tuple[int, int]] = []

    while explorer_edges.More():
        edge = explorer_edges.Current()
        vertices_on_edge: list[int] = []
        vertex_explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
        while vertex_explorer.More():
            point = BRep_Tool.Pnt(vertex_explorer.Current())
            coord = (round(point.X(), 6), round(point.Y(), 6), round(point.Z(), 6))
            if coord not in vertex_map:
                vertex_map[coord] = len(vertex_map)
            vertices_on_edge.append(vertex_map[coord])
            vertex_explorer.Next()
        if len(vertices_on_edge) == 2:
            edges.append((vertices_on_edge[0], vertices_on_edge[1]))
        explorer_edges.Next()
    return vertex_map, edges


def calculate_cyclomatic_complexity(vertex_map: dict, edges: list[tuple[int, int]]) -> int:
    graph = nx.Graph()
    graph.add_nodes_from(vertex_map.values())
    graph.add_edges_from(edges)
    if graph.number_of_nodes() == 0:
        return 0
    return graph.number_of_edges() - graph.number_of_nodes() + 2 * nx.number_connected_components(graph)


def evaluate_step_file(step_file_path: str | Path) -> dict[str, float | int | bool] | None:
    is_valid, shape = check_step_valid_solid(step_file_path, return_shape=True)
    if not is_valid or shape is None:
        return None

    vertex_map, edges = extract_edges_and_vertices(shape)
    face_list = get_primitives(shape, TopAbs_FACE)
    curvature_samples: list[float] = []
    for face in face_list:
        surf_adaptor = BRepAdaptor_Surface(face)
        u_min, u_max = surf_adaptor.FirstUParameter(), surf_adaptor.LastUParameter()
        v_min, v_max = surf_adaptor.FirstVParameter(), surf_adaptor.LastVParameter()
        for u in np.linspace(u_min, u_max, 4):
            for v in np.linspace(v_min, v_max, 4):
                props = BRepLProp_SLProps(surf_adaptor, u, v, 2, 1e-6)
                if props.IsCurvatureDefined():
                    curvature_samples.append(abs(props.MeanCurvature()))

    curvature_samples = remove_outliers_zscore(curvature_samples)
    mean_curvature = float(np.mean(curvature_samples)) if curvature_samples else float("nan")
    if len(face_list) == 0 or np.isnan(mean_curvature):
        return None

    return {
        "is_valid_solid": True,
        "num_faces": int(len(face_list)),
        "num_edges": int(len(edges)),
        "num_vertices": int(len(vertex_map)),
        "cyclomatic_complexity": int(calculate_cyclomatic_complexity(vertex_map, edges)),
        "mean_curvature": mean_curvature,
    }


def evaluate_sample(sample: EvalSample) -> dict[str, float | int | bool] | None:
    step_file = first_existing_file([sample.pred_dir / "recon_brep.step", *sorted(sample.pred_dir.glob("*.step"))])
    if step_file is None:
        return None
    return evaluate_step_file(step_file)
