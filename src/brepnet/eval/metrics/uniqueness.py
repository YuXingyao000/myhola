"""Unique / novel graph metrics.

职责：把每个样本的 face 采样点和 face adjacency 建成 graph，再做图同构比较。
Unique 衡量生成集中互相不重复的比例；Novel 衡量生成结果是否不在训练集中。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import ray
from tqdm import tqdm

from src.brepnet.eval.metrics.point_cloud_set import normalize_pc
from src.brepnet.eval.metrics.validity import check_step_valid_soild, load_data_with_prefix


def real2bit(data: np.ndarray, n_bits: int = 8, min_range: float = -1, max_range: float = 1) -> np.ndarray:
    range_quantize = 2**n_bits - 1
    data_quantize = (data - min_range) * range_quantize / (max_range - min_range)
    return np.clip(data_quantize, a_min=0, a_max=range_quantize).astype(int)


def build_graph(faces: np.ndarray, faces_adj: list[list[int]], n_bit: int = 4) -> nx.Graph:
    faces_bits = faces if n_bit < 0 else real2bit(faces, n_bits=n_bit)
    graph = nx.Graph()
    for face_idx, face_bit in enumerate(faces_bits):
        graph.add_node(face_idx, shape_geometry=face_bit)
    graph.add_edges_from((pair[0], pair[1]) for pair in faces_adj)
    return graph


def is_graph_identical(graph1: nx.Graph, graph2: nx.Graph, atol: float | None = None) -> bool:
    if atol is None:
        return nx.is_isomorphic(
            graph1,
            graph2,
            node_match=lambda n1, n2: np.array_equal(n1["shape_geometry"], n2["shape_geometry"]),
        )
    return nx.is_isomorphic(
        graph1,
        graph2,
        node_match=lambda n1, n2: np.allclose(n1["shape_geometry"], n2["shape_geometry"], atol=atol, rtol=0),
    )


def is_graph_identical_batch(graph_pair_list, atol: float | None = None) -> list[bool]:
    return [is_graph_identical(graph1, graph2, atol=atol) for graph1, graph2 in graph_pair_list]


is_graph_identical_remote = ray.remote(is_graph_identical_batch)


def find_connected_components(matrix: np.ndarray) -> list[list[int]]:
    n = len(matrix)
    visited = [False] * n
    components: list[list[int]] = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        component = []
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            component.append(node)
            for neighbor in range(n):
                if matrix[node][neighbor] and not visited[neighbor]:
                    stack.append(neighbor)
        components.append(component)
    return components


def compute_gen_unique(
    graph_list: list[nx.Graph],
    is_use_ray: bool = False,
    batch_size: int = 100000,
    atol: float | None = None,
) -> tuple[float, np.ndarray]:
    n = len(graph_list)
    unique_graph_idx = list(range(n))
    pair_0, pair_1 = np.triu_indices(n, k=1)
    check_pairs = list(zip(pair_0, pair_1))
    deduplicate_matrix = np.zeros((n, n), dtype=bool)

    if not is_use_ray:
        iterator = ((pair, is_graph_identical(graph_list[pair[0]], graph_list[pair[1]], atol=atol)) for pair in tqdm(check_pairs))
    else:
        ray.init(ignore_reinit_error=True)
        futures = []
        for i in tqdm(range(0, len(check_pairs), batch_size)):
            batch_pairs = check_pairs[i : i + batch_size]
            futures.append(is_graph_identical_remote.remote([(graph_list[a], graph_list[b]) for a, b in batch_pairs], atol))
        iterator_items = []
        for batch_start, result in zip(range(0, len(check_pairs), batch_size), ray.get(futures)):
            iterator_items.extend(zip(check_pairs[batch_start : batch_start + batch_size], result))
        ray.shutdown()
        iterator = iter(iterator_items)

    for (idx1, idx2), is_identical in iterator:
        if not is_identical:
            continue
        if idx2 in unique_graph_idx:
            unique_graph_idx.remove(idx2)
        deduplicate_matrix[idx1, idx2] = True
        deduplicate_matrix[idx2, idx1] = True

    unique_ratio = len(unique_graph_idx) / n if n else 0.0
    return unique_ratio, deduplicate_matrix


def load_data_from_npz(data_npz_file: str | Path) -> tuple[np.ndarray, list[list[int]]]:
    data_npz = np.load(data_npz_file, allow_pickle=True)
    if "face_edge_adj" in data_npz:
        faces = data_npz["pred_face"]
        face_edge_adj = data_npz["face_edge_adj"]
        faces_adj_pair = []
        for face_idx1 in range(face_edge_adj.shape[0]):
            for face_idx2 in range(face_idx1 + 1, face_edge_adj.shape[0]):
                if len(set(face_edge_adj[face_idx1]).intersection(set(face_edge_adj[face_idx2]))) > 0:
                    faces_adj_pair.append([face_idx1, face_idx2])
        return faces, faces_adj_pair

    if "sample_points_faces" in data_npz and "edge_face_connectivity" in data_npz:
        face_points = data_npz["sample_points_faces"]
        edge_face_connectivity = data_npz["edge_face_connectivity"]
    elif "pred_face" in data_npz and "pred_edge_face_connectivity" in data_npz:
        face_points = data_npz["pred_face"]
        edge_face_connectivity = data_npz["pred_edge_face_connectivity"]
    else:
        raise ValueError(f"Invalid data format: {data_npz_file}")

    faces_adj_pair = [[int(face_idx1), int(face_idx2)] for _, face_idx1, face_idx2 in edge_face_connectivity]
    if face_points.shape[-1] != 3:
        face_points = face_points[..., :3]
    src_shape = face_points.shape
    return normalize_pc(face_points.reshape(-1, 3)).reshape(src_shape), faces_adj_pair


def load_and_build_graph(
    data_npz_file_list: list[str],
    gen_post_data_root: str | Path | None = None,
    n_bit: int = 4,
) -> tuple[list[nx.Graph], list[str]]:
    graph_list, prefix_list = [], []
    for data_npz_file in data_npz_file_list:
        folder_name = os.path.basename(os.path.dirname(data_npz_file))
        if gen_post_data_root:
            step_file_list = load_data_with_prefix(Path(gen_post_data_root) / folder_name, ".step")
            if len(step_file_list) == 0 or not check_step_valid_soild(step_file_list[0]):
                continue
        faces, faces_adj_pair = load_data_from_npz(data_npz_file)
        graph_list.append(build_graph(faces, faces_adj_pair, n_bit))
        prefix_list.append(folder_name)
    return graph_list, prefix_list


load_and_build_graph_remote = ray.remote(load_and_build_graph)


def evaluate_unique(
    fake_root: str | Path,
    fake_post: str | Path | None = None,
    *,
    n_bit: int = 4,
    atol: float | None = None,
    use_ray: bool = False,
    load_batch_size: int = 400,
    compute_batch_size: int = 200000,
    min_face: int | None = None,
) -> dict[str, Any]:
    if atol is not None:
        n_bit = -1
    gen_data_npz_file_list = load_data_with_prefix(fake_root, "data.npz")
    if use_ray:
        ray.init(ignore_reinit_error=True)
        futures = [
            load_and_build_graph_remote.remote(gen_data_npz_file_list[i : i + load_batch_size], fake_post, n_bit)
            for i in range(0, len(gen_data_npz_file_list), load_batch_size)
        ]
        gen_graph_list, gen_prefix_list = [], []
        for graph_batch, prefix_batch in ray.get(futures):
            gen_graph_list.extend(graph_batch)
            gen_prefix_list.extend(prefix_batch)
        ray.shutdown()
    else:
        gen_graph_list, gen_prefix_list = load_and_build_graph(gen_data_npz_file_list, fake_post, n_bit)

    if min_face is not None:
        keep = [idx for idx, graph in enumerate(gen_graph_list) if graph.number_of_nodes() >= min_face]
        gen_graph_list = [gen_graph_list[idx] for idx in keep]
        gen_prefix_list = [gen_prefix_list[idx] for idx in keep]

    unique_ratio, deduplicate_matrix = compute_gen_unique(gen_graph_list, use_ray, compute_batch_size, atol=atol)
    components = [
        [gen_prefix_list[idx] for idx in component]
        for component in find_connected_components(deduplicate_matrix)
        if len(component) > 1
    ]
    return {
        "unique_ratio": unique_ratio,
        "num_graphs": len(gen_graph_list),
        "duplicate_components": components,
        "prefix": gen_prefix_list,
    }


def is_graph_identical_list(graph: nx.Graph, graph2_path_list: list[str], n_bit: int = 4, atol: float | None = None) -> bool:
    graph2_list, _ = load_and_build_graph(graph2_path_list, n_bit=n_bit)
    return any(is_graph_identical(graph, graph2, atol=atol) for graph2 in graph2_list)


is_graph_identical_list_remote = ray.remote(is_graph_identical_list)


def evaluate_novel_with_nearest(
    fake_root: str | Path,
    fake_post: str | Path,
    train_root: str | Path,
    *,
    n_bit: int = 4,
    atol: float | None = None,
    use_ray: bool = False,
) -> dict[str, Any]:
    """Evaluate novelty using each generated sample's `nearest.txt` candidates.

    Full generated-vs-training all-pairs graph matching is usually too expensive.
    The historical workflow first writes nearest training candidates under
    `fake_post/sample/nearest.txt`; this function preserves that protocol.
    """
    if atol is not None:
        n_bit = -1
    gen_graph_list, gen_prefix_list = load_and_build_graph(load_data_with_prefix(fake_root, "data.npz"), fake_post, n_bit)
    is_identical = np.zeros(len(gen_graph_list), dtype=bool)

    def nearest_paths(prefix: str) -> list[str]:
        nearest_txt = Path(fake_post) / prefix / "nearest.txt"
        if not nearest_txt.exists():
            return []
        lines = nearest_txt.read_text(encoding="utf-8").splitlines()
        return [str(Path(train_root) / line.strip().split(" ")[0] / "data.npz") for line in lines[2:] if line.strip()]

    if use_ray:
        ray.init(ignore_reinit_error=True)
        task_items = []
        for graph, prefix in zip(gen_graph_list, gen_prefix_list):
            paths = nearest_paths(prefix)
            task_items.append(is_graph_identical_list_remote.remote(graph, paths, n_bit, atol) if paths else None)
        for idx, ref in enumerate(task_items):
            is_identical[idx] = bool(ray.get(ref)) if ref is not None else False
        ray.shutdown()
    else:
        for idx, (graph, prefix) in enumerate(tqdm(list(zip(gen_graph_list, gen_prefix_list)), desc="Novel")):
            paths = nearest_paths(prefix)
            is_identical[idx] = is_graph_identical_list(graph, paths, n_bit=n_bit, atol=atol) if paths else False

    identical_folder = np.array(gen_prefix_list)[is_identical].tolist()
    return {
        "novel_ratio": float(np.sum(~is_identical) / len(gen_graph_list)) if gen_graph_list else 0.0,
        "num_graphs": len(gen_graph_list),
        "non_novel_prefix": identical_folder,
    }
