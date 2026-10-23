"""GT-free consistency measurements on saved face grids and half-edge curves.

The curve and triangle mathematics is adapted from ``legacy_failure_upstream``
and the UV-boundary failure census. These measurements describe the predicted
geometry only; they neither identify hidden geometry errors nor prove validity.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {"count": len(array), "mean": float(array.mean()) if len(array) else None,
            "p95": float(np.quantile(array, .95)) if len(array) else None,
            "max": float(array.max()) if len(array) else None}


def _triangles(grid: np.ndarray) -> np.ndarray:
    a, b, c, d = grid[:-1, :-1], grid[1:, :-1], grid[1:, 1:], grid[:-1, 1:]
    return np.concatenate((np.stack((a, b, c), -2).reshape(-1, 3, 3),
                           np.stack((a, c, d), -2).reshape(-1, 3, 3)))


def point_triangle_distances(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Nearest Euclidean distances to finite triangles, including degenerate ones."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac = b - a, c - a
    delta = points[:, None] - a[None]
    normal = np.cross(ab, ac)
    norm_sq = np.sum(normal * normal, -1)
    safe_norm_sq = np.where(norm_sq > 1e-30, norm_sq, 1.0)
    signed = np.sum(delta * normal[None], -1)
    projected = delta - (signed / safe_norm_sq[None])[..., None] * normal[None]
    d00, d01, d11 = np.sum(ab * ab, -1), np.sum(ab * ac, -1), np.sum(ac * ac, -1)
    d20, d21 = np.sum(projected * ab[None], -1), np.sum(projected * ac[None], -1)
    u = (d11[None] * d20 - d01[None] * d21) / safe_norm_sq[None]
    v = (d00[None] * d21 - d01[None] * d20) / safe_norm_sq[None]
    inside = (norm_sq[None] > 1e-30) & (u >= 0) & (v >= 0) & (u + v <= 1)
    best = np.where(inside, signed * signed / safe_norm_sq[None], np.inf)
    for start, end in ((a, b), (b, c), (c, a)):
        vector = end - start
        offset = points[:, None] - start[None]
        fraction = np.sum(offset * vector[None], -1) / np.maximum(np.sum(vector * vector, -1), 1e-30)[None]
        residual = offset - np.clip(fraction, 0, 1)[..., None] * vector[None]
        best = np.minimum(best, np.sum(residual * residual, -1))
    return np.sqrt(np.maximum(best.min(-1), 0))


def _arc_samples(curve: np.ndarray, count: int = 64) -> np.ndarray:
    cumulative = np.r_[0., np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=-1))]
    if cumulative[-1] <= 1e-15:
        return np.repeat(curve[:1], count, axis=0)
    keep = np.r_[True, np.diff(cumulative) > 0]
    positions = np.linspace(0, cumulative[-1], count)
    return np.stack([np.interp(positions, cumulative[keep], curve[keep, axis]) for axis in range(3)], -1)


def _point_polyline_distances(points: np.ndarray, curve: np.ndarray) -> np.ndarray:
    vector = curve[1:] - curve[:-1]
    offset = points[:, None] - curve[None, :-1]
    fraction = np.sum(offset * vector[None], -1) / np.maximum(np.sum(vector * vector, -1), 1e-30)[None]
    residual = offset - np.clip(fraction, 0, 1)[..., None] * vector[None]
    return np.sqrt(np.sum(residual * residual, -1).min(-1))


def _curve_distance(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    forward = _point_polyline_distances(_arc_samples(first), second)
    backward = _point_polyline_distances(_arc_samples(second), first)
    return {"mean": float((forward.mean() + backward.mean()) / 2),
            "p95": float(max(np.quantile(forward, .95), np.quantile(backward, .95))),
            "max": float(max(forward.max(), backward.max()))}


def measure_raw_consistency(raw_path: Path, *, distance_threshold: float = .01) -> dict[str, Any]:
    """Inspect a legacy half-edge NPZ without GT face ordering or correspondence.

    ``distance_threshold`` is a normalized XYZ length, not squared Chamfer or a
    physical-unit tolerance. Distances to triangulated grids are discrete proxies.
    Invalid array contracts raise; callers retain the case as a measurement error.
    """
    if not np.isfinite(distance_threshold) or distance_threshold <= 0:
        raise ValueError("distance_threshold must be positive and finite")
    with np.load(raw_path, allow_pickle=False) as data:
        if "representation" in data and str(data["representation"].item()) != "legacy_half_edges":
            raise ValueError("Half-edge consistency requires legacy_half_edges representation")
        faces = np.asarray(data["pred_face"][..., :3], dtype=np.float64)
        curves = np.asarray(data["pred_edge"][..., :3], dtype=np.float64)
        conn = np.asarray(data["pred_edge_face_connectivity"])
    if faces.ndim != 4 or min(faces.shape[1:3]) < 2 or faces.shape[-1] != 3 or not len(faces):
        raise ValueError("Expected nonempty face grids [F,U,V,3], U,V >= 2")
    if curves.ndim != 3 or curves.shape[-1] != 3 or curves.shape[1] < 2:
        raise ValueError("Expected curves [E,P,3], P >= 2")
    if not np.isfinite(faces).all() or not np.isfinite(curves).all():
        raise ValueError("Nonfinite predicted geometry")
    # Original HoLa exports integral connectivity using float32 arrays.
    if (conn.ndim != 2 or conn.shape[1] != 3 or not np.issubdtype(conn.dtype, np.number)
            or not np.isfinite(conn).all() or not np.equal(conn, np.floor(conn)).all()):
        raise ValueError("Expected integral connectivity [R,3]")
    conn = conn.astype(np.int64)
    groups: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    directed: Counter = Counter()
    neighbors = [set() for _ in faces]
    invalid_rows, self_rows, owner_records = [], [], []
    meshes = [_triangles(face) for face in faces]
    for index, values in enumerate(conn):
        edge, first, second = map(int, values)
        if not (0 <= edge < len(curves) and 0 <= first < len(faces) and 0 <= second < len(faces)):
            invalid_rows.append(index)
            continue
        directed[first, second] += 1
        if first == second:
            self_rows.append(index)
        else:
            groups[tuple(sorted((first, second)))].append((edge, first, second))
            neighbors[first].add(second)
            neighbors[second].add(first)
        for owner in sorted({first, second}):
            distances = point_triangle_distances(curves[edge], meshes[owner])
            owner_records.append({"row_index": index, "edge_id": edge, "owner_face_id": owner,
                                  "mean": float(distances.mean()), "p95": float(np.quantile(distances, .95)),
                                  "max": float(distances.max())})
    components, unseen = [], set(range(len(faces)))
    while unseen:
        pending, component = [min(unseen)], []
        while pending:
            face = pending.pop()
            if face not in unseen:
                continue
            unseen.remove(face)
            component.append(face)
            pending.extend(neighbors[face] & unseen)
        components.append(sorted(component))
    reciprocal = []
    for pair, rows in groups.items():
        if len(rows) == 2 and rows[0][1:] == rows[1][1:][::-1]:
            reciprocal.append({"face_pair": list(pair), "edge_ids": [row[0] for row in rows],
                               **_curve_distance(curves[rows[0][0]], curves[rows[1][0]])})
    one_direction = [list(pair) for pair, rows in groups.items() if len({row[1:] for row in rows}) == 1]
    structure = {
        "face_count": len(faces), "curve_count": len(curves), "connectivity_row_count": len(conn),
        "invalid_connectivity_rows": invalid_rows, "self_connection_rows": self_rows,
        "one_direction_pairs": one_direction,
        "duplicate_directed_count": sum(max(count - 1, 0) for count in directed.values()),
        "unsupported_multiplicity_pairs": [list(pair) for pair, rows in groups.items() if len(rows) > 2],
        "isolated_face_indices": [index for index, links in enumerate(neighbors) if not links],
        "connected_component_count": len(components), "connected_components": components,
    }
    owner = {"records": owner_records, "p95_per_edge_owner": _summary([row["p95"] for row in owner_records]),
             "max_distance": max((row["max"] for row in owner_records), default=None),
             "above_threshold_count": sum(row["p95"] > distance_threshold for row in owner_records)}
    twin = {"records": reciprocal, "p95_per_pair": _summary([row["p95"] for row in reciprocal]),
            "max_distance": max((row["max"] for row in reciprocal), default=None),
            "above_threshold_count": sum(row["p95"] > distance_threshold for row in reciprocal)}
    return {
        "status": "ok", "representation": "legacy_half_edges", "distance_threshold": distance_threshold,
        "distance_units": "normalized XYZ Euclidean length", "structure": structure,
        "owner_mesh_distance": owner, "reciprocal_curve_distance": twin,
        "observation_flags": {
            "invalid_connectivity": bool(invalid_rows), "self_connection": bool(self_rows),
            "one_direction_pair": bool(one_direction), "duplicate_directed_pair": structure["duplicate_directed_count"] > 0,
            "isolated_face": bool(structure["isolated_face_indices"]), "disconnected_face_graph": len(components) > 1,
            "reciprocal_curve_gap": twin["above_threshold_count"] > 0 if reciprocal else None,
            "edge_owner_mesh_gap": owner["above_threshold_count"] > 0 if owner_records else None,
        },
        "measurement_notes": {
            "owner": "Every stored curve point to each assigned owner grid triangulation; p95 per edge-owner.",
            "reciprocal": "64 arc-length samples to opposite original polyline; maximum of directional p95.",
            "scope": "Discrete geometric proxies and graph observations; no GT and no causal or validity claim.",
        },
    }


def inspect_saved_step(step_path: Path) -> dict[str, Any]:
    """Read native STEP topology without healing, tessellation, or overwriting."""
    from tools.legacy_failure_geometry import native_diagnostics
    from tools.vae_compare_post import read_step_at_precision

    shape = read_step_at_precision(Path(step_path), mode=0)
    diagnostics, _, _ = native_diagnostics(shape)
    return {"status": "ok", "native": diagnostics, "read_precision_mode": 0,
            "scope": "Saved STEP only; no repair and no GT correspondence."}
