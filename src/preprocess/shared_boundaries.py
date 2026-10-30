"""Explicit adapters for the legacy merged-boundary representation.

This is not native STEP edge extraction. One unordered pair of distinct faces
has exactly two reciprocal legacy rows and at most one shared boundary. Closed
curves retain the original samples (the last point repeats the first within the
declared endpoint tolerance); no resampling or geometry averaging is performed.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


SHARED_BOUNDARY_REPRESENTATION = "shared_boundary_v1"


def _fail(sample_id: str, message: str) -> None:
    raise ValueError(f"shared boundary sample {sample_id!r}: {message}")


def validate_legacy_geometry(
    face_points: np.ndarray,
    edge_points: np.ndarray,
    connectivity: np.ndarray,
    sample_id: str,
) -> None:
    """Validate before the legacy loader can drop seams or normalize geometry."""
    faces, edges, conn = map(np.asarray, (face_points, edge_points, connectivity))
    if faces.ndim != 4 or faces.shape[-1] != 6 or faces.shape[0] == 0:
        _fail(sample_id, f"expected nonempty [F,U,V,6] face samples, got {faces.shape}")
    if min(faces.shape[1:3]) < 2 or not np.isfinite(faces).all():
        _fail(sample_id, "face samples must be finite with at least 2x2 points")
    if edges.ndim != 3 or edges.shape[-1] != 6 or edges.shape[1] < 2:
        _fail(sample_id, f"expected [E,L,6], L>=2 edge samples, got {edges.shape}")
    if not np.isfinite(edges).all():
        _fail(sample_id, "edge samples contain NaN/Inf")
    if conn.ndim != 2 or conn.shape[1] != 3 or not np.issubdtype(conn.dtype, np.integer):
        _fail(sample_id, f"expected integer [R,3] connectivity, got {conn.shape}/{conn.dtype}")
    if len(conn):
        if (conn[:, 0] < 0).any() or (conn[:, 0] >= len(edges)).any():
            _fail(sample_id, "edge index outside edge sample array")
        if (conn[:, 1:] < 0).any() or (conn[:, 1:] >= len(faces)).any():
            _fail(sample_id, "face index outside face sample array")
        if (conn[:, 1] == conn[:, 2]).any():
            _fail(sample_id, "self-face/seam boundaries are unsupported by shared_boundary_v1")
        if len(np.unique(conn[:, 0])) != len(conn):
            _fail(sample_id, "a directed edge index occurs more than once")
    if set(conn[:, 0].tolist()) != set(range(len(edges))):
        _fail(sample_id, "edge samples and directed connectivity do not have one-to-one coverage")


def _point_to_polyline_max(points: np.ndarray, polyline: np.ndarray) -> float:
    starts = polyline[:-1]
    vectors = polyline[1:] - starts
    delta = points[:, None, :] - starts[None, :, :]
    lengths2 = np.sum(vectors * vectors, axis=-1)
    fractions = np.sum(delta * vectors[None], axis=-1) / np.maximum(lengths2, 1e-20)[None]
    fractions = np.clip(fractions, 0.0, 1.0)
    distances = np.linalg.norm(delta - fractions[..., None] * vectors[None], axis=-1)
    return float(distances.min(axis=1).max())


def build_shared_boundary_targets(
    face_points: np.ndarray,
    edge_points: np.ndarray,
    connectivity: np.ndarray,
    *,
    sample_id: str,
    reciprocal_tolerance: float,
    closed_tolerance: float,
    allow_closed_phase: bool,
) -> dict[str, Any]:
    """Group reciprocal rows without altering the old encoder inputs.

    The default comparison is maximum pointwise L2 up to reversal. If enabled,
    two closed curves may also agree as polylines despite different start phase;
    then symmetric maximum vertex-to-polyline distance is used. It is a sampled
    consistency check, not proof of equality of the original analytic curves.
    All tolerances are in the input's global XYZ units.
    """
    if not np.isfinite(reciprocal_tolerance) or reciprocal_tolerance < 0:
        _fail(sample_id, "reciprocal_tolerance must be finite and nonnegative")
    if not np.isfinite(closed_tolerance) or closed_tolerance < 0:
        _fail(sample_id, "closed_tolerance must be finite and nonnegative")
    validate_legacy_geometry(face_points, edge_points, connectivity, sample_id)
    edges = np.asarray(edge_points)[..., :3]
    conn = np.asarray(connectivity)
    groups: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for edge, face1, face2 in conn:
        groups[tuple(sorted((int(face1), int(face2))))].append((int(edge), int(face1), int(face2)))

    curves, face_pairs, closed, representatives, residuals, phase_used = [], [], [], [], [], []
    for pair, rows in sorted(groups.items()):
        if len(rows) != 2 or rows[0][1:] != rows[1][1:][::-1]:
            _fail(sample_id, f"face pair {pair} requires exactly two reciprocal rows; got {rows}")
        rows.sort(key=lambda row: row[0])
        first, second = (edges[row[0]].astype(np.float64) for row in rows)
        first_closed = np.linalg.norm(first[0] - first[-1]) <= closed_tolerance
        second_closed = np.linalg.norm(second[0] - second[-1]) <= closed_tolerance
        if first_closed != second_closed:
            _fail(sample_id, f"face pair {pair} disagrees on curve closure")
        distance = min(
            float(np.linalg.norm(first - second, axis=-1).max()),
            float(np.linalg.norm(first - second[::-1], axis=-1).max()),
        )
        used_polyline = bool(allow_closed_phase and first_closed and distance > reciprocal_tolerance)
        if used_polyline:
            distance = max(_point_to_polyline_max(first, second), _point_to_polyline_max(second, first))
        if distance > reciprocal_tolerance:
            _fail(sample_id, f"face pair {pair} reciprocal residual {distance:.8g} exceeds {reciprocal_tolerance:.8g}")
        curves.append(edges[rows[0][0]])
        face_pairs.append(pair)
        closed.append(first_closed)
        representatives.append(rows[0][0])
        residuals.append(distance)
        phase_used.append(used_polyline)

    return {
        "curves": np.asarray(curves, dtype=np.float32).reshape(-1, edges.shape[1], 3),
        "face_pairs": np.asarray(face_pairs, dtype=np.int64).reshape(-1, 2),
        "is_closed": np.asarray(closed, dtype=bool),
        "source_edge_ids": np.asarray(representatives, dtype=np.int64),
        "reciprocal_residual": np.asarray(residuals, dtype=np.float32),
        "closed_phase_used": np.asarray(phase_used, dtype=bool),
    }


def shared_prediction_diagnostics(
    num_faces: int,
    edge_points: np.ndarray,
    connectivity: np.ndarray,
    confidence: np.ndarray,
) -> dict[str, int]:
    """Count raw errors without silently deduplicating or selecting queries."""
    edges, conn, scores = map(np.asarray, (edge_points, connectivity, confidence))
    if edges.ndim != 3 or edges.shape[-1] != 3 or edges.shape[1] < 2:
        raise ValueError(f"shared_boundary_v1 requires pred_edge [Q,L,3], got {edges.shape}")
    if conn.shape != (len(edges), 3) or not np.issubdtype(conn.dtype, np.integer):
        raise ValueError("shared_boundary_v1 requires integer pred_edge_face_connectivity [Q,3]")
    if scores.shape != (len(edges),):
        raise ValueError("shared_boundary_v1 requires pred_edge_confidence [Q]")
    if not np.array_equal(np.sort(conn[:, 0]), np.arange(len(edges))):
        raise ValueError("shared_boundary_v1 requires each predicted edge index exactly once")
    valid_ids = ((conn[:, 1:] >= 0) & (conn[:, 1:] < num_faces)).all(axis=1)
    self_assigned = conn[:, 1] == conn[:, 2]
    pairs = [tuple(sorted(row[1:].tolist())) for row in conn[valid_ids & ~self_assigned]]
    return {
        "num_predicted_queries": len(edges),
        "num_invalid_face_assignments": int((~valid_ids).sum()),
        "num_self_assignments": int(self_assigned.sum()),
        "num_duplicate_pair_assignments": len(pairs) - len(set(pairs)),
        "num_distinct_valid_pairs": len(set(pairs)),
        "num_nonfinite_curves": int((~np.isfinite(edges).all(axis=(1, 2))).sum()),
        "num_invalid_confidences": int((~np.isfinite(scores) | (scores < 0) | (scores > 1)).sum()),
    }
