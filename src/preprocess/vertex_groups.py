"""Native vertex sidecars and endpoint pairs; no OCC dependency at training time."""
from itertools import combinations
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 1


def canonical_edges(connectivity):
    """Select the actual i<j row; preserve its source edge ID and point order."""
    rows = connectivity[connectivity[:, 1] < connectivity[:, 2]]
    rows = rows[np.lexsort((rows[:, 2], rows[:, 1]))]
    return rows[:, 0], rows[:, 1:]


def corner_pairs(source_edge_ids, endpoint_vertex_ids):
    """Unordered cross-curve endpoint pairs, grouped by native vertex identity.

    [N,2,2]: N pairs, two members, each (raw edge ID, endpoint slot 0/1).
    A closed curve's own two ends never form a cross-curve corner pair.
    """
    pairs, groups, vertices = [], [], []
    for vertex in np.unique(endpoint_vertex_ids[endpoint_vertex_ids >= 0]):
        members = np.argwhere(endpoint_vertex_ids == vertex)
        local_pairs = [((int(source_edge_ids[a]), int(s)), (int(source_edge_ids[b]), int(t)))
                       for (a, s), (b, t) in combinations(members, 2) if a != b]
        if local_pairs:
            groups.extend([len(vertices)] * len(local_pairs))
            pairs.extend(local_pairs)
            vertices.append(int(vertex))
    return {
        "corner_pairs": np.asarray(pairs, dtype=np.int64).reshape(-1, 2, 2),
        "corner_pair_groups": np.asarray(groups, dtype=np.int64),
        "corner_group_vertices": np.asarray(vertices, dtype=np.int64),
    }


def load_vertex_groups(path, *, source_sha256, connectivity):
    """Verify source/order at the dataset boundary before exposing loss indices."""
    with np.load(Path(path), allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    if arrays["schema_version"].item() != SCHEMA_VERSION:
        raise ValueError(f"Unsupported vertex group schema: {path}")
    if arrays["source_sha256"].item() != source_sha256:
        raise ValueError(f"Vertex group source digest mismatch: {path}")
    ids, face_pairs = canonical_edges(connectivity)
    for name, expected in (("source_edge_ids", ids), ("face_pairs", face_pairs)):
        if not np.array_equal(arrays[name], expected):
            raise ValueError(f"Vertex group {name} mismatch: {path}")
    vertices = arrays["endpoint_vertex_ids"]
    if vertices.dtype.kind not in "iu" or vertices.shape != (len(ids), 2) or (vertices < -1).any():
        raise ValueError(f"Invalid endpoint vertex IDs: {path}")
    # Deriving pairs is cheap and detects corrupted indices, duplicate pairs,
    # or accidental inclusion of the two ends of one closed curve.
    expected_pairs = corner_pairs(ids, vertices)
    for key, value in expected_pairs.items():
        if arrays[key].dtype.kind not in "iu" or not np.array_equal(arrays[key], value):
            raise ValueError(f"Invalid {key}: {path}")
    return expected_pairs
