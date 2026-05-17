"""Rotation protocol shared by data generation and evaluation.

The old image extraction code enumerated 4 x 4 x 4 Euler rotations, but
those 64 indices collapse to the 24 proper rotations of a cube.  New data
should use these 24 matrices directly.  The legacy helpers are kept only to
map old 64-view assets onto the canonical 24-view protocol.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.spatial.transform import Rotation


def legacy_64_rotation_matrix(index: int) -> np.ndarray:
    """Return the matrix used by the historical 64-view extractor."""
    if index < 0 or index >= 64:
        raise ValueError(f"legacy rotation index must be in [0, 63], got {index}")
    angles = np.array([index % 4, index // 4 % 4, index // 16], dtype=np.float32)
    matrix = Rotation.from_euler("xyz", angles * np.pi / 2).as_matrix().T
    return np.rint(matrix).astype(np.float32)


def _matrix_key(matrix: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in np.rint(matrix).astype(np.int8).reshape(-1))


@lru_cache(maxsize=1)
def get_octahedral_rotation_matrices() -> tuple[np.ndarray, ...]:
    """Return the 24 cube rotations, identity first.

    The order follows the first occurrence in the old 64-view enumeration.
    This makes migration from historical `view_id in [0, 63]` deterministic
    while keeping `rotation_id == 0` as the canonical identity view.
    """
    seen: set[tuple[int, ...]] = set()
    rotations: list[np.ndarray] = []
    for legacy_index in range(64):
        matrix = legacy_64_rotation_matrix(legacy_index)
        key = _matrix_key(matrix)
        if key in seen:
            continue
        seen.add(key)
        rotations.append(matrix)
    if len(rotations) != 24:
        raise ValueError(f"Expected 24 unique cube rotations, got {len(rotations)}")
    if not np.array_equal(rotations[0], np.eye(3, dtype=np.float32)):
        raise ValueError("rotation_id=0 must be identity")
    return tuple(rotations)


@lru_cache(maxsize=1)
def legacy_64_to_rotation_id() -> tuple[int, ...]:
    """Map each historical 64-view id to a canonical 24-view rotation id."""
    rotations = get_octahedral_rotation_matrices()
    key_to_id = {_matrix_key(matrix): idx for idx, matrix in enumerate(rotations)}
    return tuple(key_to_id[_matrix_key(legacy_64_rotation_matrix(i))] for i in range(64))


OCTAHEDRAL_ROTATIONS = get_octahedral_rotation_matrices()
