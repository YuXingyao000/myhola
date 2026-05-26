"""Rotation protocol shared by data generation, training, and evaluation.

CANONICAL ORDER: The 24 proper rotations of the cube are defined by enumerating
orthonormal axis pairs from the 6 signed axis directions. This is the same order
used by:
  - Blender render_cube24.py (generate_cube_rotations)
  - dataset.py (_build_cube24_rotation_matrices)

Index 0 is NOT identity — it is a 180° rotation about Y. Identity is at index 18.
This is intentional: the Blender renders and cached latents both use this ordering,
and all downstream code (eval, inference) must align to it.

Legacy 64-view helpers are kept only for data migration from old euler64 assets.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.spatial.transform import Rotation


# ═══════════════════════════════════════════════════════════════
# Canonical 24-rotation definition (Blender/dataset order)
# ═══════════════════════════════════════════════════════════════

_AXIS_DIRECTIONS = (
    np.array([1.0, 0.0, 0.0]),
    np.array([-1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([0.0, -1.0, 0.0]),
    np.array([0.0, 0.0, 1.0]),
    np.array([0.0, 0.0, -1.0]),
)


@lru_cache(maxsize=1)
def get_cube24_rotation_matrices() -> tuple[np.ndarray, ...]:
    """Return the 24 cube rotations in Blender/dataset canonical order.

    Algorithm: enumerate z-axis from 6 directions, for each z pick orthogonal y,
    compute x = cross(y, z). Deduplicate. Same as render_cube24.py and dataset.py.

    Index 0: z=+X, y=+Y → x=[0,0,-1] → 180° Y-rotation (NOT identity)
    Index 18: z=+Z, y=+Y → x=+X → IDENTITY
    """
    rotations: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for z in _AXIS_DIRECTIONS:
        for y in _AXIS_DIRECTIONS:
            if abs(float(np.dot(z, y))) > 1e-6:
                continue
            x = np.cross(y, z)
            matrix = np.stack([x, y, z], axis=1).astype(np.float32)
            key = tuple(int(round(v)) for v in matrix.flatten())
            if key in seen:
                continue
            seen.add(key)
            rotations.append(matrix)
    if len(rotations) != 24:
        raise ValueError(f"Expected 24 cube rotations, got {len(rotations)}")
    return tuple(rotations)


# The canonical rotation list used everywhere (eval, dataset, Blender)
OCTAHEDRAL_ROTATIONS = get_cube24_rotation_matrices()

# Convenience: which index is identity?
IDENTITY_ROTATION_ID = next(
    i for i, m in enumerate(OCTAHEDRAL_ROTATIONS)
    if np.allclose(m, np.eye(3))
)


# ═══════════════════════════════════════════════════════════════
# Legacy 64-view helpers (for data migration only)
# ═══════════════════════════════════════════════════════════════

def legacy_64_rotation_matrix(index: int) -> np.ndarray:
    """Return the matrix used by the historical 64-view extractor."""
    if index < 0 or index >= 64:
        raise ValueError(f"legacy rotation index must be in [0, 63], got {index}")
    angles = np.array([index % 4, index // 4 % 4, index // 16], dtype=np.float32)
    matrix = Rotation.from_euler("xyz", angles * np.pi / 2).as_matrix()
    return np.rint(matrix).astype(np.float32)


def _matrix_key(matrix: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in np.rint(matrix).astype(np.int8).reshape(-1))


@lru_cache(maxsize=1)
def cube24_to_euler64_mapping() -> tuple[int, ...]:
    """Map cube24_id → euler64_id. Length 24."""
    cube_mats = get_cube24_rotation_matrices()
    euler_mats = [legacy_64_rotation_matrix(i) for i in range(64)]
    mapping = []
    for cm in cube_mats:
        for j, em in enumerate(euler_mats):
            if np.allclose(cm, em, atol=1e-6):
                mapping.append(j)
                break
        else:
            raise RuntimeError("Cube rotation has no Euler-64 counterpart")
    return tuple(mapping)


@lru_cache(maxsize=1)
def euler64_to_cube24_mapping() -> tuple[int | None, ...]:
    """Map euler64_id → cube24_id (None if not a unique rotation). Length 64."""
    cube_mats = get_cube24_rotation_matrices()
    euler_mats = [legacy_64_rotation_matrix(i) for i in range(64)]
    mapping: list[int | None] = [None] * 64
    for eid in range(64):
        for cid, cm in enumerate(cube_mats):
            if np.allclose(euler_mats[eid], cm, atol=1e-6):
                mapping[eid] = cid
                break
    return tuple(mapping)
