"""Identity-first cube-24 rotation protocol.

The runtime data contract uses 24 proper cube rotations. Rotation id 0 is the
identity view and matches the single-view FLUX basis. Legacy Euler-64 ids are
kept out of training, inference, and evaluation code; migration/debug scripts
own any old-id conversion.
"""

from __future__ import annotations

import numpy as np


NUM_CUBE24_VIEWS = 24
IDENTITY_ROTATION_ID = 0

CUBE24_ROTATION_MATRICES: tuple[np.ndarray, ...] = (
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
    np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32),
    np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32),
    np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32),
    np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float32),
    np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], dtype=np.float32),
    np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]], dtype=np.float32),
    np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]], dtype=np.float32),
    np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32),
    np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float32),
    np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32),
    np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=np.float32),
    np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
    np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32),
    np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]], dtype=np.float32),
    np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=np.float32),
    np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]], dtype=np.float32),
    np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32),
    np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float32),
    np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float32),
    np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float32),
    np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32),
    np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float32),
    np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], dtype=np.float32),
)

# Kept for older imports that name the cube group this way. The order is the
# identity-first runtime order, not raw Blender file order.
OCTAHEDRAL_ROTATIONS = CUBE24_ROTATION_MATRICES


def get_cube24_rotation_matrices() -> tuple[np.ndarray, ...]:
    """Return the runtime identity-first cube-24 rotation matrices."""

    return CUBE24_ROTATION_MATRICES


def cube24_rotation_matrix(rotation_id: int) -> np.ndarray:
    """Return a rotation matrix for an identity-first cube-24 id."""

    if rotation_id < 0 or rotation_id >= NUM_CUBE24_VIEWS:
        raise ValueError(f"rotation_id must be in [0, 23], got {rotation_id}")
    return CUBE24_ROTATION_MATRICES[int(rotation_id)]
