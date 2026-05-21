"""Small shared utilities required by legacy HoLa-BRep modules."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def check_dir(v_path):
    """Create a directory if needed and return it as a ``Path``."""
    path = Path(v_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_check_dir(v_path):
    """Compatibility alias for callers that expect directory creation."""
    return check_dir(v_path)


def export_point_cloud(v_path, v_points, v_colors: Optional[np.ndarray] = None):
    """Export xyz points to a PLY file for debugging."""
    path = Path(v_path)
    check_dir(path.parent)

    points = np.asarray(v_points, dtype=np.float32).reshape(-1, 3)
    if v_colors is not None:
        colors = np.asarray(v_colors).reshape(-1, 3)
        if colors.max() <= 1.0:
            colors = colors * 255
        colors = np.clip(colors, 0, 255).astype(np.uint8)
        vertices = np.empty(
            points.shape[0],
            dtype=[
                ("x", "f4"), ("y", "f4"), ("z", "f4"),
                ("red", "u1"), ("green", "u1"), ("blue", "u1"),
            ],
        )
        vertices["red"], vertices["green"], vertices["blue"] = colors.T
    else:
        vertices = np.empty(points.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    vertices["x"], vertices["y"], vertices["z"] = points.T

    from plyfile import PlyData, PlyElement

    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(str(path))
    return path
