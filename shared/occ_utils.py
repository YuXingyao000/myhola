"""Compatibility OCC helpers used by legacy HoLa-BRep modules."""

from __future__ import annotations

from typing import Any

from OCC.Core.TopAbs import (
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_SHELL,
    TopAbs_SOLID,
    TopAbs_VERTEX,
    TopAbs_WIRE,
)
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods


def _cast_topods_shape(shape: Any, primitive_type: int):
    if primitive_type == TopAbs_FACE:
        return topods.Face(shape)
    if primitive_type == TopAbs_EDGE:
        return topods.Edge(shape)
    if primitive_type == TopAbs_VERTEX:
        return topods.Vertex(shape)
    if primitive_type == TopAbs_WIRE:
        return topods.Wire(shape)
    if primitive_type == TopAbs_SHELL:
        return topods.Shell(shape)
    if primitive_type == TopAbs_SOLID:
        return topods.Solid(shape)
    return shape


def get_primitives(shape: Any, primitive_type: int, v_remove_half: bool = False):
    """Collect OCC sub-shapes of ``primitive_type``.

    ``v_remove_half=True`` keeps only orientation-independent unique shapes.
    This matches the historical helper used for counting undirected edges and
    vertices in validity metrics.
    """
    explorer = TopExp_Explorer(shape, primitive_type)
    primitives = []
    unique_shapes = []

    while explorer.More():
        current = explorer.Current()
        if v_remove_half and any(current.IsSame(prev) for prev in unique_shapes):
            explorer.Next()
            continue
        if v_remove_half:
            unique_shapes.append(current)
        primitives.append(_cast_topods_shape(current, primitive_type))
        explorer.Next()

    return primitives


def disable_occ_log():
    """Best-effort suppression of verbose OpenCascade messages."""
    try:
        from OCC.Core.Message import Message

        Message.DefaultMessenger().ClearPrinters()
    except Exception:
        pass

