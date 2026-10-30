"""Read-only OCC topology and BRepCheck diagnostics of an as-read STEP shape."""
from __future__ import annotations

from collections import Counter

SHAPE_NAMES = {0: "compound", 1: "compsolid", 2: "solid", 3: "shell", 4: "face",
               5: "wire", 6: "edge", 7: "vertex", 8: "shape"}
FREE_EDGE_DEFINITION = "one edge-in-face occurrence, excluding degenerate and seam edges; topological identity only"


def primitive_map(shape, kind):
    from OCC.Core.TopExp import topexp
    from OCC.Core.TopTools import TopTools_IndexedMapOfShape
    result = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, kind, result)
    return result


def _shape_list(items):
    # PythonOCC 7.8.1 lists are not Python iterables.
    from OCC.Core.TopTools import TopTools_ListIteratorOfListOfShape
    iterator = TopTools_ListIteratorOfListOfShape(items)
    result = []
    while iterator.More():
        result.append(iterator.Value())
        iterator.Next()
    return result


def _status_names(statuses):
    from OCC.Core.BRepCheck import BRepCheck_ListIteratorOfListOfStatus, BRepCheck_Status
    # Constants are IntEnums, but this iterator returns plain ints in 7.8.1.
    names = {int(value): name for name, value in BRepCheck_Status.__members__.items()}
    iterator = BRepCheck_ListIteratorOfListOfStatus(statuses)
    result = []
    while iterator.More():
        value = int(iterator.Value())
        result.append(names.get(value, f"BRepCheck_UnknownStatus_{value}"))
        iterator.Next()
    return result


def native_diagnostics(shape) -> tuple[dict, list, dict]:
    """Inspect the as-read shape before adding any triangulations.

    Return diagnostics, per-edge incidence flags, and stable primitive maps.
    A valid open shell is correctly distinguished from an invalid solid.
    """
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.ShapeAnalysis import ShapeAnalysis_Shell
    from OCC.Core.TopAbs import (TopAbs_VERTEX, TopAbs_EDGE, TopAbs_WIRE,
                                 TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID)
    from OCC.Core.TopExp import topexp
    from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
    from OCC.Core.TopoDS import topods

    kinds = {"vertices": TopAbs_VERTEX, "edges": TopAbs_EDGE, "wires": TopAbs_WIRE,
             "faces": TopAbs_FACE, "shells": TopAbs_SHELL, "solids": TopAbs_SOLID}
    maps = {name: primitive_map(shape, kind) for name, kind in kinds.items()}
    analyzer = BRepCheck_Analyzer(shape)
    result = {
        "shape_type": int(shape.ShapeType()), "shape_type_name": SHAPE_NAMES[int(shape.ShapeType())],
        "native_brepcheck_valid": bool(analyzer.IsValid()),
        "native_is_solid_type": shape.ShapeType() == TopAbs_SOLID,
        "counts": {name: items.Size() for name, items in maps.items()},
    }
    result["native_valid_solid"] = result["native_is_solid_type"] and result["native_brepcheck_valid"]
    ancestors = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, ancestors)
    edge_info = []
    for index in range(1, maps["edges"].Size() + 1):
        edge = topods.Edge(maps["edges"].FindKey(index))
        ancestor_index = ancestors.FindIndex(edge)
        parents = _shape_list(ancestors.FindFromIndex(ancestor_index)) if ancestor_index else []
        face_ids = [maps["faces"].FindIndex(face) - 1 for face in parents]
        degenerate = bool(BRep_Tool.Degenerated(edge))
        seam = any(BRep_Tool.IsClosed(edge, topods.Face(face)) for face in parents)
        degree = len(parents)
        edge_info.append({"id": index - 1, "face_occurrences": face_ids,
                          "unique_face_ids": sorted(set(face_ids)), "incidence_degree": degree,
                          "degenerate": degenerate, "seam": bool(seam),
                          "free": degree == 1 and not degenerate and not seam,
                          "nonmanifold": degree > 2 and not degenerate})
    result["incidence"] = {
        "degree_histogram": dict(Counter(str(item["incidence_degree"]) for item in edge_info)),
        **{name + "_edges": sum(bool(item[name]) for item in edge_info)
           for name in ("free", "nonmanifold", "degenerate", "seam")},
        "orphan_edges": sum(item["incidence_degree"] == 0 for item in edge_info),
        "definition": FREE_EDGE_DEFINITION, "edges": edge_info,
    }
    try:
        shell = ShapeAnalysis_Shell()
        shell.CheckOrientedShells(shape, True, True)
        result["shell_orientation"] = {
            "has_bad_edges": bool(shell.HasBadEdges()), "has_free_edges": bool(shell.HasFreeEdges()),
            "bad_edge_count": primitive_map(shell.BadEdges(), TopAbs_EDGE).Size(),
            "free_edge_count": primitive_map(shell.FreeEdges(), TopAbs_EDGE).Size(),
        }
    except Exception as exc:
        result["shell_orientation"] = {"error": f"{type(exc).__name__}: {exc}"}

    # Include contextual errors: Status() alone can miss edge-in-face failures.
    bad, errors = [], []
    for kind, items in maps.items():
        for index in range(1, items.Size() + 1):
            try:
                checked = analyzer.Result(items.FindKey(index))
                if checked is None:
                    continue
                statuses = [s for s in _status_names(checked.Status()) if s != "BRepCheck_NoError"]
                contexts = []
                checked.InitContextIterator()
                while checked.MoreShapeInContext():
                    context = checked.ContextualShape()
                    names = [s for s in _status_names(checked.StatusOnShape()) if s != "BRepCheck_NoError"]
                    if names:
                        contexts.append({"shape_type": SHAPE_NAMES[int(context.ShapeType())], "statuses": names})
                    checked.NextShapeInContext()
                if statuses or contexts:
                    bad.append({"kind": kind, "id": index - 1, "statuses": statuses, "contexts": contexts})
            except Exception as exc:
                errors.append({"kind": kind, "id": index - 1, "error": f"{type(exc).__name__}: {exc}"})
    result["brepcheck_subshapes"] = {"with_errors": bad, "inspection_errors": errors}
    return result, edge_info, maps


