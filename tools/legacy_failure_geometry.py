"""Read-only OCC diagnostics and view assets for the frozen legacy failure set.

The saved STEP is never repaired or overwritten. Rendering may retain only a
subset of its faces, and records every missing face explicitly. Run from the
repository root with ``python -m tools.legacy_failure_geometry --help``.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import time
import traceback

import numpy as np

SCHEMA = "legacy_failure_geometry_v1"
SHAPE_NAMES = {0: "compound", 1: "compsolid", 2: "solid", 3: "shell", 4: "face",
               5: "wire", 6: "edge", 7: "vertex", 8: "shape"}
VIEWS = [
    {"name": "isometric", "elev": 24, "azim": -58},
    {"name": "front", "elev": 0, "azim": -90},
    {"name": "side", "elev": 0, "azim": 0},
    {"name": "top", "elev": 90, "azim": -90},
]
SETTINGS = {
    "native_step_read_precision_mode": 0,
    "mesh_linear_deflection": 0.02,
    "mesh_angular_deflection": 0.15,
    "max_triangles_per_face": 200000,
    "edge_samples": 64,
    "stored_decimal_places": 7,
    "raw_closed_endpoint_threshold": 1e-5,
    "raw_distance": "Euclidean normalized XYZ; nearest 16x16 surface-grid samples, not exact surfaces",
    "free_edge": "one edge-in-face occurrence, excluding degenerate and seam edges; topological identity only",
    "nonmanifold_edge": "more than two edge-in-face occurrences, excluding degenerate edges",
}


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    temporary.replace(path)


def digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flat_numbers(values) -> list:
    return np.asarray(values, dtype=np.float64).round(SETTINGS["stored_decimal_places"]).reshape(-1).tolist()


def numeric_summary(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "mean": None, "max": None, "p90": None}
    return {"count": len(values), "mean": float(values.mean()), "max": float(values.max()),
            "p90": float(np.quantile(values, 0.9))}


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
        "definition": SETTINGS["free_edge"], "edges": edge_info,
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


def edge_points(edge, samples=64):
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_REVERSED
    from OCC.Core.TopoDS import topods
    if BRep_Tool.Degenerated(edge):
        vertices = primitive_map(edge, TopAbs_VERTEX)
        if not vertices.Size():
            raise ValueError("Degenerate edge has no topological vertex")
        point = BRep_Tool.Pnt(topods.Vertex(vertices.FindKey(1)))
        return np.repeat(np.asarray(point.Coord())[None], samples, axis=0)
    curve = BRepAdaptor_Curve(edge)
    first, last = curve.FirstParameter(), curve.LastParameter()
    if not np.isfinite([first, last]).all():
        raise ValueError("Curve has a nonfinite parameter domain")
    if edge.Orientation() == TopAbs_REVERSED:
        first, last = last, first
    points = np.asarray([curve.Value(float(u)).Coord() for u in np.linspace(first, last, samples)])
    if not np.isfinite(points).all():
        raise ValueError("Curve samples contain nonfinite coordinates")
    return points


def wire_gaps(maps) -> dict:
    """Ordered native curve endpoint gaps, with incomplete traversals explicit."""
    from OCC.Core.BRepTools import BRepTools_WireExplorer
    from OCC.Core.TopAbs import TopAbs_WIRE, TopAbs_EDGE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods
    records = []
    for face_index in range(1, maps["faces"].Size() + 1):
        face = topods.Face(maps["faces"].FindKey(face_index))
        wires = primitive_map(face, TopAbs_WIRE)
        for wire_index in range(1, wires.Size() + 1):
            wire = topods.Wire(wires.FindKey(wire_index))
            row = {"face_id": face_index - 1, "wire_id": maps["wires"].FindIndex(wire) - 1,
                   "wire_closed_flag": bool(wire.Closed())}
            try:
                all_edges = TopExp_Explorer(wire, TopAbs_EDGE)
                expected = 0
                while all_edges.More():
                    expected += 1
                    all_edges.Next()
                explorer = BRepTools_WireExplorer(wire, face)
                ends = []
                while explorer.More():
                    ends.append(edge_points(topods.Edge(explorer.Current()), 2))
                    explorer.Next()
                row.update(edge_occurrences=expected, visited_edges=len(ends), traversal_complete=len(ends) == expected)
                if ends:
                    ends = np.asarray(ends)
                    gaps = np.linalg.norm(ends[:, -1] - np.roll(ends[:, 0], -1, axis=0), axis=-1)
                    row.update(endpoint_gaps=flat_numbers(gaps), **numeric_summary(gaps))
                else:
                    row.update(numeric_summary([]))
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            records.append(row)
    return {"description": "Oriented 3D curve end-to-next-start distance in each face wire; includes final-to-first. Not a solid-closure test.",
            "wires": records}


def mesh_face(face) -> tuple[dict, dict]:
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopAbs import TopAbs_REVERSED
    from OCC.Core.TopLoc import TopLoc_Location
    BRepMesh_IncrementalMesh(face, SETTINGS["mesh_linear_deflection"], False,
                             SETTINGS["mesh_angular_deflection"], False)
    location = TopLoc_Location()
    mesh = BRep_Tool.Triangulation(face, location)
    if mesh is None or mesh.NbTriangles() == 0:
        return {"positions": [], "indices": [], "status": "no_triangulation"}, {"status": "no_triangulation"}
    count = mesh.NbTriangles()
    if count > SETTINGS["max_triangles_per_face"]:
        info = {"status": "mesh_budget_exceeded", "triangle_count": count}
        return {"positions": [], "indices": [], **info}, info
    vertices = np.asarray([mesh.Node(i).Transformed(location.Transformation()).Coord()
                           for i in range(1, mesh.NbNodes() + 1)])
    triangles = np.asarray([mesh.Triangle(i).Get() for i in range(1, count + 1)], dtype=np.int64) - 1
    if not np.isfinite(vertices).all():
        raise ValueError("Triangulation has nonfinite vertices")
    if triangles.min() < 0 or triangles.max() >= len(vertices):
        raise ValueError("Triangulation has invalid vertex indices")
    if face.Orientation() == TopAbs_REVERSED:
        triangles = triangles[:, ::-1]
    points = vertices[triangles]
    areas2 = np.linalg.norm(np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1)
    usable = np.isfinite(areas2) & (areas2 > 0)
    info = {"status": "ok" if usable.any() else "zero_mesh_area", "triangle_count": count,
            "rendered_triangles": int(usable.sum()), "degenerate_triangles": int((~usable).sum())}
    if not usable.any():
        return {"positions": [], "indices": [], **info}, info
    return {"positions": flat_numbers(vertices), "indices": triangles[usable].reshape(-1).tolist(), **info}, info


def panel_counts(panel):
    counts = panel.setdefault("counts", {})
    counts.update(faces=len(panel["faces"]), edges=len(panel["edges"]),
                  meshed_faces=sum(f.get("status") == "ok" for f in panel["faces"]),
                  missing_mesh_faces=sum(f.get("status") != "ok" for f in panel["faces"]),
                  unreadable_edges=sum(e.get("status") != "ok" for e in panel["edges"]))


def empty_panel(label, status="pending", error=None):
    return {"label": label, "status": status, "faces": [], "edges": [], "counts": {}, "error": error}


def extract_step(path: Path, panel: dict, diagnostic: dict, checkpoint=lambda: None):
    from OCC.Core.TopoDS import topods
    from tools.vae_compare_post import read_step_at_precision
    shape = read_step_at_precision(path, mode=0)
    native, info, maps = native_diagnostics(shape)
    diagnostic.update(native)
    diagnostic["wire_endpoint_gaps"] = wire_gaps(maps)
    panel["counts"].update(native["counts"])
    panel["native_valid_solid"] = native["native_valid_solid"]
    panel["shape_type_name"] = native["shape_type_name"]
    panel["faces"] = [{"id": i, "positions": [], "indices": [], "status": "not_processed"}
                      for i in range(maps["faces"].Size())]
    for item in info:
        row = {**item, "points": []}
        try:
            points = edge_points(topods.Edge(maps["edges"].FindKey(item["id"] + 1)), SETTINGS["edge_samples"])
            row.update(points=flat_numbers(points), status="ok")
        except Exception as exc:
            row.update(status="error", error=f"{type(exc).__name__}: {exc}")
        panel["edges"].append(row)
    panel.update(status="meshing")
    diagnostic.update(stage="meshing", meshing_faces=[])
    panel_counts(panel)
    checkpoint()
    for index in range(maps["faces"].Size()):
        try:
            mesh, report = mesh_face(topods.Face(maps["faces"].FindKey(index + 1)))
        except Exception as exc:
            report = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            mesh = {"positions": [], "indices": [], **report}
        panel["faces"][index] = {"id": index, **mesh}
        diagnostic["meshing_faces"].append({"id": index, **report})
        panel_counts(panel)
        # Preserve already readable edges and faces if a later OCC mesh hangs.
        if (index + 1) % 5 == 0 or report["status"] != "ok":
            checkpoint()
    panel_counts(panel)
    panel["status"] = "ok" if panel["counts"]["missing_mesh_faces"] == 0 else "partial_mesh"
    diagnostic.update(stage="complete", status="ok", render_status=panel["status"], counts=dict(panel["counts"]))
    checkpoint()


def raw_panel_and_diagnostics(path: Path) -> tuple[dict, dict]:
    from scipy.spatial import cKDTree
    from src.brepnet.data.shared_boundaries import _point_to_polyline_max
    with np.load(path, allow_pickle=False) as data:
        faces = data["pred_face"][..., :3].astype(np.float64)
        curves = data["pred_edge"][..., :3].astype(np.float64)
        conn = data["pred_edge_face_connectivity"]
    if faces.ndim != 4 or faces.shape[1:] != (16, 16, 3):
        raise ValueError("Raw faces must have shape [F,16,16,3]")
    if curves.ndim != 3 or curves.shape[1:] != (16, 3) or conn.shape != (len(curves), 3):
        raise ValueError("Raw curves/connectivity have incompatible shape")
    if not np.isfinite(faces).all() or not np.isfinite(curves).all():
        raise ValueError("Raw geometry contains nonfinite points")
    if not np.issubdtype(conn.dtype, np.integer):
        raise ValueError("Connectivity must be integral")
    panel = empty_panel("Legacy raw", "ok")
    a = np.arange(256).reshape(16, 16)[:-1, :-1].reshape(-1)
    triangles = np.stack([np.stack((a, a + 16, a + 17), -1), np.stack((a, a + 17, a + 1), -1)], 1)
    for index, face in enumerate(faces):
        panel["faces"].append({"id": index, "positions": flat_numbers(face),
                               "indices": triangles.reshape(-1).tolist(), "status": "ok"})
    for index, curve in enumerate(curves):
        panel["edges"].append({"id": index, "points": flat_numbers(curve), "status": "ok",
                               "free": False, "nonmanifold": False, "degenerate": False, "seam": False})
    panel_counts(panel)
    report = {"status": "ok", "representation": "legacy_half_edges", "counts": dict(panel["counts"]),
              "description": "Untrimmed 16x16 surface grids and all raw directed half-edges. Native free-edge flags do not apply."}
    valid = ((conn[:, 0] >= 0) & (conn[:, 0] < len(curves)) &
             (conn[:, 1:] >= 0).all(-1) & (conn[:, 1:] < len(faces)).all(-1))
    report["invalid_connectivity_rows"] = int((~valid).sum())
    groups = defaultdict(list)
    trees = [cKDTree(face.reshape(-1, 3)) for face in faces]
    distances = []
    for edge, f1, f2 in conn[valid]:
        edge, f1, f2 = int(edge), int(f1), int(f2)
        groups[tuple(sorted((f1, f2)))].append((edge, f1, f2))
        sides = [numeric_summary(tree.query(curves[edge], workers=1)[0]) for tree in (trees[f1], trees[f2])]
        distances.append({"edge_id": edge, "face_pair": [f1, f2], "sides": sides})
    reciprocal = []
    unsupported = []
    for pair, rows in sorted(groups.items()):
        if len(rows) != 2 or rows[0][1:] != rows[1][1:][::-1] or pair[0] == pair[1]:
            unsupported.append({"pair": list(pair), "rows": len(rows)})
            continue
        first, second = curves[rows[0][0]], curves[rows[1][0]]
        is_closed = [bool(np.linalg.norm(c[0] - c[-1]) <= SETTINGS["raw_closed_endpoint_threshold"])
                     for c in (first, second)]
        item = {"pair": list(pair), "edge_ids": [rows[0][0], rows[1][0]], "closed": is_closed,
                "bidirectional_point_to_polyline_max": max(_point_to_polyline_max(first, second),
                                                           _point_to_polyline_max(second, first))}
        if not any(is_closed):
            direct = np.linalg.norm(first[[0, -1]] - second[[0, -1]], axis=-1)
            reversed_gap = np.linalg.norm(first[[0, -1]] - second[[-1, 0]], axis=-1)
            chosen = direct if direct.mean() <= reversed_gap.mean() else reversed_gap
            item["endpoint_gap_mean"] = float(chosen.mean())
            item["endpoint_gap_max"] = float(chosen.max())
        else:
            item["endpoint_gap_mean"] = item["endpoint_gap_max"] = None
        reciprocal.append(item)
    report["reciprocal"] = {
        "pairs": reciprocal, "unilateral_or_ambiguous_groups": unsupported,
        "polyline_max": numeric_summary([r["bidirectional_point_to_polyline_max"] for r in reciprocal]),
        "open_endpoint_mean": numeric_summary([r["endpoint_gap_mean"] for r in reciprocal if r["endpoint_gap_mean"] is not None]),
        "note": "Closed curves omit endpoint discrepancy because their sampled origin is arbitrary; polyline distance is a discrete approximation, not exact CAD distance.",
    }
    report["curve_to_face_grid"] = {
        "definition": SETTINGS["raw_distance"], "edges": distances,
        "side_mean": numeric_summary([side["mean"] for row in distances for side in row["sides"]]),
        "side_max": numeric_summary([side["max"] for row in distances for side in row["sides"]]),
    }
    return panel, report


def common_bounds(panels) -> dict:
    points = []
    for panel in panels.values():
        points.extend(np.asarray(f["positions"], dtype=np.float64).reshape(-1, 3)
                      for f in panel["faces"] if f["positions"])
        points.extend(np.asarray(e["points"], dtype=np.float64).reshape(-1, 3)
                      for e in panel["edges"] if e["points"])
    if not points:
        return {"min": [-1, -1, -1], "max": [1, 1, 1], "center": [0, 0, 0], "radius": 1.1, "source": "empty_default"}
    points = np.concatenate(points)
    low, high = points.min(0), points.max(0)
    return {"min": flat_numbers(low), "max": flat_numbers(high), "center": flat_numbers((low + high) / 2),
            "radius": max(float((high - low).max()) * 0.55, 0.01), "source": "union_gt_raw_post_no_alignment"}


def persist(root: Path, asset: dict, diagnostics: dict):
    sid = asset["id"]
    asset["bounds"] = common_bounds(asset["panels"])
    atomic_json(root / "visualizations" / f"{sid}.json", asset)
    atomic_json(root / "diagnostics" / f"{sid}.json", diagnostics)


def geometry_case(case: dict, root: Path, comparison: Path, data_root: Path):
    sid = case["id"]
    source = {"gt": data_root / sid / "normalized_shape.step",
              "raw": comparison / "raw_predictions/posterior_mean/legacy" / sid / "data.npz",
              "post": comparison / "post/optimized/legacy" / sid / "recon_brep.step"}
    asset = {"schema": SCHEMA, "id": sid, "views": VIEWS, "settings": SETTINGS,
             "panels": {"gt": empty_panel("GT STEP"), "raw": empty_panel("Legacy raw"),
                        "post": empty_panel("Legacy optimized STEP")}}
    diagnostics = {"schema": SCHEMA, "id": sid, "status": "running", "settings": SETTINGS,
                   "sources": {key: {"path": str(path.resolve()), "sha256": digest(path)} for key, path in source.items()},
                   "panels": {}}
    persist(root, asset, diagnostics)
    for key in ("raw", "gt", "post"):
        diagnostics["stage"] = key
        diagnostic = diagnostics["panels"].setdefault(key, {"status": "running"})
        persist(root, asset, diagnostics)
        try:
            if key == "raw":
                panel, report = raw_panel_and_diagnostics(source[key])
                asset["panels"][key] = panel
                diagnostics["panels"][key] = report
            else:
                extract_step(source[key], asset["panels"][key], diagnostic,
                             lambda: persist(root, asset, diagnostics))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            asset["panels"][key].update(status="error", error=error)
            diagnostic.update(status="error", error=error)
            traceback.print_exc()
        persist(root, asset, diagnostics)
    diagnostics.update(status="complete", stage="complete")
    persist(root, asset, diagnostics)
    return asset, diagnostics


def _draw_panel(ax, panel, bounds, view):
    from matplotlib.colors import LightSource
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
    triangles = []
    for face in panel["faces"]:
        if face["indices"]:
            vertices = np.asarray(face["positions"]).reshape(-1, 3)
            triangles.append(vertices[np.asarray(face["indices"]).reshape(-1, 3)])
    raw = panel["label"] == "Legacy raw"
    ax.computed_zorder = not raw
    if triangles:
        surface = Poly3DCollection(np.concatenate(triangles), facecolors="#9bb8ce", edgecolor="none",
                                   alpha=0.30 if raw else 0.86, zsort="average", zorder=1,
                                   shade=True, lightsource=LightSource(azdeg=315, altdeg=40))
        ax.add_collection3d(surface)
    for flag, color, width in [("ordinary", "#24394b", 0.45), ("free", "#d62828", 1.6), ("nonmanifold", "#8d3bb8", 2.0)]:
        curves = [np.asarray(edge["points"]).reshape(-1, 3) for edge in panel["edges"] if edge["points"] and
                  ((flag == "ordinary" and not edge.get("free") and not edge.get("nonmanifold")) or edge.get(flag, False))]
        if curves:
            ax.add_collection3d(Line3DCollection(curves, colors=color, linewidths=width, alpha=0.9, zorder=2))
    for setter, center in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), bounds["center"]):
        setter(center - bounds["radius"], center + bounds["radius"])
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=view["elev"], azim=view["azim"])
    ax.set_axis_off()


def render_case(asset: dict, root: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [asset["panels"][key] for key in ("gt", "raw", "post")]
    figure, axes = plt.subplots(4, 3, figsize=(12, 13), subplot_kw={"projection": "3d"}, squeeze=False)
    for row, view in enumerate(VIEWS):
        for column, panel in enumerate(panels):
            ax = axes[row, column]
            _draw_panel(ax, panel, asset["bounds"], view)
            count = panel["counts"]
            title = f"{panel['label']} | {view['name']}\n{count.get('meshed_faces', 0)}/{count.get('faces', 0)} faces rendered"
            if count.get("missing_mesh_faces", 0):
                title += f"; {count['missing_mesh_faces']} missing"
            if panel["status"] in {"error", "pending", "meshing"}:
                title += f" | {panel['status']}"
            ax.set_title(title, fontsize=9, pad=0)
    figure.suptitle(f"{asset['id']} | same coordinates and camera | red: native free edges; purple: nonmanifold", fontsize=12, y=0.992)
    figure.text(0.5, 0.012, "Raw: untrimmed grids, all half-edges. Missing STEP meshes remain missing; rendered surfaces do not establish a solid.",
                ha="center", fontsize=8)
    figure.subplots_adjust(left=0, right=1, bottom=0.035, top=0.955, hspace=0.12, wspace=0)
    figure.savefig(root / "visualizations" / f"{asset['id']}.png", dpi=120, facecolor="white")
    plt.close(figure)


def render_contacts(root: Path, cases: list):
    """Ten cases per sheet, each showing one synchronized GT/raw/post view."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for offset in range(0, len(cases), 10):
        group = cases[offset:offset + 10]
        figure, axes = plt.subplots(len(group), 3, figsize=(12, 3 * len(group)),
                                    subplot_kw={"projection": "3d"}, squeeze=False)
        for row, case in enumerate(group):
            asset = json.loads((root / "visualizations" / f"{case['id']}.json").read_text())
            for column, key in enumerate(("gt", "raw", "post")):
                panel = asset["panels"][key]
                _draw_panel(axes[row, column], panel, asset["bounds"], VIEWS[0])
                counts = panel["counts"]
                axes[row, column].set_title(f"{asset['id']} | {panel['label']}\n"
                                            f"faces {counts.get('meshed_faces', 0)}/{counts.get('faces', 0)} | {panel['status']}",
                                            fontsize=9, pad=0)
        figure.suptitle(f"Legacy failures | cases {offset + 1}-{offset + len(group)} | red: free edges", fontsize=12, y=0.997)
        figure.subplots_adjust(left=0, right=1, top=0.985, bottom=0.005, hspace=0.15, wspace=0)
        figure.savefig(root / "visualizations" / f"contact_sheet_{offset // 10 + 1:02d}.png", dpi=110, facecolor="white")
        plt.close(figure)


def _worker(case, root, comparison, data_root):
    os.setsid()
    sid = case["id"]
    log = root / "metadata" / "geometry_logs" / f"{sid}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    stream = open(log, "w", buffering=1)
    os.dup2(stream.fileno(), 1)
    os.dup2(stream.fileno(), 2)
    started = time.monotonic()
    asset, diagnostics = geometry_case(case, root, comparison, data_root)
    diagnostics.update(stage="rendering")
    persist(root, asset, diagnostics)
    try:
        render_case(asset, root)
        diagnostics["static_image_status"] = "ok"
    except Exception as exc:
        diagnostics["static_image_status"] = "error"
        diagnostics["static_image_error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    diagnostics.update(stage="complete", elapsed_seconds=time.monotonic() - started)
    persist(root, asset, diagnostics)


def finalize_failed_worker(root, sid, status, elapsed, exitcode):
    path = root / "diagnostics" / f"{sid}.json"
    report = json.loads(path.read_text()) if path.exists() else {"schema": SCHEMA, "id": sid, "panels": {}}
    report.update(status=status, elapsed_seconds=elapsed, worker_exit_code=exitcode)
    atomic_json(path, report)
    asset_path = root / "visualizations" / f"{sid}.json"
    if asset_path.exists():
        asset = json.loads(asset_path.read_text())
    else:
        asset = {"schema": SCHEMA, "id": sid, "views": VIEWS, "settings": SETTINGS,
                 "panels": {"gt": empty_panel("GT STEP"), "raw": empty_panel("Legacy raw"),
                            "post": empty_panel("Legacy optimized STEP")}}
        asset["bounds"] = common_bounds(asset["panels"])
    for panel in asset["panels"].values():
        if panel["status"] in {"pending", "meshing"}:
            panel.update(status=status, error=f"Diagnostic worker {status}; partial geometry retained")
    atomic_json(asset_path, asset)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--contacts-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.timeout <= 0:
        parser.error("workers and timeout must be positive")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    manifest_path = args.root / "selected_cases.json"
    cases = json.loads(manifest_path.read_text())["cases"]
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Manifest has duplicate IDs")
    if args.ids:
        if not set(args.ids) <= set(ids):
            raise ValueError("Requested ID is not in frozen manifest")
        cases = [case for case in cases if case["id"] in args.ids]
    if args.contacts_only:
        render_contacts(args.root, cases)
        return
    protocol = {"schema": SCHEMA, "manifest_sha256": digest(manifest_path), "settings": SETTINGS,
                "comparison_root": str(args.comparison_root.resolve()), "data_root": str(args.data_root.resolve()),
                "views": VIEWS, "timeout": args.timeout, "threads_per_worker": 1,
                "source_sha256": digest(Path(__file__))}
    protocol_path = args.root / "metadata" / "geometry_protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Geometry protocol changed; preserve the existing output and use a new directory")
    atomic_json(protocol_path, protocol)
    pending = []
    for case in cases:
        report = args.root / "diagnostics" / f"{case['id']}.json"
        if report.exists():
            if not args.resume:
                raise FileExistsError(f"Existing diagnostic: {report}; use --resume")
            status = json.loads(report.read_text()).get("status")
            if status in {"complete", "timeout", "crash"}:
                continue
        pending.append(case)
    context = mp.get_context("fork")
    active = []
    completed = len(cases) - len(pending)
    try:
        while pending or active:
            while pending and len(active) < args.workers:
                case = pending.pop(0)
                process = context.Process(target=_worker, args=(case, args.root, args.comparison_root, args.data_root))
                process.start()
                active.append((process, case["id"], time.monotonic()))
            for process, sid, started in list(active):
                elapsed = time.monotonic() - started
                if process.is_alive() and elapsed < args.timeout:
                    continue
                if process.is_alive():
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.join()
                    finalize_failed_worker(args.root, sid, "timeout", elapsed, process.exitcode)
                else:
                    process.join()
                    if process.exitcode:
                        finalize_failed_worker(args.root, sid, "crash", elapsed, process.exitcode)
                active.remove((process, sid, started))
                completed += 1
                print(f"geometry {completed}/{len(cases)} {sid}", flush=True)
            if active:
                time.sleep(0.1)
    finally:
        for process, sid, started in active:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                if process.is_alive():
                    process.kill()
            process.join()
            finalize_failed_worker(args.root, sid, "interrupted", time.monotonic() - started, process.exitcode)
    for case in cases:
        if not (args.root / "visualizations" / f"{case['id']}.png").exists():
            # Rendering a retained partial asset never retries OCC or changes
            # the diagnostic outcome. This also supplies a timeout placeholder.
            asset = json.loads((args.root / "visualizations" / f"{case['id']}.json").read_text())
            render_case(asset, args.root)
    # This stays outside the numeric protocol; screenshots never affect validity.
    render_contacts(args.root, cases)


if __name__ == "__main__":
    main()
