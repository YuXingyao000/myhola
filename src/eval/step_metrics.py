"""Native STEP validity, deterministic primitive sampling and squared Chamfer distance.

Shared by VAE comparisons and diffusion/preference evaluation. OCC, NumPy and
trimesh are imported inside each function so CPU workers can configure
threading first.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

from src.common.io import fingerprint, sha256_file
from src.common.step_io import read_step_at_precision

SAMPLING = {
    "version": "native_step_per_primitive_v1",
    "face_points_per_face": 1024,
    "face_distribution": "triangle-area-weighted within each face; equal count per face",
    "mesh_linear_deflection": 0.01,
    "mesh_angular_deflection": 0.1,
    "edge_points_per_edge": 128,
    "edge_distribution": "uniform native curve parameter including endpoints",
    "degenerate_edges": "repeat a real topological vertex; never use an origin fallback",
    "coordinates": "as stored; no alignment, scaling or rotation",
    "cd": "mean(pred-to-gt squared nearest distance) + mean(gt-to-pred squared nearest distance)",
    "read_precision_mode": 0,
    "seed": 42,
}


def stable_seed(sample_id: str, stage: str) -> int:
    return int.from_bytes(hashlib.sha256(f"42:{sample_id}:{stage}".encode()).digest()[:4], "little")


def squared_chamfer(predicted, target) -> dict[str, float]:
    import numpy as np
    from scipy.spatial import cKDTree
    predicted, target = np.asarray(predicted, dtype=np.float64), np.asarray(target, dtype=np.float64)
    for name, points in (("predicted", predicted), ("target", target)):
        if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
            raise ValueError(f"{name} point cloud must be finite nonempty [N,3]")
    forward = cKDTree(target).query(predicted, workers=1)[0]
    backward = cKDTree(predicted).query(target, workers=1)[0]
    accuracy = float(np.mean(forward ** 2))
    completeness = float(np.mean(backward ** 2))
    return {"accuracy": accuracy, "completeness": completeness, "cd": accuracy + completeness}


def native_validity(shape) -> dict[str, Any]:
    """Analyze the as-read shape before any ShapeFix mutation or triangulation."""
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX, TopAbs_WIRE
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopoDS import topods
    from src.common.occ_utils import get_primitives
    solid_type = shape.ShapeType() == TopAbs_SOLID
    checked = bool(BRepCheck_Analyzer(shape).IsValid())
    primitives = {"face": get_primitives(shape, TopAbs_FACE, v_remove_half=True),
                  "edge": get_primitives(shape, TopAbs_EDGE, v_remove_half=True),
                  "vertex": get_primitives(shape, TopAbs_VERTEX, v_remove_half=True)}
    tolerances = {}
    for name, cast in (("face", topods.Face), ("edge", topods.Edge), ("vertex", topods.Vertex)):
        values = [float(BRep_Tool.Tolerance(cast(item))) for item in primitives[name]]
        tolerances[f"native_{name}_tolerance_max"] = max(values) if values else None
    wire_counts = [len(get_primitives(face, TopAbs_WIRE, v_remove_half=True)) for face in primitives["face"]]
    return {
        "step_readable": True,
        "shape_type": int(shape.ShapeType()),
        "native_is_solid_type": solid_type,
        "native_brepcheck_valid": checked,
        "native_valid_solid": bool(solid_type and checked),
        "num_faces": len(primitives["face"]),
        "num_edges": len(primitives["edge"]),
        "num_vertices": len(primitives["vertex"]),
        "num_face_wire_occurrences": sum(wire_counts),
        "num_faces_with_multiple_wires": sum(n > 1 for n in wire_counts),
        **tolerances,
    }


def compatibility_validity(path: Path) -> bool:
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.ShapeFix import ShapeFix_ShapeTolerance
    from OCC.Core.TopAbs import TopAbs_SOLID
    shape = read_step_at_precision(path, mode=1, precision=0.1)
    if shape.ShapeType() != TopAbs_SOLID:
        return False
    ShapeFix_ShapeTolerance().SetTolerance(shape, 0.1)
    return bool(BRepCheck_Analyzer(shape).IsValid())


def sample_native_geometry(shape, seed: int) -> dict[str, Any]:
    """Deterministic samples of native STEP faces/edges, including locations.

    No face is silently skipped. A failed primitive makes this object's geometry
    metric unavailable, independently of its already recorded solid validity.
    """
    import numpy as np
    import trimesh
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
    from OCC.Core.TopLoc import TopLoc_Location
    from src.common.occ_utils import get_primitives
    faces = get_primitives(shape, TopAbs_FACE, v_remove_half=True)
    edges = get_primitives(shape, TopAbs_EDGE, v_remove_half=True)
    if not faces or not edges:
        raise ValueError("STEP geometry comparison requires nonempty native faces and edges")
    face_points, edge_points = [], []
    for index, face in enumerate(faces):
        BRepMesh_IncrementalMesh(face, SAMPLING["mesh_linear_deflection"], False, SAMPLING["mesh_angular_deflection"])
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, location)
        if triangulation is None or triangulation.NbTriangles() == 0:
            raise ValueError(f"Native face {index} has no triangulation")
        transform = location.Transformation()
        vertices = [triangulation.Node(i).Transformed(transform).Coord() for i in range(1, triangulation.NbNodes() + 1)]
        triangles = [triangulation.Triangle(i).Get() for i in range(1, triangulation.NbTriangles() + 1)]
        mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(triangles) - 1, process=False)
        if not np.isfinite(mesh.area) or mesh.area <= 0:
            raise ValueError(f"Native face {index} has no finite positive mesh area")
        points, _ = trimesh.sample.sample_surface(mesh, SAMPLING["face_points_per_face"], seed=(seed + index) % (2 ** 32))
        face_points.append(points)
    degenerate_count = 0
    for index, edge in enumerate(edges):
        if BRep_Tool.Degenerated(edge):
            vertices = get_primitives(edge, TopAbs_VERTEX, v_remove_half=True)
            if not vertices:
                raise ValueError(f"Degenerate native edge {index} has no real vertex")
            points = np.repeat(np.asarray(BRep_Tool.Pnt(vertices[0]).Coord())[None], SAMPLING["edge_points_per_edge"], axis=0)
            degenerate_count += 1
        else:
            curve = BRepAdaptor_Curve(edge)
            start, end = curve.FirstParameter(), curve.LastParameter()
            if not np.isfinite([start, end]).all():
                raise ValueError(f"Native edge {index} has an unbounded parameter interval")
            points = np.asarray([curve.Value(float(u)).Coord() for u in np.linspace(start, end, SAMPLING["edge_points_per_edge"])])
        edge_points.append(points)
    face_points, edge_points = np.concatenate(face_points), np.concatenate(edge_points)
    if not np.isfinite(face_points).all() or not np.isfinite(edge_points).all():
        raise ValueError("STEP sampling produced NaN/Inf")
    return {"face_points": face_points, "edge_points": edge_points,
            "num_faces": len(faces), "num_edges": len(edges), "num_degenerate_edges": degenerate_count}


def load_gt_samples(path: Path, cache_root: Path, sample_id: str) -> dict[str, Any]:
    import numpy as np
    seed = stable_seed(sample_id, "geometry")
    key = fingerprint({"step_sha256": sha256_file(path), "sampling": SAMPLING, "seed": seed})
    cache = cache_root / f"{sample_id}-{key}.npz"
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as data:
            return {name: data[name] for name in data.files}
    data = sample_native_geometry(read_step_at_precision(path), seed)
    cache_root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache_root, prefix=sample_id + ".", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **data)
    os.replace(temporary, cache)
    return data


def validate_raw_prediction(path: Path) -> dict[str, Any]:
    import numpy as np
    with np.load(path, allow_pickle=False) as data:
        face = data["pred_face"]
        edge = data["pred_edge"]
        conn = data["pred_edge_face_connectivity"]
        representation = data["representation"].item() if "representation" in data else "legacy_half_edges"
        if face.ndim != 4 or face.shape[-1] != 3 or not len(face) or min(face.shape[1:3]) < 2:
            raise ValueError("pred_face must be nonempty [F,U,V,3] with U,V>=2")
        if edge.ndim != 3 or edge.shape[-1] != 3 or edge.shape[1] < 2:
            raise ValueError("pred_edge must be [E,L,3] with L>=2")
        if not np.isfinite(face).all() or not np.isfinite(edge).all():
            raise ValueError("Raw geometry contains NaN/Inf")
        if conn.shape != (len(edge), 3) or not np.issubdtype(conn.dtype, np.integer):
            raise ValueError("Raw connectivity must be integer [E,3]")
        if not np.array_equal(np.sort(conn[:, 0]), np.arange(len(edge))):
            raise ValueError("Raw edge IDs must cover the curve array exactly once")
        if representation == "shared_boundary_v1":
            from src.preprocess.shared_boundaries import shared_prediction_diagnostics
            # Invalid/self face assignments remain raw diagnostics; the existing
            # explicit shared adapter handles them using predictions only.
            shared_prediction_diagnostics(len(face), edge, conn, data["pred_edge_confidence"])
        elif representation == "legacy_half_edges":
            if (conn[:, 1:] < 0).any() or (conn[:, 1:] >= len(face)).any():
                raise ValueError("Legacy raw connectivity has an invalid face ID")
        else:
            raise ValueError(f"Unknown prediction representation {representation!r}")
    return {"representation": representation, "raw_num_faces": len(face), "raw_num_edges": len(edge)}
