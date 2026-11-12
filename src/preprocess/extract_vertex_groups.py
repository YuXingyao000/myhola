"""Offline native STEP vertex labels for the retained, directed NPZ curves.

Run with explicit roots and split lists. OCC is only needed by this generator,
not by the training loader or loss. Failed objects remain in the report.
"""
from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from OCC.Extend.DataExchange import read_step_file
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_WIRE
from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.TopoDS import topods
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepTools import BRepTools_WireExplorer

from src.common.io import sha256_file
from src.preprocess.shared_boundaries import validate_legacy_geometry
from src.preprocess.vertex_groups import SCHEMA_VERSION, canonical_edges, corner_pairs


def _unique_id(shape, shapes):
    for i, other in enumerate(shapes):
        if shape.IsSame(other):
            return i
    shapes.append(shape)
    return len(shapes) - 1


def _explore(shape, kind, cast):
    ex = TopExp_Explorer(shape, kind)
    while ex.More():
        yield cast(ex.Current())
        ex.Next()


def read_native_topology(shape):
    """Keep edge owner occurrences: a seam has two occurrences on one face."""
    faces = list(_explore(shape, TopAbs_FACE, topods.Face))
    edges, vertices, records = [], [], []
    for fid, face in enumerate(faces):
        for wire in _explore(face, TopAbs_WIRE, topods.Wire):
            ex = BRepTools_WireExplorer(wire)
            while ex.More():
                edge = topods.Edge(ex.Current())
                eid = _unique_id(edge, edges)
                if eid == len(records):
                    vids = [_unique_id(topexp.FirstVertex(edge, True), vertices),
                            _unique_id(topexp.LastVertex(edge, True), vertices)]
                    records.append({"vertices": vids, "owners": []})
                records[eid]["owners"].append(fid)
                ex.Next()
    xyz = np.asarray([BRep_Tool.Pnt(v).Coord() for v in vertices]).reshape(-1, 3)
    return faces, records, xyz


def match_native_faces(faces, gt, tolerance):
    """Reproduce the extractor's [v,u] grid, not a new surface fitting rule."""
    grids = []
    for face in faces:
        surface = BRepAdaptor_Surface(face)
        us = np.linspace(surface.FirstUParameter(), surface.LastUParameter(), 16)
        vs = np.linspace(surface.FirstVParameter(), surface.LastVParameter(), 16)
        grids.append([surface.Value(u, v).Coord() for v in vs for u in us])
    grids = np.asarray(grids).reshape(-1, 16, 16, 3)
    if len(grids) != len(gt):
        raise ValueError("STEP/NPZ face counts differ")
    cost = np.linalg.norm(grids[:, None] - gt[None], axis=-1).max(axis=(2, 3))
    rows, cols = linear_sum_assignment(cost)
    max_error = float(cost[rows, cols].max(initial=0))
    if max_error > tolerance or np.any((cost <= tolerance).sum(axis=1) != 1):
        raise ValueError(f"STEP/NPZ face correspondence failed; max error {max_error}")
    return dict(zip(rows.tolist(), cols.tolist())), max_error


def chain_vertices(records):
    """Open chain: terminal vertices only. Closed chain: retain cut candidates."""
    degree, neighbors = Counter(), defaultdict(set)
    for edge in records:
        a, b = edge["vertices"]
        degree.update((a, b))
        neighbors[a].add(b)
        neighbors[b].add(a)
    if not degree:
        raise ValueError("NPZ face pair has no native edges")
    reached, stack = set(), [min(degree)]
    while stack:
        v = stack.pop()
        if v not in reached:
            reached.add(v)
            stack.extend(neighbors[v] - reached)
    terminals = [v for v, count in degree.items() if count == 1]
    if len(reached) != len(degree):
        raise ValueError("Disconnected native edges for one face pair")
    if len(terminals) == 2 and all(count in (1, 2) for count in degree.values()):
        return sorted(terminals), False
    if all(count == 2 for count in degree.values()):
        return sorted(degree), True
    raise ValueError("Branched native edges for one face pair")


def bind_endpoints(points, candidates, vertices, tolerance):
    """Coordinate matching selects an existing native ID; it never defines IDs."""
    distances = np.linalg.norm(points[:, None] - vertices[candidates][None], axis=-1)
    within = distances <= tolerance
    nearest = distances.argmin(axis=1)
    valid = within.sum(axis=1) == 1
    ids = np.where(valid, np.asarray(candidates)[nearest], -1)
    return ids, distances[np.arange(2), nearest]


def extract_vertex_groups(source, step, *, face_tolerance=1e-5, vertex_tolerance=1e-5):
    with np.load(source, allow_pickle=False) as data:
        gt = data["sample_points_faces"]
        curves = data["sample_points_lines"]
        conn = data["edge_face_connectivity"]
    validate_legacy_geometry(gt, curves, conn, Path(source).parent.name)
    shape = read_step_file(str(step), verbosity=False)
    faces, records, vertices = read_native_topology(shape)
    mapping, face_error = match_native_faces(faces, gt[..., :3].astype(float), face_tolerance)
    native_pairs, seam_count = defaultdict(list), 0
    for record in records:
        owners = [mapping[f] for f in record["owners"]]
        if len(owners) != 2:
            raise ValueError("Native edge must have two owner occurrences")
        if owners[0] == owners[1]:
            seam_count += 1
        else:
            native_pairs[tuple(sorted(owners))].append(record)
    source_ids, pairs = canonical_edges(conn)
    endpoint_ids = np.full((len(pairs), 2), -1, dtype=np.int64)
    errors = np.zeros((len(pairs), 2), dtype=np.float64)
    closed = np.zeros(len(pairs), dtype=bool)
    merged_count = 0
    for k, (edge, pair) in enumerate(zip(source_ids, pairs)):
        native = native_pairs[tuple(pair)]
        candidates, closed[k] = chain_vertices(native)
        endpoint_ids[k], errors[k] = bind_endpoints(
            curves[edge, [0, -1], :3].astype(float), candidates, vertices, vertex_tolerance,
        )
        merged_count += len(native) > 1
    params = {"face_tolerance": face_tolerance, "vertex_tolerance": vertex_tolerance}
    return {
        "schema_version": np.array(SCHEMA_VERSION),
        "source_sha256": np.array(sha256_file(source)), "step_sha256": np.array(sha256_file(step)),
        "parameters_json": np.array(json.dumps(params, sort_keys=True)),
        "source_edge_ids": source_ids, "face_pairs": pairs, "endpoint_vertex_ids": endpoint_ids,
        "endpoint_binding_error": errors, "is_closed": closed,
        "face_grid_max_error": np.array(face_error),
        "seam_count": np.array(seam_count), "merged_curve_count": np.array(merged_count),
        **corner_pairs(source_ids, endpoint_ids),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--face-tolerance", type=float, default=1e-5)
    parser.add_argument("--vertex-tolerance", type=float, default=1e-5)
    args = parser.parse_args()
    if args.output_root.exists():
        parser.error("Output directory exists; choose a new label version")
    ids = list(dict.fromkeys(sid for split in args.split for sid in split.read_text().split()))
    if any(len(sid) != 8 or not sid.isdigit() for sid in ids):
        parser.error("Split must contain eight-digit CAD IDs")
    if not all(np.isfinite(t) and t > 0 for t in (args.face_tolerance, args.vertex_tolerance)):
        parser.error("Tolerances must be finite and positive")
    args.output_root.mkdir(parents=True)
    results = []
    for index, sid in enumerate(ids):
        try:
            folder = args.data_root / sid
            data = extract_vertex_groups(folder / "data.npz", folder / "normalized_shape.step",
                                         face_tolerance=args.face_tolerance, vertex_tolerance=args.vertex_tolerance)
            target = args.output_root / sid
            target.mkdir()
            np.savez_compressed(target / "vertex_groups.npz", **data)
            results.append({"id": sid, "status": "ok", "curves": len(data["source_edge_ids"]),
                            "groups": len(data["corner_group_vertices"]), "pairs": len(data["corner_pairs"]),
                            "unmatched_endpoints": int((data["endpoint_vertex_ids"] < 0).sum())})
        except Exception as exc:
            results.append({"id": sid, "status": "failed", "error": str(exc)})
        print(f"{index + 1}/{len(ids)} {results[-1]}", flush=True)
    report = {"parameters": {key: str(value) for key, value in vars(args).items()},
              "split_sha256": {str(p): sha256_file(p) for p in args.split},
              "generator_sha256": sha256_file(Path(__file__)), "samples": results}
    (args.output_root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return int(any(row["status"] == "failed" for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
