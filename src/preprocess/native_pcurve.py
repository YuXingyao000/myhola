"""Export native coedges, including seams, in the retained face-grid charts.

python -m src.preprocess.native_pcurve --data-root ... --output-root ...
    --split ... --workers 8 --report ...
"""

import argparse
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from src.common.io import sha256_file

SCHEMA = "native_pcurve_v4"


def pair_geometry_key(a, b, uv):
    """Symmetric UV order: independent of face IDs and rigid XYZ rotation."""
    keys = [tuple(np.r_[uv[h].mean(axis=0), uv[h].reshape(-1)]) for h in (a, b)]
    return tuple(sorted(keys))


def extract_native(source, vertex_labels):
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Curve2d, BRepAdaptor_Surface
    from OCC.Core.BRepTools import BRepTools_WireExplorer
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_WIRE, TopAbs_REVERSED
    from OCC.Core.TopExp import TopExp_Explorer, topexp
    from OCC.Core.TopoDS import topods
    from OCC.Core.gp import gp_Pnt, gp_Vec
    from OCC.Extend.DataExchange import read_step_file
    from src.preprocess.extract_vertex_groups import match_native_faces

    def explore(shape, kind, cast):
        ex = TopExp_Explorer(shape, kind)
        while ex.More():
            yield cast(ex.Current())
            ex.Next()

    with np.load(source / "data.npz") as old:
        face_points = old["sample_points_faces"].copy()
    shape = read_step_file(str(source / "normalized_shape.step"), verbosity=False)
    faces = list(explore(shape, TopAbs_FACE, topods.Face))
    with np.load(vertex_labels) as labels:
        if str(labels["source_sha256"]) != sha256_file(source / "data.npz") or str(
            labels["step_sha256"]
        ) != sha256_file(source / "normalized_shape.step"):
            raise ValueError("reviewed_face_correspondence_provenance_mismatch")
        face_tolerance = float(json.loads(str(labels["parameters_json"]))["face_tolerance"])
    face_map, face_error = match_native_faces(faces, face_points[..., :3], face_tolerance)
    edges, rows, loops = [], [], []
    trace_error = 0.0
    trace_exceeds_tolerance = 0
    for native_owner, face in enumerate(faces):
        owner = face_map[native_owner]
        surface = BRepAdaptor_Surface(face)
        lo = np.array([surface.FirstUParameter(), surface.FirstVParameter()])
        hi = np.array([surface.LastUParameter(), surface.LastVParameter()])
        if not np.isfinite(np.r_[lo, hi]).all() or (hi <= lo).any():
            raise ValueError("invalid_face_chart")
        for wire in explore(face, TopAbs_WIRE, topods.Wire):
            loop = []
            ex = BRepTools_WireExplorer(wire, face)
            while ex.More():
                edge = topods.Edge(ex.Current())
                eid = next((i for i, other in enumerate(edges) if edge.IsSame(other)), len(edges))
                if eid == len(edges):
                    edges.append(edge)
                if BRep_Tool.Degenerated(edge):
                    raise ValueError("degenerate_edge")
                curve, pc = BRepAdaptor_Curve(edge), BRepAdaptor_Curve2d(edge, face)
                t = np.linspace(curve.FirstParameter(), curve.LastParameter(), 16)
                reverse = edge.Orientation() == TopAbs_REVERSED
                if reverse:
                    t = t[::-1]
                uv = np.asarray([pc.Value(float(a)).Coord() for a in t])
                xyz, tangent = [], []
                for a in t:
                    point, derivative = gp_Pnt(), gp_Vec()
                    curve.D1(float(a), point, derivative)
                    xyz.append(point.Coord())
                    tangent.append(derivative.Coord())
                xyz, tangent = np.asarray(xyz), np.asarray(tangent)
                tangent *= -1 if reverse else 1
                tangent /= np.maximum(np.linalg.norm(tangent, axis=-1, keepdims=True), 1e-12)
                trace = np.asarray([surface.Value(float(u), float(v)).Coord() for u, v in uv])
                local_trace_error = float(np.linalg.norm(trace - xyz, axis=-1).max())
                trace_error = max(trace_error, local_trace_error)
                edge_tolerance = BRep_Tool.Tolerance(edge)
                # Native STEP is a tolerance-based representation; equal curve
                # parameters can still have a nonzero trace residual. Preserve
                # those labels and report their error floor instead of silently
                # changing geometry or filtering the fixed CAD split.
                trace_exceeds_tolerance += local_trace_error > edge_tolerance + 2e-6
                start_tolerance = BRep_Tool.Tolerance(topexp.FirstVertex(edge, True))
                end_tolerance = BRep_Tool.Tolerance(topexp.LastVertex(edge, True))
                normalized_uv = (uv - lo) / (hi - lo)
                if not np.isfinite(normalized_uv).all():
                    raise ValueError("nonfinite_uv")
                loop.append(len(rows))
                rows.append(
                    dict(
                        owner=owner,
                        edge=eid,
                        uv=normalized_uv,
                        start_tolerance=start_tolerance,
                        end_tolerance=end_tolerance,
                        trace=trace,
                        points=np.concatenate([xyz, tangent], axis=-1),
                    )
                )
                ex.Next()
            if not loop:
                raise ValueError("empty_wire")
            loops.append(loop)
    if not rows:
        raise ValueError("no_native_boundary")
    owner = np.array([r["owner"] for r in rows], dtype=np.int64)
    uv = np.stack([r["uv"] for r in rows])
    points = np.stack([r["points"] for r in rows])
    nxt = np.full(len(rows), -1, dtype=np.int64)
    for loop in loops:
        nxt[loop] = loop[1:] + loop[:1]
    by_edge = defaultdict(list)
    for h, row in enumerate(rows):
        by_edge[row["edge"]].append(h)
    mate = np.full_like(nxt, -1)
    pairs = defaultdict(list)
    for uses in by_edge.values():
        if len(uses) != 2:
            raise ValueError("edge_use_count_not_two")
        a, b = uses
        # Side 0 uses the lower owner; for a seam choose its UV branch explicitly.
        key_a = (int(owner[a]), *uv[a].mean(axis=0), *uv[a].reshape(-1))
        key_b = (int(owner[b]), *uv[b].mean(axis=0), *uv[b].reshape(-1))
        if key_b < key_a:
            a, b = b, a
        mate[a], mate[b] = b, a
        pairs[(int(owner[a]), int(owner[b]))].append((a, b))
    slot, side = np.zeros_like(nxt), np.zeros_like(nxt)
    for group in pairs.values():
        # Both directions refer to the same slot. Never rank by the smaller
        # face index: that would change the target under a face permutation.
        group.sort(key=lambda ab: pair_geometry_key(*ab, uv))
        for index, (a, b) in enumerate(group):
            slot[[a, b]] = index
            side[b] = 1
    xyz_gap = float(np.linalg.norm(points[:, -1, :3] - points[nxt, 0, :3], axis=-1).max())
    uv_gap = float(np.linalg.norm(uv[:, -1] - uv[nxt, 0], axis=-1).max())
    mate_gap = float(np.linalg.norm(points[..., :3] - points[mate, ::-1, :3], axis=-1).max())
    native_tolerance = np.array([r["end_tolerance"] + rows[k]["start_tolerance"] for r, k in zip(rows, nxt)]) + 4e-6
    endpoint_gaps = np.linalg.norm(points[:, -1, :3] - points[nxt, 0, :3], axis=-1)
    traces = np.stack([r["trace"] for r in rows])
    trace_gaps = np.linalg.norm(traces[:, -1] - traces[nxt, 0], axis=-1)
    if mate_gap > 2e-6:
        raise ValueError(f"mate_parameter_direction_inconsistent: gap={mate_gap}")
    endpoint_exceeds_tolerance = int(
        np.count_nonzero((endpoint_gaps > native_tolerance) | (trace_gaps > native_tolerance))
    )
    vertices = np.full_like(nxt, -1)
    rotate = nxt[mate]
    vertex_count = 0
    for h in range(len(rows)):
        if vertices[h] >= 0:
            continue
        k = h
        while vertices[k] < 0:
            vertices[k] = vertex_count
            k = rotate[k]
        vertex_count += 1
    arrays = dict(
        schema=np.asarray(SCHEMA),
        sample_points_faces=face_points.astype(np.float32),
        sample_points_lines=points.astype(np.float32),
        pcurve_uv=uv.astype(np.float32),
        owner=owner,
        mate=mate,
        next=nxt,
        slot=slot,
        side=side,
        vertex=vertices,
        gt_next_uv_gap=np.asarray(uv_gap),
        gt_next_xyz_gap=np.asarray(xyz_gap),
        gt_trace_error=np.asarray(trace_error),
        face_match_error=np.asarray(face_error),
        face_tolerance=np.asarray(face_tolerance),
        gt_trace_exceeds_tolerance=np.asarray(trace_exceeds_tolerance),
        gt_endpoint_exceeds_tolerance=np.asarray(endpoint_exceeds_tolerance),
        source_npz_sha256=np.asarray(sha256_file(source / "data.npz")),
        source_step_sha256=np.asarray(sha256_file(source / "normalized_shape.step")),
    )
    info = dict(
        faces=len(faces),
        coedges=len(rows),
        vertices=vertex_count,
        seams=sum(i == j for i, j in pairs),
        max_slots=max(map(len, pairs.values())),
        face_error=face_error,
        trace_error=trace_error,
        uv_gap=uv_gap,
        trace_exceeds_tolerance=trace_exceeds_tolerance,
        endpoint_exceeds_tolerance=endpoint_exceeds_tolerance,
    )
    return arrays, info


def export_one(task):
    source, output, sample, vertex_root = map(Path, task)
    destination = output / sample / "data.npz"
    try:
        if destination.exists():
            with np.load(destination) as data:
                if (
                    str(data["schema"]) != SCHEMA
                    or str(data["source_npz_sha256"]) != sha256_file(source / sample / "data.npz")
                    or str(data["source_step_sha256"]) != sha256_file(source / sample / "normalized_shape.step")
                ):
                    raise ValueError("existing_output_provenance_mismatch")
                info = dict(max_slots=int(data["slot"].max()) + 1, coedges=len(data["owner"]))
        else:
            arrays, info = extract_native(source / sample, vertex_root / sample / "vertex_groups.npz")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, **arrays)
            temporary.replace(destination)
        return dict(id=str(sample), status="ok", **info)
    except Exception as exc:
        return dict(id=str(sample), status="failed", error=f"{type(exc).__name__}: {exc}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--vertex-root",
        type=Path,
        required=True,
        help="Reviewed STEP/NPZ correspondence tolerances, provenance-checked",
    )
    parser.add_argument("--split", type=Path, action="append", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.resolve() == args.data_root.resolve():
        raise ValueError("Native export requires a separate output root")
    ids = list(dict.fromkeys(s for p in args.split for s in p.read_text().splitlines() if s.strip()))
    tasks = [(args.data_root, args.output_root, s, args.vertex_root) for s in ids]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = []
        for result in pool.map(export_one, tasks, chunksize=8):
            results.append(result)
            if result["status"] == "failed":
                print(json.dumps(result), flush=True)
            if len(results) % 100 == 0:
                print(
                    json.dumps(
                        dict(done=len(results), total=len(ids), statuses=dict(Counter(r["status"] for r in results)))
                    ),
                    flush=True,
                )
    report = dict(
        schema=SCHEMA,
        data_root=str(args.data_root),
        output_root=str(args.output_root),
        splits={str(p): sha256_file(p) for p in args.split},
        total=len(ids),
        success=sum(r["status"] == "ok" for r in results),
        max_slots=max((r.get("max_slots", 0) for r in results), default=0),
        rows=results,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
