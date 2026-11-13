"""Cache native coedge neighbors for unchanged VAE directed curve records.

The generator reads STEP only offline. Failures stay in report.json; no CAD is
filtered, no geometric nearest-neighbor fallback creates topology.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

import numpy as np
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.BRepTools import BRepTools_WireExplorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_WIRE, TopAbs_REVERSED
from OCC.Core.TopoDS import topods
from OCC.Extend.DataExchange import read_step_file

from src.common.io import sha256_file
from src.preprocess.coedge_topology import (
    SCHEMA_VERSION, curve_orientation, quotient_coedges, validate_coedge_topology,
)
from src.preprocess.extract_vertex_groups import _explore, _unique_id, match_native_faces
from src.preprocess.shared_boundaries import validate_legacy_geometry


def native_coedges(shape):
    """Keep each wire occurrence, including the two uses of one seam edge."""
    faces = list(_explore(shape, TopAbs_FACE, topods.Face))
    unique_edges, occurrences, owners, following = [], [], [], []
    uses = defaultdict(list)
    for fid, face in enumerate(faces):
        for wire in _explore(face, TopAbs_WIRE, topods.Wire):
            first = len(occurrences)
            explorer = BRepTools_WireExplorer(wire)
            while explorer.More():
                edge = topods.Edge(explorer.Current())
                uses[_unique_id(edge, unique_edges)].append(len(occurrences))
                occurrences.append(edge)
                owners.append(fid)
                following.append(len(occurrences))
                explorer.Next()
            if len(occurrences) > first:
                following[-1] = first
    mate = np.full(len(occurrences), -1, dtype=np.int64)
    for members in uses.values():
        if len(members) != 2:
            raise ValueError("Native edge must have exactly two coedge occurrences")
        a, b = members
        mate[a], mate[b] = b, a
    return faces, occurrences, np.asarray(owners, dtype=np.int64), np.asarray(following, dtype=np.int64), mate


def native_segments(occurrences, chain, segments_per_edge):
    starts, ends = [], []
    for h in chain:
        edge = occurrences[h]
        curve = BRepAdaptor_Curve(edge)
        parameters = np.linspace(curve.FirstParameter(), curve.LastParameter(), segments_per_edge + 1)
        if edge.Orientation() == TopAbs_REVERSED:
            parameters = parameters[::-1]
        points = np.asarray([curve.Value(float(t)).Coord() for t in parameters])
        starts.append(points[:-1])
        ends.append(points[1:])
    return np.concatenate(starts), np.concatenate(ends)


def source_metadata(source, step, vertex_labels, segments_per_edge, minimum_cosine):
    source_digest, step_digest = sha256_file(source), sha256_file(step)
    with np.load(vertex_labels, allow_pickle=False) as labels:
        if (labels["source_sha256"].item() != source_digest
                or labels["step_sha256"].item() != step_digest):
            raise ValueError("Native vertex labels refer to different NPZ/STEP sources")
        face_tolerance = json.loads(labels["parameters_json"].item())["face_tolerance"]
    parameters = {"face_tolerance": face_tolerance, "segments_per_edge": segments_per_edge,
                  "minimum_orientation_cosine": minimum_cosine}
    return {"schema_version": np.array(SCHEMA_VERSION),
            "source_sha256": np.array(source_digest), "step_sha256": np.array(step_digest),
            "vertex_labels_sha256": np.array(sha256_file(vertex_labels)),
            "parameters_json": np.array(json.dumps(parameters, sort_keys=True))}, parameters


def extract_coedge_topology(source, step, vertex_labels, *, segments_per_edge=64, minimum_cosine=0.5):
    metadata, parameters = source_metadata(source, step, vertex_labels, segments_per_edge, minimum_cosine)
    with np.load(source, allow_pickle=False) as data:
        faces, curves, connectivity = (data[key] for key in
                                       ("sample_points_faces", "sample_points_lines", "edge_face_connectivity"))
    validate_legacy_geometry(faces, curves, connectivity, Path(source).parent.name)
    shape = read_step_file(str(step), verbosity=False)
    native_faces, occurrences, owner, following, mate = native_coedges(shape)
    mapping, face_error = match_native_faces(native_faces, faces[..., :3].astype(float), parameters["face_tolerance"])
    owner = np.asarray([mapping[int(f)] for f in owner], dtype=np.int64)
    quotient = quotient_coedges(owner, following, mate, connectivity)
    chains = quotient.pop("native_chains")
    orientation, scores, errors = [], [], []
    for raw_id, chain in enumerate(chains):
        starts, ends = native_segments(occurrences, chain, segments_per_edge)
        direction, score, error = curve_orientation(curves[raw_id, :, :3], starts, ends, minimum_cosine)
        orientation.append(direction)
        scores.append(score)
        errors.append(error)
    ordered = connectivity[np.argsort(connectivity[:, 0])]
    result = {**metadata, **{key: np.asarray(value) for key, value in quotient.items()},
              "source_edge_ids": ordered[:, 0], "owner_faces": ordered[:, 1],
              "orientation": np.asarray(orientation, dtype=np.int8),
              "orientation_cosine": np.asarray(scores), "orientation_projection_error": np.asarray(errors),
              "native_chain_lengths": np.asarray([len(chain) for chain in chains], dtype=np.int64),
              "face_grid_max_error": np.array(face_error)}
    validate_coedge_topology(result, source_sha256=metadata["source_sha256"].item(), connectivity=connectivity)
    return result


def process_sample(job):
    sid, data_root, vertex_root, output_root, segments, threshold, resume = job
    source = Path(data_root) / sid / "data.npz"
    step = Path(data_root) / sid / "normalized_shape.step"
    labels = Path(vertex_root) / sid / "vertex_groups.npz"
    destination = Path(output_root) / sid / "coedge_topology.npz"
    try:
        if destination.exists():
            if not resume:
                raise FileExistsError(f"Refusing to overwrite {destination}")
            metadata, _ = source_metadata(source, step, labels, segments, threshold)
            with np.load(destination, allow_pickle=False) as cache:
                arrays = {key: cache[key] for key in cache.files}
            with np.load(source, allow_pickle=False) as data:
                connectivity = data["edge_face_connectivity"]
            validate_coedge_topology(arrays, source_sha256=metadata["source_sha256"].item(),
                                    step_sha256=metadata["step_sha256"].item(), connectivity=connectivity)
            for key in ("vertex_labels_sha256", "parameters_json"):
                if arrays[key].item() != metadata[key].item():
                    raise ValueError(f"Existing coedge cache {key} mismatch; choose a new output version")
            reused = True
        else:
            arrays = extract_coedge_topology(source, step, labels, segments_per_edge=segments, minimum_cosine=threshold)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, **arrays)
            temporary.replace(destination)
            reused = False
        return {"id": sid, "status": "ok", "reused": reused, "rows": len(arrays["orientation"]),
                "seam_occurrences": int(arrays["seam_removed_count"]),
                "merged_rows": int((arrays["native_chain_lengths"] > 1).sum()),
                "reversed_rows": int((arrays["orientation"] < 0).sum()),
                "min_abs_orientation_cosine": float(np.abs(arrays["orientation_cosine"]).min(initial=1)),
                "max_projection_error": float(arrays["orientation_projection_error"].max(initial=0)),
                "face_grid_max_error": float(arrays["face_grid_max_error"])}
    except Exception as exc:
        return {"id": sid, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--vertex-group-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--segments-per-edge", type=int, default=64)
    parser.add_argument("--minimum-cosine", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true", help="Reuse only source/parameter-verified existing sidecars")
    args = parser.parse_args()
    if args.output_root.exists() and not args.resume:
        parser.error("Output root exists; choose a new version or explicitly --resume")
    if args.workers < 1 or args.segments_per_edge < 2 or not 0 < args.minimum_cosine <= 1:
        parser.error("Require workers >= 1, segments >= 2 and 0 < minimum cosine <= 1")
    ids = list(dict.fromkeys(sid for split in args.split for sid in split.read_text().split()))
    if any(len(sid) != 8 or not sid.isdigit() for sid in ids):
        parser.error("Split must contain eight-digit CAD IDs")
    args.output_root.mkdir(parents=True, exist_ok=True)
    jobs = [(sid, str(args.data_root), str(args.vertex_group_root), str(args.output_root),
             args.segments_per_edge, args.minimum_cosine, args.resume) for sid in ids]
    pool = ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) if args.workers > 1 else None
    results = []
    try:
        iterator = pool.map(process_sample, jobs, chunksize=8) if pool else map(process_sample, jobs)
        with (args.output_root / "progress.jsonl").open("w") as log:
            for index, result in enumerate(iterator, 1):
                results.append(result)
                log.write(json.dumps(result) + "\n")
                log.flush()
                if result["status"] != "ok" or index % 100 == 0 or index == len(ids):
                    print(f"{index}/{len(ids)} {result}", flush=True)
    finally:
        if pool:
            pool.shutdown()
    report = {"parameters": {key: str(value) for key, value in vars(args).items()},
              "split_sha256": {str(p): sha256_file(p) for p in args.split},
              "generator_sha256": sha256_file(Path(__file__)),
              "topology_sha256": sha256_file(Path(__file__).with_name("coedge_topology.py")),
              "total": len(ids), "success": sum(row["status"] == "ok" for row in results), "samples": results}
    (args.output_root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return int(report["success"] != report["total"])


if __name__ == "__main__":
    raise SystemExit(main())
