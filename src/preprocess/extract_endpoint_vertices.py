"""Bind both directed uses to native STEP vertices without changing geometry."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np

from src.common.io import sha256_file
from src.preprocess.endpoint_vertices import SCHEMA_VERSION, load_endpoint_vertices
from src.preprocess.model_lists import load_model_ids
from src.preprocess.vertex_groups import canonical_edges, load_vertex_groups


def extract_endpoint_vertices(source, step, canonical_labels):
    # OCC is an offline dependency; the loader above remains NumPy-only.
    from OCC.Extend.DataExchange import read_step_file
    from src.preprocess.extract_vertex_groups import (
        bind_endpoints, chain_vertices, match_native_faces, read_native_topology,
    )

    with np.load(source, allow_pickle=False) as data:
        faces = data['sample_points_faces']
        curves = data['sample_points_lines']
        connectivity = data['edge_face_connectivity']
    source_digest, step_digest = sha256_file(source), sha256_file(step)
    load_vertex_groups(canonical_labels, source_sha256=source_digest, connectivity=connectivity)
    with np.load(canonical_labels, allow_pickle=False) as previous:
        if previous['step_sha256'].item() != step_digest:
            raise ValueError('Canonical vertex labels refer to a different STEP')
        parameters = json.loads(previous['parameters_json'].item())
        canonical_vertices = previous['endpoint_vertex_ids']
    if not np.isfinite(faces).all() or not np.isfinite(curves).all():
        raise ValueError('Nonfinite source geometry')

    native_faces, records, vertices = read_native_topology(read_step_file(str(step), verbosity=False))
    mapping, face_error = match_native_faces(
        native_faces, faces[..., :3].astype(float), parameters['face_tolerance'],
    )
    native_pairs = defaultdict(list)
    for record in records:
        owners = [mapping[f] for f in record['owners']]
        if len(owners) != 2:
            raise ValueError('Native edge must have two owner occurrences')
        if owners[0] != owners[1]:
            native_pairs[tuple(sorted(owners))].append(record)
    chains = {pair: chain_vertices(edges) for pair, edges in native_pairs.items()}

    connectivity = connectivity[connectivity[:, 1] != connectivity[:, 2]]
    endpoint_ids = np.full((len(connectivity), 2), -1, dtype=np.int64)
    errors = np.zeros((len(connectivity), 2), dtype=np.float64)
    closed = np.zeros(len(connectivity), dtype=bool)
    for row, (eid, owner, neighbor) in enumerate(connectivity):
        candidates, closed[row] = chains[tuple(sorted((owner, neighbor)))]
        endpoint_ids[row], errors[row] = bind_endpoints(
            curves[eid, [0, -1], :3].astype(float), candidates, vertices, parameters['vertex_tolerance'],
        )
    canonical_ids, _ = canonical_edges(connectivity)
    by_raw_edge = {eid: row for row, eid in enumerate(connectivity[:, 0])}
    matched = endpoint_ids[[by_raw_edge[eid] for eid in canonical_ids]]
    if not np.array_equal(matched, canonical_vertices):
        raise ValueError('Native IDs disagree with the audited canonical labels')
    return {
        'schema_version': np.array(SCHEMA_VERSION),
        'source_sha256': np.array(source_digest), 'step_sha256': np.array(step_digest),
        'canonical_labels_sha256': np.array(sha256_file(canonical_labels)),
        'parameters_json': np.array(json.dumps(parameters, sort_keys=True)),
        'source_edge_ids': connectivity[:, 0], 'face_pairs': connectivity[:, 1:],
        'endpoint_vertex_ids': endpoint_ids, 'endpoint_valid': endpoint_ids >= 0,
        'endpoint_binding_error': errors, 'is_closed': closed,
        'native_vertex_xyz': vertices, 'face_grid_max_error': np.array(face_error),
    }


def _extract_job(job):
    sid, data_root, reference_root, output_root = job
    folder = data_root / sid
    try:
        arrays = extract_endpoint_vertices(
            folder / 'data.npz', folder / 'normalized_shape.step',
            reference_root / sid / 'vertex_groups.npz',
        )
        target = output_root / sid
        target.mkdir()
        path = target / 'endpoint_vertices.npz'
        np.savez_compressed(path, **arrays)
        conn = np.column_stack((arrays['source_edge_ids'], arrays['face_pairs']))
        load_endpoint_vertices(path, source_path=folder / 'data.npz', connectivity=conn)
        valid = arrays['endpoint_valid']
        return dict(id=sid, status='ok', halfedges=len(conn), endpoints=valid.size,
                    unknown=int((~valid).sum()), closed_halfedges=int(arrays['is_closed'].sum()),
                    max_valid_binding_error=float(arrays['endpoint_binding_error'][valid].max(initial=0)))
    except Exception as error:
        return dict(id=sid, status='failed', error=repr(error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--canonical-vertex-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--split', type=Path, action='append', required=True)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    ids = list(dict.fromkeys(sid for split in args.split for sid in load_model_ids(split)))
    args.output_root.mkdir(parents=True, exist_ok=False)
    started, rows = time.monotonic(), []
    jobs = [(sid, args.data_root, args.canonical_vertex_root, args.output_root) for sid in ids]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        with (args.output_root / 'records.jsonl').open('x') as stream:
            for row in pool.map(_extract_job, jobs, chunksize=8):
                rows.append(row)
                stream.write(json.dumps(row) + '\n')
                if len(rows) % 500 == 0 or row['status'] != 'ok':
                    stream.flush()
                    print(json.dumps(dict(completed=len(rows), total=len(ids),
                                          failed=sum(r['status'] != 'ok' for r in rows),
                                          seconds=time.monotonic() - started)), flush=True)
    failed = [row for row in rows if row['status'] != 'ok']
    report = dict(count=len(rows), failed=failed,
                  endpoints=sum(r.get('endpoints', 0) for r in rows),
                  unknown=sum(r.get('unknown', 0) for r in rows),
                  seconds=time.monotonic() - started,
                  split_sha256={str(p): sha256_file(p) for p in args.split},
                  generator_sha256=sha256_file(Path(__file__)),
                  parameters={k: [str(x) for x in v] if isinstance(v, list) else str(v)
                              for k, v in vars(args).items()})
    (args.output_root / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    return int(bool(failed))


if __name__ == '__main__':
    raise SystemExit(main())
