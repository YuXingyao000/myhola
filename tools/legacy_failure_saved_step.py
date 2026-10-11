"""Inspect original saved STEP free edges without healing or triangulation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

from tools.vae_compare_post import (atomic_json, configure_cpu, fingerprint, read_json,
                                    read_step_at_precision, run_isolated_jobs, sha256_file)
from tools.failure_census_contract import case_method, case_path, manifest_method

THRESHOLDS = (0.002, 0.006, 0.01, 0.08)


def point_polyline_distance(points, polyline):
    points, polyline = np.asarray(points, dtype=float), np.asarray(polyline, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or polyline.ndim != 2 or polyline.shape[1] != 3:
        raise ValueError('Expected [N,3] points and polyline')
    if not len(points) or len(polyline) < 2 or not np.isfinite(points).all() or not np.isfinite(polyline).all():
        raise ValueError('Expected nonempty finite geometry and at least two polyline points')
    delta = polyline[1:]-polyline[:-1]
    length_sq = np.sum(delta*delta, axis=-1)
    offsets = points[:, None, :]-polyline[None, :-1, :]
    parameter = np.sum(offsets*delta[None], axis=-1)/np.where(length_sq > 0, length_sq, 1)[None]
    parameter = np.clip(parameter, 0, 1)
    return np.sqrt(np.min(np.sum((offsets-parameter[..., None]*delta[None])**2, axis=-1), axis=1))


def proximity(a, b):
    forward, backward = point_polyline_distance(a, b), point_polyline_distance(b, a)
    return {'p95': float(max(np.percentile(forward, 95), np.percentile(backward, 95))),
            'max': float(max(forward.max(), backward.max())),
            'mean': float((forward.mean()+backward.mean())/2)}


def nearby_free_edges(edge_info, samples):
    ids = sorted(samples)
    pairs = []
    by_id = {edge['id']: edge for edge in edge_info}
    for offset, first in enumerate(ids):
        a = samples[first]
        for second in ids[offset+1:]:
            if set(by_id[first]['unique_face_ids']) & set(by_id[second]['unique_face_ids']):
                continue
            b = samples[second]
            gap = np.maximum(np.maximum(a.min(0)-b.max(0), b.min(0)-a.max(0)), 0)
            if np.linalg.norm(gap) > max(THRESHOLDS):
                continue
            values = proximity(a, b)
            if values['p95'] <= max(THRESHOLDS):
                pairs.append({'edge_ids': [first, second],
                              'face_ids': [by_id[first]['unique_face_ids'], by_id[second]['unique_face_ids']],
                              **values})
    thresholds = {}
    for threshold in THRESHOLDS:
        accepted = [pair for pair in pairs if pair['p95'] <= threshold]
        thresholds[str(threshold)] = {'pair_count': len(accepted),
            'involved_edge_count': len({edge for pair in accepted for edge in pair['edge_ids']})}
    return {'sampled_free_edges': len(ids), 'pairs': pairs, 'thresholds': thresholds}


def execute(job, status_path, log_path):
    os.setsid()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('ab', buffering=0) as handle:
        os.dup2(handle.fileno(), 1)
        os.dup2(handle.fileno(), 2)
        start = time.monotonic()
        result = {'id': job['sample_id'], 'method': job['method'], 'schema': 'legacy_saved_step_v1',
                  'protocol_fingerprint': job['protocol_fingerprint'], 'step_sha256': job['step_sha256'],
                  'status': 'running', 'sampling_errors': []}
        try:
            from OCC.Core.TopoDS import topods
            from tools.legacy_failure_geometry import native_diagnostics, edge_points
            shape = read_step_at_precision(Path(job['path']), mode=0)
            diagnostic, info, maps = native_diagnostics(shape)
            result['native'] = diagnostic
            samples = {}
            for edge in info:
                if not edge['free']:
                    continue
                try:
                    samples[edge['id']] = edge_points(topods.Edge(maps['edges'].FindKey(edge['id']+1)), 128)
                except Exception as exc:
                    result['sampling_errors'].append({'edge_id': edge['id'], 'error': str(exc)})
            result['free_edge_proximity'] = nearby_free_edges(info, samples)
            result['status'] = 'ok' if not result['sampling_errors'] else 'partial'
        except Exception as exc:
            traceback.print_exc()
            result.update(status='error', error=f'{type(exc).__name__}: {exc}')
        result['elapsed_seconds'] = time.monotonic()-start
        atomic_json(Path(job['output']), result)
        atomic_json(status_path, {'status': 'post_error' if result['status']=='error' else 'ok', 'sample_id': job['sample_id'],
                                  'protocol_fingerprint': job['protocol_fingerprint']})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--only-ids', nargs='+')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 8 or not 0 < args.timeout <= 180:
        parser.error('workers must be 1..8 and timeout 0..180 seconds')
    root = args.root.resolve()
    manifest = read_json(args.manifest)
    cases = manifest['cases']
    method = manifest_method(cases)
    if len({case['id'] for case in cases}) != len(cases):
        raise ValueError('Duplicate manifest IDs')
    if args.only_ids:
        if set(args.only_ids)-{case['id'] for case in cases}:
            raise ValueError('Requested ID outside frozen manifest')
        cases = [case for case in cases if case['id'] in set(args.only_ids)]
    protocol = {'schema': 'legacy_saved_step_v1', 'method': method, 'manifest_sha256': sha256_file(args.manifest),
                'source_sha256': {p: sha256_file(Path(p)) for p in
                    ['tools/legacy_failure_saved_step.py', 'tools/legacy_failure_geometry.py', 'tools/vae_compare_post.py', 'tools/failure_census_contract.py']},
                'curve_samples': 128, 'thresholds': THRESHOLDS, 'workers': args.workers,
                'timeout_seconds': args.timeout, 'read_precision_mode': 0,
                'distance': 'max of directed p95 Euclidean point-to-polyline distances, normalized XYZ length',
                'scope': 'Different topological free edges on different faces; exclude seam/degenerate; no repair or meshes',
                'limitations': 'Finite curve sampling; proximity does not prove two edges should be merged.'}
    key = fingerprint(protocol)
    metadata = root/'metadata'/'saved_step'
    previous = read_json(metadata/'protocol.json')
    if previous and (fingerprint(previous) != key or not args.resume):
        raise ValueError('Existing output requires --resume and identical protocol')
    atomic_json(metadata/'protocol.json', protocol)
    jobs = []
    for case in cases:
        source = case_path(case, 'post_step')
        out = root/'saved_step'/f"{case['id']}.json"
        old = read_json(out)
        sha = sha256_file(source) if source.exists() else None
        if old:
            if old.get('protocol_fingerprint') != key or old.get('step_sha256') != sha:
                raise ValueError('Changed input or protocol')
            if args.resume:
                continue
        jobs.append({'sample_id': case['id'], 'method': case_method(case), 'mode': 'optimized', 'path': str(source),
                     'output': str(out), 'step_sha256': sha,
                     'raw_sha256': sha256_file(case_path(case, 'raw_npz')), 'protocol_fingerprint': key})
    configure_cpu()
    from tools.legacy_failure_geometry import native_diagnostics  # Preload once before fork.
    from OCC.Core.STEPControl import STEPControl_Reader
    run_isolated_jobs(jobs, metadata, args.workers, args.timeout, process_target=execute)
    counts = {}
    for case in cases:
        out = root/'saved_step'/f"{case['id']}.json"
        row = read_json(out)
        status = read_json(metadata/'samples'/case_method(case)/f"{case['id']}.json")
        if status.get('status') in ('timeout', 'crash'):
            row.update(id=case['id'], method=case_method(case), status=status['status'], error=status.get('error'), protocol_fingerprint=key)
            atomic_json(out, row)
        state = row.get('status', 'missing')
        counts[state] = counts.get(state, 0)+1
    print(json.dumps({'method': method, 'cases': len(cases), 'status_counts': counts}))


if __name__ == '__main__':
    main()
