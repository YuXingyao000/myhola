"""Independently audit the fixed failure-viewer payload and source identities.

This script imports no payload-builder or replay implementation. It compares
against the original evaluation, checks geometric references and distribution
arithmetic, and never runs or changes OCC post-processing.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
REPO = HERE.parents[4]
GT = Path('/mnt/d/data/deepcad_v7')
SPLIT = REPO / 'checkpoints/joint_boundary/validation/val_contract_audit/accepted.txt'
STAGES = {'trimming': 307, 'solid_construction': 6, 'roundtrip_boundary': 3}
CONNECTOR_ATOL = 2e-6


def read(path):
    if str(path).endswith('.gz'):
        with gzip.open(path, 'rt') as handle:
            return json.load(handle)
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def average(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def describe(values):
    measured = [value for value in values if value is not None]
    return {'intended': len(values), 'measured': len(measured), 'missing': len(values) - len(measured),
            'mean': average(measured), 'median': float(np.median(measured)) if measured else None,
            'p90': float(np.quantile(measured, .9)) if measured else None,
            'p95': float(np.quantile(measured, .95)) if measured else None,
            'max': float(max(measured)) if measured else None}


class Audit:
    def __init__(self):
        self.failures = []
        self.checks = Counter()
        self.max_connector_error = {'fitted_owner': 0., 'raw_endpoint': 0., 'fitted_endpoint': 0.}
        self.counts = Counter()

    def check(self, condition, category, message):
        self.checks[category] += 1
        if not bool(condition):
            self.failures.append({'category': category, 'message': message})

    def close(self, actual, expected, category, message, atol=1e-12):
        if actual is None or expected is None:
            self.check(actual is None and expected is None, category, message)
        else:
            self.check(np.isfinite(actual) and np.isfinite(expected)
                       and np.isclose(actual, expected, rtol=1e-11, atol=atol), category, message)

    def source(self, path, expected, message):
        self.check(path.is_file() and sha(path) == expected, 'source_sha256', message)

    def geometry(self, sid, name, view):
        faces, edges = view['faces'], view['edges']
        for index, grid in enumerate(faces):
            array = np.asarray(grid)
            self.check(array.ndim == 3 and array.shape[-1] == 3 and min(array.shape[:2]) >= 2
                       and np.isfinite(array).all(), 'finite_geometry', f'{sid}/{name}/face/{index}')
        for index, points in enumerate(edges):
            array = np.asarray(points)
            self.check(array.ndim == 2 and array.shape[-1] == 3 and len(array) >= 2
                       and np.isfinite(array).all(), 'finite_geometry', f'{sid}/{name}/edge/{index}')
        edge_owners = {}
        for row in view['connectivity']:
            correct = len(row) == 3 and all(isinstance(value, int) for value in row)
            self.check(correct and 0 <= row[0] < len(edges) and all(0 <= f < len(faces) for f in row[1:]),
                       'connectivity', f'{sid}/{name}/{row}')
            edge_owners.setdefault(row[0], set()).update(row[1:])
        if name == 'fitted':
            expected_relations = Counter((edge, owner) for edge, first, second in view['connectivity'] for owner in (first, second))
            observed_relations = Counter((item['edge'], item['owner']) for item in view.get('owner_records', []))
            audit_message = f'{sid}/fitted/complete owner records'
            self.check(observed_relations == expected_relations, 'owner_mapping_complete', audit_message)
            expected_endpoints = Counter((edge, owner, tip) for edge, first, second in view['connectivity'] for owner in (first, second) for tip in (0, -1))
            observed_endpoints = Counter((item['edge'], item['owner'], item['endpoint']) for item in view.get('endpoint_records', []))
            self.check(observed_endpoints == expected_endpoints, 'endpoint_mapping_complete', f'{sid}/fitted/complete endpoint records')
        for record in view.get('owner_records', []):
            edge, owner = record['edge'], record['owner']
            self.check(owner in edge_owners.get(edge, set()), 'owner_mapping', f'{sid}/{name}/{edge}/{owner}')
            if name != 'fitted':
                continue
            distances, closest = record['distances'], record['closest']
            self.check(len(distances) == len(closest) == len(edges[edge]) == 64,
                       'projection_samples', f'{sid}/{edge}/{owner}')
            valid = [d for d in distances if d is not None]
            self.check(record['valid_points'] == len(valid) and record['unknown_points'] == 64 - len(valid),
                       'projection_unknown', f'{sid}/{edge}/{owner}')
            self.counts['projection_points'] += len(distances)
            self.counts['projection_unknown'] += len(distances) - len(valid)
            for key, expected in [('mean', np.mean(valid) if len(valid) == 64 else None),
                                  ('p95', np.quantile(valid, .95) if len(valid) == 64 else None),
                                  ('max', max(valid) if len(valid) == 64 else None)]:
                self.close(record[key], expected, 'projection_statistics', f'{sid}/{edge}/{owner}/{key}')
            for index, (point, target, distance) in enumerate(zip(edges[edge], closest, distances)):
                self.check((target is None) == (distance is None), 'projection_unknown', f'{sid}/{edge}/{owner}/{index}')
                if target is None or distance is None:
                    continue
                self.check(np.asarray(target).shape == (3,) and np.isfinite(target).all()
                           and np.isfinite(distance) and distance >= 0, 'finite_projection', f'{sid}/{edge}/{owner}/{index}')
                error = abs(float(np.linalg.norm(np.asarray(point) - target)) - distance)
                self.max_connector_error['fitted_owner'] = max(self.max_connector_error['fitted_owner'], error)
                self.check(error <= CONNECTOR_ATOL, 'projection_connector', f'{sid}/{edge}/{owner}/{index}: {error}')
        for index, record in enumerate(view.get('endpoint_records', [])):
            edge, owner, tip = record['edge'], record['owner'], record['endpoint']
            self.check(owner in edge_owners.get(edge, set()) and tip in (0, -1),
                       'endpoint_mapping', f'{sid}/{name}/{index}')
            distance = record['distance']
            neighbor, neighbor_tip = record.get('neighbor_edge'), record.get('neighbor_endpoint')
            if distance is None:
                self.counts[name + '_endpoint_unknown'] += 1
                self.check(neighbor is None and neighbor_tip is None, 'endpoint_unknown', f'{sid}/{name}/{index}')
                continue
            self.check(np.isfinite(distance) and distance >= 0, 'finite_endpoint', f'{sid}/{name}/{index}')
            self.check(neighbor is not None and neighbor_tip in (0, -1)
                       and owner in edge_owners.get(neighbor, set()), 'endpoint_neighbor_mapping', f'{sid}/{name}/{index}')
            if neighbor is not None and neighbor_tip in (0, -1):
                error = abs(float(np.linalg.norm(np.asarray(edges[edge][tip]) - edges[neighbor][neighbor_tip])) - distance)
                if name in ('raw', 'fitted'):
                    self.max_connector_error[name + '_endpoint'] = max(self.max_connector_error[name + '_endpoint'], error)
                self.check(error <= CONNECTOR_ATOL, 'endpoint_connector', f'{sid}/{name}/{index}: {error}')


def validate(stage):
    audit = Audit()
    history = read(EVAL / 'failure_analysis/summary.json')
    groups = history['paired']['ids']
    expected_ids = sorted(groups['regressed'] + groups['both_failed'])
    population = SPLIT.read_text().split()
    base = read(HERE / 'data/base_manifest.json')
    audit.check(len(expected_ids) == len(set(expected_ids)) == 316 and len(population) == len(set(population)) == 3495,
                'fixed_denominator', 'Fixed failure/population IDs')
    audit.check(base['ids'] == expected_ids and base['population'] == population, 'fixed_denominator', 'Base manifest ID lists')
    audit.source(SPLIT, base['split_sha256'], 'Validation split')
    audit.source(EVAL / 'failure_analysis/summary.json', base['summary_source_sha256'], 'Original summary')
    audit.check(base['checkpoint_sha256'] == history['checkpoint_sha256'], 'checkpoint_identity', 'Model checkpoint SHA')
    with (EVAL / 'failure_analysis/samples.csv').open() as handle:
        rows = {row['id']: row for row in csv.DictReader(handle) if row['method'] == 'no_edge_feature64'}
    expected_all_metrics = {}
    raw_caches = {}
    for sid in population:
        cache = read(EVAL / f'failure_analysis/raw/no_edge_feature64/{sid}.json')
        raw_caches[sid] = cache
        metrics = {'raw_mean': average(item['mean'] for item in cache['owner']['records']),
                   'raw_p95': cache['owner']['p95_max'], 'raw_max': cache['owner']['max'],
                   'endpoint_max': cache['endpoints']['max'],
                   'endpoint_mean': average(item['distance'] for item in cache['endpoints']['records']),
                   'endpoint_unknown': sum(item['distance'] is None for item in cache['endpoints']['records']),
                   'endpoint_empty_owner_faces': cache['structure']['face_count'] - len({item['owner'] for item in cache['endpoints']['records']})}
        expected_all_metrics[sid] = metrics
        for metric, value in metrics.items():
            audit.close(base['metrics_all'][sid].get(metric), value, 'population_cache_metrics', f'{sid}/{metric}')
        audit.check(cache['endpoints']['unknown_count'] + cache['endpoints']['unmatchable_count'] == metrics['endpoint_unknown'] + metrics['endpoint_empty_owner_faces'],
                    'raw_unknown_census', sid)
    case_metrics, stages, cohorts, reasons = {}, Counter(), Counter(), Counter()
    replay_matches = 0
    for sid in expected_ids:
        case_path = HERE / f'data/{"cases" if stage == "full" else "base"}/{sid}.json.gz'
        if not case_path.is_file():
            audit.check(False, 'case_missing', str(case_path))
            continue
        case = read(case_path)
        cache = raw_caches[sid]
        sources = base['source_hashes'][sid]
        raw_path = EVAL / f'raw_predictions/posterior_mean/no_edge_feature64/{sid}/data.npz'
        gt_path = GT / sid / 'data.npz'
        diag_path = EVAL / f'post/optimized/no_edge_feature64/{sid}/post_diagnostics.json'
        diagnostics = read(diag_path)
        meta = read(EVAL / f'metadata/post/optimized/samples/no_edge_feature64/{sid}.json')
        for key, path in [('raw', raw_path), ('gt', gt_path)]:
            audit.source(path, sources[key]['sha256'], f'{sid}/{key}')
            audit.check(sources[key]['sha256'] == cache['sources'][key]['sha256'], 'cached_source_identity', f'{sid}/{key}')
        audit.source(diag_path, sources['diagnostics_sha256'], f'{sid}/diagnostics')
        audit.source(diag_path, meta['post_diagnostics_sha256'], f'{sid}/original-metadata-diagnostics')
        audit.source(EVAL / f'failure_analysis/raw/no_edge_feature64/{sid}.json', sources['cache_sha256'], f'{sid}/raw-cache')
        audit.check(case['id'] == sid and case['diagnostics'] == diagnostics and meta['native_valid_solid'] is False,
                    'historical_identity', sid)
        audit.check(case['stage'] == {'roundtrip_or_boundary': 'roundtrip_boundary'}.get(rows[sid]['terminal_stage'], rows[sid]['terminal_stage']) and case['cohort'] == rows[sid]['cohort_vs_source32'],
                    'historical_outcome', sid)
        stages[case['stage']] += 1
        cohorts[case['cohort']] += 1
        failed_faces = [face['face_index'] for face in diagnostics['paths'][-1]['faces'] if face['accepted_attempt'] is None]
        audit.check(case['failed_faces'] == failed_faces, 'failed_faces', sid)
        audit.counts['failed_faces'] += len(failed_faces)
        expected_reasons = []
        for face in diagnostics['paths'][-1]['faces']:
            if face['accepted_attempt'] is None:
                if not face['input_edge_ids']:
                    reason = 'no_input_edges'
                elif not face['attempts']:
                    reason = 'no_attempt_recorded'
                else:
                    reason = face['attempts'][-1].get('rejection') or face['attempts'][-1]['status']
                expected_reasons.append(reason)
        audit.check(case['reasons'] == expected_reasons, 'failure_reasons', sid)
        reasons.update(case['reasons'])
        for metric, value in expected_all_metrics[sid].items():
            audit.close(case['metrics'].get(metric), value, 'case_raw_metrics', f'{sid}/{metric}')
        for name, path, keyset in [('raw', raw_path, ('pred_face', 'pred_edge', 'pred_edge_face_connectivity')),
                                  ('gt', gt_path, ('sample_points_faces', 'sample_points_lines', 'edge_face_connectivity'))]:
            view = case['views'][name]
            with np.load(path, allow_pickle=False) as array:
                audit.check(view['faces'] == np.round(array[keyset[0]][..., :3].astype(np.float64), 6).tolist(),
                            'raw_display_identity', f'{sid}/{name}/faces')
                audit.check(view['edges'] == np.round(array[keyset[1]][..., :3].astype(np.float64), 6).tolist(),
                            'raw_display_identity', f'{sid}/{name}/edges')
                audit.check(view['connectivity'] == array[keyset[2]].astype(int).tolist(),
                            'raw_display_identity', f'{sid}/{name}/connectivity')
            expected_owner = cache['owner' if name == 'raw' else 'gt_owner']['records']
            audit.check(view['owner_records'] == expected_owner, 'raw_cache_records', f'{sid}/{name}/owner')
        raw_endpoints = case['views']['raw']['endpoint_records']
        audit.check([{key: value for key, value in item.items() if key not in ('neighbor_edge', 'neighbor_endpoint')}
                     for item in raw_endpoints] == cache['endpoints']['records'], 'raw_cache_records', f'{sid}/endpoints')
        audit.check(sum(item['distance'] is None for item in raw_endpoints) == case['metrics']['endpoint_unknown'],
                    'case_endpoint_unknown', sid)
        for name, view in case['views'].items():
            audit.geometry(sid, name, view)
        if stage == 'full':
            replay = read(HERE / f'replay/{sid}.json.gz')
            replay_matches += bool(case['replay']['matches_history'])
            audit.check(case['replay']['status'] == replay['status'] and case['replay']['matches_history'] == replay['matches_history'],
                        'replay_identity', sid)
            audit.source(raw_path, replay['sources']['raw']['sha256'], f'{sid}/replay-raw')
            audit.source(diag_path, replay['sources']['diagnostics']['sha256'], f'{sid}/replay-diagnostics')
            audit.check(replay['matches_history'] is True and replay['mismatch_fields'] == [], 'replay_history_match', sid)
            for key in ('faces', 'selected_edge_count', 'periodic_edge_replacements'):
                audit.check(replay['report'].get(key) == diagnostics['paths'][-1].get(key), 'replay_history_fields', f'{sid}/{key}')
            fitted = case['views'].get('fitted')
            audit.check(fitted is not None, 'fitted_view_present', sid)
            if fitted is not None:
                adjacency = [[] for _ in fitted['faces']]
                for edge, first, second in fitted['connectivity']:
                    if first != second:
                        adjacency[first].append(edge)
                        adjacency[second].append(edge)
                for face in diagnostics['paths'][-1]['faces']:
                    audit.check(face['input_edge_ids'] == adjacency[face['face_index']], 'post_selected_edge_mapping', f'{sid}/face/{face["face_index"]}')
                audit.check(len(fitted['edges']) == diagnostics['paths'][-1]['selected_edge_count'], 'post_selected_edge_count', sid)
                audit.check(fitted['periodic_edge_replacements'] == diagnostics['paths'][-1]['periodic_edge_replacements'],
                            'post_periodic_edge_mapping', sid)
                replay_view = replay['view']
                audit.check(fitted['faces'] == np.round(np.asarray(replay_view['faces'], dtype=np.float64), 6).tolist(), 'fitted_display_identity', f'{sid}/faces')
                audit.check(fitted['edges'] == [np.round(np.asarray(edge, dtype=np.float64), 6).tolist() for edge in replay_view['edges']], 'fitted_display_identity', f'{sid}/edges')
                audit.check(fitted['connectivity'] == replay_view['connectivity'], 'fitted_display_identity', f'{sid}/connectivity')
                audit.check(fitted['endpoint_other_edge_records'] == replay_view['endpoint_records'], 'fitted_cache_records', f'{sid}/original endpoints')
                audit.counts['fitted_original_endpoint_unknown'] += sum(item['distance'] is None for item in fitted['endpoint_other_edge_records'])
                audit.check(len(fitted['endpoint_records']) == len(replay_view['endpoint_records']), 'fitted_cache_records', f'{sid}/candidate endpoint count')
                candidate_distances = []
                for candidate, original in zip(fitted['endpoint_records'], replay_view['endpoint_records']):
                    own_gap = float(np.linalg.norm(np.asarray(replay_view['edges'][original['edge']][0]) - replay_view['edges'][original['edge']][-1]))
                    choose_self = original['distance'] is None or own_gap < original['distance']
                    expected_distance = own_gap if choose_self else original['distance']
                    candidate_distances.append(expected_distance)
                    audit.close(candidate['own_seam_distance'], own_gap, 'endpoint_candidate_metadata', f'{sid}/own seam')
                    audit.close(candidate['original_neighbor_distance'], original['distance'], 'endpoint_candidate_metadata', f'{sid}/original distance')
                    audit.close(candidate['distance'], expected_distance, 'endpoint_candidate_minimum', f'{sid}/candidate distance')
                    audit.check(candidate['original_neighbor_edge'] == original['neighbor_edge'] and candidate['original_neighbor_endpoint'] == original['neighbor_endpoint'], 'endpoint_candidate_metadata', f'{sid}/original neighbor')
                    expected_edge = original['edge'] if choose_self else original['neighbor_edge']
                    expected_tip = (-1 if original['endpoint'] == 0 else 0) if choose_self else original['neighbor_endpoint']
                    audit.check(candidate['neighbor_edge'] == expected_edge and candidate['neighbor_endpoint'] == expected_tip, 'endpoint_candidate_minimum', f'{sid}/candidate neighbor')
                    audit.check(all(candidate[key] == original[key] for key in ('edge', 'owner', 'endpoint')), 'endpoint_candidate_identity', sid)
                audit.close(case['metrics']['replay_endpoint_mean'], average(candidate_distances), 'case_fitted_metrics', f'{sid}/candidate mean')
                audit.close(case['metrics']['replay_endpoint_max'], max(candidate_distances) if candidate_distances else None, 'case_fitted_metrics', f'{sid}/candidate max')
                audit.check(len(fitted['owner_records']) == len(replay_view['owner_records']), 'fitted_cache_records', f'{sid}/owner count')
                for packed_record, source_record in zip(fitted['owner_records'], replay_view['owner_records']):
                    expected_record = dict(source_record)
                    expected_record['closest'] = [np.round(np.asarray(point, dtype=np.float64), 6).tolist() if point is not None else None for point in source_record['closest']]
                    audit.check(packed_record == expected_record, 'fitted_cache_records', f'{sid}/owner/{source_record["edge"]}/{source_record["owner"]}')
                for target, source in [('replay_mean', 'owner_mean'), ('replay_p95', 'owner_p95_max'), ('replay_max', 'owner_max'), ('replay_endpoint_other_mean', 'endpoint_mean'), ('replay_endpoint_other_max', 'endpoint_max'), ('replay_endpoint_other_unknown', 'unknown_endpoint_count')]:
                    audit.close(case['metrics'].get(target), replay['metrics'].get(source), 'case_fitted_metrics', f'{sid}/{target}')
        case_metrics[sid] = case['metrics']
        audit.counts['cases_checked'] += 1
    audit.check(dict(stages) == STAGES, 'fixed_denominator', f'Case stages: {dict(stages)}')
    audit.check(dict(cohorts) == {'both_failed': 287, 'regressed': 29}, 'fixed_denominator', f'Case cohorts: {dict(cohorts)}')
    audit.check(audit.counts['failed_faces'] == 767, 'fixed_denominator', f'Failed faces: {audit.counts["failed_faces"]}')
    audit.check(dict(reasons) == base['reasons'], 'failure_reason_census', str(dict(reasons)))
    if stage == 'full':
        manifest = read(HERE / 'data/manifest.json')
        summary = read(HERE / 'summary.json')
        audit.check(manifest['summary'] == summary, 'manifest_identity', 'Summary')
        audit.check(sorted(row['id'] for row in manifest['cases']) == expected_ids, 'fixed_denominator', 'Manifest cases')
        audit.check((summary['population'], summary['valid'], summary['failed'], summary['failed_faces']) == (3495, 3179, 316, 767),
                    'fixed_denominator', 'Summary counts')
        audit.check(summary['replay_match'] == replay_matches == 316, 'replay_match_census', str(replay_matches))
        audit.check(summary['projection_sample_count'] == audit.counts['projection_points']
                    and summary['projection_unknown_count'] == audit.counts['projection_unknown'], 'projection_census', 'Summary projection sample counts')
        for row in manifest['cases']:
            for metric, expected in case_metrics[row['id']].items():
                audit.close(row.get(metric), expected, 'manifest_case_metrics', f'{row["id"]}/{metric}')
        failed_set = set(expected_ids)
        for metric, distribution in manifest['distributions'].items():
            for label, is_failed in [('valid', False), ('failed', True)]:
                if metric.startswith('replay_'):
                    values = [case_metrics[sid].get(metric) for sid in expected_ids] if is_failed else [None] * 3179
                else:
                    values = [expected_all_metrics[sid][metric] for sid in population if (sid in failed_set) == is_failed]
                expected_values = sorted(value for value in values if value is not None)
                actual_values = sorted(distribution[label])
                audit.check(len(actual_values) == len(expected_values) and np.allclose(actual_values, expected_values, rtol=1e-12, atol=1e-14),
                            'distribution_values', f'{metric}/{label}')
                audit.check(distribution['unknown_' + label] == sum(value is None for value in values),
                            'distribution_denominator', f'{metric}/{label}')
                for key, value in describe(values).items():
                    audit.close(summary['distributions'][metric][label][key], value, 'distribution_statistics', f'{metric}/{label}/{key}')
        for label, is_failed in [('valid', False), ('failed', True)]:
            expected = sum(expected_all_metrics[sid]['endpoint_unknown'] > 0 for sid in population if (sid in failed_set) == is_failed)
            audit.check(summary['raw_endpoint_partial_unknown_objects'][label] == expected, 'population_endpoint_unknown', label)
    preflight = read(EVAL / 'validation/post_preflight.json')
    for relative, expected in preflight['post_algorithm_files'].items():
        audit.source(REPO / relative, expected, 'Unchanged production ' + relative)
    report = {'passed': not audit.failures, 'complete_payload': stage == 'full', 'stage': stage,
              'completed_utc': datetime.now(timezone.utc).isoformat(), 'script_sha256': sha(Path(__file__)),
              'spec_sha256': sha(HERE / 'SPEC.md'), 'schema_sha256': sha(HERE / 'SCHEMA.md'),
              'payload_files_sha256': {str(path.relative_to(HERE)): sha(path) for path in
                  ([HERE / 'data/base_manifest.json'] + ([HERE / 'data/manifest.json', HERE / 'summary.json'] if stage == 'full' else [])
                   + sorted((HERE / 'data' / ('cases' if stage == 'full' else 'base')).glob('*.json.gz')))},
              'checks': dict(audit.checks), 'counts': dict(audit.counts), 'stage_counts': dict(stages),
              'cohort_counts': dict(cohorts), 'reason_counts': dict(reasons),
              'display_connector_atol': CONNECTOR_ATOL, 'max_display_connector_error': audit.max_connector_error,
              'failure_count': len(audit.failures), 'failures': audit.failures,
              'limitations': ['Raw 16-point-to-grid distances are diagnostic proxies, not OCC tolerances.',
                              'Fitted geometry is an independent replay checked against per-face history, not an original saved snapshot.',
                              'Fitted measurements sample 64 parameter points; they do not certify supremum distance or wire closure.',
                              'Inclusive endpoint candidate is the minimum of own seam and historical local neighbor; it is a lower bound and may be degenerate self closure.',
                              'No fitted measurements were performed for the 3179 valid control objects.',
                              'This validator does not rerun model inference, post-processing, or browser interactions.']}
    output = HERE / 'validation/payload.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: report[key] for key in ('passed', 'complete_payload', 'counts', 'stage_counts', 'failure_count', 'max_display_connector_error')}, ensure_ascii=False))
    if audit.failures:
        print(json.dumps(audit.failures[:20], ensure_ascii=False, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('base', 'full'), default='full')
    args = parser.parse_args()
    sys.exit(validate(args.stage))
