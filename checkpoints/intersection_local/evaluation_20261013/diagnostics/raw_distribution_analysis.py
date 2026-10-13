"""Frozen paired raw distribution diagnostic; see RAW_DISTRIBUTION_SPEC.md."""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'diagnostics'
METHODS = ('intersection', 'local')
MODES = ('posterior_mean', 'archived_latents')
METRICS = ('face_mae', 'face_cd_sq', 'edge_cd_sq', 'edge_pred_to_gt_sq', 'edge_gt_to_pred_sq')
COMPLEXITY = ('gt_face_count', 'gt_boundary_count', 'pred_face_count', 'raw_edge_count')
BINS = ((1, 5), (6, 10), (11, 15), (16, 20), (21, 30), (31, None))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def value(row, key):
    if row.get(key) in (None, ''):
        return None
    number = float(row[key])
    if not math.isfinite(number) or number < 0:
        raise ValueError(f'Invalid {key}: {row.get(key)}')
    return number


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {'count': 0, **{key: None for key in ('mean', 'median', 'p90', 'p95', 'p99', 'max')}}
    return {'count': len(values), 'mean': float(values.mean()), 'median': float(np.median(values)),
            **{f'p{q}': float(np.percentile(values, q)) for q in (90, 95, 99)}, 'max': float(values.max())}


def ratio(a, b):
    return float(a / b) if b else None


def write_csv(path, rows):
    with path.open('x', newline='') as handle:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    destinations = ('raw_distribution.json', 'raw_distribution_per_object.csv', 'raw_distribution_extremes.csv')
    if any((OUT / name).exists() for name in destinations):
        raise FileExistsError('Diagnostics already exist')
    result = {'spec_sha256': digest(OUT / 'RAW_DISTRIBUTION_SPEC.md'),
              'script_sha256': digest(Path(__file__)), 'methods': METHODS, 'modes': {},
              'trimming_note': 'Diagnostic only; remove ceil(1% * n) IDs ranked by local error from BOTH methods. Original primary mean remains unchanged.',
              'correlation_note': 'Spearman association with counts is descriptive, not causal; archived GT count describes the target CAD, not predicted face alignment.'}
    all_rows, extremes = [], []
    for mode in MODES:
        path = ROOT / 'raw_metrics' / mode / 'per_object.csv'
        summary_path = path.with_name('summary.json')
        summary = json.loads(summary_path.read_text())
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        lookup = {(row['method'], row['model_id']): row for row in rows}
        ids = sorted({row['model_id'] for row in rows})
        assert len(ids) == 2417 and len(rows) == len(lookup) == 2 * len(ids)
        assert set(lookup) == {(method, sample) for method in METHODS for sample in ids}
        for sample in ids:
            a, b = (lookup[method, sample] for method in METHODS)
            assert a['shared_input_status'] == b['shared_input_status'] == 'verified'
            assert a['input_latent_sha256'] == b['input_latent_sha256']
        table = {}
        for sample in ids:
            row = {'mode': mode, 'model_id': sample}
            for method in METHODS:
                source = lookup[method, sample]
                for key in (*METRICS, *COMPLEXITY, 'topology_f1', 'legacy_reciprocal_cd_sq_mean', 'unique_pair_count'):
                    row[f'{method}_{key}'] = value(source, key)
            for key in METRICS:
                old, new = (row[f'{method}_{key}'] for method in METHODS)
                row[f'{key}_delta'] = new - old if old is not None and new is not None else None
                row[f'{key}_ratio'] = ratio(new, old) if old is not None and new is not None else None
            table[sample] = row
            all_rows.append(row)
        mode_result = {'intended_count': len(ids), 'source_csv': str(path), 'source_csv_sha256': digest(path),
                       'source_summary_sha256': digest(summary_path), 'metrics': {}, 'complexity_groups': {}}
        for key in METRICS:
            common = [sample for sample in ids if all(table[sample][f'{method}_{key}'] is not None for method in METHODS)]
            old = np.array([table[sample][f'intersection_{key}'] for sample in common])
            new = np.array([table[sample][f'local_{key}'] for sample in common])
            delta = new - old
            record = {'common_count': len(common), 'missing_count': len(ids) - len(common),
                      'intersection': stats(old), 'local': stats(new)}
            if not common:
                mode_result['metrics'][key] = record
                continue
            for method in METHODS:
                reference = summary['methods'][method]['metrics'][key]['mean']
                assert math.isclose(record[method]['mean'], reference, rel_tol=1e-10, abs_tol=1e-14), (mode, key, method)
            record.update(mean_ratio=ratio(new.mean(), old.mean()), median_ratio=ratio(np.median(new), np.median(old)),
                          win_count=int(np.sum(delta < -1e-12)), loss_count=int(np.sum(delta > 1e-12)),
                          tie_count=int(np.sum(np.abs(delta) <= 1e-12)), win_fraction=float(np.mean(delta < -1e-12)),
                          paired_ratio=stats([ratio(b, a) for a, b in zip(old, new) if a > 0]),
                          paired_delta=stats(delta), tail={})
            ordered = sorted(range(len(common)), key=lambda i: (-new[i], common[i]))
            for fraction in (.01, .05, .10):
                count = math.ceil(fraction * len(common))
                top = ordered[:count]
                remain = ordered[count:]
                record['tail'][f'top_{int(fraction * 100)}pct'] = {
                    'count': count, 'local_error_share': ratio(new[top].sum(), new.sum()),
                    'same_ids_baseline_error_share': ratio(old[top].sum(), old.sum()),
                    'same_ids_positive_delta_share': ratio(np.maximum(delta[top], 0).sum(), np.maximum(delta, 0).sum()),
                    'retained_count': len(remain), 'trimmed_intersection_mean': float(old[remain].mean()),
                    'trimmed_local_mean': float(new[remain].mean()), 'trimmed_mean_ratio': ratio(new[remain].mean(), old[remain].mean()),
                    'removed_ids': [common[i] for i in top] if fraction == .01 else None,
                }
            positive_rank = sorted(range(len(common)), key=lambda i: (-max(delta[i], 0), common[i]))[:math.ceil(.01 * len(common))]
            record['largest_positive_delta_1pct_share'] = ratio(np.maximum(delta[positive_rank], 0).sum(), np.maximum(delta, 0).sum())
            record['spearman'] = {}
            for ckey in COMPLEXITY:
                cvalues = np.array([table[sample][f'local_{ckey}'] for sample in common])
                record['spearman'][ckey] = {}
                for target_name, target in (('local_error', new), ('delta', delta)):
                    rho = float(spearmanr(cvalues, target).statistic) if len(set(cvalues)) > 1 and len(set(target)) > 1 else None
                    record['spearman'][ckey][target_name] = {'rho': rho, 'count': len(common)}
            mode_result['metrics'][key] = record
            for rank, index in enumerate(ordered[:20], 1):
                sample = common[index]
                extremes.append({'mode': mode, 'metric': key, 'local_error_rank': rank,
                                 'model_id': sample, 'intersection_error': float(old[index]),
                                 'local_error': float(new[index]), 'delta': float(delta[index]),
                                 'ratio': ratio(new[index], old[index]),
                                 **{c: table[sample][f'local_{c}'] for c in COMPLEXITY}})
        for ckey in ('gt_face_count', 'pred_face_count'):
            groups = []
            for low, high in BINS:
                group = [sample for sample in ids if table[sample][f'local_{ckey}'] >= low and (high is None or table[sample][f'local_{ckey}'] <= high)]
                entry = {'range': f'{low}-{high}' if high else f'{low}+', 'count': len(group), 'metrics': {}}
                for key in ('face_cd_sq', 'edge_cd_sq'):
                    paired = [sample for sample in group if all(table[sample][f'{method}_{key}'] is not None for method in METHODS)]
                    old = [table[sample][f'intersection_{key}'] for sample in paired]
                    new = [table[sample][f'local_{key}'] for sample in paired]
                    entry['metrics'][key] = {'intersection': stats(old), 'local': stats(new),
                                             'mean_ratio': ratio(np.mean(new), np.mean(old)) if paired else None,
                                             'win_fraction': float(np.mean(np.array(new) < np.array(old) - 1e-12)) if paired else None}
                groups.append(entry)
            assert sum(row['count'] for row in groups) == len(ids)
            mode_result['complexity_groups'][ckey] = groups
        result['modes'][mode] = mode_result
    with (OUT / 'raw_distribution.json').open('x') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
    write_csv(OUT / 'raw_distribution_per_object.csv', all_rows)
    write_csv(OUT / 'raw_distribution_extremes.csv', extremes)
    for mode, record in result['modes'].items():
        print(mode)
        for key in ('face_mae', 'face_cd_sq', 'edge_cd_sq'):
            row = record['metrics'][key]
            if row['common_count']:
                print(key, json.dumps({name: row[name] for name in ('intersection', 'local', 'mean_ratio', 'median_ratio', 'win_fraction')}, ensure_ascii=False))
                print('trim1%', row['tail']['top_1pct'])
        print('edge spearman', record['metrics']['edge_cd_sq']['spearman'])
        print('facecount groups', [(g['range'], g['count'], g['metrics']['edge_cd_sq']['mean_ratio'], g['metrics']['edge_cd_sq']['win_fraction']) for g in record['complexity_groups']['gt_face_count']])


if __name__ == '__main__':
    main()
