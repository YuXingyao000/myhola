"""Descriptive same-ID subset diagnostic; never changes main 2417 denominator."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    return {'count': len(a), 'mean': float(a.mean()), 'median': float(np.median(a)),
            'p90': float(np.percentile(a, 90)), 'p95': float(np.percentile(a, 95))}


def main():
    source = ROOT / 'raw_metrics/posterior_mean/per_object.csv'
    with source.open() as handle:
        rows = list(csv.DictReader(handle))
    table = {(x['method'], x['model_id']): x for x in rows}
    ids = sorted({x['model_id'] for x in rows})
    assert len(table) == len(rows) == 4834 and len(ids) == 2417
    cohorts = {'both_adjacency_exact': [], 'at_least_one_adjacency_inexact': []}
    for sample in ids:
        a, b = (table[m, sample] for m in ('intersection', 'local'))
        assert a['input_latent_sha256'] == b['input_latent_sha256']
        exact = all(float(x['topology_fp']) == 0 and float(x['topology_fn']) == 0 for x in (a, b))
        cohorts['both_adjacency_exact' if exact else 'at_least_one_adjacency_inexact'].append(sample)
    result = {'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'note': 'Descriptive only. Equal predicted-vs-GT undirected face-pair sets for both methods; does not establish coedge loop correctness or causal isolation. Main result retains all 2417 IDs.',
              'cohorts': {}}
    for cohort, samples in cohorts.items():
        record = {'count': len(samples), 'ids': samples, 'metrics': {}}
        for metric in ('face_mae', 'face_cd_sq', 'edge_cd_sq', 'edge_pred_to_gt_sq', 'edge_gt_to_pred_sq'):
            a = np.array([float(table['intersection', x][metric]) for x in samples])
            b = np.array([float(table['local', x][metric]) for x in samples])
            record['metrics'][metric] = {'intersection': stats(a), 'local': stats(b),
                                          'local_mean_over_intersection': float(b.mean()/a.mean()),
                                          'local_improved_count': int((b < a - 1e-12).sum())}
        result['cohorts'][cohort] = record
    with (ROOT / 'diagnostics/adjacency_subset.json').open('x') as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for cohort, record in result['cohorts'].items():
        print(cohort, record['count'], json.dumps(record['metrics']['edge_cd_sq']))


if __name__ == '__main__':
    main()
