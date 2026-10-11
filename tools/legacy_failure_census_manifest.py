"""Freeze every legacy posterior-mean failure, without sampling or repairs."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from tools.legacy_failure_select import make_paths, read_table, file_sha256


def partition_population(raw: dict, post: dict, fit: dict) -> tuple[list[str], list[str], dict]:
    """Require equal legacy identity sets and explicit, independently read validity."""
    sets = [{item for method, item in table if method == "legacy"} for table in (raw, post, fit)]
    if not sets[0] or any(ids != sets[0] for ids in sets[1:]):
        raise ValueError("Legacy identities must be nonempty and identical across all three tables")
    ids = sorted(sets[0])
    for item in ids:
        for table in (post, fit):
            if type(table['legacy', item].get('native_valid_solid')) is not bool:
                raise ValueError(f"Unknown native validity for {item}; do not silently call it false")
    failed = [item for item in ids if post['legacy', item]['native_valid_solid'] is False]
    transitions = Counter(
        f"fit_{'valid' if fit['legacy', item]['native_valid_solid'] else 'failed'}_"
        f"optimized_{'valid' if post['legacy', item]['native_valid_solid'] else 'failed'}"
        for item in ids
    )
    return ids, failed, dict(transitions)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args(argv)
    comparison, data_root, root = (value.resolve() for value in
                                   (args.comparison_root, args.data_root, args.output_root))
    source_paths = {'raw_metrics': comparison / 'raw_metrics/posterior_mean/per_object.csv',
                    'optimized': comparison / 'post/optimized/samples.csv',
                    'fit_only': comparison / 'post/fit_only/samples.csv'}
    raw = read_table(source_paths['raw_metrics'], 'model_id')
    post = read_table(source_paths['optimized'], 'sample_id')
    fit = read_table(source_paths['fit_only'], 'sample_id')
    ids, failed, transitions = partition_population(raw, post, fit)
    if len(ids) != 2417 or len(failed) != 369:
        raise ValueError('This frozen study requires the original 2417 / 369 population')
    cases = []
    for item in failed:
        paths = make_paths(item, comparison, data_root)
        # Keep only legacy assets; the new decoder is not part of this census.
        paths = {name: path for name, path in paths.items() if 'joint' not in name}
        cases.append({'id': item, 'gt_face_count': raw['legacy', item]['gt_face_count'],
                      'raw_legacy': raw['legacy', item], 'post_legacy': post['legacy', item],
                      'fit_legacy': fit['legacy', item], 'paths': paths,
                      'path_exists': {name: Path(path).is_file() for name, path in paths.items()}})
    text_ids = '\n'.join(failed) + '\n'
    manifest = {
        'schema': 'legacy_failure_census_manifest_v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'comparison_root': str(comparison), 'data_root': str(data_root),
        'sources': {name: {'path': str(path), 'sha256': file_sha256(path)}
                    for name, path in source_paths.items()},
        'population': {'original_test': 2424, 'contract_unsupported': 7,
                       'accepted_test': len(ids), 'native_valid': len(ids)-len(failed),
                       'native_failed': len(failed), 'selected': len(failed)},
        'scope': 'All legacy / posterior_mean / optimized native-invalid-solid outputs; not diffusion samples',
        'selection_protocol': {'method': 'census: native_valid_solid is False', 'sampling': False,
                               'diagnostic_failures_remain_in_denominator': True},
        'face_identity_contract': {'gt_order_preserved': True, 'face_deduplication': 'not_applied',
                                   'evidence': 'tools.vae_compare_inference.decode_legacy passes v_deduplicated=False'},
        'fit_only_vs_optimized_all_accepted': transitions,
        'failed_ids_sha256': hashlib.sha256(text_ids.encode()).hexdigest(), 'cases': cases,
    }
    outputs = {'manifest.json': json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+'\n',
               'failed_ids.txt': text_ids}
    if any((root/name).exists() for name in outputs):
        raise FileExistsError('Frozen census files already exist; use a new output root')
    root.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        (root/name).write_text(content)
    print(json.dumps({'population': manifest['population'], 'transitions': transitions,
                      'missing_assets': {case['id']: [k for k,v in case['path_exists'].items() if not v]
                                         for case in cases if not all(case['path_exists'].values())}}, indent=2))


if __name__ == '__main__':
    main()
