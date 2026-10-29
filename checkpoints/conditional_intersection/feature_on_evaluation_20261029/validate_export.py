"""Validate transferred raw files and common CAD identities before OCC."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
import json
from pathlib import Path
import numpy as np

CODE = Path('/mnt/d/python_uv_train/checkpoints/conditional_intersection/feature_on_evaluation_20261029')
HERE = Path('/root/hola_eval/conditional_feature_on_20261029')
ROOT = CODE.parents[2]
REFERENCE = Path('/root/hola_eval/conditional_feature_on_20261029/baseline')


def sha(path):
    digest = hashlib.sha256()
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOATIME), 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True)
    args = parser.parse_args()
    method = args.method
    ids = (ROOT/'checkpoints/joint_boundary/validation/val_contract_audit/accepted.txt').read_text().split()
    metadata = HERE/'metadata/export'/method
    rows = [json.loads(line) for line in (metadata/'records.jsonl').read_text().splitlines()]
    completed = json.loads((metadata/'completion.json').read_text())
    assert completed['source_unchanged'] and completed['checkpoint_unchanged']
    assert len(rows) == len(ids) == len(set(row['id'] for row in rows)) == 3495
    assert set(row['id'] for row in rows) == set(ids)
    assert Counter(row['status'] for row in rows) == {'ok': 3495}
    def check(row):
        name = row['id']
        raw = HERE/'raw_predictions/posterior_mean'/method/name/'data.npz'
        baseline = REFERENCE/'raw_predictions/posterior_mean/shared_curve_uv'/name/'data.npz'
        source = Path('/mnt/d/data/deepcad_v7')/name/'data.npz'
        assert sha(raw) == row['raw_sha256'], name
        assert sha(source) == row['input_source_sha256'], name
        with np.load(raw, allow_pickle=False) as new, np.load(baseline, allow_pickle=False) as old:
            assert str(new['input_source_sha256']) == row['input_source_sha256'], name
            assert str(old['input_source_sha256']) == row['input_source_sha256'], name
        return name
    with ThreadPoolExecutor(max_workers=8) as pool:
        found = list(pool.map(check, rows))
    result = dict(passed=True,count=len(found),method=method,transferred_raw_sha_matches=True,
        source_cad_sha_matches_142_and_baseline=True,ids_unique_and_complete=True,
        source_unchanged=completed['source_unchanged'],checkpoint_unchanged=completed['checkpoint_unchanged'],
        export_seconds=completed['elapsed_seconds'],records_sha256=sha(metadata/'records.jsonl'))
    output=HERE/'validation'/f'{method}_transfer.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
