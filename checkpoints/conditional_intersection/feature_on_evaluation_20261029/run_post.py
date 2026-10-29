"""Run unchanged recover_v1 with explicit conditional checkpoint identities."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CODE = Path('/mnt/d/python_uv_train/checkpoints/conditional_intersection/feature_on_evaluation_20261029')
HERE = Path('/root/hola_eval/conditional_feature_on_20261029')
ROOT = HERE / 'repository'
REFERENCE = Path('/root/hola_eval/conditional_feature_on_20261029/baseline')
SPLIT = ROOT / 'checkpoints/joint_boundary/validation/val_contract_audit/accepted.txt'
sys.path.insert(0, str(ROOT))
from tools import vae_compare_post as post


def link(target, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        assert destination.resolve() == target.resolve(), destination
    else:
        destination.symlink_to(target, target_is_directory=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', required=True, choices=('conditional_best415', 'conditional_last5395'))
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    os.chdir(ROOT)
    ids = SPLIT.read_text().split()
    assert len(ids) == len(set(ids)) == 3495
    assert post.sha256_file(SPLIT) == 'ae607bfe5499e919d1e68a5110c0c211daf1158aedbe57a4756eda99f6f3f0ed'
    previous = json.loads((REFERENCE / 'metadata/post/optimized/protocol.json').read_text())
    files = {str(p): post.sha256_file(p) for p in sorted(Path('src/brepnet/post').glob('*.py'))}
    assert files == previous['post_algorithm_files'] and len(files) == 11
    output = HERE / args.method
    candidate = HERE / 'raw_predictions/posterior_mean' / args.method
    found = {p.parent.name for p in candidate.glob('*/data.npz')}
    assert found == set(ids), f'Incomplete raw export: {len(found)}'
    transfer = json.loads((HERE / 'validation' / f'{args.method}_transfer.json').read_text())
    assert transfer['passed'] and transfer['count'] == 3495
    link(candidate, output / 'raw_predictions/posterior_mean' / args.method)
    link(REFERENCE / 'raw_predictions/posterior_mean/shared_curve_uv', output / 'raw_predictions/posterior_mean/shared_curve_uv')
    link(REFERENCE / 'post/optimized/shared_curve_uv', output / 'post/optimized/shared_curve_uv')
    cache = output / 'metadata/gt_step_samples'
    cache.mkdir(parents=True, exist_ok=True)
    for source in (REFERENCE / 'metadata/gt_step_samples').glob('*.npz'):
        dest = cache / source.name
        if not dest.exists():
            os.link(source, dest)
    post.BASELINE_METHODS = (*post.BASELINE_METHODS, 'shared_curve_uv')
    post.CANDIDATE_METHODS = (*post.CANDIDATE_METHODS, args.method)
    sys.argv = [str(Path(__file__)), '--comparison-root', str(output), '--data-root', '/mnt/d/data/deepcad_v7',
        '--split-list', str(SPLIT), '--baseline-method', 'shared_curve_uv', '--candidate-method', args.method,
        '--methods', args.method, '--reference-root', str(REFERENCE), '--mode', 'optimized',
        '--post-policy', 'recover_v1', '--workers', str(args.workers), '--timeout', '180',
        '--max-optimize-iter', '200', '--original-test-count', '3505']
    metadata = output / 'metadata/execution'
    metadata.mkdir(parents=True, exist_ok=False)
    import OCC
    post.atomic_json(metadata / 'launch.json', dict(command=sys.argv, python=sys.executable,
        python_version=sys.version, pythonocc=OCC.VERSION, source_hashes=files,
        runner_sha256=post.sha256_file(Path(post.__file__)), wrapper_sha256=post.sha256_file(Path(__file__)),
        data_root_mapping={'logical': '/mnt/d/data/deepcad_v7', 'physical': str(HERE/'gt'), 'verification': str(HERE/'validation/staging.json')},
        git_commit=json.loads((HERE/'metadata/export'/args.method/'protocol.json').read_text())['git_commit'],
        source_repository='/mnt/d/python_uv_train', physical_repository=str(ROOT),
        environment_staging=str(HERE/'validation/environment_staging.json'),
        started_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid()))
    original_scheduler = post.run_isolated_jobs
    def schedule_staged(jobs, metadata, workers, timeout):
        jobs = [dict(job, data_root=str(HERE / 'gt')) for job in jobs]
        return original_scheduler(jobs, metadata, workers, timeout)
    post.run_isolated_jobs = schedule_staged
    started = time.monotonic()
    code = post.main()
    unchanged = all(post.sha256_file(Path(k)) == v for k,v in files.items())
    post.atomic_json(metadata / 'completion.json', dict(returncode=code, elapsed_seconds=time.monotonic()-started,
        production_sources_unchanged=unchanged, finished_utc=datetime.now(timezone.utc).isoformat()))
    assert code == 0 and unchanged


if __name__ == '__main__':
    main()
