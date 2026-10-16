"""Frozen Oct13 orchestration only; production model and metric tools unchanged."""
from pathlib import Path
import argparse
import datetime
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
PYTHON = '/mnt/d/anaconda3/envs/torch/bin/python'
CHECKPOINT = REPO / 'checkpoints/intersection_local/full_capacity/20261012T141134.581695Z/weights/epoch=0042-step=00015609-val_loss=0.098798.ckpt'
SPLIT = REPO / 'checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt'
REFERENCE = REPO / 'checkpoints/intersection_refinement/evaluation_20261011'
ORCHESTRATION = ROOT / 'metadata/orchestration'
ENV = {**os.environ, **{name: '1' for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}}

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def run_jobs(label, commands):
    manifest = ORCHESTRATION / (label + '_jobs.json')
    if manifest.exists():
        raise FileExistsError(manifest)
    jobs = []
    for name, command in commands:
        log = ORCHESTRATION / (name + '.log')
        handle = log.open('x')
        process = subprocess.Popen(command, cwd=REPO, env=ENV, stdout=handle, stderr=subprocess.STDOUT)
        jobs.append((process, handle, {'name': name, 'command': command, 'log': str(log), 'pid': process.pid, 'started_utc': now()}))
    manifest.write_text(json.dumps([record for _, _, record in jobs], indent=2) + '\n')
    while any(process.poll() is None for process, _, _ in jobs):
        print(json.dumps({'phase': label, 'utc': now(), 'running': sum(process.poll() is None for process, _, _ in jobs)}), flush=True)
        time.sleep(30)
    for process, handle, record in jobs:
        handle.close()
        record.update(exit_code=process.returncode, completed_utc=now())
    manifest.write_text(json.dumps([record for _, _, record in jobs], indent=2) + '\n')
    if any(process.returncode for process, _, _ in jobs):
        raise RuntimeError(f'{label} has failed jobs; inspect manifests/logs, never silently retry')

def inference(mode, count=8, smoke=False):
    base = [PYTHON, '-m', 'tools.intersection_evaluate_inference', '--method', 'local', '--checkpoint', str(CHECKPOINT),
            '--legacy-checkpoint', '/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt',
            '--reference-root', str(REPO / 'checkpoints/joint_boundary/comparison_20261011'),
            '--data-root', '/mnt/d/data/deepcad_v7', '--split-list', str(SPLIT),
            '--output-root', str(ROOT / 'validation/inference_smoke' if smoke else ROOT), '--mode', mode]
    return [(f'{"smoke" if smoke else "inference"}-{mode}-{shard}', base + ['--num-shards', str(1209 if smoke else 8),
             '--shard-index', str(shard), '--device', f'cuda:{shard}']) for shard in range(count)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('smoke', 'inference', 'metrics', 'metrics_posterior_mean', 'metrics_archived_latents', 'post'))
    args = parser.parse_args()
    ORCHESTRATION.mkdir(parents=True, exist_ok=True)
    if args.phase == 'smoke':
        run_jobs('smoke', inference('posterior_mean', count=1, smoke=True))
    elif args.phase == 'inference':
        for mode in ('posterior_mean', 'archived_latents'):
            run_jobs('inference_' + mode, inference(mode))
    elif args.phase.startswith('metrics'):
        modes = ('posterior_mean', 'archived_latents') if args.phase == 'metrics' else (args.phase.removeprefix('metrics_'),)
        run_jobs(args.phase, [(f'metrics-{mode}', [PYTHON, '-m', 'tools.vae_compare_metrics', '--baseline-method', 'intersection',
                  '--candidate-method', 'local', '--comparison-root', str(ROOT), '--data-root', '/mnt/d/data/deepcad_v7',
                  '--split-list', str(SPLIT), '--mode', mode, '--workers', '16']) for mode in modes])
    else:
        run_jobs('post', [(f'post-{mode}', [PYTHON, '-m', 'tools.vae_compare_post', '--baseline-method', 'intersection',
                  '--candidate-method', 'local', '--methods', 'local', '--reference-root', str(REFERENCE),
                  '--comparison-root', str(ROOT), '--data-root', '/mnt/d/data/deepcad_v7', '--split-list', str(SPLIT),
                  '--mode', mode, '--workers', '32', '--timeout', '180', '--max-optimize-iter', '200',
                  '--original-test-count', '2424']) for mode in ('fit_only', 'optimized')])

if __name__ == '__main__':
    main()
