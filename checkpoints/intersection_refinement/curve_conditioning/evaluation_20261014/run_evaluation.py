"""Run the frozen curve-conditioning comparison using existing evaluation tools."""
from pathlib import Path
import argparse
import datetime
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
PYTHON = '/mnt/d/anaconda3/envs/torch/bin/python'
CHECKPOINT = Path('/mnt/d/python/checkpoints/intersection_refinement/curve_conditioning/20261013T083729.740439Z/weights/epoch=0157-step=00057354-val_loss=0.038204.ckpt')
SPLIT = REPO / 'checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt'
REFERENCE = Path('/mnt/d/python/checkpoints/intersection_refinement/evaluation_20261011')
LATENT_REFERENCE = Path('/mnt/d/python/checkpoints/joint_boundary/comparison_20261011')
ORCHESTRATION = ROOT / 'metadata/orchestration'
ENV = {**os.environ, **{name: '1' for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def run_jobs(label, commands):
    ORCHESTRATION.mkdir(parents=True, exist_ok=True)
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


def inference(mode, smoke=False):
    base = [PYTHON, '-m', 'tools.intersection_evaluate_inference', '--method', 'conditioned', '--checkpoint', str(CHECKPOINT),
            '--legacy-checkpoint', '/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt',
            '--reference-root', str(LATENT_REFERENCE), '--data-root', '/mnt/d/data/deepcad_v7',
            '--split-list', str(SPLIT), '--output-root', str(ROOT / 'validation/inference_smoke' if smoke else ROOT), '--mode', mode]
    return [(f'{"smoke" if smoke else "inference"}-{mode}-{shard}', base + ['--num-shards', str(1209 if smoke else 8),
             '--shard-index', str(shard), '--device', f'cuda:{shard}']) for shard in range(1 if smoke else 8)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('smoke', 'inference_posterior_mean', 'inference_archived_latents',
                                         'metrics_posterior_mean', 'metrics_archived_latents', 'post'))
    args = parser.parse_args()
    if args.phase == 'smoke':
        commands = inference('posterior_mean', smoke=True)
    elif args.phase.startswith('inference_'):
        commands = inference(args.phase.removeprefix('inference_'))
    elif args.phase.startswith('metrics_'):
        mode = args.phase.removeprefix('metrics_')
        commands = [(args.phase, [PYTHON, '-m', 'tools.vae_compare_metrics', '--baseline-method', 'intersection',
                     '--candidate-method', 'conditioned', '--comparison-root', str(ROOT), '--data-root', '/mnt/d/data/deepcad_v7',
                     '--split-list', str(SPLIT), '--mode', mode, '--workers', '16'])]
    else:
        commands = [(f'post-{mode}', [PYTHON, '-m', 'tools.vae_compare_post', '--baseline-method', 'intersection',
                     '--candidate-method', 'conditioned', '--methods', 'conditioned', '--reference-root', str(REFERENCE),
                     '--comparison-root', str(ROOT), '--data-root', '/mnt/d/data/deepcad_v7', '--split-list', str(SPLIT),
                     '--mode', mode, '--workers', '32', '--timeout', '180', '--max-optimize-iter', '200',
                     '--original-test-count', '2424']) for mode in ('fit_only', 'optimized')]
    run_jobs(args.phase, commands)


if __name__ == '__main__':
    main()
