#!/usr/bin/env bash
set -uo pipefail
cd /mnt/d/python
export PYTHONPATH=/mnt/d/python
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
/mnt/d/anaconda3/envs/torch/bin/python -u -m src.brepnet.eval.photo_baseline --run-dir /mnt/d/python/checkpoints/diffusion/photo_baseline/evaluation_20261026 --photo-val-best /mnt/d/python/checkpoints/diffusion/photo_baseline/evaluation_20261026/weights/photo_val_best.ckpt --photo-latest /mnt/d/python/checkpoints/diffusion/photo_baseline/evaluation_20261026/weights/photo_latest.ckpt > /mnt/d/python/checkpoints/diffusion/photo_baseline/evaluation_20261026/logs/coordinator.log 2>&1
evaluation_exit_code=$?
printf '%s\n' "$evaluation_exit_code" > /mnt/d/python/checkpoints/diffusion/photo_baseline/evaluation_20261026/logs/exit_code
exit "$evaluation_exit_code"
