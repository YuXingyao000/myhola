#!/usr/bin/env bash
set -uo pipefail
cd /mnt/d/python
export PYTHONPATH=/mnt/d/python
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1

photo_full_run=/mnt/d/python/checkpoints/diffusion/photo_baseline/evaluation_full_20261026
/mnt/d/anaconda3/envs/torch/bin/python -u -m src.brepnet.eval.photo_full_validation \
    --run-dir "$photo_full_run" \
    --reuse-run /mnt/d/python/checkpoints/diffusion/photo_baseline/evaluation_20261026 \
    --canonical-validation-list /mnt/d/python/src/brepnet/data/list/deduplicated_deepcad_validation_7_30.txt \
    --training-run /mnt/d/python/checkpoints/diffusion_photo_baseline/20261025T125340Z \
    --gpus 0 1 2 3 4 5 6 7 \
    --cpu-workers 32 > "$photo_full_run/logs/coordinator.log" 2>&1
photo_full_exit=$?
printf '%s\n' "$photo_full_exit" > "$photo_full_run/logs/exit_code"
exit "$photo_full_exit"
