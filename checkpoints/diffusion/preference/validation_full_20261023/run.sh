#!/usr/bin/env bash
set -uo pipefail

cd /mnt/d/python
run_dir=${1:?Usage: bash run.sh /absolute/path/to/new_run}
mkdir -p "$(dirname "$run_dir")"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=2

/mnt/d/anaconda3/envs/torch/bin/python -u -m src.brepnet.eval.preference_compare \
    --run-dir "$run_dir" \
    --original-checkpoint /mnt/d/ckpt/HoLa/Diffusion_svr_sq30_1500k.ckpt \
    --preference-checkpoint /mnt/d/python/checkpoints/diffusion_preference_svr/20261023T030527Z/weights/steps/step=00005000.ckpt \
    --split-list /mnt/d/python/src/brepnet/data/list/deduplicated_deepcad_validation_7_30.txt \
    --expected-count 3505 \
    --reuse-run-dir /mnt/d/python/checkpoints/diffusion/preference_eval/20261023T062400Z \
    --raw-data-root /mnt/d/data/deepcad_v7 \
    --condition-data-root /mnt/d/data/deepcad_v7_cond \
    --gpus 0 1 2 3 4 5 6 7 \
    --cpu-workers 16 \
    --seed 42 > "${run_dir}.log" 2>&1
run_status=$?
printf '%s\n' "$run_status" > "${run_dir}.exit_code.txt"
exit "$run_status"
