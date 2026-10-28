#!/usr/bin/env bash
set -uo pipefail
cd /mnt/d/python
if [[ $# -ne 1 ]]; then
    echo 'Usage: bash RUN_TRAINING.sh /absolute/new/run/path' >&2
    exit 2
fi
photo_dpo_run=$(realpath -m "$1")
if [[ -e "$photo_dpo_run/metadata/run.json" ]]; then
    echo 'Existing run: use explicit trainer.resume_from_checkpoint, not new initialization.' >&2
    exit 2
fi
mkdir -p "$photo_dpo_run/logs" "$photo_dpo_run/metadata"
exec 9>"$photo_dpo_run/metadata/training.lock"
flock -n 9 || exit 73
export PYTHONPATH=/mnt/d/python
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1

/mnt/d/anaconda3/envs/torch/bin/python -u -m src.brepnet.train \
    --config-name train_diffusion_preference_photo \
    trainer.run_dir="$photo_dpo_run" \
    trainer.devices=7 trainer.batch_size=8 trainer.seed=42 \
    trainer.max_steps=5000 trainer.precision=32-true trainer.learning_rate=1e-5 \
    >> "$photo_dpo_run/logs/console.log" 2>&1
photo_dpo_exit=$?
printf '%s\n' "$photo_dpo_exit" > "$photo_dpo_run/metadata/exit_code.txt"
exit "$photo_dpo_exit"
