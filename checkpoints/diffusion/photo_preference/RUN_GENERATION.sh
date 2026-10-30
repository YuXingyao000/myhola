#!/usr/bin/env bash
set -uo pipefail
cd /mnt/d/python
if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo 'Usage: bash RUN_GENERATION.sh /absolute/run/path [--resume]' >&2
    exit 2
fi
photo_pair_run=$(realpath -m "$1")
photo_pair_resume=()
if [[ $# -eq 2 ]]; then
    if [[ "$2" != '--resume' ]]; then
        echo 'Only --resume is accepted as the second argument.' >&2
        exit 2
    fi
    photo_pair_resume=(--resume)
elif [[ -e "$photo_pair_run" ]]; then
    echo 'Existing run requires --resume; preserving its logs and artifacts.' >&2
    exit 2
fi
mkdir -p "$(dirname "$photo_pair_run")"
export PYTHONPATH=/mnt/d/python
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1

/mnt/d/anaconda3/envs/torch/bin/python -u -m src.brepnet.data.preference_generate \
    --config configs/datagen/preference_photo.yaml \
    --run-dir "$photo_pair_run" "${photo_pair_resume[@]}" \
    >> "${photo_pair_run}.console.log" 2>&1
photo_pair_exit=$?
printf '%s\n' "$photo_pair_exit" > "${photo_pair_run}.exit_code.txt"
exit "$photo_pair_exit"
