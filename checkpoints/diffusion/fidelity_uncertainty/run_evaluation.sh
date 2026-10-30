#!/usr/bin/env bash
# Frozen step-5000 validation comparison; launch manually when GPUs are available.
set -euo pipefail
if [[ $# -lt 1 ]]; then
    echo "Usage: bash $0 OUTPUT_DIR [--check-only | --resume]" >&2
    exit 2
fi
evaluation_output=$1
shift
cd /mnt/d/python
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
exec /mnt/d/anaconda3/envs/torch/bin/python -u -m src.brepnet.eval.preference_ablation \
    --old-run /mnt/d/python/checkpoints/diffusion/preference_eval/20261023T090834Z \
    --old-audit /mnt/d/python/checkpoints/diffusion/failure_analysis/runs/20261023T105637Z_full \
    --original-checkpoint /mnt/d/ckpt/HoLa/Diffusion_svr_sq30_1500k.ckpt \
    --preference-checkpoint /mnt/d/python/checkpoints/diffusion_preference_svr_observation/20261024T040949.398883Z/weights/steps/step=00005000.ckpt \
    --split-list /mnt/d/python/src/brepnet/data/list/deduplicated_deepcad_validation_7_30.txt \
    --raw-data-root /mnt/d/data/deepcad_v7 \
    --condition-data-root /mnt/d/data/deepcad_v7_cond \
    --output-dir "$evaluation_output" \
    --gpus 0 1 2 3 4 5 6 7 --cpu-workers 16 --audit-workers 16 \
    --expected-count 3505 --expected-step 5000 --seed 42 --bootstrap-repeats 10000 \
    --font /root/.cache/hola-report-fonts/SourceHanSansSC-Regular.otf "$@"
