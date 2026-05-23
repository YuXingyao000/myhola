#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/d/data/new_cond_results/exp_plan_0502_depth}"
CKPT="${CKPT:-/mnt/d/data/new_cond_ckpt/0502_deepcad_flux_single_view_align_depth.ckpt}"
AE="${AE:-/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt}"
LATENT_ROOT="${LATENT_ROOT:-/mnt/d/data/ae_cache/1119_deepcad_aug1_11k}"
COND_ROOT="${COND_ROOT:-/mnt/d/data/deepcad_v6_cond}"
TEST_LIST="${TEST_LIST:-src/brepnet/data/list/deduplicated_deepcad_testing_7_30_sample_100.txt}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-32}"
BASE_SEED="${BASE_SEED:-20260509}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES

FORCE_ARGS=()
if [[ "${FORCE:-0}" == "1" ]]; then
  FORCE_ARGS=(--force)
fi

python -m src.brepnet.experiments.condition_debug.denoise_gt_probe \
  --checkpoint "$CKPT" \
  --autoencoder-weights "$AE" \
  --latent-root "$LATENT_ROOT" \
  --cond-root "$COND_ROOT" \
  --test-list "$TEST_LIST" \
  --output-root "$ROOT/E3_denoise_probe" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --base-seed "$BASE_SEED" \
  --timesteps 50 100 200 500 800 \
  --condition-modes normal shuffle zero \
  "${FORCE_ARGS[@]}"

