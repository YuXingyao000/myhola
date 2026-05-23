#!/usr/bin/env bash
set -euo pipefail

: "${GT_ROOT:?Set GT_ROOT to the root containing <prefix>/normalized_shape.step}"

ROOT="${ROOT:-/mnt/d/data/new_cond_results/exp_plan_0502_depth}"
CKPT="${CKPT:-/mnt/d/data/new_cond_ckpt/0502_deepcad_flux_single_view_align_depth.ckpt}"
AE="${AE:-/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt}"
LATENT_ROOT="${LATENT_ROOT:-/mnt/d/data/ae_cache/1119_deepcad_aug1_11k}"
COND_ROOT="${COND_ROOT:-/mnt/d/data/deepcad_v6_cond}"
TEST_LIST="${TEST_LIST:-src/brepnet/data/list/deduplicated_deepcad_testing_7_30_sample_100.txt}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-32}"
NUM_CPUS="${NUM_CPUS:-32}"
BASE_SEED="${BASE_SEED:-20260509}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES

FORCE_ARGS=()
if [[ "${FORCE:-0}" == "1" ]]; then
  FORCE_ARGS=(--force)
fi

EXP_ROOT="$ROOT/E2_condition_sensitivity"

for mode in normal shuffle zero; do
  OUT="$EXP_ROOT/$mode/raw"
  POST="$EXP_ROOT/$mode/post"
  python -m src.brepnet.experiments.condition_debug.sample_condition \
    --checkpoint "$CKPT" \
    --autoencoder-weights "$AE" \
    --latent-root "$LATENT_ROOT" \
    --cond-root "$COND_ROOT" \
    --test-list "$TEST_LIST" \
    --output-root "$OUT" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --base-seed "$BASE_SEED" \
    --num-samples 1 \
    --condition-mode "$mode" \
    "${FORCE_ARGS[@]}"

  python -m src.brepnet.experiments.condition_debug.post_eval \
    --data-root "$OUT" \
    --post-root "$POST" \
    --gt-root "$GT_ROOT" \
    --list "$TEST_LIST" \
    --num-cpus "$NUM_CPUS" \
    "${FORCE_ARGS[@]}"

  python -m src.brepnet.experiments.condition_debug.summarize_eval \
    --post-root "$POST" \
    --list "$TEST_LIST" \
    --output-json "$EXP_ROOT/${mode}_summary.json" \
    --output-csv "$EXP_ROOT/${mode}_summary.csv"
done

