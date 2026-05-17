#!/usr/bin/env bash
set -euo pipefail

: "${GT_ROOT:?Set GT_ROOT to the root containing <prefix>/normalized_shape.step}"

ROOT="${ROOT:-/mnt/d/data/new_cond_results/exp_plan_0502_depth}"
CKPT="${CKPT:-/mnt/d/data/new_cond_ckpt/0502_deepcad_flux_single_view_align_depth.ckpt}"
AE="${AE:-/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt}"
FACE_Z="${FACE_Z:-/mnt/d/data/ae_cache/1119_deepcad_aug1_11k}"
COND_ROOT="${COND_ROOT:-/mnt/d/data/deepcad_v6_cond}"
TEST_LIST="${TEST_LIST:-src/brepnet/data/list/deduplicated_deepcad_testing_7_30_sample_100.txt}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-32}"
NUM_CPUS="${NUM_CPUS:-32}"
BASE_SEED="${BASE_SEED:-20260509}"
K="${K:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES

FORCE_ARGS=()
if [[ "${FORCE:-0}" == "1" ]]; then
  FORCE_ARGS=(--force)
fi

RAW_ROOT="$ROOT/E1_ksample/raw"
POST_ROOT="$ROOT/E1_ksample/post"
BASELINE_POST="$ROOT/E0_baseline_post"

python -m src.brepnet.experiments.condition_debug.sample_condition \
  --checkpoint "$CKPT" \
  --autoencoder-weights "$AE" \
  --face-z "$FACE_Z" \
  --cond-root "$COND_ROOT" \
  --test-list "$TEST_LIST" \
  --output-root "$RAW_ROOT" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --base-seed "$BASE_SEED" \
  --num-samples "$K" \
  --layout sample-subdirs \
  --condition-mode normal \
  "${FORCE_ARGS[@]}"

for sample_idx in $(seq 0 $((K - 1))); do
  sample_name=$(printf "s%02d" "$sample_idx")
  python -m src.brepnet.experiments.condition_debug.post_eval \
    --data-root "$RAW_ROOT/$sample_name" \
    --post-root "$POST_ROOT/$sample_name" \
    --gt-root "$GT_ROOT" \
    --list "$TEST_LIST" \
    --num-cpus "$NUM_CPUS" \
    "${FORCE_ARGS[@]}"
done

python -m src.brepnet.experiments.condition_debug.summarize_ksample_oracle \
  --baseline-post-root "$BASELINE_POST" \
  --ksample-post-root "$POST_ROOT" \
  --list "$TEST_LIST" \
  --num-samples "$K" \
  --layout sample-subdirs \
  --output-json "$ROOT/E1_ksample/oracle_summary.json" \
  --output-csv "$ROOT/E1_ksample/oracle_summary.csv"

