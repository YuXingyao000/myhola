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
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES

CFG_SCHEDULE="${CFG_SCHEDULE:-high-t}"
CFG_SCALE="${CFG_SCALE:-2.0}"
CFG_MID_SCALE="${CFG_MID_SCALE:-1.5}"
CFG_LOW_SCALE="${CFG_LOW_SCALE:-1.0}"
CFG_HIGH_T="${CFG_HIGH_T:-500}"
CFG_MID_T="${CFG_MID_T:-200}"

CFG_SCALE_TAG="${CFG_SCALE//./p}"
CFG_MID_SCALE_TAG="${CFG_MID_SCALE//./p}"
CFG_LOW_SCALE_TAG="${CFG_LOW_SCALE//./p}"
CFG_TAG="${CFG_TAG:-cfg_${CFG_SCHEDULE}_s${CFG_SCALE_TAG}_m${CFG_MID_SCALE_TAG}_l${CFG_LOW_SCALE_TAG}}"

OUT="$ROOT/CFG_compare"
CFG_RAW="$OUT/$CFG_TAG/raw"
CFG_POST="$OUT/$CFG_TAG/post"
NO_CFG_POST="${NO_CFG_POST:-$ROOT/E0_baseline_post}"

FORCE_ARGS=()
if [[ "${FORCE:-0}" == "1" ]]; then
  FORCE_ARGS=(--force)
fi

if [[ "${RUN_NO_CFG:-0}" == "1" ]]; then
  NO_CFG_RAW="$OUT/no_cfg/raw"
  NO_CFG_POST="$OUT/no_cfg/post"
  python -m src.brepnet.experiments.condition_debug.sample_condition \
    --checkpoint "$CKPT" \
    --autoencoder-weights "$AE" \
    --face-z "$FACE_Z" \
    --cond-root "$COND_ROOT" \
    --test-list "$TEST_LIST" \
    --output-root "$NO_CFG_RAW" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --base-seed "$BASE_SEED" \
    --num-samples 1 \
    --condition-mode normal \
    "${FORCE_ARGS[@]}"

  python -m src.brepnet.experiments.condition_debug.post_eval \
    --data-root "$NO_CFG_RAW" \
    --post-root "$NO_CFG_POST" \
    --gt-root "$GT_ROOT" \
    --list "$TEST_LIST" \
    --num-cpus "$NUM_CPUS" \
    "${FORCE_ARGS[@]}"

  python -m src.brepnet.experiments.condition_debug.summarize_eval \
    --post-root "$NO_CFG_POST" \
    --list "$TEST_LIST" \
    --output-json "$OUT/no_cfg_summary.json" \
    --output-csv "$OUT/no_cfg_summary.csv"
fi

if [[ ! -d "$NO_CFG_POST" ]]; then
  echo "Missing no-CFG post root: $NO_CFG_POST" >&2
  echo "Run E0 first, set NO_CFG_POST, or set RUN_NO_CFG=1." >&2
  exit 1
fi

python -m src.brepnet.experiments.condition_debug.sample_condition_cfg \
  --checkpoint "$CKPT" \
  --autoencoder-weights "$AE" \
  --face-z "$FACE_Z" \
  --cond-root "$COND_ROOT" \
  --test-list "$TEST_LIST" \
  --output-root "$CFG_RAW" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --base-seed "$BASE_SEED" \
  --num-samples 1 \
  --cfg-schedule "$CFG_SCHEDULE" \
  --cfg-scale "$CFG_SCALE" \
  --cfg-mid-scale "$CFG_MID_SCALE" \
  --cfg-low-scale "$CFG_LOW_SCALE" \
  --cfg-high-t "$CFG_HIGH_T" \
  --cfg-mid-t "$CFG_MID_T" \
  "${FORCE_ARGS[@]}"

python -m src.brepnet.experiments.condition_debug.post_eval \
  --data-root "$CFG_RAW" \
  --post-root "$CFG_POST" \
  --gt-root "$GT_ROOT" \
  --list "$TEST_LIST" \
  --num-cpus "$NUM_CPUS" \
  "${FORCE_ARGS[@]}"

python -m src.brepnet.experiments.condition_debug.summarize_eval \
  --post-root "$CFG_POST" \
  --list "$TEST_LIST" \
  --output-json "$OUT/${CFG_TAG}_summary.json" \
  --output-csv "$OUT/${CFG_TAG}_summary.csv"

python -m src.brepnet.experiments.condition_debug.compare_eval_roots \
  --baseline-post-root "$NO_CFG_POST" \
  --candidate-post-root "$CFG_POST" \
  --baseline-name no_cfg \
  --candidate-name "$CFG_TAG" \
  --list "$TEST_LIST" \
  --output-json "$OUT/${CFG_TAG}_compare.json" \
  --output-csv "$OUT/${CFG_TAG}_compare.csv"
