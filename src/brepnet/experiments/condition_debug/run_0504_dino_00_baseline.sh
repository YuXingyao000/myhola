#!/usr/bin/env bash
set -euo pipefail

: "${GT_ROOT:?Set GT_ROOT to the root containing <prefix>/normalized_shape.step}"

ROOT="${ROOT:-/mnt/d/data/new_cond_results/exp_plan_0504_dino}"
CKPT="${CKPT:-/mnt/d/data/new_cond_ckpt/0504_deepcad_flux_single_view_align_dino.ckpt}"
AE="${AE:-/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt}"
FACE_Z="${FACE_Z:-/mnt/d/data/ae_cache/1119_deepcad_aug1_11k}"
COND_ROOT="${COND_ROOT:-/mnt/d/data/deepcad_v6_cond}"
TEST_LIST="${TEST_LIST:-src/brepnet/data/list/deduplicated_deepcad_testing_7_30_sample_100.txt}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-32}"
NUM_CPUS="${NUM_CPUS:-32}"
BASE_SEED="${BASE_SEED:-20260509}"
BETA_SCHEDULE="${BETA_SCHEDULE:-squaredcos_cap_v2}"
CONTROLNET="${CONTROLNET:-0}"
CONTROLNET_BASE_CONDITION_DIM="${CONTROLNET_BASE_CONDITION_DIM:-256}"
CONTROLNET_CONDITIONING_SCALE="${CONTROLNET_CONDITIONING_SCALE:-1.0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES

FORCE_ARGS=()
if [[ "${FORCE:-0}" == "1" ]]; then
  FORCE_ARGS=(--force)
fi

MODEL_ARGS=(--beta-schedule "$BETA_SCHEDULE")
if [[ "$CONTROLNET" == "1" ]]; then
  MODEL_ARGS+=(
    --controlnet
    --controlnet-base-condition-dim "$CONTROLNET_BASE_CONDITION_DIM"
    --controlnet-conditioning-scale "$CONTROLNET_CONDITIONING_SCALE"
  )
fi

OUT="$ROOT/E0_baseline"
POST="${OUT}_post"
EVAL_LIST="$OUT/_metadata/generated_prefixes.txt"

python -m src.brepnet.experiments.condition_debug.sample_condition \
  --checkpoint "$CKPT" \
  --autoencoder-weights "$AE" \
  --face-z "$FACE_Z" \
  --cond-root "$COND_ROOT" \
  --test-list "$TEST_LIST" \
  --output-root "$OUT" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --base-seed "$BASE_SEED" \
  --num-samples 1 \
  --condition-mode normal \
  "${MODEL_ARGS[@]}" \
  "${FORCE_ARGS[@]}"

python -m src.brepnet.experiments.condition_debug.post_eval \
  --data-root "$OUT" \
  --post-root "$POST" \
  --gt-root "$GT_ROOT" \
  --list "$EVAL_LIST" \
  --num-cpus "$NUM_CPUS" \
  "${FORCE_ARGS[@]}"

python -m src.brepnet.experiments.condition_debug.summarize_eval \
  --post-root "$POST" \
  --list "$EVAL_LIST" \
  --output-json "$ROOT/E0_baseline_summary.json" \
  --output-csv "$ROOT/E0_baseline_summary.csv"
