#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/d/data/new_cond_results/exp_plan_0502_depth}"
TEST_LIST="${TEST_LIST:-src/brepnet/data/list/deduplicated_deepcad_testing_7_30_sample_100.txt}"
POST_ROOT="${POST_ROOT:-$ROOT/E0_baseline_post}"

python -m src.brepnet.experiments.condition_debug.summarize_face_count \
  --post-root "$POST_ROOT" \
  --list "$TEST_LIST" \
  --output-json "$ROOT/E5_face_count/face_count_summary.json" \
  --output-csv "$ROOT/E5_face_count/face_count_summary.csv"

