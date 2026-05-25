#!/usr/bin/env bash
# ============================================================================
# 2026-05-25 Dataset Quality Evaluation
# ============================================================================
#
# Evaluate FLUX-generated images against OCC gray-model renders.
# Metrics: Silhouette IoU, DINOv2 Cosine, CLIP Realism Score
#
# Prerequisites:
#   - condition_root/{model_id}/imgs.npz must exist (OCC renders, key: svr_imgs)
#   - condition_root/{model_id}/single_view.npz must exist (FLUX output, key: flux)
#
# ============================================================================

set -euo pipefail
cd /mnt/d/python

COND_ROOT="/mnt/d/data/deepcad_v6_cond"
MODEL_LIST="src/brepnet/data/list/deduplicated_deepcad_training_7_30.txt"
OUTPUT_DIR="experiments/2026-05-25/quality_results"

mkdir -p "${OUTPUT_DIR}"

case "${1:-help}" in

  # --- IoU only (fast, no GPU needed) ---
  iou)
    echo "=== Silhouette IoU only (CPU) ==="
    python -m src.brepnet.eval.quality_metrics \
      --condition-root "${COND_ROOT}" \
      --model-list "${MODEL_LIST}" \
      --output "${OUTPUT_DIR}/iou_only.json"
    ;;

  # --- IoU + DINO cosine (needs GPU, ~1 hour for full dataset) ---
  iou-dino)
    echo "=== IoU + DINOv2 Cosine (GPU) ==="
    python -m src.brepnet.eval.quality_metrics \
      --condition-root "${COND_ROOT}" \
      --model-list "${MODEL_LIST}" \
      --compute-dino \
      --output "${OUTPUT_DIR}/iou_dino.json"
    ;;

  # --- IoU + CLIP realism (needs GPU) ---
  iou-clip)
    echo "=== IoU + CLIP Realism (GPU) ==="
    python -m src.brepnet.eval.quality_metrics \
      --condition-root "${COND_ROOT}" \
      --model-list "${MODEL_LIST}" \
      --compute-clip \
      --output "${OUTPUT_DIR}/iou_clip.json"
    ;;

  # --- Full evaluation: IoU + DINO + CLIP ---
  full)
    echo "=== Full Quality Eval: IoU + DINO + CLIP (GPU, slow) ==="
    python -m src.brepnet.eval.quality_metrics \
      --condition-root "${COND_ROOT}" \
      --model-list "${MODEL_LIST}" \
      --compute-dino \
      --compute-clip \
      --output "${OUTPUT_DIR}/full_quality.json"
    ;;

  # --- Quick test on 100 models ---
  test)
    echo "=== Quick test (first 100 models, IoU only) ==="
    head -100 "${MODEL_LIST}" > /tmp/quality_test_list.txt
    python -m src.brepnet.eval.quality_metrics \
      --condition-root "${COND_ROOT}" \
      --model-list /tmp/quality_test_list.txt \
      --output "${OUTPUT_DIR}/test_100.json"
    ;;

  # --- Test set evaluation ---
  test-set)
    echo "=== Evaluate on test set ==="
    python -m src.brepnet.eval.quality_metrics \
      --condition-root "${COND_ROOT}" \
      --model-list src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt \
      --compute-dino \
      --compute-clip \
      --output "${OUTPUT_DIR}/test_set_quality.json"
    ;;

  help|*)
    echo "Usage: bash experiments/2026-05-25/run_quality_eval.sh <command>"
    echo ""
    echo "Commands:"
    echo "  test        - Quick test (100 models, IoU only, no GPU)"
    echo "  iou         - Full dataset, IoU only (fast, CPU)"
    echo "  iou-dino    - IoU + DINOv2 cosine (GPU)"
    echo "  iou-clip    - IoU + CLIP realism (GPU)"
    echo "  full        - All metrics (IoU + DINO + CLIP, slow)"
    echo "  test-set    - Evaluate test split with all metrics"
    echo ""
    echo "Output: ${OUTPUT_DIR}/*.json"
    echo ""
    echo "Metrics explained:"
    echo "  Silhouette IoU  — shape preservation (foreground mask overlap)"
    echo "  DINO Cosine     — semantic similarity (same object? [0.5~0.9 good])"
    echo "  CLIP Realism    — how real does it look? (vs 'a real photograph') [>0.25 good]"
    ;;
esac
