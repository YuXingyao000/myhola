#!/bin/bash
# ============================================================
# HoLa-BRep Inference Script
# ============================================================
# Usage: bash scripts/infer.sh <mode> [extra overrides...]
#
# Modes:
#   baseline         - Inference with original white-model pipeline
#   with_mapper      - Inference with Feature Mapper on real photos
#   two_stage        - Image → Point Cloud → BRep
# ============================================================

set -e

MODE="${1:?Usage: bash scripts/infer.sh <mode> [overrides...]}"
shift

DIFFUSION_CKPT="${DIFFUSION_CKPT:-/path/to/diffusion.ckpt}"
MAPPER_CKPT="${MAPPER_CKPT:-/path/to/feature_mapper.pt}"
INPUT_DIR="${INPUT_DIR:-/path/to/input_images}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/inference}"

case "$MODE" in

  baseline)
    echo "[*] Running baseline inference"
    python -m src.brepnet.inference \
        --checkpoint "$DIFFUSION_CKPT" \
        --input "$INPUT_DIR" \
        --output "$OUTPUT_DIR/baseline" \
        --condition single_img \
        "$@"
    ;;

  with_mapper)
    echo "[*] Running inference with Feature Mapper"
    python -m src.brepnet.inference \
        --checkpoint "$DIFFUSION_CKPT" \
        --mapper "$MAPPER_CKPT" \
        --input "$INPUT_DIR" \
        --output "$OUTPUT_DIR/with_mapper" \
        --condition single_img \
        "$@"
    ;;

  two_stage)
    echo "[*] Running two-stage: Image → PC → BRep"
    echo "    Step 1: Generate point clouds with TRELLIS/InstantMesh (manual)"
    echo "    Step 2: Run PC-conditioned diffusion"
    python -m src.brepnet.inference \
        --checkpoint "$DIFFUSION_CKPT" \
        --input "$INPUT_DIR" \
        --output "$OUTPUT_DIR/two_stage" \
        --condition pc \
        "$@"
    ;;

  *)
    echo "Unknown mode: $MODE"
    echo "Available: baseline | with_mapper | two_stage"
    exit 1
    ;;
esac

echo "[*] Done. Results in: $OUTPUT_DIR"
