#!/bin/bash
# ============================================================
# HoLa-BRep Inference Script
# ============================================================
# Usage: bash scripts/infer.sh <mode> [extra overrides...]
#
# Modes:
#   baseline         - Inference with original white-model pipeline
#   two_stage        - Image → Point Cloud → BRep
# ============================================================

set -e

MODE="${1:?Usage: bash scripts/infer.sh <mode> [overrides...]}"
shift

DIFFUSION_CKPT="${DIFFUSION_CKPT:-/path/to/diffusion.ckpt}"
VAE_CKPT="${VAE_CKPT:-/path/to/vae.ckpt}"
INPUT_DIR="${INPUT_DIR:-/path/to/input_images}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/inference}"

case "$MODE" in

  baseline)
    echo "[*] Running baseline inference"
    python -m src.brepnet.inference \
        --diffusion-weights "$DIFFUSION_CKPT" \
        --autoencoder-weights "$VAE_CKPT" \
        --input "$INPUT_DIR" \
        --output-dir "$OUTPUT_DIR/baseline" \
        --condition single_img \
        "$@"
    ;;

  two_stage)
    echo "[*] Running two-stage: Image → PC → BRep"
    echo "    Step 1: Generate point clouds with TRELLIS/InstantMesh (manual)"
    echo "    Step 2: Run PC-conditioned diffusion"
    python -m src.brepnet.inference \
        --diffusion-weights "$DIFFUSION_CKPT" \
        --autoencoder-weights "$VAE_CKPT" \
        --input "$INPUT_DIR" \
        --output-dir "$OUTPUT_DIR/two_stage" \
        --condition point_cloud \
        "$@"
    ;;

  *)
    echo "Unknown mode: $MODE"
    echo "Available: baseline | two_stage"
    exit 1
    ;;
esac

echo "[*] Done. Results in: $OUTPUT_DIR"
