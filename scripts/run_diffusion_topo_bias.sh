#!/bin/bash
# ============================================================
# Experiment: Diffusion with Topology Attention Bias
# ============================================================
# Adds GT adjacency matrix as attention bias to the denoising backbone.
# Adjacent faces get positive bias → attend more to each other.
# Bias is timestep-dependent: strong at high noise, weak at low noise.
#
# Usage:
#   bash scripts/run_diffusion_topo_bias.sh [extra overrides...]
#
# Example:
#   bash scripts/run_diffusion_topo_bias.sh trainer.gpus=4 model.topo_scale=3.0
# ============================================================

set -e

# ─── Paths (modify these for your machine) ─────────────────
DATA_ROOT="${DATA_ROOT:-/path/to/your/data}"
FACE_Z_DIR="${FACE_Z_DIR:-/path/to/cached_latents}"
COND_ROOT="${COND_ROOT:-/path/to/condition_data}"
VAE_CKPT="${VAE_CKPT:-/path/to/vae_checkpoint.ckpt}"
OUTPUT_DIR="./outputs/diffusion_topo_bias"

echo "============================================"
echo " Diffusion + Topology Attention Bias"
echo " topo_scale: adjustable via model.topo_scale=X"
echo " dataset.load_topology=true"
echo "============================================"

python -m src.brepnet.train \
    experiment=train_diffusion_topo_bias \
    dataset.data_root="$DATA_ROOT" \
    dataset.face_z_dir="$FACE_Z_DIR" \
    dataset.cond_root="$COND_ROOT" \
    dataset.train_dataset="$DATA_ROOT/list/train.txt" \
    dataset.val_dataset="$DATA_ROOT/list/val.txt" \
    dataset.test_dataset="$DATA_ROOT/list/test.txt" \
    model.autoencoder.weights="$VAE_CKPT" \
    trainer.output_dir="$OUTPUT_DIR" \
    "$@"

echo ""
echo "============================================"
echo " Done. Checkpoint in: $OUTPUT_DIR/diffusion_topo_bias/checkpoints/"
echo "============================================"
