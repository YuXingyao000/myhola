#!/bin/bash
# Oracle topology upper bound: cross-attention + CLIP-style alignment +
# GT face-adjacency self-attention bias with random padding.

set -e

DATA_ROOT="${DATA_ROOT:-/path/to/your/data}"
LATENT_ROOT="${LATENT_ROOT:-/path/to/cached_latents}"
COND_ROOT="${COND_ROOT:-/path/to/condition_data}"
VAE_CKPT="${VAE_CKPT:-/path/to/vae_checkpoint.ckpt}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/diffusion_oracle_topology}"

python -m src.brepnet.train \
    --config-name train_diffusion_oracle_topology \
    dataset.data_root="$DATA_ROOT" \
    dataset.latent_root="$LATENT_ROOT" \
    dataset.condition_root="$COND_ROOT" \
    dataset.train_dataset="$DATA_ROOT/list/train.txt" \
    dataset.val_dataset="$DATA_ROOT/list/val.txt" \
    dataset.test_dataset="$DATA_ROOT/list/test.txt" \
    model.autoencoder.checkpoint="$VAE_CKPT" \
    trainer.output_dir="$OUTPUT_DIR" \
    "$@"
