#!/bin/bash
# ============================================================
# HoLa-BRep Training Scripts
# ============================================================
# Usage: bash scripts/train.sh <mode> [extra hydra overrides...]
#
# Modes:
#   vae              - Train VAE (Stage 1)
#   diffusion_white  - Train diffusion on white-model images
#   diffusion_real   - Train diffusion on real photos (100%)
#
# Examples:
#   bash scripts/train.sh vae dataset.data_root=/data/deepcad
#   bash scripts/train.sh diffusion_white trainer.devices=8
# ============================================================

set -e

MODE="${1:?Usage: bash scripts/train.sh <mode> [overrides...]}"
shift  # remaining args become hydra overrides

# ─── Paths (modify these for your machine) ─────────────────
DATA_ROOT="${DATA_ROOT:-/path/to/your/data}"
LATENT_ROOT="${LATENT_ROOT:-/path/to/cached_latents}"
COND_ROOT="${COND_ROOT:-/path/to/condition_data}"
VAE_CKPT="${VAE_CKPT:-/path/to/vae_checkpoint.ckpt}"

# ─── Common args ───────────────────────────────────────────
COMMON=(
    dataset.data_root="$DATA_ROOT"
    dataset.train_dataset="$DATA_ROOT/list/train.txt"
    dataset.val_dataset="$DATA_ROOT/list/val.txt"
    dataset.test_dataset="$DATA_ROOT/list/test.txt"
)

# ─── Dispatch ──────────────────────────────────────────────
case "$MODE" in

  vae)
    echo "[*] Training VAE (Stage 1)"
    python -m src.brepnet.train \
        --config-name train_vae \
        "${COMMON[@]}" \
        "$@"
    ;;

  diffusion_white)
    echo "[*] Training Diffusion - white model images"
    python -m src.brepnet.train \
        --config-name train_diffusion_white \
        dataset.latent_root="$LATENT_ROOT" \
        dataset.condition_root="$COND_ROOT" \
        model.autoencoder.checkpoint="$VAE_CKPT" \
        "${COMMON[@]}" \
        "$@"
    ;;

  diffusion_real)
    echo "[*] Training Diffusion - real photos (100%)"
    python -m src.brepnet.train \
        --config-name train_diffusion_real \
        dataset.latent_root="$LATENT_ROOT" \
        dataset.condition_root="$COND_ROOT" \
        model.autoencoder.checkpoint="$VAE_CKPT" \
        "${COMMON[@]}" \
        "$@"
    ;;

  *)
    echo "Unknown mode: $MODE"
    echo "Available: vae | diffusion_white | diffusion_real"
    exit 1
    ;;
esac
