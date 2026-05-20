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
#   feature_mapper   - Train Feature Domain Mapper (Strategy B)
#   distill          - Knowledge Distillation (Strategy A)
#
# Examples:
#   bash scripts/train.sh vae dataset.data_root=/data/deepcad
#   bash scripts/train.sh feature_mapper trainer.gpus=2
#   bash scripts/train.sh distill strategy.teacher.checkpoint=/ckpt/white.ckpt
# ============================================================

set -e

MODE="${1:?Usage: bash scripts/train.sh <mode> [overrides...]}"
shift  # remaining args become hydra overrides

# ─── Paths (modify these for your machine) ─────────────────
DATA_ROOT="${DATA_ROOT:-/path/to/your/data}"
FACE_Z_DIR="${FACE_Z_DIR:-/path/to/cached_latents}"
COND_ROOT="${COND_ROOT:-/path/to/condition_data}"
VAE_CKPT="${VAE_CKPT:-/path/to/vae_checkpoint.ckpt}"
TEACHER_CKPT="${TEACHER_CKPT:-/path/to/white_model_diffusion.ckpt}"

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
        experiment=train_vae \
        "${COMMON[@]}" \
        "$@"
    ;;

  diffusion_white)
    echo "[*] Training Diffusion - white model images"
    python -m src.brepnet.train \
        experiment=train_diffusion_white \
        dataset.face_z_dir="$FACE_Z_DIR" \
        dataset.cond_root="$COND_ROOT" \
        model.autoencoder.weights="$VAE_CKPT" \
        "${COMMON[@]}" \
        "$@"
    ;;

  diffusion_real)
    echo "[*] Training Diffusion - real photos (100%)"
    python -m src.brepnet.train \
        experiment=train_diffusion_real \
        dataset.face_z_dir="$FACE_Z_DIR" \
        dataset.cond_root="$COND_ROOT" \
        model.autoencoder.weights="$VAE_CKPT" \
        "${COMMON[@]}" \
        "$@"
    ;;

  feature_mapper)
    echo "[*] Training Feature Domain Mapper (Strategy B)"
    python -m src.brepnet.train \
        experiment=train_feature_mapper \
        dataset.cond_root="$COND_ROOT" \
        "${COMMON[@]}" \
        "$@"
    ;;

  distill)
    echo "[*] Training Knowledge Distillation (Strategy A)"
    python -m src.brepnet.train \
        experiment=train_distill \
        strategy.teacher.checkpoint="$TEACHER_CKPT" \
        dataset.face_z_dir="$FACE_Z_DIR" \
        dataset.cond_root="$COND_ROOT" \
        model.autoencoder.weights="$VAE_CKPT" \
        "${COMMON[@]}" \
        "$@"
    ;;

  *)
    echo "Unknown mode: $MODE"
    echo "Available: vae | diffusion_white | diffusion_real | feature_mapper | distill"
    exit 1
    ;;
esac
