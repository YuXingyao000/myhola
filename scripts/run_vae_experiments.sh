#!/bin/bash
# ============================================================
# VAE Experiments
# ============================================================
# Train the maintained VAE variants.
# ============================================================

set -e

MODE="${1:?Usage: bash scripts/run_vae_experiments.sh <mode>}"
shift

DATA_ROOT="${DATA_ROOT:-/path/to/your/data}"
OUTPUT_DIR="./outputs/vae_experiments"

COMMON=(
    dataset.data_root="$DATA_ROOT"
    dataset.train_dataset="$DATA_ROOT/list/train.txt"
    dataset.val_dataset="$DATA_ROOT/list/val.txt"
    trainer.precision="bf16-mixed"
)

case "$MODE" in

  baseline)
    echo "[*] Training baseline VAE (AutoEncoder)"
    python -m src.brepnet.train \
        --config-name train_vae \
        trainer.exp_name="vae_baseline" \
        trainer.output_dir="$OUTPUT_DIR" \
        "${COMMON[@]}" "$@"
    ;;

  light)
    echo "[*] Training light VAE (AutoEncoder_light)"
    python -m src.brepnet.train \
        --config-name train_vae \
        model=vae_light \
        trainer.exp_name="vae_light" \
        trainer.output_dir="$OUTPUT_DIR" \
        "${COMMON[@]}" "$@"
    ;;

  light_exp)
    echo "[*] Training experimental light VAE (AutoEncoder_light_exp)"
    python -m src.brepnet.train \
        --config-name train_vae \
        model=vae_light_exp \
        trainer.exp_name="vae_light_exp" \
        trainer.output_dir="$OUTPUT_DIR" \
        "${COMMON[@]}" "$@"
    ;;

  *)
    echo "Unknown mode: $MODE"
    echo ""
    echo "Available:"
    echo "  baseline        - 原始 AutoEncoder (对比用)"
    echo "  light           - AutoEncoder_light"
    echo "  light_exp       - AutoEncoder_light_exp"
    exit 1
    ;;
esac
