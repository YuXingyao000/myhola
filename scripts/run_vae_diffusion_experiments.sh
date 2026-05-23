#!/bin/bash
# ============================================================
# VAE & Diffusion 实验集
# ============================================================
# Usage: bash scripts/run_vae_diffusion_experiments.sh <exp> [overrides...]
#
# Experiments:
#   v0_diagnose         - 诊断 latent 分布 (几分钟)
#   v1_kl_sweep         - KL weight sweep (训4个VAE)
#   v2_zero_pad         - Zero-pad + classifier diffusion
#   d1_best_vae         - 在最佳VAE上训diffusion
#   d2_zero_pad_best    - Zero-pad + 最佳VAE上训diffusion
# ============================================================

set -e

MODE="${1:?Usage: bash scripts/run_vae_diffusion_experiments.sh <exp> [overrides...]}"
shift

DATA_ROOT="${DATA_ROOT:-/path/to/your/data}"
COND_ROOT="${COND_ROOT:-/path/to/condition_data}"
OUTPUT_DIR="./outputs/vae_diffusion"

COMMON_VAE=(
    dataset.data_root="$DATA_ROOT"
    dataset.train_dataset="$DATA_ROOT/list/train.txt"
    dataset.val_dataset="$DATA_ROOT/list/val.txt"
    dataset.test_dataset="$DATA_ROOT/list/test.txt"
    trainer.precision="bf16-mixed"
)

case "$MODE" in

  # ─── V0: 诊断 latent 分布 ────────────────────────────
  v0_diagnose)
    echo "[V0] Diagnosing latent space distribution..."
    python -c "
import torch
import numpy as np
from pathlib import Path
from scipy.stats import kurtosis, shapiro

# 加载缓存的 latent stats
latent_root = '${LATENT_ROOT:-/path/to/cached_latents}'
all_z = []
for npz in sorted(Path(latent_root).glob('**/*.npy')):
    z = np.load(npz)
    if z.ndim == 2:  # [num_faces, 32] or [num_faces, 64]
        all_z.append(z[:, :32])  # take mean only (first 32 dims)
    elif z.ndim == 1:
        all_z.append(z[:32].reshape(1, -1))

all_z = np.concatenate(all_z, axis=0)
print(f'Total faces: {all_z.shape[0]}')
print(f'Latent dim:  {all_z.shape[1]}')
print()
print(f'Global mean:    {all_z.mean():.6f}  (ideal: ~0)')
print(f'Global std:     {all_z.std():.6f}   (ideal: ~1)')
print(f'Per-dim std:    min={all_z.std(0).min():.4f}  max={all_z.std(0).max():.4f}')
print(f'Per-dim mean:   min={all_z.mean(0).min():.4f}  max={all_z.mean(0).max():.4f}')
print(f'Kurtosis (avg): {kurtosis(all_z, axis=0).mean():.4f}  (ideal: ~0 for normal)')
print()

# 每个维度的分布健康检查
dead_dims = (all_z.std(0) < 0.01).sum()
print(f'Dead dims (std < 0.01): {dead_dims}/{all_z.shape[1]}')
low_var = (all_z.std(0) < 0.1).sum()
print(f'Low-var dims (std < 0.1): {low_var}/{all_z.shape[1]}')
print()

# 范围
print(f'Value range: [{all_z.min():.4f}, {all_z.max():.4f}]')
print(f'99% percentile: [{np.percentile(all_z, 0.5):.4f}, {np.percentile(all_z, 99.5):.4f}]')
print()

# 建议
if all_z.std() < 0.5:
    print('⚠️  Latent std << 1: 分布过于集中, KL 正则化不足')
    print('    → 建议增大 gaussian_weights 到 1e-3 或 1e-2')
elif all_z.std() > 2.0:
    print('⚠️  Latent std >> 1: 分布过于分散')
    print('    → diffusion 从 N(0,1) 出发距离过远')
else:
    print('✓  Latent std 在合理范围内')

if dead_dims > 0:
    print(f'⚠️  有 {dead_dims} 个维度几乎是常数 → 信息容量浪费')
"
    ;;

  # ─── V1: KL Weight Sweep ─────────────────────────────
  v1_kl_sweep)
    echo "[V1] KL Weight Sweep: training 4 VAEs in sequence"
    echo "     (如果有多卡可以改成并行)"

    for KL in 1e-6 1e-4 1e-3 1e-2; do
        KL_TAG=$(echo $KL | tr '-' 'm' | tr '.' 'p')
        echo ""
        echo "──── Training VAE with gaussian_weights=$KL ────"
        python -m src.brepnet.train \
            --config-name train_vae \
            "${COMMON_VAE[@]}" \
            model.gaussian_weights=$KL \
            trainer.exp_name="vae_kl_${KL_TAG}" \
            trainer.output_dir="$OUTPUT_DIR/v1_kl_sweep" \
            "$@"
    done

    echo ""
    echo "Done. Compare reconstruction CD across:"
    echo "  $OUTPUT_DIR/v1_kl_sweep/vae_kl_*"
    ;;

  # ─── V2: Zero-Pad + Classifier Diffusion ─────────────
  v2_zero_pad)
    VAE_CKPT="${VAE_CKPT:-/path/to/vae.ckpt}"
    LATENT_ROOT="${LATENT_ROOT:-/path/to/cached_latents}"
    echo "[V2] Training Diffusion with zero-padding + classifier"
    python -m src.brepnet.train \
        --config-name train_diffusion_white \
        model.padding.type=zero \
        model.autoencoder.checkpoint="$VAE_CKPT" \
        condition=single_img \
        dataset.name=Diffusion_dataset \
        dataset.latent_root="$LATENT_ROOT" \
        dataset.condition_root="$COND_ROOT" \
        dataset.data_root="$DATA_ROOT" \
        dataset.max_faces=30 \
        dataset.padding=zero \
        trainer.exp_name="diffusion_zero_pad" \
        trainer.output_dir="$OUTPUT_DIR/v2_zero_pad" \
        trainer.batch_size=64 \
        "$@"
    ;;

  # ─── D1: Diffusion on Best VAE ──────────────────────
  d1_best_vae)
    BEST_VAE_CKPT="${BEST_VAE_CKPT:-$OUTPUT_DIR/v1_kl_sweep/vae_kl_1em3/checkpoints/last.ckpt}"
    LATENT_ROOT="${LATENT_ROOT:-$OUTPUT_DIR/d1_cached_latents}"
    echo "[D1] Step 1: Re-extract latents from best VAE..."
    echo "     (需要你先手动用 best VAE 提取新的 latent cache)"
    echo "     VAE checkpoint: $BEST_VAE_CKPT"
    echo "     Output cache dir: $LATENT_ROOT"
    echo ""
    echo "[D1] Step 2: Train diffusion on new latents..."
    python -m src.brepnet.train \
        --config-name train_diffusion_white \
        model.padding.type=random \
        model.autoencoder.checkpoint="$BEST_VAE_CKPT" \
        condition=single_img \
        dataset.name=Diffusion_dataset \
        dataset.latent_root="$LATENT_ROOT" \
        dataset.condition_root="$COND_ROOT" \
        dataset.data_root="$DATA_ROOT" \
        dataset.padding=random \
        trainer.exp_name="diffusion_best_vae" \
        trainer.output_dir="$OUTPUT_DIR/d1_best_vae" \
        trainer.batch_size=64 \
        "$@"
    ;;

  # ─── D2: Zero-pad + Best VAE ────────────────────────
  d2_zero_pad_best)
    BEST_VAE_CKPT="${BEST_VAE_CKPT:-$OUTPUT_DIR/v1_kl_sweep/vae_kl_1em3/checkpoints/last.ckpt}"
    LATENT_ROOT="${LATENT_ROOT:-$OUTPUT_DIR/d1_cached_latents}"
    echo "[D2] Diffusion: Best VAE + Zero-pad + Classifier"
    python -m src.brepnet.train \
        --config-name train_diffusion_white \
        model.padding.type=zero \
        model.autoencoder.checkpoint="$BEST_VAE_CKPT" \
        condition=single_img \
        dataset.name=Diffusion_dataset \
        dataset.latent_root="$LATENT_ROOT" \
        dataset.condition_root="$COND_ROOT" \
        dataset.data_root="$DATA_ROOT" \
        dataset.padding=zero \
        dataset.max_faces=30 \
        trainer.exp_name="diffusion_best_vae_zero_pad" \
        trainer.output_dir="$OUTPUT_DIR/d2_zero_pad_best" \
        trainer.batch_size=64 \
        "$@"
    ;;

  *)
    echo "Unknown experiment: $MODE"
    echo ""
    echo "Available:"
    echo "  v0_diagnose        - 诊断 latent 分布 (几分钟, 不需要训练)"
    echo "  v1_kl_sweep        - KL weight sweep (训4个VAE)"
    echo "  v2_zero_pad        - Zero-pad diffusion (不改VAE)"
    echo "  d1_best_vae        - 在最佳KL的VAE上训diffusion"
    echo "  d2_zero_pad_best   - 最佳VAE + zero-pad diffusion"
    exit 1
    ;;
esac
