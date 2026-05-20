#!/bin/bash
# ============================================================
# VAE Experiments: Topology-Guided & TokenVAE variants
# ============================================================
# 验证 VAE 侧的改进方向
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

  # ─── 实验1：原始VAE (baseline对比) ─────────────────────
  baseline)
    echo "[*] Training baseline VAE (AutoEncoder_1119)"
    python -m src.brepnet.train \
        model=vae_1119 \
        experiment=train_vae \
        trainer.exp_name="vae_baseline" \
        trainer.output_dir="$OUTPUT_DIR" \
        "${COMMON[@]}" "$@"
    ;;

  # ─── 实验2：TokenVAE (attention替代CNN) ────────────────
  token_vae)
    echo "[*] Training TokenVAE (attention-based encoder/decoder)"
    python -m src.brepnet.train \
        model=vae_1119 \
        model.name=AutoEncoder_1119_TokenVAE \
        model.primitive_token_num=8 \
        model.primitive_token_hidden=256 \
        model.primitive_token_depth=6 \
        experiment=train_vae \
        trainer.exp_name="vae_token_k8_d6" \
        trainer.output_dir="$OUTPUT_DIR" \
        "${COMMON[@]}" "$@"
    ;;

  # ─── 实验3：TokenVAE 更大容量 ─────────────────────────
  token_vae_large)
    echo "[*] Training TokenVAE (larger: k=16, depth=8)"
    python -m src.brepnet.train \
        model=vae_1119 \
        model.name=AutoEncoder_1119_TokenVAE \
        model.primitive_token_num=16 \
        model.primitive_token_hidden=256 \
        model.primitive_token_depth=8 \
        model.primitive_token_heads=8 \
        experiment=train_vae \
        trainer.exp_name="vae_token_k16_d8" \
        trainer.output_dir="$OUTPUT_DIR" \
        "${COMMON[@]}" "$@"
    ;;

  # ─── 实验4：Disentangled Latent VAE ───────────────────
  disentangled)
    echo "[*] Training Disentangled Latent VAE (topo|geom split)"
    echo "    注意: 需要修改 vae.py 中的 sample() 方法使用 DisentangledLatentProjection"
    echo "    参见 models/topology_guided.py"
    python -m src.brepnet.train \
        model=vae_1119 \
        model.disentangled_latent=true \
        model.topo_dim=16 \
        model.geom_dim=16 \
        experiment=train_vae \
        trainer.exp_name="vae_disentangled_16_16" \
        trainer.output_dir="$OUTPUT_DIR" \
        "${COMMON[@]}" "$@"
    ;;

  *)
    echo "Unknown mode: $MODE"
    echo ""
    echo "Available:"
    echo "  baseline        - 原始 AutoEncoder_1119 (对比用)"
    echo "  token_vae       - Attention-based encoder/decoder (k=8)"
    echo "  token_vae_large - 更大容量 TokenVAE (k=16)"
    echo "  disentangled    - 拓扑-几何解耦 latent"
    exit 1
    ;;
esac
