#!/usr/bin/env bash
# ============================================================================
# 2026-05-25 Experiment: Topology Bias with REAL PHOTOS (ratio=1.0)
# ============================================================================
#
# Context:
#   Previous topo_bias experiments (2026-05-22) used real_photo_ratio=0.0 (pure white
#   model renders). The loss looked good, BUT that doesn't prove topo_bias helps
#   for real-photo → B-Rep (which is the thesis contribution).
#
# Bug found and fixed today:
#   dataset.py had real_photo_ratio HARDCODED to 0.2 (config value was IGNORED).
#   Fixed: prepare_condition() now reads v_real_photo_ratio from config.
#   Also fixed: has_required_condition_files() no longer requires single_view.npz
#   when ratio=0.0.
#
# Experiment matrix:
#   A. Baseline (no topo bias) + real_photo_ratio=1.0  → proves gap exists
#   B. Topo bias + real_photo_ratio=1.0                → proves topo bias helps on real
#   C. (already done) Topo bias + ratio=0.0            → upper bound reference
#
# IMPORTANT: This requires FLUX-generated data in condition_root!
#   Each model needs: {condition_root}/{model_id}/single_view.npz with "flux" key.
#   If data doesn't exist, run the DataGeneration pipeline first.
#
# ============================================================================

set -euo pipefail

# --- Shared settings ---
OUTPUT_DIR="/mnt/d/data/diffusion_topo_experiments"
AE_CKPT="/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt"
DATA_ROOT="/mnt/d/data/deepcad_v6"
LATENT_ROOT="/mnt/d/data/ae_cache/1119_deepcad_aug1_11k"
COND_ROOT="/mnt/d/data/deepcad_v6_cond"
TRAIN_LIST="src/brepnet/data/list/deduplicated_deepcad_training_7_30.txt"
VAL_LIST="src/brepnet/data/list/deduplicated_deepcad_validation_7_30.txt"
TEST_LIST="src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt"

COMMON_ARGS=(
    trainer.devices=8
    trainer.batch_size=64
    trainer.learning_rate=1e-4
    trainer.max_steps=100000
    trainer.check_val_every_n_epoch=1
    trainer.num_sanity_val_steps=0
    trainer.output_dir="${OUTPUT_DIR}"
    trainer.wandb.enabled=true
    trainer.wandb.project=hola-brep
    model.autoencoder.checkpoint="${AE_CKPT}"
    model.latent.use_cached_latents=true
    model.denoiser.hidden_dim=768
    model.padding.max_faces=30
    model.autoencoder.in_channels=6
    model.noise.beta_schedule=squaredcos_cap_v2
    dataset.name=Diffusion_dataset
    dataset.data_root="${DATA_ROOT}"
    dataset.latent_root="${LATENT_ROOT}"
    dataset.condition_root="${COND_ROOT}"
    dataset.train_dataset="${TRAIN_LIST}"
    dataset.val_dataset="${VAL_LIST}"
    dataset.test_dataset="${TEST_LIST}"
    dataset.cached_condition=false
    dataset.is_aug=0
    dataset.max_faces=30
    dataset.scale_factor=200
    hydra.job.chdir=false
)

case "${1:-help}" in
  # --- Experiment A: Baseline (no topo bias) on real photos ---
  baseline-real)
    echo "=== Experiment A: Baseline (no topo bias), real_photo_ratio=1.0 ==="
    python -m src.brepnet.train \
        --config-name train_diffusion_baseline_real \
        "${COMMON_ARGS[@]}" \
        trainer.exp_name=20260525_baseline_real_ratio1.0 \
        trainer.wandb.name=20260525_baseline_real_ratio1.0 \
        dataset.real_photo_ratio=1.0 \
        dataset.load_topology=false \
        model.topology_bias.enabled=false
    ;;

  # --- Experiment B: Topo bias on real photos ---
  topo-bias-real)
    echo "=== Experiment B: Topo bias, real_photo_ratio=1.0 ==="
    python -m src.brepnet.train \
        --config-name train_diffusion_topo_bias_real \
        "${COMMON_ARGS[@]}" \
        trainer.exp_name=20260525_topo_bias_real_ratio1.0 \
        trainer.wandb.name=20260525_topo_bias_real_ratio1.0 \
        dataset.real_photo_ratio=1.0 \
        dataset.load_topology=true \
        model.topology_bias.enabled=true \
        model.topology_bias.scale=2.0
    ;;

  # --- Experiment C: Topo bias on pure white (corrected - now config actually works) ---
  topo-bias-white)
    echo "=== Experiment C: Topo bias, real_photo_ratio=0.0 (CORRECTED) ==="
    python -m src.brepnet.train \
        --config-name train_diffusion_topo_bias \
        "${COMMON_ARGS[@]}" \
        trainer.exp_name=20260525_topo_bias_white_ratio0.0_corrected \
        trainer.wandb.name=20260525_topo_bias_white_ratio0.0_corrected \
        dataset.real_photo_ratio=0.0 \
        dataset.load_topology=true \
        model.topology_bias.enabled=true \
        model.topology_bias.scale=2.0
    ;;

  # --- Quick sanity check (2 GPUs, small data, verify code works) ---
  sanity)
    echo "=== Sanity check: 1 GPU, 100 steps, verify real_photo_ratio takes effect ==="
    python -m src.brepnet.train \
        --config-name train_diffusion_topo_bias_real \
        trainer.devices=1 \
        trainer.batch_size=8 \
        trainer.learning_rate=1e-4 \
        trainer.max_steps=100 \
        trainer.check_val_every_n_epoch=999 \
        trainer.num_sanity_val_steps=0 \
        trainer.output_dir="${OUTPUT_DIR}" \
        trainer.wandb.enabled=false \
        trainer.exp_name=20260525_sanity_real_ratio \
        model.autoencoder.checkpoint="${AE_CKPT}" \
        model.latent.use_cached_latents=true \
        model.denoiser.hidden_dim=768 \
        model.padding.max_faces=30 \
        model.autoencoder.in_channels=6 \
        model.noise.beta_schedule=squaredcos_cap_v2 \
        dataset.name=Diffusion_dataset \
        dataset.data_root="${DATA_ROOT}" \
        dataset.latent_root="${LATENT_ROOT}" \
        dataset.condition_root="${COND_ROOT}" \
        dataset.train_dataset="${TRAIN_LIST}" \
        dataset.val_dataset="${VAL_LIST}" \
        dataset.test_dataset="${TEST_LIST}" \
        dataset.cached_condition=false \
        dataset.is_aug=0 \
        dataset.max_faces=30 \
        dataset.scale_factor=1 \
        dataset.real_photo_ratio=1.0 \
        dataset.load_topology=true \
        model.topology_bias.enabled=true \
        model.topology_bias.scale=2.0 \
        hydra.job.chdir=false
    ;;

  help|*)
    echo "Usage: bash experiments/2026-05-25/run_real_photo_topo_bias.sh <command>"
    echo ""
    echo "Commands:"
    echo "  sanity           - Quick 100-step check (1 GPU) to verify code changes work"
    echo "  baseline-real    - Exp A: No topo bias + real_photo_ratio=1.0"
    echo "  topo-bias-real   - Exp B: Topo bias + real_photo_ratio=1.0"
    echo "  topo-bias-white  - Exp C: Topo bias + ratio=0.0 (corrected config)"
    echo ""
    echo "Run sanity first to verify FLUX data exists and code works!"
    ;;
esac
