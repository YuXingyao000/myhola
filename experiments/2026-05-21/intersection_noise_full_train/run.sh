#!/usr/bin/env bash
# Train only the VAE neural intersection with training-time latent-pair noise.
#
# Defaults target the DeepCAD/1119-light setup used by recent experiment records.
# Override variables before the command, or append Hydra overrides after it.

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/mnt/d/data/deepcad_v6}"
TRAIN_LIST="${TRAIN_LIST:-src/brepnet/data/list/deduplicated_deepcad_training_7_30.txt}"
VAL_LIST="${VAL_LIST:-src/brepnet/data/list/deduplicated_deepcad_validation_7_30.txt}"
TEST_LIST="${TEST_LIST:-src/brepnet/data/list/deduplicated_deepcad_testing_7_30_sample_100.txt}"
INIT_CKPT="${INIT_CKPT:-/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/d/data/vae_intersection_experiments}"

MODEL_CONFIG="${MODEL_CONFIG:-vae_light}"
NOISE_STD="${NOISE_STD:-0.1}"
GPUS="${GPUS:-8}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"
MAX_STEPS="${MAX_STEPS:-20000}"

python -m src.brepnet.train \
    --config-name train_vae \
    model="${MODEL_CONFIG}" \
    trainer.init_from_checkpoint="${INIT_CKPT}" \
    trainer.output_dir="${OUTPUT_DIR}" \
    trainer.exp_name="20260521_intersection_noise_${NOISE_STD}_${MODEL_CONFIG}" \
    trainer.devices="${GPUS}" \
    trainer.batch_size="${BATCH_SIZE}" \
    trainer.learning_rate="${LR}" \
    trainer.max_steps="${MAX_STEPS}" \
    trainer.num_sanity_val_steps=0 \
    trainer.check_val_every_n_epoch=1 \
    trainer.vae_validation_inference=false \
    trainer.wandb.enabled=true \
    trainer.wandb.project=hola-brep \
    trainer.wandb.name="20260521_intersection_noise_${NOISE_STD}_${MODEL_CONFIG}" \
    model.trainable_scope=intersection \
    model.intersection_noise_std="${NOISE_STD}" \
    dataset.name=AutoEncoder_dataset3 \
    dataset.data_root="${DATA_ROOT}" \
    dataset.train_dataset="${TRAIN_LIST}" \
    dataset.val_dataset="${VAL_LIST}" \
    dataset.test_dataset="${TEST_LIST}" \
    dataset.is_overfit=false \
    dataset.is_aug=0 \
    dataset.condition_root=null \
    dataset.cached_condition=false \
    dataset.num_points=10000 \
    dataset.scale_factor=1 \
    "$@"
