#!/usr/bin/env bash
# Run on 142. New 64-D run, only edge feature matching disabled, capped at 30k.
set -euo pipefail
cd /mnt/d/python
export WANDB_MODE=online
experiment_run_dir=/mnt/d/python/checkpoints/latent_expansion/d64_no_edge_feature/20261023T034042Z
exec /mnt/d/anaconda3/envs/torch/bin/python -u -m src.brepnet.train \
  --config-name train_vae_latent_expansion_no_edge_feature \
  dataset.raw_data_root=/mnt/d/data/deepcad_v7 \
  dataset.edge_uv_root=/mnt/d/data/deepcad_v7_uv/v1 \
  dataset.train_dataset=checkpoints/joint_boundary/validation/train_contract_audit/accepted.txt \
  dataset.val_dataset=checkpoints/joint_boundary/validation/val_contract_audit/accepted.txt \
  dataset.test_dataset=checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt \
  'model.init_latent_expansion_checkpoint="/mnt/d/python/checkpoints/shared_curve_uv/candidate/20261019T064906.049992Z/weights/epoch=0116-step=00042471-val_loss=0.036517.ckpt"' \
  trainer.devices=8 trainer.batch_size=16 trainer.seed=42 \
  trainer.learning_rate=2e-5 trainer.precision=bf16-mixed \
  trainer.max_steps=30000 trainer.max_epochs=10000 trainer.max_time=null \
  trainer.run_dir="$experiment_run_dir" \
  trainer.wandb.enabled=true trainer.wandb.offline=false "$@"
