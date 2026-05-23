#!/usr/bin/env bash
set -euo pipefail

# Evaluate the 0504 single-view model with the same cached-latent target
# distribution used during training.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

python -m src.brepnet.train --config-name train_diffusion_white \
  trainer.evaluate=true \
  trainer.resume_from_checkpoint=/mnt/d/data/new_cond_ckpt/0504_deepcad_flux_single_view_align_dino.ckpt \
  trainer.test_output_dir=/mnt/d/data/new_cond_results/0504_deepcad_flux_single_view_align_dino_cached_z \
  trainer.num_workers=32 \
  trainer.batch_size=64 \
  trainer.devices=8 \
  trainer.precision=bf16-mixed \
  dataset.name=Diffusion_dataset \
  dataset.latent_root=/mnt/d/data/ae_cache/1119_deepcad_aug1_11k \
  dataset.condition_root=/mnt/d/data/deepcad_v6_cond \
  dataset.train_dataset=src/brepnet/data/list/deduplicated_deepcad_training_7_30.txt \
  dataset.val_dataset=src/brepnet/data/list/deduplicated_deepcad_validation_7_30.txt \
  dataset.test_dataset=src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt \
  dataset.cached_condition=false \
  dataset.is_aug=0 \
  dataset.max_faces=30 \
  model.latent.use_cached_latents=true \
  model.denoiser.hidden_dim=768 \
  model.padding.max_faces=30 \
  model.autoencoder.checkpoint=/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt \
  model.autoencoder.in_channels=6 \
  model.noise.beta_schedule=squaredcos_cap_v2 \
  hydra.job.chdir=false
