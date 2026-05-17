#!/usr/bin/env bash
set -euo pipefail

# Evaluate the 0504 single-view model with the same cached-latent target
# distribution used during training.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

python -m src.brepnet.train_diffusion \
  trainer.evaluate=true \
  trainer.resume_from_checkpoint=/mnt/d/data/new_cond_ckpt/0504_deepcad_flux_single_view_align_dino.ckpt \
  trainer.test_output_dir=/mnt/d/data/new_cond_results/0504_deepcad_flux_single_view_align_dino_cached_z \
  trainer.num_worker=32 \
  trainer.batch_size=64 \
  trainer.gpu=8 \
  trainer.accelerator=bf16-mixed \
  dataset.name=Diffusion_dataset \
  dataset.face_z=/mnt/d/data/ae_cache/1119_deepcad_aug1_11k \
  dataset.cond_root=/mnt/d/data/deepcad_v6_cond \
  dataset.train_dataset=src/brepnet/data/list/deduplicated_deepcad_training_7_30.txt \
  dataset.val_dataset=src/brepnet/data/list/deduplicated_deepcad_validation_7_30.txt \
  dataset.test_dataset=src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt \
  dataset.cached_condition=false \
  dataset.is_aug=0 \
  dataset.num_max_faces=30 \
  dataset.condition=\[single_img\] \
  model.name=Diffusion_condition \
  model.stored_z=true \
  model.diffusion_latent=768 \
  model.num_max_faces=30 \
  model.autoencoder=AutoEncoder_1119_light \
  model.autoencoder_weights=/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt \
  model.in_channels=6 \
  model.beta_schedule=squaredcos_cap_v2 \
  model.condition=\[single_img\] \
  hydra.job.chdir=false
