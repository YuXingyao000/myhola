#!/usr/bin/env bash
set -euo pipefail

# Evaluate the 16 real-photo conditions against the cached latent of the
# DeepCAD model they all depict: 00104204.  The latent_root directory contains
# symlinks real_photo_XX_4 -> 00104204_4, so Diffusion_dataset can keep using
# real_photo_XX as the condition prefix while reading the shared GT z.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python -m src.brepnet.train_diffusion \
  trainer.gpu=1 \
  trainer.evaluate=true \
  trainer.test_output_dir=/mnt/d/python/src/brepnet/results/flux_real_photo_00104204_cached_z \
  trainer.resume_from_checkpoint=/mnt/d/data/new_cond_ckpt/0504_deepcad_flux_single_view_align_dino.ckpt \
  trainer.num_worker=2 \
  trainer.batch_size=6 \
  trainer.accelerator=bf16-mixed \
  dataset.name=Diffusion_dataset \
  dataset.face_z=/mnt/d/python/src/brepnet/data/DataGenerationRefactored/real_photo/prepared/latent_root \
  dataset.cond_root=/mnt/d/python/src/brepnet/data/DataGenerationRefactored/real_photo/prepared/cond_root \
  dataset.train_dataset=/mnt/d/python/src/brepnet/data/DataGenerationRefactored/real_photo/prepared/test_list.txt \
  dataset.val_dataset=/mnt/d/python/src/brepnet/data/DataGenerationRefactored/real_photo/prepared/test_list.txt \
  dataset.test_dataset=/mnt/d/python/src/brepnet/data/DataGenerationRefactored/real_photo/prepared/test_list.txt \
  dataset.num_max_faces=30 \
  dataset.condition=\[single_img\] \
  dataset.is_aug=0 \
  dataset.cached_condition=false \
  model.name=Diffusion_condition \
  model.num_max_faces=30 \
  model.autoencoder=AutoEncoder_1119_light \
  model.diffusion_latent=768 \
  model.in_channels=6 \
  model.beta_schedule=squaredcos_cap_v2 \
  model.autoencoder_weights=/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt \
  model.condition=\[single_img\] \
  model.stored_z=true \
  model.is_aug=false \
  hydra.job.chdir=false
