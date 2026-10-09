cd /mnt/d/python

# Experiment 0: GT topology with a differentiable 1x-to-2x attention multiplier.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m src.brepnet.train \
  --config-name train_diffusion_oracle_topology_log_multiplier \
  trainer.devices=8 \
  trainer.batch_size=64 \
  trainer.num_workers=8 \
  trainer.max_steps=100000 \
  trainer.output_dir=experiments/2026-06-28 \
  trainer.exp_name=outputs_oracle_topology_log_multiplier_real \
  trainer.wandb.enabled=true \
  trainer.wandb.project=hola-brep \
  trainer.wandb.name=oracle_topology_log_multiplier_real_v7_cube24 \
  dataset.raw_data_root=/mnt/d/data/deepcad_v7 \
  dataset.cached_latent_root=/mnt/d/data/ae_cache/1119_deepcad_aug1_11k_24 \
  dataset.condition_data_root=/mnt/d/data/deepcad_v7_cond \
  dataset.train_dataset=src/brepnet/data/list/deduplicated_deepcad_training_7_30.txt \
  dataset.val_dataset=src/brepnet/data/list/deduplicated_deepcad_validation_7_30.txt \
  dataset.test_dataset=src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt \
  dataset.real_photo_ratio=1.0 \
  dataset.validation_real_photo_ratio=1.0 \
  dataset.is_aug=false \
  dataset.is_overfit=false \
  dataset.load_topology=true \
  dataset.epoch_scale_factor=200 \
  model.autoencoder.checkpoint=/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt \
  model.topology_bias.enabled=true \
  model.topology_bias.source=gt_adjacency \
  model.topology_bias.mode=soft_bias \
  model.topology_bias.parameterization=log_multiplier \
  model.topology_bias.max_multiplier=2.0 \
  hydra.job.chdir=false
