#!/usr/bin/env bash
# Diffusion topology bias ablation: test different topo_scale values.
# Run these sequentially or in parallel on different GPUs.

# Scale = 1.0 (weak bias)
python -m src.brepnet.train \
    --config-name train_diffusion_topo_bias \
    trainer.devices=8 \
    trainer.batch_size=64 \
    trainer.learning_rate=1e-4 \
    trainer.max_steps=100000 \
    trainer.output_dir=/mnt/d/data/diffusion_topo_experiments \
    trainer.exp_name=20260522_topo_scale1 \
    trainer.wandb.enabled=true \
    trainer.wandb.project=hola-brep \
    trainer.wandb.name=20260522_topo_scale1 \
    model.autoencoder.checkpoint=/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt \
    model.topology_bias.scale=1.0 \
    model.latent.use_cached_latents=true \
    model.denoiser.hidden_dim=768 \
    model.padding.max_faces=30 \
    model.autoencoder.in_channels=6 \
    model.noise.beta_schedule=squaredcos_cap_v2 \
    dataset.name=Diffusion_dataset \
    dataset.data_root=/mnt/d/data/deepcad_v6 \
    dataset.latent_root=/mnt/d/data/ae_cache/1119_deepcad_aug1_11k \
    dataset.condition_root=/mnt/d/data/deepcad_v6_cond \
    dataset.train_dataset=src/brepnet/data/list/deduplicated_deepcad_training_7_30.txt \
    dataset.val_dataset=src/brepnet/data/list/deduplicated_deepcad_validation_7_30.txt \
    dataset.test_dataset=src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt \
    dataset.load_topology=true \
    dataset.cached_condition=false \
    dataset.is_aug=0 \
    dataset.max_faces=30 \
    dataset.scale_factor=200 \
    hydra.job.chdir=false

# Scale = 5.0 (strong bias)
python -m src.brepnet.train \
    --config-name train_diffusion_topo_bias \
    trainer.devices=8 \
    trainer.batch_size=64 \
    trainer.learning_rate=1e-4 \
    trainer.max_steps=100000 \
    trainer.output_dir=/mnt/d/data/diffusion_topo_experiments \
    trainer.exp_name=20260522_topo_scale5 \
    trainer.wandb.enabled=true \
    trainer.wandb.project=hola-brep \
    trainer.wandb.name=20260522_topo_scale5 \
    model.autoencoder.checkpoint=/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt \
    model.topology_bias.scale=5.0 \
    model.latent.use_cached_latents=true \
    model.denoiser.hidden_dim=768 \
    model.padding.max_faces=30 \
    model.autoencoder.in_channels=6 \
    model.noise.beta_schedule=squaredcos_cap_v2 \
    dataset.name=Diffusion_dataset \
    dataset.data_root=/mnt/d/data/deepcad_v6 \
    dataset.latent_root=/mnt/d/data/ae_cache/1119_deepcad_aug1_11k \
    dataset.condition_root=/mnt/d/data/deepcad_v6_cond \
    dataset.train_dataset=src/brepnet/data/list/deduplicated_deepcad_training_7_30.txt \
    dataset.val_dataset=src/brepnet/data/list/deduplicated_deepcad_validation_7_30.txt \
    dataset.test_dataset=src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt \
    dataset.load_topology=true \
    dataset.cached_condition=false \
    dataset.is_aug=0 \
    dataset.max_faces=30 \
    dataset.scale_factor=200 \
    hydra.job.chdir=false
