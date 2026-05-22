#!/usr/bin/env bash
# Diffusion training with GT topology attention bias.
# Adjacent faces get positive attention bias → structure-aware generation.
# Bias is timestep-dependent: strong at high noise, weak at low noise.

python -m src.brepnet.train \
    model=diffusion_cross_attn \
    condition=single_img \
    experiment=train_diffusion_topo_bias \
    trainer.gpus=8 \
    trainer.batch_size=64 \
    trainer.learning_rate=1e-4 \
    trainer.max_steps=100000 \
    trainer.check_val_every_n_epoch=1 \
    trainer.num_sanity_val_steps=0 \
    trainer.output_dir=/mnt/d/data/diffusion_topo_experiments \
    trainer.exp_name=20260522_diffusion_topo_bias_scale2 \
    trainer.wandb.enabled=true \
    trainer.wandb.project=hola-brep \
    trainer.wandb.name=20260522_diffusion_topo_bias_scale2 \
    model.autoencoder.weights=/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt \
    model.topo_scale=2.0 \
    model.stored_z=true \
    model.diffusion_latent=768 \
    model.num_max_faces=30 \
    model.autoencoder=AutoEncoder_1119_light \
    model.in_channels=6 \
    model.beta_schedule=squaredcos_cap_v2 \
    'model.condition=[single_img]' \
    dataset.name=Diffusion_dataset \
    dataset.data_root=/mnt/d/data/deepcad_v6 \
    dataset.face_z_dir=/mnt/d/data/ae_cache/1119_deepcad_aug1_11k \
    dataset.cond_root=/mnt/d/data/deepcad_v6_cond \
    dataset.train_dataset=src/brepnet/data/list/deduplicated_deepcad_training_7_30.txt \
    dataset.val_dataset=src/brepnet/data/list/deduplicated_deepcad_validation_7_30.txt \
    dataset.test_dataset=src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt \
    dataset.load_topology=true \
    dataset.cached_condition=true \
    dataset.is_aug=0 \
    dataset.num_max_faces=30 \
    dataset.scale_factor=200 \
    'dataset.condition=[single_img]' \
    hydra.job.chdir=false
