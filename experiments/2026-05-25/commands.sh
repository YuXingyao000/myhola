#!/usr/bin/env bash
# 2026-05-25 所有命令 (直接复制粘贴)
cd /mnt/d/python

# ═══════════════════════════════════════════════════════════════
# 1. 数据生成
# ═══════════════════════════════════════════════════════════════

# Blender 渲染 (cube24, 8 GPU)
python -m src.brepnet.data.datagen.run_blender mode=cube24

# FLUX 生成 (动态 prompt)
python -m src.brepnet.data.datagen.run_flux mode=cube24

# 打包成训练用 npz
python -m src.brepnet.data.datagen.run_pack mode=cube24

# ═══════════════════════════════════════════════════════════════
# 2. 数据集质量评估 (SAM2 分割 FLUX 前景 → 与 OCC mask 比 IoU)
#    注意: 必须从非项目根目录运行, 否则 configs/ 会和 SAM2 的 Hydra 冲突
# ═══════════════════════════════════════════════════════════════

# IoU only (需要 GPU, SAM2)
cd /tmp && PYTHONPATH=/mnt/d/python python -m src.brepnet.eval.quality_metrics \
    --condition-root /mnt/d/data/deepcad_v6_cond \
    --model-list /mnt/d/python/src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt \
    --sam2-checkpoint /mnt/d/model/sam2.1_hiera_large.pt \
    --output /mnt/d/python/experiments/2026-05-25/quality_results/test_iou.json

# IoU + DINO + CLIP (全指标)
cd /tmp && PYTHONPATH=/mnt/d/python python -m src.brepnet.eval.quality_metrics \
    --condition-root /mnt/d/data/deepcad_v6_cond \
    --model-list /mnt/d/python/src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt \
    --sam2-checkpoint /mnt/d/model/sam2.1_hiera_large.pt \
    --compute-dino --compute-clip \
    --output /mnt/d/python/experiments/2026-05-25/quality_results/test_full.json

# ═══════════════════════════════════════════════════════════════
# 3. Diffusion 训练
# ═══════════════════════════════════════════════════════════════

# Topo bias + 纯白模 (已跑过, 此处用修正后的代码重跑)
python -m src.brepnet.train --config-name train_diffusion_topo_bias \
    trainer.devices=8 trainer.batch_size=64 trainer.max_steps=100000 \
    trainer.output_dir=/mnt/d/data/diffusion_topo_experiments \
    trainer.exp_name=20260525_topo_bias_white_corrected \
    trainer.wandb.enabled=true trainer.wandb.project=hola-brep \
    model.autoencoder.checkpoint=/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt \
    model.latent.use_cached_latents=true model.denoiser.hidden_dim=768 \
    model.padding.max_faces=30 model.autoencoder.in_channels=6 \
    model.noise.beta_schedule=squaredcos_cap_v2 \
    dataset.data_root=/mnt/d/data/deepcad_v6 \
    dataset.latent_root=/mnt/d/data/ae_cache/1119_deepcad_aug1_11k \
    dataset.condition_root=/mnt/d/data/deepcad_v6_cond \
    dataset.train_dataset=src/brepnet/data/list/deduplicated_deepcad_training_7_30.txt \
    dataset.val_dataset=src/brepnet/data/list/deduplicated_deepcad_validation_7_30.txt \
    dataset.real_photo_ratio=0.0 dataset.load_topology=true \
    dataset.cached_condition=false dataset.is_aug=0 dataset.scale_factor=200 \
    hydra.job.chdir=false

# Topo bias + 真实照片 (ratio=1.0)
python -m src.brepnet.train --config-name train_diffusion_topo_bias_real \
    trainer.devices=8 trainer.batch_size=64 trainer.max_steps=100000 \
    trainer.output_dir=/mnt/d/data/diffusion_topo_experiments \
    trainer.exp_name=20260525_topo_bias_real \
    trainer.wandb.enabled=true trainer.wandb.project=hola-brep \
    model.autoencoder.checkpoint=/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt \
    model.latent.use_cached_latents=true model.denoiser.hidden_dim=768 \
    model.padding.max_faces=30 model.autoencoder.in_channels=6 \
    model.noise.beta_schedule=squaredcos_cap_v2 \
    dataset.data_root=/mnt/d/data/deepcad_v6 \
    dataset.latent_root=/mnt/d/data/ae_cache/1119_deepcad_aug1_11k \
    dataset.condition_root=/mnt/d/data/deepcad_v6_cond \
    dataset.train_dataset=src/brepnet/data/list/deduplicated_deepcad_training_7_30.txt \
    dataset.val_dataset=src/brepnet/data/list/deduplicated_deepcad_validation_7_30.txt \
    dataset.real_photo_ratio=1.0 dataset.load_topology=true \
    dataset.cached_condition=false dataset.is_aug=0 dataset.scale_factor=200 \
    hydra.job.chdir=false

# Baseline 无 topo bias + 真实照片 (对照组)
python -m src.brepnet.train --config-name train_diffusion_baseline_real \
    trainer.devices=8 trainer.batch_size=64 trainer.max_steps=100000 \
    trainer.output_dir=/mnt/d/data/diffusion_topo_experiments \
    trainer.exp_name=20260525_baseline_real \
    trainer.wandb.enabled=true trainer.wandb.project=hola-brep \
    model.autoencoder.checkpoint=/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt \
    model.latent.use_cached_latents=true model.denoiser.hidden_dim=768 \
    model.padding.max_faces=30 model.autoencoder.in_channels=6 \
    model.noise.beta_schedule=squaredcos_cap_v2 \
    dataset.data_root=/mnt/d/data/deepcad_v6 \
    dataset.latent_root=/mnt/d/data/ae_cache/1119_deepcad_aug1_11k \
    dataset.condition_root=/mnt/d/data/deepcad_v6_cond \
    dataset.train_dataset=src/brepnet/data/list/deduplicated_deepcad_training_7_30.txt \
    dataset.val_dataset=src/brepnet/data/list/deduplicated_deepcad_validation_7_30.txt \
    dataset.real_photo_ratio=1.0 dataset.load_topology=false \
    dataset.cached_condition=false dataset.is_aug=0 dataset.scale_factor=200 \
    hydra.job.chdir=false

# ═══════════════════════════════════════════════════════════════
# 4. Inference (加载 checkpoint, 生成 npz)
# ═══════════════════════════════════════════════════════════════

# 把 checkpoint 路径替换为实际训练出来的
python -m src.brepnet.train --config-name train_diffusion_topo_bias \
    trainer.evaluate=true trainer.devices=1 trainer.batch_size=64 \
    trainer.resume_from_checkpoint=/mnt/d/data/new_cond_ckpt/0524_oracle_topology_soft_bias.ckpt \
    trainer.test_output_dir=/mnt/d/data/new_cond_results/inference/topo_bias_white \
    model.autoencoder.checkpoint=/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt \
    model.latent.use_cached_latents=true model.denoiser.hidden_dim=768 \
    model.padding.max_faces=30 model.autoencoder.in_channels=6 \
    model.noise.beta_schedule=squaredcos_cap_v2 \
    dataset.data_root=/mnt/d/data/deepcad_v6 \
    dataset.latent_root=/mnt/d/data/ae_cache/1119_deepcad_aug1_11k \
    dataset.condition_root=/mnt/d/data/deepcad_v6_cond \
    dataset.test_dataset=src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt \
    dataset.real_photo_ratio=1.0 dataset.load_topology=true \
    dataset.cached_condition=false dataset.is_aug=0 dataset.scale_factor=1 \
    hydra.job.chdir=false
