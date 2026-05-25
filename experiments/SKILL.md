---
name: write-experiment-commands
description: Write runnable experiment commands into experiments/{date}/commands.sh following the project's simple copy-paste format.
---

# Write Experiment Commands

Use this skill when the user asks to add new training, inference, evaluation, or data generation commands.

## Format Rules

1. **One folder per date**: `experiments/YYYY-MM-DD/`
2. **每个文件夹必须有 `NOTE.md`**: 记录当天实验目的、哪些必做、哪些可选
3. **`commands.sh`**: 可直接复制粘贴的命令
4. **Flat structure**: no case/esac, no functions, no variables except paths at the top
5. **Section headers**: use `# ═══` dividers with short section titles
6. **Each command is self-contained**: can be copied as-is to terminal
7. **Comments above each command**: one line explaining what it does
8. **Working directory**: always `cd /mnt/d/python` at the top

## NOTE.md Template

每个 `experiments/YYYY-MM-DD/` 下必须有 `NOTE.md`：

```markdown
# YYYY-MM-DD 实验笔记

## 今日目标
一句话说清楚今天要验证什么。

## 必做
- [ ] 实验名 — 为什么做 (一句话)
- [ ] ...

## 可选
- [ ] 实验名 — 如果必做的结果好再做
- [ ] ...

## 依赖 / 前置条件
- 哪些数据必须已经生成
- 哪些 checkpoint 必须存在

## 预期结果
- 如果假设成立，应该看到什么
- 如果假设不成立，说明什么
```

## Template

```bash
#!/usr/bin/env bash
# YYYY-MM-DD 实验命令
cd /mnt/d/python

# ═══════════════════════════════════════════════════════════════
# 1. Section Name
# ═══════════════════════════════════════════════════════════════

# 简短说明
python -m src.brepnet.xxx \
    --arg1 value1 \
    --arg2 value2
```

## Common Command Patterns

### Training (Diffusion)

```bash
python -m src.brepnet.train --config-name train_diffusion_xxx \
    trainer.devices=8 trainer.batch_size=64 trainer.max_steps=100000 \
    trainer.output_dir=/mnt/d/data/diffusion_topo_experiments \
    trainer.exp_name=YYYYMMDD_experiment_name \
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
```

### Inference (load checkpoint → generate npz)

```bash
python -m src.brepnet.train --config-name train_diffusion_xxx \
    trainer.evaluate=true trainer.devices=1 trainer.batch_size=16 \
    trainer.resume_from_checkpoint=/path/to/checkpoints/last.ckpt \
    trainer.test_output_dir=/path/to/inference_output \
    model.autoencoder.checkpoint=/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt \
    model.latent.use_cached_latents=true model.denoiser.hidden_dim=768 \
    model.padding.max_faces=30 model.autoencoder.in_channels=6 \
    model.noise.beta_schedule=squaredcos_cap_v2 \
    dataset.data_root=/mnt/d/data/deepcad_v6 \
    dataset.latent_root=/mnt/d/data/ae_cache/1119_deepcad_aug1_11k \
    dataset.condition_root=/mnt/d/data/deepcad_v6_cond \
    dataset.test_dataset=src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt \
    dataset.cached_condition=false dataset.is_aug=0 dataset.scale_factor=1 \
    hydra.job.chdir=false
```

### Data Generation (Hydra-based)

```bash
# Blender
python -m src.brepnet.data.datagen.run_blender mode=cube24

# FLUX
python -m src.brepnet.data.datagen.run_flux mode=cube24

# Pack
python -m src.brepnet.data.datagen.run_pack mode=cube24
```

### Quality Evaluation

```bash
python -m src.brepnet.eval.quality_metrics \
    --condition-root /mnt/d/data/deepcad_v6_cond \
    --model-list src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt \
    --compute-dino --compute-clip \
    --output experiments/YYYY-MM-DD/quality_results/xxx.json
```

## Key Paths (current setup)

| What | Path |
|------|------|
| AE checkpoint | `/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt` |
| Latent cache | `/mnt/d/data/ae_cache/1119_deepcad_aug1_11k` |
| CAD data | `/mnt/d/data/deepcad_v6` |
| Condition images | `/mnt/d/data/deepcad_v6_cond` |
| Training list | `src/brepnet/data/list/deduplicated_deepcad_training_7_30.txt` |
| Validation list | `src/brepnet/data/list/deduplicated_deepcad_validation_7_30.txt` |
| Test list | `src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt` |
| Experiment output | `/mnt/d/data/diffusion_topo_experiments/` |

## Key Config Names

| Config | Use |
|--------|-----|
| `train_diffusion_topo_bias` | Topo bias, ratio=0.0 (白模) |
| `train_diffusion_topo_bias_real` | Topo bias, ratio=1.0 (真实照片) |
| `train_diffusion_baseline_real` | 无 topo bias, ratio=1.0 (对照组) |
| `train_diffusion_white` | 原始 HoLa-BRep 白模训练 |
| `train_vae` | VAE 训练 |

## DO NOT

- Do not use bash variables, case statements, or functions
- Do not write wrapper scripts (run_xxx.sh calling run_yyy.sh)
- Do not add `--help` sections or usage text
- Do not create multiple variants of the same command with minor differences — write one canonical command with a comment noting what to change
