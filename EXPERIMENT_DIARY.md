# Experiment Diary

This file is the running log for HoLa-BRep thesis experiments. Add newest entries near the top. Each entry should record purpose, code/config changes, exact command or launcher, output path, checkpoint path, result, and follow-up.

## 2026-05-22 - VAE Latent Noise Sensitivity Retest

Purpose: test whether the VAE fine-tuned with intersection noise is more robust when GT cached face latents are perturbed by isotropic Gaussian noise.

Checkpoint:

```text
/mnt/d/data/new_cond_ckpt/20260521_intersection_noise_0p1_vae_1119_light.ckpt
```

Script:

```text
experiments/2026-05-22/noise_sensitivity_retest/latent_sensitivity.py
```

Default output:

```text
/mnt/d/data/latent_sensitivity/20260522_intersection_noise_0p1_vae_1119_light_test100
```

Noise levels: `0.0 0.05 0.10 0.15 0.20 0.30 0.50`.

Status: script updated for the refactored repo. It now loads `src.brepnet.models.vae`, uses `AutoEncoder_1119_light`, enables neural intersection, and defaults to the copied fine-tuned checkpoint.

Expected metric: `success_summary.csv` should be compared against the previous baseline where `sigma=0.10` had about `44/100` valid solid success.

## 2026-05-22 - Diffusion Topology-Aware Plan

Purpose: isolate whether topology-aware diffusion helps face-slot survival and edge assembly.

Baseline organization: start from a zero-pad baseline before adding topology bias. Zero padding keeps real faces in slots `0..N-1` and padding in `N..29`, which makes topology adjacency meaningful. Random padding is not a clean baseline for topology-aware attention because face slots are shuffled/duplicated.

Planned sequence:

```text
1. zero-pad diffusion baseline
2. zero-pad + GT topology bias
3. zero-pad + predicted topology bias
```

Status: planned, not run yet. Refactored diffusion still needs interface cleanup before long runs.

## 2026-05-22 - Smoke Data Pack

Purpose: provide a small local dataset for Claude Code smoke tests without touching the full dataset.

Location:

```text
smoke_data/deepcad5
```

Model IDs: `00572443`, `00261287`, `00416460`, `00684055`, `00475715`.

Contents: original data folders, cached condition folders, VAE cached latent features under `ae_cache/1119_deepcad_aug1_11k`, and train/val/test list files. Only `_0` VAE cache was copied, so use `dataset.is_aug=0`.

## 2026-05-21 - Intersection Noise Fine-Tuning

Purpose: make the VAE neural intersection module less brittle to latent perturbations while keeping the pretrained encoder/decoder fixed.

Code/config changes:

```text
src/brepnet/models/vae.py              adds training-time intersection_noise_std
src/brepnet/train.py                   supports init_from_checkpoint and trainable_scope
configs/model/vae_1119_light.yaml      exposes intersection_noise_std and trainable_scope
experiments/2026-05-21/intersection_noise_full_train/run.sh
```

Training setup:

```text
model=vae_1119_light
model.intersection_noise_std=0.1
model.trainable_scope=intersection
trainer.init_from_checkpoint=/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt
trainer.gpus=8
trainer.batch_size=32
trainer.max_steps=20000
dataset.is_overfit=false
trainer.wandb.enabled=true
```

Frozen weights: all parameters except `inter.*` and `classifier.*`.

Output root:

```text
/mnt/d/data/vae_intersection_experiments/20260521_intersection_noise_0p1_vae_1119_light/20260521_intersection_noise_0p1_vae_1119_light
```

Best validation checkpoint found from Lightning metadata:

```text
/mnt/d/data/vae_intersection_experiments/20260521_intersection_noise_0p1_vae_1119_light/20260521_intersection_noise_0p1_vae_1119_light/checkpoints/epoch=0101-val/loss=0.0374.ckpt
```

Copied checkpoint:

```text
/mnt/d/data/new_cond_ckpt/20260521_intersection_noise_0p1_vae_1119_light.ckpt
```

Notes: checkpoint filenames created nested `epoch=...-val/loss=...ckpt` folders because the monitored metric name contained `/`.

