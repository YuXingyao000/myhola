# HoLa-BRep Conditional Diffusion Experiment Record

This document records the main commands, controlled experiments, and conclusions from the recent conditional diffusion debugging work.

## Context

Initial observation:

- The conditional diffusion model can generate plausible CAD-like BRep outputs.
- Failure cases often contain faces that should not exist.
- Some generated faces are locally reasonable but misplaced in the assembled model.
- Chamfer Distance, F-score, and Valid Rate do not improve reliably under the existing evaluation in `src/brepnet/eval/eval_condition.py`.

Working interpretation:

- The image encoder is not the only bottleneck.
- The current image feature and HoLa-BRep holistic latent space have a gap.
- The existing CLIP-style alignment loss helped Valid Rate substantially in earlier experiments, suggesting global image-CAD latent alignment is useful.
- However, global alignment is not enough to control face-slot existence, face placement, or topology assembly.

Important baseline clarification:

- `0504` is the true DINO baseline.
- `0502` / `0502_depth` naming was misleading in earlier scripts and corresponds to a Depth Anything V2 encoder experiment.

## Baseline Testing Command Template

Original 100-model testing command pattern:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m src.brepnet.train_diffusion \
    trainer.evaluate=true \
    trainer.resume_from_checkpoint=/mnt/d/data/new_cond_ckpt/0502_deepcad_flux_single_view_align_depth.ckpt \
    trainer.test_output_dir=/mnt/d/data/new_cond_results/0502_depth \
    trainer.num_worker=32 \
    trainer.batch_size=64 \
    trainer.gpu=8 \
    trainer.accelerator=bf16-mixed \
    dataset.name=Diffusion_dataset \
    dataset.face_z=/mnt/d/data/ae_cache/1119_deepcad_aug1_11k \
    dataset.cond_root=/mnt/d/data/deepcad_v6_cond \
    dataset.test_dataset=src/brepnet/data/list/deduplicated_deepcad_testing_7_30_sample_100.txt \
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
```

Harness scripts used for controlled diagnostics:

```bash
export GT_ROOT=/mnt/d/data/deepcad_v6
export TEST_LIST=src/brepnet/data/list/deduplicated_deepcad_testing_7_30_sample_100.txt
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

FORCE=1 src/brepnet/experiments/condition_debug/run_0502_depth_00_baseline.sh
FORCE=1 src/brepnet/experiments/condition_debug/run_0502_depth_02_condition_sensitivity.sh
FORCE=1 src/brepnet/experiments/condition_debug/run_0502_depth_03_denoise_probe.sh
src/brepnet/experiments/condition_debug/run_0502_depth_05_face_count.sh
```

## 0504 DINO Baseline

Checkpoint:

```text
/mnt/d/data/new_cond_ckpt/0504_deepcad_flux_single_view_align_dino.ckpt
```

Result root:

```text
/mnt/d/data/new_cond_results/exp_plan_0504_dino_baseline
```

E0 summary:

| Metric | Value |
|---|---:|
| Valid Rate | 0.60 |
| Success | 60 / 100 |
| Face CD | 0.215834 |
| Edge CD | 0.331102 |
| Vertex CD | 0.743193 |
| Face F-score | 0.214979 |
| FE F-score | 0.533406 |
| EV F-score | 0.537639 |
| Abs Face Count Error | 1.575758 |

E2 condition sensitivity:

| Mode | Valid | Face CD | FE | EV | Abs Face Err |
|---|---:|---:|---:|---:|---:|
| normal | 0.59 | 0.2151 | 0.5307 | 0.5364 | 1.5253 |
| shuffle | 0.54 | 0.3031 | 0.3641 | 0.3706 | 5.2600 |
| zero | 0.69 | 0.2380 | 0.4645 | 0.4258 | 5.3300 |

Interpretation:

- Condition is used. Shuffle and zero conditions degrade geometry/topology metrics.
- However, condition use does not guarantee valid BRep assembly.

E5 face-count grouping:

| Direction | Count | Valid | FE | EV |
|---|---:|---:|---:|---:|
| exact | 49 | 0.8980 | 0.6472 | 0.6472 |
| under | 29 | 0.3793 | 0.4282 | 0.4521 |
| over | 21 | 0.2381 | 0.4131 | 0.4002 |

## 0510 Image Face/Edge Count Auxiliary

Checkpoint:

```text
/mnt/d/data/new_cond_ckpt/0510_deepcad_flux_single_view_align_dino.ckpt
```

Result:

| Metric | 0510 |
|---|---:|
| Valid Rate | 0.43 |
| FE | 0.5010 |
| EV | 0.5005 |
| Face F-score | 0.2076 |
| Abs Face Count Error | 1.6869 |

Valid transitions versus 0504:

```text
valid -> valid: 37
invalid -> valid: 6
valid -> invalid: 23
both invalid: 34
```

Conclusion:

- Global image-level face/edge count auxiliary was harmful.
- It does not solve face-slot survival or topology assembly.

## 0511 Negative Condition Margin

Training command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m src.brepnet.train_diffusion \
    trainer.check_val_every_n_epoch=1 \
    trainer.num_worker=32 \
    trainer.batch_size=64 \
    trainer.accelerator=bf16-mixed \
    dataset.name=Diffusion_dataset \
    trainer.resume_from_checkpoint=/mnt/d/data/new_cond_ckpt/0502_deepcad_flux_single_view_align_depth.ckpt \
    dataset.face_z=/mnt/d/data/ae_cache/1119_deepcad_aug1_11k \
    dataset.cond_root=/mnt/d/data/deepcad_v6_cond \
    dataset.train_dataset=src/brepnet/data/list/deduplicated_deepcad_training_7_30.txt \
    dataset.val_dataset=src/brepnet/data/list/deduplicated_deepcad_validation_7_30.txt \
    dataset.test_dataset=src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt \
    dataset.cached_condition=false \
    dataset.is_aug=0 \
    dataset.num_max_faces=30 \
    dataset.scale_factor=200 \
    dataset.condition=\[single_img\] \
    trainer.gpu=8 \
    trainer.wandb=true \
    trainer.exp_name=0511_deepcad_flux_single_view_align_dino_neg_cond \
    model.name=Diffusion_condition \
    model.diffusion_latent=768 \
    model.num_max_faces=30 \
    model.autoencoder=AutoEncoder_1119_light \
    model.autoencoder_weights=/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt \
    model.in_channels=6 \
    model.beta_schedule=squaredcos_cap_v2 \
    model.condition=\[single_img\] \
    model.lambda_cond_margin=0.1 \
    model.cond_margin=0.01 \
    model.cond_margin_negative=shuffle \
    hydra.job.chdir=false
```

Checkpoint:

```text
/mnt/d/data/new_cond_ckpt/0511_deepcad_dino_cond_margin_s0p1_m0p01_shuffle.ckpt
```

E0 summary:

| Metric | 0504 | 0511 |
|---|---:|---:|
| Valid Rate | 0.60 | 0.61 |
| Face CD | 0.215834 | 0.215836 |
| Edge CD | 0.331102 | 0.329809 |
| Vertex CD | 0.743193 | 0.742550 |
| Face F-score | 0.214979 | 0.213269 |
| FE | 0.533406 | 0.532767 |
| EV | 0.537639 | 0.551410 |
| Abs Face Count Error | 1.575758 | 1.383838 |

Valid transitions versus 0504:

```text
0504 valid -> 0511 valid: 51
0504 invalid -> 0511 valid: 10
0504 valid -> 0511 invalid: 9
both invalid: 30
```

Conclusion:

- Slight net Valid Rate gain, but not robust.
- Diffusion loss increased relative to 0504 and validation overfitting appeared earlier.
- The margin branch did not clearly improve condition discrimination in E3.

## 0513 Timestep-Gated Condition Margin

Code commit:

```text
7bc89c1 Add timestep-gated condition margin experiment
```

Checkpoint:

```text
/mnt/d/data/new_cond_ckpt/0513_deepcad_dino_cond_margin_s0p1_m0p01_shuffle_t300.ckpt
```

Key config:

```text
model.lambda_cond_margin=0.1
model.cond_margin=0.01
model.cond_margin_negative=shuffle
model.cond_margin_t_min=300
model.cond_margin_t_max=999
```

Evaluation command:

```bash
cd /mnt/d/python
conda activate torch

export GT_ROOT=/mnt/d/data/deepcad_v6
export CKPT=/mnt/d/data/new_cond_ckpt/0513_deepcad_dino_cond_margin_s0p1_m0p01_shuffle_t300.ckpt
export ROOT=/mnt/d/data/new_cond_results/exp_plan_0513_dino_cond_margin_s0p1_m0p01_shuffle_t300
export TEST_LIST=src/brepnet/data/list/deduplicated_deepcad_testing_7_30_sample_100.txt
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

FORCE=1 src/brepnet/experiments/condition_debug/run_0502_depth_00_baseline.sh
FORCE=1 src/brepnet/experiments/condition_debug/run_0502_depth_02_condition_sensitivity.sh
FORCE=1 src/brepnet/experiments/condition_debug/run_0502_depth_03_denoise_probe.sh
src/brepnet/experiments/condition_debug/run_0502_depth_05_face_count.sh
```

E0 summary:

| Metric | 0504 | 0511 | 0513 |
|---|---:|---:|---:|
| Valid Rate | 0.60 | 0.61 | 0.56 |
| Face CD | 0.215834 | 0.215836 | 0.209635 |
| Edge CD | 0.331102 | 0.329809 | 0.323618 |
| Vertex CD | 0.743193 | 0.742550 | 0.737812 |
| Face F-score | 0.214979 | 0.213269 | 0.218506 |
| FE | 0.533406 | 0.532767 | 0.529123 |
| EV | 0.537639 | 0.551410 | 0.539719 |
| Abs Face Count Error | 1.575758 | 1.383838 | 1.373737 |

E2 condition sensitivity for 0513:

| Mode | Valid | Face CD | FE | EV | Abs Face Err |
|---|---:|---:|---:|---:|---:|
| normal | 0.57 | 0.2099 | 0.5289 | 0.5399 | 1.3535 |
| shuffle | 0.56 | 0.3002 | 0.3676 | 0.3656 | 5.1000 |
| zero | 0.30 | 0.2364 | 0.4068 | 0.3870 | 4.8400 |

E5 face-count grouping for 0513:

| Direction | Count | Valid | FE | EV |
|---|---:|---:|---:|---:|
| exact | 54 | 0.8704 | 0.6474 | 0.6661 |
| under | 26 | 0.3077 | 0.4306 | 0.4490 |
| over | 19 | 0.0526 | 0.3278 | 0.3047 |

Conclusion:

- Timestep gating improved Chamfer and face-count error.
- It reduced Valid Rate and did not improve FE/EV.
- It suggests condition margin can make latent geometry closer, but does not solve BRep legality or topology assembly.

## Current Addition Tag Experiment

Hypothesis:

```text
Some extra faces may come from random padding / duplicated latent slots not being filtered at inference.
```

Baseline behavior:

- Training pads each CAD to `num_max_faces=30` by repeating real face latents.
- Inference generates 30 face slots.
- With `pad_method=random`, duplicate slots are filtered only by a latent-distance threshold.
- If duplicate slots do not collapse tightly enough, they become extra generated faces.

Current code change:

- `dataset.addition_tag=true` appends a keep/drop tag to each stored latent slot.
- Mandatory original slots get `+1`.
- Repeated padding slots get `-1`.
- `diffusion_model.get_z()` preserves the tag under `stored_z=true`.
- Inference filters slots by `tag > 0` before the existing deduplication step.

Sanity check:

```text
00007186 face_features shape = (30, 65)
tag: -1 count = 20
tag: +1 count = 10
```

Training command:

```bash
cd /mnt/d/python
conda activate torch

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m src.brepnet.train_diffusion \
    trainer.check_val_every_n_epoch=1 \
    trainer.num_worker=32 \
    trainer.batch_size=64 \
    trainer.accelerator=bf16-mixed \
    trainer.gpu=8 \
    trainer.wandb=true \
    trainer.exp_name=0514_deepcad_dino_addition_tag \
    trainer.resume_from_checkpoint=none \
    dataset.name=Diffusion_dataset \
    dataset.face_z=/mnt/d/data/ae_cache/1119_deepcad_aug1_11k \
    dataset.cond_root=/mnt/d/data/deepcad_v6_cond \
    dataset.train_dataset=src/brepnet/data/list/deduplicated_deepcad_training_7_30.txt \
    dataset.val_dataset=src/brepnet/data/list/deduplicated_deepcad_validation_7_30.txt \
    dataset.test_dataset=src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt \
    dataset.cached_condition=false \
    dataset.is_aug=0 \
    dataset.num_max_faces=30 \
    dataset.scale_factor=200 \
    dataset.condition=\[single_img\] \
    dataset.addition_tag=true \
    model.name=Diffusion_condition \
    model.stored_z=true \
    model.diffusion_latent=768 \
    model.num_max_faces=30 \
    model.autoencoder=AutoEncoder_1119_light \
    model.autoencoder_weights=/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt \
    model.in_channels=6 \
    model.beta_schedule=squaredcos_cap_v2 \
    model.condition=\[single_img\] \
    model.addition_tag=true \
    model.lambda_cond_margin=0.0 \
    hydra.job.chdir=false
```

Expected success signal:

- Valid Rate improves.
- Over-count cases decrease.
- Abs Face Count Error decreases.
- FE/EV improves or at least does not degrade.

Failure signal:

- Face count gets closer but Valid Rate does not improve.
- Tag filtering becomes too aggressive and under-count cases increase.
- FE/EV degrades, suggesting the issue is topology/assembly rather than duplicate slot survival.

## Current Direction

The three failed auxiliary directions suggest that global image-level constraints are too weak:

- global face/edge count auxiliary
- negative-condition margin
- timestep-gated negative-condition margin

The next experiment focuses on slot-level survival:

```text
global HoLa alignment helps enter the CAD latent manifold;
slot-level keep/drop modeling may be needed to remove extra faces.
```

Direct VAE modification is intentionally deferred. The current hypothesis can still be tested at the diffusion/dataset interface without changing HoLa-BRep's VAE.
