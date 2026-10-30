# Claude Code Guidelines — BRepNet Source

The canonical instructions for this directory are `AGENTS.md`; read it before changing code. This file is a summary only. The repo-wide entry point is the root `CLAUDE.md`, and the research goals/collaboration rules are in the root `AGENTS.md`.

## Entry Points and Configuration

- Training: `python -m src.train --config-name <config>`. `UnifiedTrainer` dispatches VAE/diffusion by `model.stage`; datasets are looked up by `dataset.name`.
- Multi-GPU is controlled by `trainer.devices`; `batch_size` and `num_workers` are per GPU. Formal evaluation and NPZ export still require `trainer.devices=1` to avoid distributed-sampler duplication.
- Runs default to `checkpoints/<exp_name>/<UTC timestamp>/`. `trainer.run_dir` names a full run; `trainer.resume_from_checkpoint` resumes it. Weight-only init starts a new run and is not resume. Never overwrite an existing run without a resume marker.
- Model factory: `models/__init__.py:build_model`. VAE/shared operators: `models/vae.py`, `models/blocks.py`. Diffusion: `diffusion/`. Legacy `models/diffusion*.py` and `models/condition_encoders.py` are compatibility re-exports only.
- Behavior belongs in `configs/model/`, `dataset/`, `condition/`, `trainer/`, `datagen/`. Do not revive the removed `model.py`, `diffusion_model.py`, or `train_diffusion.py` entry points.

## Checkpoint Contracts

- Full current models use strict loading. `strict=False` is not a migration strategy and must not hide old/new key mismatches.
- `load_diffusion_state_dict` may omit only the separately loaded VAE and strictly checks every other key; a diffusion checkpoint never overrides the separate AE.
- Resume skips component initialization. Init-from-checkpoint paths are explicit, per-variant, and do not restore optimizer state.
- Changing VAE or normalization requires stating whether old latent caches, AE checkpoints, and diffusion checkpoints remain compatible.

## Diffusion Data Flow

1. `dataset.py:Diffusion_dataset` reads the cache and pads; `diffusion/latents.py:LatentProvider` takes mean or samples from std.
2. `[B, F, 32]` is noised (default DDPM 1000 steps, epsilon prediction, L2 loss); losses/sampling live in `losses.py` / `sampling.py`.
3. `conditioning.py` routes single-image by default. `multi_img` and `text` are explicitly rejected; a YAML option does not mean the path is verified.
4. `image_encoder.py` uses frozen DINOv2 ViT-L (257 tokens for 224 input) with a trainable projection.
5. `denoiser.py` projects `[B,30,32]` to 768, adds the timestep embedding, runs 24 complete TransformerEncoderLayers, and after each one applies an independent image cross-attention from `fusion.py`.
6. Sampling filters/deduplicates face latents, then decodes with the frozen VAE. Diffusion loss does not directly optimize OCC solid validity or reprojection.

Current configs: `configs/train_diffusion_layerwise_svr.yaml` (SVR-only, photo ratio 0, reads only `imgs.npz["svr_imgs"]`) and `configs/train_diffusion_layerwise_real.yaml` (generated-photo condition). `condition_fuser` configs are explicitly rejected; old diffusion checkpoints cannot resume under this architecture.

## Face Padding and Topology

- Random padding duplicates and permutes real faces into 30 slots; `RandomFacePadder.postprocess()` deduplicates by latent distance, so face count is indirect.
- Any face-level label or adjacency must use the same padding indices. Oracle adjacency is reordered in the dataset with that mapping.
- The learned topology predictor has no explicit slot correspondence; do not treat GT face order or GT face count as deployable inference inputs. `source: gt_adjacency` is oracle-only.
- Cache statistics are mean/std, not log-variance.

## Known Config Drift

| Area | Check before use |
|---|---|
| `configs/train_diffusion_real.yaml` | Inherits `is_overfit=true` and `validation_real_photo_ratio=0.0`; `scale_factor` is not read by the diffusion loader |
| `configs/train_diffusion_baseline_real.yaml` | Fixes overfit/validation but has historical machine paths and 8-GPU settings |
| `AutoEncoder_dataset3` | Accepts canonical and legacy field names; conflicting definitions error |
| cached condition | The image branch reads `imgs`; a cached flag does not make cached features a substitute |
| `baseline` filenames | Do not imply the old architecture; reproduce old models from the baseline branch/tag |

## Verification

Run only tests relevant to the change under `tests/` (it is not a dependency-free unified suite). Check imports, config consumption, real small-batch forward/backward, finite gradients, permutation consistency, and train/inference interfaces where applicable. Research validity is decided by fixed-protocol downstream results, not smoke tests. Document-only changes do not require training or model downloads.
