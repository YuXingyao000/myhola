# Repository Guidelines

## Project Structure & Module Organization

This repository contains the HoLa-BRep research codebase for B-Rep VAE and conditional diffusion experiments, now organized around a config-driven training stack. Core source lives under `src/brepnet/`. Current diffusion model code is modularized in `src/brepnet/models/`: `diffusion.py` assembles the model, `diffusion_latents.py` handles cached latent stats, `diffusion_padding.py` handles face padding, `diffusion_condition.py` builds condition encoders, `diffusion_denoiser.py` contains the denoiser, condition fuser, and topology self-attention bias, and `condition_encoders.py` contains image/point encoders. VAE code remains in `vae.py`.

Hydra configs live under `configs/`, grouped by `model`, `dataset`, `condition`, `trainer`, `datagen`, and top-level experiment configs such as `train_diffusion_real.yaml` and `train_diffusion_learned_topology.yaml`. Data preparation scripts are in `src/brepnet/data/` and `tools/`, evaluation code is in `src/brepnet/eval/`, and PythonOCC post-processing is in `src/brepnet/post/`.

When touching `experiments/`, read `experiments/SKILL.md` first. Experiment folders are note-first: one `experiments/YYYY-MM-DD/` folder per day, with a mandatory `NOTE.md` containing per-experiment records for why, implementation, and result.

## Build, Test, and Development Commands

Create the environment and install the package:

```bash
conda env create -f environment.yml
conda activate hola-brep
pip install -e .
```

Prefer direct Hydra entry points for training because the current stack is config-driven:

```bash
python -m src.brepnet.train --config-name train_diffusion_real \
    dataset.raw_data_root=/mnt/d/data/deepcad_v7 \
    dataset.cached_latent_root=/mnt/d/data/ae_cache/1119_deepcad_aug1_11k_24 \
    dataset.condition_data_root=/mnt/d/data/deepcad_v7_cond \
    dataset.train_dataset=src/brepnet/data/list/deduplicated_deepcad_training_7_30.txt \
    dataset.val_dataset=src/brepnet/data/list/deduplicated_deepcad_validation_7_30.txt \
    dataset.test_dataset=src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt \
    model.autoencoder.checkpoint=/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt \
    hydra.job.chdir=false
```

Use config names such as `train_diffusion_real`, `train_diffusion_oracle_topology`, `train_diffusion_learned_topology`, `train_diffusion_topo_bias`, and `train_vae`. Before documenting or relying on a command, verify the relevant dataset class fields in `src/brepnet/dataset.py`; some VAE paths still have legacy field names.

## Coding Style & Naming Conventions

Use Python 3.10+, four-space indentation, and descriptive snake_case names for functions, variables, config keys, and script files. Classes use PascalCase. Prefer small, explicit modules over adding to legacy monoliths. Surface tunables through Hydra YAML and dot-list overrides; avoid hard-coded machine paths in source code.

For diffusion changes, preserve the module boundaries in `src/brepnet/models/`. Add config fields in the relevant config group and read them directly from the module that owns the behavior. Avoid implicit runtime guessing when a config contract can be explicit.

## Data Contracts

Current DeepCAD v7 diffusion experiments generally use:

| What | Path |
|------|------|
| Raw B-Rep data | `/mnt/d/data/deepcad_v7` |
| Condition data | `/mnt/d/data/deepcad_v7_cond` |
| 24-view latent cache | `/mnt/d/data/ae_cache/1119_deepcad_aug1_11k_24` |
| AE checkpoint | `/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt` |

Condition data uses `deepcad_v7_cond/{model_id}/imgs.npz` with keys `svr_imgs` and `sketch_imgs`, plus `real_photo.npz` with key `flux`. Do not call this `svr.npz` unless the repository data contract changes.

## Testing Guidelines

There is no strict coverage gate yet. Use lightweight import and smoke tests before long runs:

```bash
python -c "from src.brepnet.models import build_model"
python -m pytest src/brepnet/test
python tools/filter_deepcad_v7_lists.py --dry-run
```

For model changes, run a short training or validation pass with reduced batch size and a small list. For data changes, run the relevant audit/filter tool and save the report under `experiments/YYYY-MM-DD/`.

## Commit & Pull Request Guidelines

The current history uses short, informal commits. Prefer concise imperative messages such as `Add topology bias config` or `Fix real photo list`. PRs should include the goal, affected configs/scripts, dataset or checkpoint assumptions, and before/after metrics when experiments are involved. Link related issues or experiment notes and include visual outputs for rendering or reconstruction changes.

## Security & Configuration Tips

Do not commit private datasets, checkpoints, W&B tokens, or absolute machine-specific paths in source code. Pass paths via Hydra overrides or environment variables. Be careful in `src/brepnet/post/` and `src/brepnet/eval/`: PythonOCC 7.8.1 APIs are version-sensitive, so preserve function signatures unless the full post-process pipeline is retested.
