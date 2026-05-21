# Repository Guidelines

## Project Structure & Module Organization

This repository contains the HoLa-BRep research codebase for B-Rep VAE and conditional diffusion experiments. Core source lives under `src/brepnet/`. Model code is organized in `src/brepnet/models/` (`vae.py`, `diffusion.py`, `condition_encoders.py`, `strategies.py`, shared `blocks.py`). Data preparation scripts are in `src/brepnet/data/`, evaluation code in `src/brepnet/eval/`, and PythonOCC post-processing in `src/brepnet/post/`. Hydra configs are under `configs/`, grouped by `model`, `condition`, `strategy`, `scheduler`, and `experiment`. Shell entry points live in `scripts/`. Keep generated checkpoints, logs, and samples out of source directories; use `outputs_brepnet_diffusion/` or an external output path.

## Build, Test, and Development Commands

Create the environment and install the package:

```bash
conda env create -f environment.yml
conda activate hola-brep
pip install -e .
```

Run common training modes through the wrapper:

```bash
bash scripts/train.sh vae dataset.data_root=/data/deepcad
bash scripts/train.sh diffusion_white trainer.gpus=8
bash scripts/train.sh feature_mapper trainer.batch_size=64
bash scripts/train.sh distill strategy.teacher.checkpoint=/ckpt/white.ckpt
```

For direct Hydra use, run `python -m src.brepnet.train experiment=train_vae ...`.

## Coding Style & Naming Conventions

Use Python 3.10+, four-space indentation, and descriptive snake_case names for functions, variables, config keys, and script files. Classes use PascalCase, especially model modules such as `DiffusionCrossAttn` and `FeatureDomainMapper`. Prefer small, explicit modules over adding to legacy monoliths. Keep Hydra config names lowercase and task-oriented, for example `train_feature_mapper.yaml`.

## Testing Guidelines

There is no strict coverage gate yet. Use lightweight import and smoke tests before long runs:

```bash
python -c "from src.brepnet.models import AutoEncoder_1119, DiffusionCrossAttn"
python -m pytest src/brepnet/test
```

For model changes, run a short training or validation pass with reduced batch size, then verify post-processing and metrics on a small test list.

## Commit & Pull Request Guidelines

The current history uses short, informal commits. Prefer concise imperative messages such as `Add topology bias config` or `Fix feature mapper training`. PRs should include the goal, affected configs/scripts, dataset or checkpoint assumptions, and before/after metrics when experiments are involved. Link related issues or experiment notes and include visual outputs for rendering or reconstruction changes.

## Security & Configuration Tips

Do not commit private datasets, checkpoints, W&B tokens, or absolute machine-specific paths. Pass paths via Hydra overrides or environment variables (`DATA_ROOT`, `FACE_Z_DIR`, `COND_ROOT`, `VAE_CKPT`). Be careful in `src/brepnet/post/` and `src/brepnet/eval/`: PythonOCC 7.8.1 APIs are version-sensitive, so preserve function signatures unless the full post-process pipeline is retested.
