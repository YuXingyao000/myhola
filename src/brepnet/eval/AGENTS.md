# Repository Guidelines

## Project Structure & Module Organization
Core training and inference code lives one level up in `src/brepnet`: `train.py`, `train_diffusion.py`, and `inference.py` are the main entry points; `model.py`, `diffusion_model.py`, and `dataset.py` define architectures and datasets. Evaluation code in this directory covers validity, novelty, complexity, point-cloud metrics, and conditioning workflows (`eval_validity.py`, `eval_unique_novel.py`, `eval_condition.py`). Legacy baseline conversion utilities live in `eval/baseline_preprocess/`, and LFD tooling plus reference assets live in `eval/lfd/evaluation_scripts/`. Keep split lists in `data/list/`, helper scripts in `scripts/`, visual analysis in `viz/`, and lightweight regression checks in `test/`.

## Build, Test, and Development Commands
- `bash install.sh`: install the geometry and PyTorch stack used by training and evaluation.
- `python train.py trainer.exp_name=my_run dataset.dataset_name=AutoEncoder_dataset3 trainer.gpu=[0]`: start an autoencoder training run.
- `python train_diffusion.py dataset.dataset_name=Diffusion_dataset_mm trainer.gpu=[0,1]`: launch diffusion training with Hydra overrides.
- `python inference.py --ckpt path/to.ckpt --split validation`: export reconstructions for downstream evaluation.
- `python eval/eval_validity.py` or `bash eval/eval_condition.sh`: run evaluation scripts from the repo root so relative paths resolve consistently.
- `pytest test -k split` or `python test/test_unzip_dataset.py`: run focused regressions.

## Coding Style & Naming Conventions
Use Python with 4-space indentation and PEP 8 spacing. Prefer `snake_case` for functions, files, and local variables; existing examples include `sample_points_gt` and `check_data_deduplicate`. Add type hints on public helpers when practical, use `Path` instead of hard-coded absolute paths, and expose tunables through Hydra/YAML rather than inline constants.

## Testing Guidelines
Write focused `test_*.py` cases under `test/`. Use temporary `.npz` fixtures or mocked geometry inputs instead of real CAD assets, and seed random behavior explicitly. If a script depends on LFD binaries or large generated meshes, document the prerequisite and expected runtime near the test or in the PR.

## Commit & Pull Request Guidelines
Recent commits are short, imperative messages such as `nano fix`, `update_dino_feat`, and `fix repeat`. Keep commit subjects brief, present tense, and under about 60 characters. PRs should state the motivation, list the commands run, mention any config or dataset-list changes, and attach screenshots or metric summaries when evaluation behavior changes.

## Data & Configuration Tips
Do not commit STEP files, generated meshes, checkpoints, or WandB artifacts. Keep dataset splits in `data/list/`, preserve large evaluation outputs in ignored directories, and note any environment variables, proxies, or external binaries needed to reproduce results.
