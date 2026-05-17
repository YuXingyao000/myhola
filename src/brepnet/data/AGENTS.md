# Repository Guidelines

## Project Structure & Module Organization
Core training and inference entrypoints live one level up from this folder: `train.py`, `train_diffusion.py`, and `inference.py`. Model definitions are in `model.py` and `diffusion_model.py`, and dataset loaders such as `AutoEncoder_dataset3` and `Diffusion_dataset_mm` are in `dataset.py`. This `data/` directory contains preprocessing scripts, dataset packing utilities, and split assets under `data/list/`. Evaluation tools live in `eval/`, helper shells in `scripts/`, visual checks in `viz/`, and regression tests in `test/`.

## Build, Test, and Development Commands
Use `bash install.sh` to install the geometry stack, including PyTorch, pythonocc, and PyTorch3D. Train the autoencoder with `python train.py trainer.exp_name=my_run dataset.dataset_name=AutoEncoder_dataset3 trainer.gpu=[0]`. Train diffusion with `python train_diffusion.py dataset.dataset_name=Diffusion_dataset_mm trainer.gpu=[0,1]`. Export reconstructions with `python inference.py --ckpt path/to.ckpt --split validation`. Run focused checks with `pytest test -k split` or direct scripts such as `python test/test_unzip_dataset.py`.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation and snake_case names such as `prepare_condition` or `normalize_coord1112`. Prefer type hints on public functions, explicit tensor device moves, and `Path` over hard-coded paths. Put tunable settings in Hydra YAML and override them with dot notation, for example `trainer.max_epochs=200`. Guard Windows-specific behavior with `if os.name == "nt"`.

## Testing Guidelines
Add focused PyTest cases named `test_*`. Replace heavy CAD inputs with temporary `.npz` fixtures or mocked Open3D meshes where possible. Seed randomness with `seed_everything` or a scoped `torch.Generator`. Document any evaluation flow that depends on LFD assets in `eval/README.md`, especially if runtime exceeds a minute.

## Commit & Pull Request Guidelines
Recent commits are short, imperative messages such as `nano fix`, `update_dino_feat`, and `fix repeat`. Keep commit titles present tense, under about 60 characters, with optional prefixes like `data:`. PRs should explain motivation, list the commands run, note config or dataset changes, and attach visual evidence such as renderings or `viz/` outputs when behavior changes are visible.

## Data & Configuration Tips
Do not commit STEP files, generated meshes, checkpoints, or WandB artifacts. Keep split definitions in `data/list/`, store large outputs in ignored directories, and note any proxy or auth environment variables needed by training scripts.
