# Repository Guidelines

## Project Structure & Module Organization
Code lives in `src/brepnet`: `train.py` and `train_diffusion.py` start Lightning jobs, while `inference.py` handles checkpoint export. Architectures are defined in `model.py` and `diffusion_model.py`; dataset classes such as `AutoEncoder_dataset3` and `Diffusion_dataset_mm` live in `dataset.py`. Supporting assets: preprocessing + split files in `data/`, evaluation + LFD tooling in `eval/`, sampling shells in `scripts/`, visualization aids in `viz/`, and regression utilities/tests in `test/`. Add new Hydra defaults below `configs/brepnet` and keep CAD/mesh artifacts out of version control.

## Build, Test, and Development Commands
- `bash install.sh` installs PyTorch 2.4/CUDA 12.4, pythonocc, PyTorch3D, and other geometry dependencies; adapt the steps if apt is unavailable.
- `python train.py trainer.exp_name=my_run dataset.dataset_name=AutoEncoder_dataset3 trainer.gpu=[0]` trains the autoencoder defined in `model.py`.
- `python train_diffusion.py dataset.dataset_name=Diffusion_dataset_mm trainer.gpu=[0,1]` launches the diffusion pipeline using `configs/brepnet/train_diffusion.yaml`.
- `python inference.py --ckpt path/to.ckpt --split validation` materializes reconstructions consumed by `eval/`.
- `pytest test -k split` or targeted scripts such as `python test/test_unzip_dataset.py` run the lightweight regression checks.

## Coding Style & Naming Conventions
Follow PEP 8, 4-space indents, and snake_case identifiers (`prepare_condition`, `normalize_coord1112`). Favor type hints on public functions, move tensors explicitly between devices, and reuse helpers from `shared.common_utils`. Surface tunables through Hydra YAML and override them with dot notation (`trainer.max_epochs=200`). Avoid hard-coded paths; prefer `Path` objects and guard platform-specific logic with `if os.name == "nt"`.

## Testing Guidelines
Prefer focused PyTest tests named `test_*`. Replace heavyweight CAD inputs with temporary `.npz` fixtures or mocked Open3D meshes, and seed randomness through `seed_everything` or scoped `torch.Generator` instances. Record any evaluation procedure that depends on LFD assets in `eval/README.md`, and note runtime expectations for scripts that exceed a minute.

## Commit & Pull Request Guidelines
Commits in this repo are short, imperative statements (`nano fix`, `update_dino_feat`). Match that style: present tense, ≤60 characters, optional scope prefixes (`data:`). PRs should explain motivation, list the commands you ran (training, `pytest`), attach qualitative evidence (renderings or `viz/` dumps), and highlight config or dataset changes so reviewers can reproduce the run.

## Data & Configuration Tips
Do not commit STEP/CAD data, generated meshes, or WandB artifacts. Store split definitions in `data/list`, keep checkpoints/logs in ignored folders, and mention any proxy or auth env vars (see `train.py`) inside the PR. When adding a dataset or condition, supply the Hydra stub plus a short usage note so agents know how to activate it.
