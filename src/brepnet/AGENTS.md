# BRepNet Source Guidelines

## Current Architecture

`src/brepnet` is the package for training, datasets, models, data tools, evaluation, and post-processing. The current training path is `python -m src.brepnet.train`; avoid reviving old entry points such as `train_diffusion.py`, `model.py`, or `diffusion_model.py` unless the files actually exist and are intentionally reintroduced.

The diffusion stack is modular and assembled from Hydra configs:

- `train.py`: unified Lightning entry point for VAE and diffusion stages.
- `dataset.py`: `AutoEncoder_dataset3`, `Diffusion_dataset`, and legacy `Diffusion_dataset_mm`.
- `models/__init__.py`: model factory.
- `models/diffusion.py`: top-level diffusion assembly, loss, and inference loop.
- `models/diffusion_latents.py`: cached latent stats or VAE encoding path.
- `models/diffusion_padding.py`: fixed-size face-token padding and validity behavior.
- `models/diffusion_condition.py`: condition encoder factory.
- `models/condition_encoders.py`: DINOv2 / Depth Anything / point-cloud condition encoders.
- `models/diffusion_denoiser.py`: cross-attention condition fuser, alignment loss, denoiser backbone, oracle/learned topology self-attention bias.
- `models/vae.py`: VAE and autoencoder variants.

## Config-Driven Development

Most behavior should be changed through `configs/`, not by adding new shell wrappers. Current config groups are:

- `configs/model/*.yaml`
- `configs/dataset/*.yaml`
- `configs/condition/*.yaml`
- `configs/trainer/*.yaml`
- top-level configs such as `train_diffusion_real.yaml`, `train_diffusion_oracle_topology.yaml`, `train_diffusion_learned_topology.yaml`, and `train_vae.yaml`

When adding a model option, add the field to the relevant config and read it explicitly from the owning module. Keep config names lowercase and task-oriented. Do not add hidden defaults in code if the experiment depends on the value.

For diffusion data, `Diffusion_dataset` currently expects:

```text
dataset.raw_data_root
dataset.cached_latent_root
dataset.condition_data_root
dataset.train_dataset
dataset.val_dataset
dataset.test_dataset
```

For current v7 data, condition files are:

```text
deepcad_v7_cond/{model_id}/imgs.npz         # keys: svr_imgs, sketch_imgs
deepcad_v7_cond/{model_id}/real_photo.npz   # key: flux
```

Cached latent features are stored under:

```text
ae_cache/1119_deepcad_aug1_11k_24/{model_id}_{rotation_id}/features.npy
```

`features.npy` is latent stats, typically `[num_faces, 64]`; `Diffusion.latent.dim=32` consumes the mean slice.

## Diffusion Maintenance Rules

Preserve the existing module boundaries:

- Put latent cache or VAE sampling changes in `diffusion_latents.py`.
- Put padding or face-mask behavior in `diffusion_padding.py`.
- Put condition encoder selection in `diffusion_condition.py` and encoder implementation in `condition_encoders.py`.
- Put denoiser, condition fusion, alignment, and topology self-attention changes in `diffusion_denoiser.py`.
- Keep top-level orchestration, training loss, and inference loop in `diffusion.py`.

Topology bias only affects denoiser self-attention. `source: gt_adjacency` reads `batch["face_adj"]`; `source: learned` reads raw condition images from `batch["conditions"]["imgs"]`. If `dataset.load_topology=true`, ensure `dataset.raw_data_root` is set and face adjacency aligns with padded face order.

The alignment branch should use clean latent features, not noisy latent features. Do not mix AR/topology predictor losses into diffusion training unless the experiment explicitly calls for a combined objective and the config documents the weighting.

## Commands

Prefer direct Hydra commands:

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

Before using VAE commands, verify `AutoEncoder_dataset3` and `configs/dataset/vae.yaml` field names. This area has legacy naming drift (`data_root` / `condition_root` vs `raw_data_root` / `condition_data_root`).

## Testing And Validation

Use small, focused checks before long jobs:

```bash
python -c "from src.brepnet.models import build_model"
python -m pytest src/brepnet/test
python tools/filter_deepcad_v7_lists.py --dry-run
```

For dataset or condition changes, run a small `Diffusion_dataset` smoke test or the audit/filter tools before training. For topology or generation changes, run a reduced-batch validation or inference pass and record the result under `experiments/YYYY-MM-DD/NOTE.md`.

## Style

Use Python 3.10+, type hints for new public helpers, `Path` for filesystem paths, and clear snake_case names. Keep generated CAD, meshes, checkpoints, logs, W&B artifacts, and samples outside source directories. Prefer narrow scripts in `tools/` for reusable data audits and one-off scripts in `experiments/YYYY-MM-DD/` for experiment-specific code.
