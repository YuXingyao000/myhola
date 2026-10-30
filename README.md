<div align="center">

# HoLa-BRep: B-Rep Generation using a Holistic Latent Representation

### ACM Transactions on Graphics (Proceedings of SIGGRAPH 2025)

[Yilin Liu]()\*, [Duoteng Xu]()\*, Xingyao Yu, Xiang Xu, Daniel Cohen-Or, Hao Zhang, Hui Huang

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2504.14257-b31b1b)](https://arxiv.org/abs/2504.14257)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://vcc.tech/research/2025/HolaBrep)
[![License](https://img.shields.io/badge/License-MIT-green)]()

</div>

---

## Overview

This repository reconstructs CAD models in **Boundary Representation (B-Rep)** from a single image via a two-stage pipeline:

1. **Stage 1 — VAE**: Encodes B-Rep geometry (faces + edges) into a compact holistic latent space that captures topology and geometry together.
2. **Stage 2 — Image-conditioned diffusion**: A frozen DINOv2 encoder conditions a 24-layer denoiser (self-attention followed by image cross-attention in every layer) that generates face latent tokens; the frozen VAE decodes them and PythonOCC post-processing builds the STEP solid.

It also contains the tooling to build paired CAD–image datasets (`src/datagen/`). Research notes and run outputs are kept outside the repository.

---

## Installation

### Prerequisites

- Linux (tested on Ubuntu 22.04) or Windows (WSL2)
- NVIDIA GPU with CUDA 12.4+
- Conda

### Setup

```bash
# 1. Clone
git clone https://github.com/your-repo/HoLa-BRep.git
cd HoLa-BRep

# 2. Create conda environment
conda env create -f environment.yml
conda activate hola-brep

# 3. Install project
pip install -e .

# 4. Build PointNet++ CUDA ops (imported by the historical SVR constructor used to
#    initialize the photo baseline / preference / spatial diffusion models)
cd thirdparty/Pointnet2_PyTorch/pointnet2_ops_lib
pip install -e .
cd ../../..
```

### Key Dependencies

| Package | Version | Note |
|---------|---------|------|
| PyTorch | 2.4.1 | CUDA 12.4 |
| pythonocc-core | 7.8.1 | `conda install` only |
| pytorch-lightning | >=2.2 | Training framework |
| diffusers | >=0.27 | Noise schedulers |
| transformers | >=4.36 | DINOv2 backbone |
| torch-geometric | >=2.5 | Graph neural networks |

---

## Data Format

### B-Rep Geometry Data (`data.npz`)

Each CAD model is stored as a NumPy compressed archive with the following arrays:

| Key | Shape | Description |
|-----|-------|-------------|
| `sample_points_faces` | `[num_faces, 16, 16, 6]` | Face surface points (xyz + normals) on a 16x16 UV grid |
| `sample_points_lines` | `[num_edges, 16, 6]` | Edge curve points (xyz + normals) with 16 samples |
| `face_adj` | `[num_faces, num_faces]` | Face adjacency matrix |
| `edge_face_connectivity` | `[num_edges, 3]` | Per-edge: (edge_id, face_id_1, face_id_2) |
| `zero_positions` | `[num_neg_pairs, 2]` | Negative face pairs (no shared edge) |

### Condition Data (per model directory)

| File | Content |
|------|---------|
| `imgs.npz` | Grayscale renders over the cube24 rotations: `svr_imgs[24, H, W, 3]` (training reads this key) |
| `real_photo.npz` | Synthetic photo-style image `flux[H, W, 3]` or `[24, H, W, 3]` (FLUX edits; the file name is historical and does not mean a real capture) |

Rotation 0 is the identity; latents, images, face adjacency and visibility share the same object, rotation and face indices.

### Cached Latent Features (for Stage 2)

Pre-computed VAE latent features stored as:
```
<cache_root>/<model_name>_<rotation_id>/features.npy  # shape: [num_faces, 64]
```
The 64 dimensions are `[mean(32), std(32)]` from the VAE's Gaussian bottleneck.

### Dataset Preparation

> Dataset download links will be available upon publication.

- `python -m src.preprocess.extract_brep` — extract B-Rep geometry from STEP files (explicit roots and model list, never overwrites; try `--dry-run` first)
- `src/preprocess/extract_visibility.py` — per-view face visibility caches
- `src/datagen/` — paired image generation (scene templates, Blender capture, review, retouch); see its `README.md`
- Fixed split lists live in `src/preprocess/splits/`

---

## Training

All training is managed through a single entry point with Hydra config composition:

```bash
python -m src.train --config-name <train_config> [overrides...]
```

### Stage 1: VAE Training

```bash
python -m src.train \
    --config-name train_vae \
    model=vae_light \
    dataset.raw_data_root=/path/to/brep_data \
    dataset.train_dataset=/path/to/train.txt \
    dataset.val_dataset=/path/to/val.txt \
    dataset.test_dataset=/path/to/test.txt \
    trainer.devices=1 \
    trainer.batch_size=4
```

### Stage 2: Diffusion Training

```bash
python -m src.train \
    --config-name train_diffusion_white \
    dataset.cached_latent_root=/path/to/cached_latents \
    dataset.condition_data_root=/path/to/condition_data \
    model.autoencoder.checkpoint=/path/to/vae.ckpt \
    trainer.devices=8 \
    trainer.batch_size=64
```

### Available Configurations

| Config Group | Options |
|---|---|
| `model` | `vae`, `vae_light`, `vae_light_exp`, `vae_joint`, `vae_intersection`, `vae_uv_intersection`, `diffusion` |
| `condition` | `single_img`, `none` |
| top-level config | `train_vae*`, `train_diffusion_layerwise_svr`, `train_diffusion_layerwise_real`, `train_diffusion_photo_baseline`, `train_diffusion_preference*`, `train_diffusion_spatial_photo`, `train_diffusion_uncond`, ... (see `configs/`) |

All entry points write to `${HOLA_RUNS_ROOT:-/mnt/d/hola_runs}/<experiment>/<UTC timestamp>/`, with separate `weights/`, `logs/`, `wandb/`, `raw_predictions/`, `visualizations/`, and `metadata/`. Set `trainer.run_dir` for an explicit full run path. Resume or evaluate with an explicit `trainer.resume_from_checkpoint`; evaluation never silently uses random weights. Checkpoints written before the 2026-10-30 layout change must be converted with `python tools/migrate_checkpoint.py OLD.ckpt NEW.ckpt` before resuming.

---

## Inference

```bash
python -m src.inference \
    --diffusion-weights /path/to/diffusion.ckpt \
    --autoencoder-weights /path/to/vae.ckpt \
    --condition single_img \
    --input /path/to/images/ \
    --output-dir /path/to/output/
```

The inference pipeline:
1. Encode the image with frozen DINOv2 and the trained projection
2. Run DDPM sampling (1000 steps) to generate face latent tokens
3. Decode latents via frozen VAE → face/edge geometry + topology
4. Post-process with PythonOCC → output STEP files

---

## Evaluation

```bash
python -m src.evaluate \
    --pred-root /path/to/predictions \
    --gt-root /path/to/ground_truth \
    --split-list /path/to/frozen_model_ids.txt \
    --metrics condition,validity
```

Solid validity is judged on the saved STEP re-read at `read.precision.mode=0` (`src/eval/step_metrics.py`); failures and timeouts stay in the denominator.

Available metrics: `validity`, `condition` (Chamfer/F1), `complexity`, `unique`, `lfd`

---

## Project Structure

```
├── configs/          Hydra configuration (model/, condition/, dataset/, trainer/, datagen/)
├── src/
│   ├── train.py      Unified training entry (VAE and diffusion)
│   ├── inference.py  Single-image inference
│   ├── evaluate.py   Evaluation dispatch
│   ├── common/       Run management, prediction export, OCC and IO helpers
│   ├── preprocess/   DeepCAD / ABC preprocessing, visibility, rotations, splits/
│   ├── datagen/      Paired image dataset generation
│   ├── datasets/     Training datasets
│   ├── preference/   Preference-pair generation for DPO
│   ├── models/       vae/, diffusion/, shared blocks
│   ├── postprocess/  PythonOCC post-processing (→ STEP)
│   └── eval/         Metrics and evaluation reports
├── tests/            Unit and interface tests
├── tools/            Maintenance CLIs (repository checks, checkpoint migration, dataset audits)
├── thirdparty/       Pointnet2_PyTorch (see Installation step 4)
├── environment.yml
└── pyproject.toml    Package, ruff and import-linter configuration
```

Development follows [AGENTS.md](AGENTS.md): short-lived branches with pull requests into `main`, and `pre-commit run --all-files` (layout, dead-module, import-layering and ruff checks) before committing.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{liu2025hola,
  title={HoLa: B-Rep Generation using a Holistic Latent Representation},
  author={Liu, Yilin and Xu, Duoteng and Yu, Xingyao and Xu, Xiang and Cohen-Or, Daniel and Zhang, Hao and Huang, Hui},
  journal={ACM Transactions on Graphics (Proceedings of SIGGRAPH)},
  volume={44},
  number={4},
  year={2025},
  publisher={ACM}
}
```

---

## Acknowledgements

This work was supported by the National Natural Science Foundation of China, Guangdong Basic and Applied Basic Research Foundation, and Shenzhen Science and Technology Program.

## License

This project is released under the MIT License.
