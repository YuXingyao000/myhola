# Claude Code Guidelines

This repository uses `AGENTS.md` files as the primary project instructions. `AGENTS.md` is authoritative; this file is the Claude Code entry point and points to it. Read the relevant `AGENTS.md` before changing files it covers. If this file and an `AGENTS.md` conflict, follow `AGENTS.md`.

## Must Read

- Whole repo: `AGENTS.md`
- Source under `src/brepnet/`: `src/brepnet/AGENTS.md` (Claude summary: `src/brepnet/CLAUDE.md`)
- Data tooling under `src/brepnet/data/`: `src/brepnet/data/AGENTS.md` (Claude summary: `src/brepnet/data/CLAUDE.md`)
- Evaluation under `src/brepnet/eval/`: `src/brepnet/eval/AGENTS.md` (Claude summary: `src/brepnet/eval/CLAUDE.md`)
- Diffusion internals: `src/brepnet/models/SKILL.md`; current implementation lives in `src/brepnet/diffusion/`
- Research records: the topic's `PLAN.md` and `README.md`/`NOTE.md` under `checkpoints/`. `experiments/` is a historical archive, and the former `skills/brepnet-experiment-*` files no longer exist — do not rely on or restore them.
- Literature baselines and how their validity/accuracy numbers are defined: `checkpoints/baselines/README.md`. Read it before putting any external number in a comparison table.

## Worktrees (as of 2026-10-29)

One Git repo, several worktrees. Each branch carries its own `AGENTS.md`/`CLAUDE.md`; read the copy in the worktree you are editing. Only `/mnt/d/python` auto-syncs to the 5090 server.

| Path | Branch | Focus |
|---|---|---|
| `/mnt/d/python` | `codex/photo-*` (main checkout) | Image-conditioned diffusion (SVR/photo, DPO, spatial memory); VAE is still the single-file `models/vae.py` and the original HoLa AE |
| `/mnt/d/python_uv_train` | `codex/uv-intersection-training` | VAE: conditional intersection / UV training (current best VAE line, 90.99% native validity on 3495 validation) |
| `/mnt/d/python_curve_eval` | `codex/vae-uv-boundary` | VAE: UV boundary, shared curve, latent expansion, OCC failure browser |
| `/mnt/d/python_room_generation` | `codex/room-generation` | Dataset/image generation (`packages/cad_scene_factory`), unconditional diffusion comparison |
| `/mnt/d/python_legacy_post_main` | detached | Frozen legacy post-processing reference; do not develop here |

The VAE branches split `models/vae.py` into the package `models/vae/` (see `checkpoints/vae_structure/README.md` on those branches). Diffusion on this checkout still uses the original HoLa AE and its 32-d latent cache; switching to a new VAE needs re-encoded caches and retrained diffusion.

## Current Project Shape

Training is config-driven:

```bash
python -m src.brepnet.train --config-name <config> [Hydra overrides]
```

Prefer Hydra overrides over adding shell wrappers for new experiments. The diffusion implementation is the package `src/brepnet/diffusion/`:

| Module | Responsibility |
|---|---|
| `model.py` | Assembly, training forward, VAE interface, strict checkpoint loading |
| `conditioning.py`, `image_encoder.py`, `image_loading.py` | Condition routing, frozen DINOv2, trainable projection |
| `denoiser.py`, `fusion.py` | 24 encoder blocks, each followed by image cross-attention |
| `latents.py`, `padding.py` | Latent cache, face packing, deduplication |
| `topology.py`, `losses.py`, `sampling.py` | Optional topology, losses, DDPM sampling |

`src/brepnet/models/diffusion*.py` and `models/condition_encoders.py` are compatibility re-exports of that package, not current entry points. `src/brepnet/models/vae.py` and `models/blocks.py` remain the VAE and shared operators. Keep behavior changes in the owning module and expose experiment knobs through `configs/`.

Current single-image path: image → frozen DINOv2 CLS/patch tokens → trainable image projection; noisy face latent + timestep embedding → 24× (TransformerEncoderLayer → image cross-attention residual) → DDPM → frozen VAE → OCC post-processing. This is the current implementation, not a binding architecture requirement, and it does not by itself imply trained or improved reconstruction quality.

## Research Records

- New plans, runs, audits, and results go under `checkpoints/`; do not append to `experiments/`.
- Each topic keeps a versioned `PLAN.md` plus `README.md` or `NOTE.md` recording why, implementation, result/status, and links.
- Runs default to `checkpoints/<trainer.exp_name>/<UTC timestamp>/`, with separate `weights/`, `logs/`, `wandb/`, `raw_predictions/`, `visualizations/`, and `metadata/`.
- Save the resolved config, Git revision, seed, and command. Resume with `trainer.resume_from_checkpoint`; weight-only initialization is a new run, not a resume. Never overwrite an existing run without a resume marker.

## Data Contract

Typical DeepCAD v7 assets (not guaranteed mounted on this machine; pass paths via Hydra overrides or environment variables):

```text
/mnt/d/data/deepcad_v7
/mnt/d/data/deepcad_v7_cond
/mnt/d/data/ae_cache/1119_deepcad_aug1_11k_24
/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt
```

- `deepcad_v7_cond/{model_id}/imgs.npz` — keys `svr_imgs`, `sketch_imgs`.
- `deepcad_v7_cond/{model_id}/real_photo.npz` — key `flux`.
- Latent cache: `{model_id}_{rotation_id}/features.npy`, usually `[num_faces, 64]` = mean(32) + std(32), not log-variance. `rotation_id=0` is identity.
- Images, latents, B-Rep face order, and visibility must use the same model/view/rotation convention. Do not use `svr.npz` or legacy `natural.npz` unless the data contract is explicitly changed.

## Validation

Before long runs, prefer small checks:

```bash
python -c "from src.brepnet.models import build_model"
python tools/filter_deepcad_v7_lists.py --dry-run
python -m pytest src/brepnet/test/test_<topic>.py
```

Run only the relevant tests under `src/brepnet/test/`; the directory is not a dependency-free unified suite. Match verification to the change: docs → paths and diffs; code → imports and focused unit behavior; model/data interfaces → real small-batch shapes, loss, gradients, and inference semantics. Do not claim research conclusions from smoke tests.

## Collaboration

- Default discussion and notes to Chinese; keep code identifiers in English.
- Separate source facts, run results, literature claims, user observations, and hypotheses.
- Architecture changes must state what information is available at training vs inference, how face correspondence/padding is handled, which failure mechanism the change targets, and what downstream evidence decides whether it stays.
- Preserve uncommitted user work. Do not force push or reset.
- The source machine auto-syncs commits to the 5090 server on the same branch (fast-forward only). A commit is not a successful sync: check `.git/hola-sync/status.json` against HEAD before reporting sync status.
- Never commit datasets, checkpoints, generated models, W&B tokens, or machine credentials.
