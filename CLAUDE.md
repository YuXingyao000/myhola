# Claude Code Guidelines

This repository uses `AGENTS.md` files as the primary project instructions. When working in Claude Code, follow this file as the entry point, then read the relevant `AGENTS.md` for the files you touch.

## Must Read

- For whole-repo context: `AGENTS.md`
- For source changes under `src/brepnet/`: `src/brepnet/AGENTS.md`
- For data tooling under `src/brepnet/data/`: `src/brepnet/data/AGENTS.md`
- For evaluation changes under `src/brepnet/eval/`: `src/brepnet/eval/AGENTS.md`
- For reporting experiments under `experiments/`: `skills/brepnet-experiment-notes/SKILL.md`
- For planning a new experiment day: `skills/brepnet-experiment-planning/SKILL.md`
- For diffusion model internals: `src/brepnet/models/SKILL.md`

## Current Project Shape

HoLa-BRep training is now config-driven. Prefer `python -m src.brepnet.train --config-name ...` with Hydra overrides over adding shell wrappers. The diffusion model is modularized across:

- `src/brepnet/models/diffusion.py`
- `src/brepnet/models/diffusion_latents.py`
- `src/brepnet/models/diffusion_padding.py`
- `src/brepnet/models/diffusion_condition.py`
- `src/brepnet/models/diffusion_denoiser.py`
- `src/brepnet/models/condition_encoders.py`

Keep behavior changes in the owning module and expose experiment knobs through `configs/`.

## Experiment Notes

When touching `experiments/`, use the note-first workflow in `skills/brepnet-experiment-notes/SKILL.md` (and `skills/brepnet-experiment-planning/SKILL.md` to plan a new day). Skills are installed into Cursor/Claude/Codex; edit the canonical copy under `skills/` and re-sync the copies (see README "Experiment Workflow & Agent Skills").

- one dated folder per day: `experiments/YYYY-MM-DD/`
- every dated folder must have `NOTE.md`
- every experiment entry must include why, implementation, and result
- any new command or script must be mentioned in that day folder's `NOTE.md`

## Data Contract

Current DeepCAD v7 diffusion experiments usually use:

```text
/mnt/d/data/deepcad_v7
/mnt/d/data/deepcad_v7_cond
/mnt/d/data/ae_cache/1119_deepcad_aug1_11k_24
```

Condition files currently use:

```text
deepcad_v7_cond/{model_id}/imgs.npz         # keys: svr_imgs, sketch_imgs
deepcad_v7_cond/{model_id}/real_photo.npz   # key: flux
```

Do not refer to `svr.npz` unless the data contract is explicitly changed.

## Validation

Before long training runs, prefer small checks:

```bash
python -c "from src.brepnet.models import build_model"
python -m pytest src/brepnet/test
python tools/filter_deepcad_v7_lists.py --dry-run
```

For dataset or experiment changes, save reports and outcomes under the relevant `experiments/YYYY-MM-DD/` folder and update `NOTE.md`.
