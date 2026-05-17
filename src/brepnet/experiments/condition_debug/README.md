# Condition Debug Experiments

This directory is an isolated workspace for conditional diffusion diagnostics.

Rules:

- Do not edit the repository's active `src/brepnet/diffusion_model.py`,
  `src/brepnet/train_diffusion.py`, `src/brepnet/dataset.py`, or other existing
  runtime files for these experiments.
- If an experiment needs changes to an existing file, copy that file into this
  directory first and modify only the copy.
- Keep generated experiment outputs outside the repository, under a configured
  result root such as `/mnt/d/data/new_cond_results/exp_plan_0502_depth`.
- Scripts in this directory should be runnable as standalone modules and should
  import the active project code unless they explicitly use a local copied
  variant.

Planned local copies, only when needed:

- `diffusion_model_condition_debug.py`
- `train_diffusion_condition_debug.py`
- `dataset_condition_debug.py`

