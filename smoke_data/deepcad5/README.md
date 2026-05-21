# DeepCAD Smoke Fixture

Five DeepCAD samples copied for repository-local smoke tests.

## Contents

- `data/<id>/`: B-Rep geometry data from `/mnt/d/data/deepcad_v6`.
- `cond/<id>/`: condition files from `/mnt/d/data/deepcad_v6_cond`.
- `ae_cache/1119_deepcad_aug1_11k/<id>_0/`: cached VAE latent features for augmentation id `0`.
- `lists/*.txt`: split files containing the five sample ids.

## Sample IDs

- `00572443`
- `00261287`
- `00416460`
- `00684055`
- `00475715`

Use `dataset.is_aug=0` when loading diffusion cached features, because this
fixture only includes the `_0` cached latent folders.
