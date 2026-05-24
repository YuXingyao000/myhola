# DataGenerationRefactored

Python-first rewrite of `DataGeneration`.

The project is split by generation logic:

- Blender render: STL/PLY model input -> Blender PNG output.
- FLUX generation: Blender PNG output -> FLUX PNG output.
- Pack: Blender + FLUX PNG output -> `natural.npz`.

Hardcoded paths live in `config.py`. Edit that file first when changing data roots,
model paths, output directories, GPU count, or render/generation defaults.

## Main Entry Points

Build model lists:

```bash
python src/brepnet/data/DataGenerationRefactored/run_lists.py
```

Split one or more model lists across machines:

```bash
python src/brepnet/data/DataGenerationRefactored/split_model_lists.py \
  src/brepnet/data/DataGenerationRefactored/render_lists/all.txt \
  --num-machines 2
```

Run Blender cube24 render:

```bash
python src/brepnet/data/DataGenerationRefactored/run_blender.py cube24
```

Run Blender on one machine's assigned list. The file is split again across local GPUs:

```bash
python src/brepnet/data/DataGenerationRefactored/run_blender.py cube24 \
  --machine-list-dir src/brepnet/data/DataGenerationRefactored/machine_lists \
  --machine-index 0 \
  --num-gpus 8
```

Run FLUX from cube24 Blender output:

```bash
python -m src.brepnet.data.DataGenerationRefactored.run_flux cube24-dynamic
```

Run FLUX on one machine's assigned single-view Blender output. The machine list is split again across local GPUs:

```bash
python -m src.brepnet.data.DataGenerationRefactored.run_flux single-view \
  --machine-list-dir /mnt/d/python/src/brepnet/data/DataGenerationRefactored/machine_lists \
  --machine-index 0 \
  --render-root /mnt/d/python/src/brepnet/data/DataGenerationRefactored/outputs/blender_single_view \
  --output-root /mnt/d/python/src/brepnet/data/DataGenerationRefactored/outputs/flux_single_view \
  --num-gpus 8
```

Pack cube24 results:

```bash
python src/brepnet/data/DataGenerationRefactored/run_pack.py cube24
```

Run the common cube24 flow end to end:

```bash
python src/brepnet/data/DataGenerationRefactored/run_pipeline.py cube24
```

For a small smoke run:

```bash
python src/brepnet/data/DataGenerationRefactored/run_pipeline.py cube24 --num-gpus 1 --max-models 2
```

## Layout

- `config.py`: editable hardcoded paths and default parameters.
- `run_lists.py`: sample train/val/test ids and create `render_lists/rank_*.txt`.
- `split_model_lists.py`: split arbitrary model lists into `machine_*.txt`.
- `run_blender.py`: launches Blender Python scripts per GPU.
- `run_flux.py`: launches FLUX generation per GPU.
- `run_pack.py`: packs PNGs into `natural.npz`.
- `blender_scripts/`: scripts executed inside Blender.
- `flux_scripts/`: scripts executed by normal Python for FLUX.
- `tools/`: smaller utility scripts.
- `external/generation/`: copied Img2Brep generation package.

Generated outputs are intentionally under `outputs/` and logs under `logs/`.
The old large output folders are not copied.
