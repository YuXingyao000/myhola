cd /mnt/d/python
set -euo pipefail

# 1. Audit the first 100 testing models for direct OCC/data.npz face-order agreement.
conda run -n torch python src/brepnet/data/extract_visibility.py \
  --mode audit \
  --model-list src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt \
  --limit 100 \
  --raw-root /mnt/d/data/deepcad_v7 \
  --condition-root /mnt/d/data/deepcad_v7_cond \
  --output-root /mnt/d/data/deepcad_v7_visibility \
  --views all \
  --image-size 224 \
  --up-mode legacy_point \
  --max-direct-cost 1e-3

# 2. Generate all 24-view visibility buffers plus debug PNGs for five testing models.
conda run -n torch python src/brepnet/data/extract_visibility.py \
  --mode generate \
  --model-list src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt \
  --limit 5 \
  --raw-root /mnt/d/data/deepcad_v7 \
  --condition-root /mnt/d/data/deepcad_v7_cond \
  --output-root /mnt/d/data/deepcad_v7_visibility \
  --views all \
  --image-size 224 \
  --up-mode legacy_point \
  --max-direct-cost 1e-3 \
  --save-debug \
  --debug-model-limit 5 \
  --debug-view-limit 24

# 3. Generate the full testing split first so tomorrow's audit has complete held-out data.
conda run -n torch python src/brepnet/data/extract_visibility.py \
  --mode generate \
  --model-list src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt \
  --raw-root /mnt/d/data/deepcad_v7 \
  --condition-root /mnt/d/data/deepcad_v7_cond \
  --output-root /mnt/d/data/deepcad_v7_visibility \
  --views all \
  --image-size 224 \
  --up-mode legacy_point \
  --max-direct-cost 1e-3

# 4. Generate the full validation split.
conda run -n torch python src/brepnet/data/extract_visibility.py \
  --mode generate \
  --model-list src/brepnet/data/list/deduplicated_deepcad_validation_7_30.txt \
  --raw-root /mnt/d/data/deepcad_v7 \
  --condition-root /mnt/d/data/deepcad_v7_cond \
  --output-root /mnt/d/data/deepcad_v7_visibility \
  --views all \
  --image-size 224 \
  --up-mode legacy_point \
  --max-direct-cost 1e-3

# 5. Generate the full training split. This is the long step and is resume-safe.
conda run -n torch python src/brepnet/data/extract_visibility.py \
  --mode generate \
  --model-list src/brepnet/data/list/deduplicated_deepcad_training_7_30.txt \
  --raw-root /mnt/d/data/deepcad_v7 \
  --condition-root /mnt/d/data/deepcad_v7_cond \
  --output-root /mnt/d/data/deepcad_v7_visibility \
  --views all \
  --image-size 224 \
  --up-mode legacy_point \
  --max-direct-cost 1e-3
