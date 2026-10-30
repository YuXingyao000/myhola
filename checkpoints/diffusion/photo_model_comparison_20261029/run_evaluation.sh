#!/usr/bin/env bash
set -euo pipefail
cd /mnt/d/python
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
exec /mnt/d/anaconda3/envs/torch/bin/python -u -m src.brepnet.eval.photo_model_comparison \
  --run-dir /mnt/d/python/checkpoints/diffusion/photo_model_comparison_20261029/run \
  --baseline-run /mnt/d/python/checkpoints/diffusion/photo_baseline/evaluation_full_20261026 \
  --original-checkpoint /mnt/d/ckpt/HoLa/Diffusion_svr_sq30_1500k.ckpt \
  --photo-checkpoint /mnt/d/python/checkpoints/diffusion/photo_baseline/evaluation_20261026/weights/photo_latest.ckpt \
  --photo-dpo5000 /mnt/d/python/checkpoints/diffusion_preference_photo/20261028T073228Z/weights/last.ckpt \
  --spatial-best /mnt/d/python/checkpoints/diffusion_spatial_photo/20261028T104221Z/weights/epoch=0022-step=00020930-val_loss=0.042389.ckpt \
  --spatial-last /mnt/d/python/checkpoints/diffusion_spatial_photo/20261028T104221Z/weights/last.ckpt \
  --gpus 0 1 2 3 4 5 6 7 --cpu-workers 32 "$@"
