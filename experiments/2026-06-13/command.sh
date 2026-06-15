cd /mnt/d/python

# 1. Train frozen-topology-decoder CVAE prior.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python experiments/2026-06-13/topology_faceadj_cvae_frozen_prior.py \
  --epochs 100 \
  --batch-size 256 \
  --num-workers 8 \
  --order-mode wl \
  --wl-rounds 3 \
  --kl-beta 0.01 \
  --kl-warmup-epochs 10 \
  --latent-mse-weight 0.1 \
  --edge-count-loss-weight 0.2 \
  --condition-root /mnt/d/data/deepcad_v7_cond \
  --image-source real_flux \
  --image-backbone dinov2 \
  --image-token-dropout 0.0 \
  --eval-generate-limit 32 \
  --eval-sample-k 0 \
  --prior-temperature 1.0 \
  --topology-checkpoint experiments/2026-06-08/outputs_corrupt_only/best.pt \
  --output-dir experiments/2026-06-13/outputs_cvae_frozen_prior

# 2. Full validation split with sample@8 after training finishes.
CUDA_VISIBLE_DEVICES=0 python experiments/2026-06-13/topology_faceadj_cvae_frozen_prior.py \
  --eval-only \
  --eval-split val \
  --checkpoint experiments/2026-06-13/outputs_cvae_frozen_prior/best.pt \
  --output-dir experiments/2026-06-13/outputs_cvae_frozen_prior \
  --batch-size 16 \
  --num-workers 8 \
  --eval-generate-limit 999999 \
  --eval-sample-k 8 \
  --prior-temperature 1.0

# 3. Full test split with sample@8 after validation eval finishes.
CUDA_VISIBLE_DEVICES=0 python experiments/2026-06-13/topology_faceadj_cvae_frozen_prior.py \
  --eval-only \
  --eval-split test \
  --checkpoint experiments/2026-06-13/outputs_cvae_frozen_prior/best.pt \
  --output-dir experiments/2026-06-13/outputs_cvae_frozen_prior \
  --batch-size 16 \
  --num-workers 8 \
  --eval-generate-limit 999999 \
  --eval-sample-k 8 \
  --prior-temperature 1.0
