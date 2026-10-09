cd /mnt/d/python
set -euo pipefail

# 1. Train visibility-aware DINO in-context topology CVAE.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python experiments/2026-06-18/topology_faceadj_cvae_visibility.py \
  --epochs 100 \
  --batch-size 256 \
  --num-workers 8 \
  --order-mode wl \
  --wl-rounds 3 \
  --kl-beta 0.001 \
  --kl-warmup-epochs 10 \
  --edge-count-loss-weight 0.2 \
  --corrupt-prob 0.15 \
  --corrupt-warmup-epochs 10 \
  --condition-root /mnt/d/data/deepcad_v7_cond \
  --visibility-root /mnt/d/data/deepcad_v7_visibility \
  --visibility-view-id 0 \
  --image-source real_flux \
  --image-backbone dinov2 \
  --image-token-dropout 0.0 \
  --eval-generate-limit 32 \
  --eval-sample-k 0 \
  --prior-temperature 1.0 \
  --init-vae-checkpoint experiments/2026-06-08/outputs_corrupt_only/best.pt \
  --best-metric cond_prior_mu_visible_edge_f1 \
  --output-dir experiments/2026-06-18/outputs_cvae_visibility

# 2. Full validation evaluation with sample@8 after training finishes.
CUDA_VISIBLE_DEVICES=0 python experiments/2026-06-18/topology_faceadj_cvae_visibility.py \
  --eval-only \
  --eval-split val \
  --checkpoint experiments/2026-06-18/outputs_cvae_visibility/best.pt \
  --output-dir experiments/2026-06-18/outputs_cvae_visibility \
  --batch-size 16 \
  --num-workers 8 \
  --eval-generate-limit 999999 \
  --eval-sample-k 8 \
  --prior-temperature 1.0

# 3. Full test evaluation with sample@8 after validation evaluation finishes.
CUDA_VISIBLE_DEVICES=0 python experiments/2026-06-18/topology_faceadj_cvae_visibility.py \
  --eval-only \
  --eval-split test \
  --checkpoint experiments/2026-06-18/outputs_cvae_visibility/best.pt \
  --output-dir experiments/2026-06-18/outputs_cvae_visibility \
  --batch-size 16 \
  --num-workers 8 \
  --eval-generate-limit 999999 \
  --eval-sample-k 8 \
  --prior-temperature 1.0
