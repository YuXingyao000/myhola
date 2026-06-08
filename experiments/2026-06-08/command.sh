# --- 06-08 Topology VAE iterations on top of 06-06 best (WL + edge_count + KL=0.001) ---
# 三个变种用三张卡并行跑，方便回家直接对比

cd /mnt/d/python

COMMON="--epochs 100 --batch-size 256 --num-workers 8 \
    --order-mode wl --wl-rounds 3 \
    --kl-beta 0.001 --kl-warmup-epochs 10 \
    --edge-count-loss-weight 0.2 \
    --eval-generate-limit 256 --prior-samples 256"

# 1. Prefix corruption only — keep 06-06 sequence, only change training prefix.
CUDA_VISIBLE_DEVICES=0 python experiments/2026-06-08/topology_faceadj_vae_corrupt.py \
    $COMMON \
    --output-dir experiments/2026-06-08/outputs_corrupt_only \
    --corrupt-prob 0.15 --corrupt-warmup-epochs 10 \
    > experiments/2026-06-08/outputs_corrupt_only.log 2>&1 &

# 2. Per-face degree token + degree-budget constrained decoding.
CUDA_VISIBLE_DEVICES=1 python experiments/2026-06-08/topology_faceadj_vae_degree.py \
    $COMMON \
    --output-dir experiments/2026-06-08/outputs_degree_only \
    --degree-loss-weight 0.2 \
    > experiments/2026-06-08/outputs_degree_only.log 2>&1 &

# 3. Corruption + degree token combined.
CUDA_VISIBLE_DEVICES=2 python experiments/2026-06-08/topology_faceadj_vae_both.py \
    $COMMON \
    --output-dir experiments/2026-06-08/outputs_both \
    --degree-loss-weight 0.2 \
    --corrupt-prob 0.15 --corrupt-warmup-epochs 10 \
    > experiments/2026-06-08/outputs_both.log 2>&1 &

wait
echo "All three runs done."

# --- Test split eval (run after best.pt is written) ---
# CUDA_VISIBLE_DEVICES=0 python experiments/2026-06-08/topology_faceadj_vae_corrupt.py \
#     --eval-only --eval-split test \
#     --checkpoint experiments/2026-06-08/outputs_corrupt_only/best.pt \
#     --output-dir experiments/2026-06-08/outputs_corrupt_only \
#     --batch-size 256 --num-workers 8 \
#     --eval-generate-limit 2424 --prior-samples 512
#
# CUDA_VISIBLE_DEVICES=0 python experiments/2026-06-08/topology_faceadj_vae_degree.py \
#     --eval-only --eval-split test \
#     --checkpoint experiments/2026-06-08/outputs_degree_only/best.pt \
#     --output-dir experiments/2026-06-08/outputs_degree_only \
#     --batch-size 256 --num-workers 8 \
#     --eval-generate-limit 2424 --prior-samples 512
#
# CUDA_VISIBLE_DEVICES=0 python experiments/2026-06-08/topology_faceadj_vae_both.py \
#     --eval-only --eval-split test \
#     --checkpoint experiments/2026-06-08/outputs_both/best.pt \
#     --output-dir experiments/2026-06-08/outputs_both \
#     --batch-size 256 --num-workers 8 \
#     --eval-generate-limit 2424 --prior-samples 512

# --- Re-evaluate 06-06 baselines with new valid_strict_ratio metric ---
# Two scripts in 06-06 had `adjacency_stats` patched on 06-08 to also report
# valid_strict_ratio = connected ∧ no_isolated. No retraining needed; just
# re-run eval-only on existing best.pt.
#
# CUDA_VISIBLE_DEVICES=0 python experiments/2026-06-06/topology_faceadj_vae.py \
#     --eval-only --eval-split test \
#     --checkpoint experiments/2026-06-06/outputs_faceadj_vae/best.pt \
#     --output-dir experiments/2026-06-06/outputs_faceadj_vae \
#     --batch-size 256 --num-workers 8 \
#     --eval-generate-limit 2424 --prior-samples 512
#
# CUDA_VISIBLE_DEVICES=0 python experiments/2026-06-06/topology_faceadj_vae.py \
#     --eval-only --eval-split test \
#     --checkpoint experiments/2026-06-06/outputs_faceadj_vae_kl001/best.pt \
#     --output-dir experiments/2026-06-06/outputs_faceadj_vae_kl001 \
#     --batch-size 256 --num-workers 8 \
#     --eval-generate-limit 2424 --prior-samples 512
#
# CUDA_VISIBLE_DEVICES=0 python experiments/2026-06-06/topology_faceadj_vae.py \
#     --eval-only --eval-split test \
#     --checkpoint experiments/2026-06-06/outputs_faceadj_vae_wl_kl001/best.pt \
#     --output-dir experiments/2026-06-06/outputs_faceadj_vae_wl_kl001 \
#     --batch-size 256 --num-workers 8 \
#     --eval-generate-limit 2424 --prior-samples 512
#
# CUDA_VISIBLE_DEVICES=0 python experiments/2026-06-06/topology_faceadj_vae_edgecount.py \
#     --eval-only --eval-split test \
#     --checkpoint experiments/2026-06-06/outputs_faceadj_vae_edgecount_wl_kl001/best.pt \
#     --output-dir experiments/2026-06-06/outputs_faceadj_vae_edgecount_wl_kl001 \
#     --batch-size 256 --num-workers 8 \
#     --eval-generate-limit 2424 --prior-samples 512
