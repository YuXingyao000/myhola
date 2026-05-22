#!/usr/bin/env bash
# Noise sensitivity re-test for the fine-tuned VAE (intersection noise training).
# Compare against the original baseline (sigma=0.10: 44% valid).
#
# Run this AFTER the intersection_noise fine-tuning finishes.
# Replace CKPT with the actual fine-tuned checkpoint path.

CKPT="${CKPT:-/mnt/d/data/vae_intersection_experiments/20260521_intersection_noise_0p1_vae_1119_light/checkpoints/last.ckpt}"

python -m src.brepnet.experiments.decoder_robustness \
    --vae-checkpoint "$CKPT" \
    --data-root /mnt/d/data/deepcad_v6 \
    --test-list src/brepnet/data/list/deduplicated_deepcad_testing_7_30_sample_100.txt \
    --output /mnt/d/data/vae_intersection_experiments/noise_sensitivity_after_finetune \
    --batch-size 4 \
    --noise-levels 0.0,0.05,0.10,0.15,0.20,0.30,0.50
