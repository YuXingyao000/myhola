#!/usr/bin/env bash
# Latent noise sensitivity re-test for the VAE fine-tuned with intersection noise.

CKPT="${CKPT:-/mnt/d/data/new_cond_ckpt/20260521_intersection_noise_0p1_vae_1119_light.ckpt}"
LATENT_ROOT="${LATENT_ROOT:-/mnt/d/data/ae_cache/1119_deepcad_aug1_11k}"
SPLIT="${SPLIT:-src/brepnet/data/list/deduplicated_deepcad_testing_7_30_sample_100.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/d/data/latent_sensitivity/20260522_intersection_noise_0p1_vae_1119_light_test100}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
NUM_CPUS="${NUM_CPUS:-40}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/d/miniconda3/envs/img2brep/bin/python}"

"${PYTHON_BIN}" experiments/2026-05-22/noise_sensitivity_retest/latent_sensitivity.py \
    --autoencoder-weights "${CKPT}" \
    --latent_root "${LATENT_ROOT}" \
    --split "${SPLIT}" \
    --output_root "${OUTPUT_ROOT}" \
    --noise_levels 0.0 0.05 0.10 0.15 0.20 0.30 0.50 \
    --limit 100 \
    --gpu_ids ${GPU_IDS} \
    --run_post \
    --post_use_ray \
    --num_cpus "${NUM_CPUS}" \
    --post_timeout 120 \
    --max_optimize_iter 200
