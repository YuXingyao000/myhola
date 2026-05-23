#!/usr/bin/env bash
# Qualitative single-view inference on a handful of custom photos.
#
# Usage:
#   bash src/brepnet/scripts/qualitative_custom_demo.sh [INPUT_DIR] [OUT_ROOT]
#
# INPUT_DIR  folder with N photos (default: /mnt/d/data/gt_photo)
# OUT_ROOT   demo workspace that will hold data_root/ cond_root/ results/
#            (default: /mnt/d/data/my_6_demo)
#
# Steps:
#   1. Pack each photo into the dataset layout expected by
#      AutoEncoder_dataset3 + prepare_condition (imgs.npz + natural.npz +
#      placeholder data.npz), via src/brepnet/data/prepare_custom_6_demo.py.
#   2. Run Diffusion inference with `single_img` condition.
#   3. Run construct_brep post-processing.
#
# Checkpoints and AE cache paths are hard-coded below. Override via env vars
# if you want to swap them:
#   CKPT=/path/to.ckpt AE_CKPT=/path/to_ae.ckpt AE_CACHE=/path/to_cached_latents \
#       bash src/brepnet/scripts/qualitative_custom_demo.sh ...

set -euo pipefail

INPUT_DIR="${1:-/mnt/d/data/gt_photo}"
OUT_ROOT="${2:-/mnt/d/data/my_6_demo}"

CKPT="${CKPT:-/mnt/d/new_cond_ckpt/0317_deepcad_natural_02_random_pick.ckpt}"
AE_CKPT="${AE_CKPT:-/mnt/d/ae_checkpoints/1119_deepcad_aug1_11k.ckpt}"
AE_CACHE="${AE_CACHE:-/mnt/d/ae_cache/1119_deepcad_aug1_11k}"
PLACEHOLDER_DATA_NPZ="${PLACEHOLDER_DATA_NPZ:-/mnt/d/data/deepcad_v6/00786107/data.npz}"

GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-6}"
NUM_WORKER="${NUM_WORKER:-2}"
NUM_CPUS_POST="${NUM_CPUS_POST:-16}"

DATA_ROOT="${OUT_ROOT}/data_root"
COND_ROOT="${OUT_ROOT}/cond_root"
LIST_PATH="${OUT_ROOT}/test_list.txt"
RESULTS_DIR="${OUT_ROOT}/results"

echo "==> [1/3] Packing photos from ${INPUT_DIR} into ${OUT_ROOT}"
python -m src.brepnet.data.prepare_custom_6_demo \
    --input_dir "${INPUT_DIR}" \
    --out_root "${OUT_ROOT}" \
    --placeholder_data_npz "${PLACEHOLDER_DATA_NPZ}"

echo "==> [2/3] Running Diffusion inference on ${LIST_PATH}"
CUDA_VISIBLE_DEVICES="${GPU}" python -m src.brepnet.train --config-name train_diffusion_white \
    trainer.devices=1 \
    trainer.evaluate=true \
    trainer.test_output_dir="${RESULTS_DIR}" \
    trainer.resume_from_checkpoint="${CKPT}" \
    trainer.num_workers="${NUM_WORKER}" \
    trainer.batch_size="${BATCH_SIZE}" \
    dataset.name=AutoEncoder_dataset3 \
    dataset.data_root="${DATA_ROOT}" \
    dataset.latent_root="${AE_CACHE}" \
    dataset.condition_root="${COND_ROOT}" \
    dataset.train_dataset="${LIST_PATH}" \
    dataset.val_dataset="${LIST_PATH}" \
    dataset.test_dataset="${LIST_PATH}" \
    dataset.max_faces=30 \
    dataset.is_aug=0 \
    dataset.cached_condition=false \
    model.padding.max_faces=30 \
    model.denoiser.hidden_dim=768 \
    model.autoencoder.in_channels=6 \
    model.noise.beta_schedule=squaredcos_cap_v2 \
    model.autoencoder.checkpoint="${AE_CKPT}" \
    model.latent.use_cached_latents=false \
    hydra.job.chdir=false

echo "==> [3/3] Reconstructing B-reps with construct_brep"
python -m src.brepnet.post.construct_brep \
    --data_root "${RESULTS_DIR}" \
    --out_root "${RESULTS_DIR}_post" \
    --use_ray \
    --num_cpus "${NUM_CPUS_POST}" \
    --drop_num 2 \
    --from_scratch

echo "==> Done."
echo "    Raw diffusion outputs : ${RESULTS_DIR}"
echo "    Post-processed B-reps : ${RESULTS_DIR}_post"
echo "    Input previews        : ${OUT_ROOT}/preview"
