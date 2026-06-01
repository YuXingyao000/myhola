#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/python
mkdir -p experiments/2026-06-01/logs

CUDA_VISIBLE_DEVICES=0,1 python experiments/2026-06-01/topology_from_svr24.py \
    --mode train \
    --output-dir experiments/2026-06-01/outputs_svr24 \
    --epochs 20 \
    --batch-size 4 \
    --num-workers 8 \
    > experiments/2026-06-01/logs/svr24.log 2>&1 &
pid_svr24=$!

CUDA_VISIBLE_DEVICES=2,3 python experiments/2026-06-01/topology_from_sketch24.py \
    --mode train \
    --output-dir experiments/2026-06-01/outputs_sketch24 \
    --epochs 20 \
    --batch-size 4 \
    --num-workers 8 \
    > experiments/2026-06-01/logs/sketch24.log 2>&1 &
pid_sketch24=$!

CUDA_VISIBLE_DEVICES=4,5 python experiments/2026-06-01/topology_from_real_photo.py \
    --mode train \
    --output-dir experiments/2026-06-01/outputs_real_photo \
    --epochs 20 \
    --batch-size 32 \
    --num-workers 8 \
    > experiments/2026-06-01/logs/real_photo.log 2>&1 &
pid_real_photo=$!

CUDA_VISIBLE_DEVICES=6,7 python experiments/2026-06-01/topology_from_svr_single.py \
    --mode train \
    --output-dir experiments/2026-06-01/outputs_svr_single \
    --epochs 20 \
    --batch-size 32 \
    --num-workers 8 \
    --view-id 0 \
    > experiments/2026-06-01/logs/svr_single.log 2>&1 &
pid_svr_single=$!

wait "${pid_svr24}"
wait "${pid_sketch24}"
wait "${pid_real_photo}"
wait "${pid_svr_single}"

echo "All topology predictor training jobs finished."
