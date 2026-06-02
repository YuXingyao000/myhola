#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

mkdir -p experiments/2026-06-01/logs
mkdir -p experiments/2026-06-01/outputs_svr24
mkdir -p experiments/2026-06-01/outputs_sketch24

SVR24_GPUS="${SVR24_GPUS:-0,1}"
SKETCH24_GPUS="${SKETCH24_GPUS:-2,3}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"

CUDA_VISIBLE_DEVICES="${SVR24_GPUS}" python experiments/2026-06-01/topology_from_svr24.py \
  --mode train \
  --output-dir experiments/2026-06-01/outputs_svr24 \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  > experiments/2026-06-01/logs/svr24_retry.log 2>&1 &
pid_svr24=$!

CUDA_VISIBLE_DEVICES="${SKETCH24_GPUS}" python experiments/2026-06-01/topology_from_sketch24.py \
  --mode train \
  --output-dir experiments/2026-06-01/outputs_sketch24 \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  > experiments/2026-06-01/logs/sketch24_retry.log 2>&1 &
pid_sketch24=$!

echo "svr24 pid=${pid_svr24}, gpus=${SVR24_GPUS}, log=experiments/2026-06-01/logs/svr24_retry.log"
echo "sketch24 pid=${pid_sketch24}, gpus=${SKETCH24_GPUS}, log=experiments/2026-06-01/logs/sketch24_retry.log"

echo "Waiting for both runs..."
wait "${pid_svr24}"
wait "${pid_sketch24}"
echo "Both 24-view topology runs finished."
