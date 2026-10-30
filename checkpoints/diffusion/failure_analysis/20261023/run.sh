#!/usr/bin/env bash
set -u
cd /mnt/d/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg
/mnt/d/anaconda3/envs/torch/bin/python -u -m src.brepnet.eval.diffusion_audit --comparison-run /mnt/d/python/checkpoints/diffusion/preference_eval/20261023T090834Z --output-dir /mnt/d/python/checkpoints/diffusion/failure_analysis/runs/20261023T105637Z_full --methods baseline dpo_5000 --workers 16 --timeout 120 --wait-seconds 7200 --distance-threshold 0.01 --resolution 224 --mesh-deflection 0.002 --mask-threshold 245 --font /root/.cache/hola-report-fonts/SourceHanSansSC-Regular.otf
status=$?
printf "%s\n" "$status" > /mnt/d/python/checkpoints/diffusion/failure_analysis/runs/20261023T105637Z_full.exit_code.txt
exit "$status"
