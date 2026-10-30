#!/usr/bin/env bash
# Full fixed-list comparison of continued DPO and DPO + geometry.
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 || ( $# -eq 2 && $2 != --resume ) ]]; then
    echo "Usage: bash $0 OUTPUT_DIR [--resume]" >&2
    exit 2
fi
cd /mnt/d/python
evaluation_output=$(realpath -m "$1")
resume_arguments=()
if [[ ${2:-} == --resume ]]; then
    test -d "$evaluation_output"
    resume_arguments=(--resume)
else
    mkdir "$evaluation_output"
fi
exec 9>"$evaluation_output/.workflow.lock"
flock -n 9 || { echo "Evaluation is already running" >&2; exit 2; }
mkdir -p "$evaluation_output/logs"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
evaluation_python=/mnt/d/anaconda3/envs/torch/bin/python
mark_stage() {
    "$evaluation_python" - "$evaluation_output" "$1" "${2:-0}" <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import os, sys
from tools.vae_compare_post import atomic_json
atomic_json(Path(sys.argv[1]) / 'status.json', {
    'status': sys.argv[2], 'exit_code': int(sys.argv[3]),
    'updated_utc': datetime.now(timezone.utc).isoformat(),
    'supervisor_pid': os.getppid(), 'intended_per_model': 3505,
    'methods': {'baseline': 'continued DPO', 'dpo_5000': 'DPO + geometry'},
})
PY
}
trap 'result=$?; if (( result != 0 )); then mark_stage failed "$result"; fi' EXIT
mark_stage comparison
"$evaluation_python" -u -m src.brepnet.eval.preference_compare \
    --original-checkpoint /mnt/d/ckpt/HoLa/Diffusion_svr_sq30_1500k.ckpt \
    --baseline-preference-checkpoint /mnt/d/python/checkpoints/diffusion_preference_continue/20261025T054547.727170Z/weights/steps/step=00005000.ckpt \
    --preference-checkpoint /mnt/d/python/checkpoints/diffusion_preference_geometry/20261025T062543.945628Z/weights/steps/step=00005000.ckpt \
    --split-list /mnt/d/python/src/brepnet/data/list/deduplicated_deepcad_validation_7_30.txt \
    --raw-data-root /mnt/d/data/deepcad_v7 \
    --condition-data-root /mnt/d/data/deepcad_v7_cond \
    --run-dir "$evaluation_output/comparison" \
    --gpus 0 1 2 3 4 5 6 7 --cpu-workers 16 --expected-count 3505 --seed 42 \
    "${resume_arguments[@]}" >"$evaluation_output/logs/comparison.log" 2>&1
mark_stage observation
# If comparison was interrupted, the observation directory does not exist yet.
audit_resume=()
if [[ -f "$evaluation_output/observation/protocol.json" ]]; then audit_resume=(--resume); fi
"$evaluation_python" -u -m src.brepnet.eval.diffusion_audit \
    --comparison-run "$evaluation_output/comparison" \
    --output-dir "$evaluation_output/observation" \
    --methods baseline dpo_5000 --workers 16 --timeout 120 \
    --distance-threshold 0.01 --resolution 224 --mesh-deflection 0.002 --mask-threshold 245 \
    --font /root/.cache/hola-report-fonts/SourceHanSansSC-Regular.otf \
    "${audit_resume[@]}" >"$evaluation_output/logs/observation.log" 2>&1
mark_stage complete
