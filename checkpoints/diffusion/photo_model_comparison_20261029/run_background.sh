#!/usr/bin/env bash
# tmux 内的持久入口；失败也留下退出码和明确状态。
set -uo pipefail
cd /mnt/d/python
run=/mnt/d/python/checkpoints/diffusion/photo_model_comparison_20261029/run
mkdir -p "$run/logs" "$run/metadata"
bash checkpoints/diffusion/photo_model_comparison_20261029/run_evaluation.sh > "$run/logs/orchestrator.log" 2>&1
evaluation_exit=$?
printf '%s\n' "$evaluation_exit" > "$run/metadata/exit_code.txt"
if [ "$evaluation_exit" -ne 0 ]; then
  /mnt/d/anaconda3/envs/torch/bin/python - "$run" "$evaluation_exit" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
from tools.vae_compare_post import atomic_json
run = Path(sys.argv[1])
path = run / 'status.json'
value = json.loads(path.read_text()) if path.is_file() else {}
value.update(status='failed', exit_code=int(sys.argv[2]), failed_at_utc=datetime.now(timezone.utc).isoformat())
atomic_json(path, value)
PY
fi
exit "$evaluation_exit"
