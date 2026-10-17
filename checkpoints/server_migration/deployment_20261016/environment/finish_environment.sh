#!/usr/bin/env bash
set -euo pipefail
mkdir -p /mnt/d/migration/logs
exec > >(tee -a /mnt/d/migration/logs/environment-finish.log) 2>&1
while [[ ! -f /mnt/d/migration/environment-bootstrap.complete ]]; do sleep 5; done
bash /mnt/d/migration/install_extensions.sh
/mnt/d/anaconda3/envs/torch/bin/python /mnt/d/migration/validate_environment.py
OMP_NUM_THREADS=1 /mnt/d/anaconda3/envs/torch/bin/torchrun --standalone --nproc-per-node=8 /mnt/d/migration/validate_nccl.py
for env_name in torch generate; do
  /mnt/d/anaconda3/envs/$env_name/bin/python -m pip check
  /mnt/d/anaconda3/bin/conda list -n "$env_name" --explicit > /mnt/d/migration/logs/$env_name-conda-explicit.txt
  /mnt/d/anaconda3/envs/$env_name/bin/python -m pip freeze > /mnt/d/migration/logs/$env_name-pip-freeze.txt
done
touch /mnt/d/migration/environment-validation.complete
