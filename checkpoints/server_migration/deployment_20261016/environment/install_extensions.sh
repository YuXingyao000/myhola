#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export PIP_CACHE_DIR=/mnt/d/migration/pip-cache
export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
export PIP_DEFAULT_TIMEOUT=120
exec > >(tee -a /mnt/d/migration/logs/extensions-install.log) 2>&1
# CUDA13.2 remains installed. This matching compiler is used only for cu128 extensions.
apt-get -o DPkg::Lock::Timeout=600 install -y --no-install-recommends cuda-nvcc-12-8 cuda-cudart-dev-12-8 cuda-cccl-12-8 libcublas-dev-12-8 libcusparse-dev-12-8 libcusolver-dev-12-8 libcurand-dev-12-8
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST='12.0'
export MAX_JOBS=8
mkdir -p /mnt/d/migration/wheels
/mnt/d/anaconda3/envs/torch/bin/python -m pip wheel --no-deps --no-build-isolation 'chamferdist==1.0.3' -w /mnt/d/migration/wheels
for env_name in torch generate; do
  /mnt/d/anaconda3/envs/$env_name/bin/python -m pip install --no-index --find-links /mnt/d/migration/wheels 'chamferdist==1.0.3'
  printf 'pythonocc-core ==7.8.1.1\nocct ==7.8.1\n' > /mnt/d/anaconda3/envs/$env_name/conda-meta/pinned
  mkdir -p /mnt/d/anaconda3/envs/$env_name/etc/conda/activate.d /mnt/d/anaconda3/envs/$env_name/etc/conda/deactivate.d
  cat > /mnt/d/anaconda3/envs/$env_name/etc/conda/activate.d/hola-cuda.sh <<'EOF'
export _HOLA_OLD_CUDA_HOME="${CUDA_HOME-}"
export _HOLA_OLD_TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST-}"
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST=12.0
EOF
  cat > /mnt/d/anaconda3/envs/$env_name/etc/conda/deactivate.d/hola-cuda.sh <<'EOF'
if [ -n "${_HOLA_OLD_CUDA_HOME-}" ]; then export CUDA_HOME="$_HOLA_OLD_CUDA_HOME"; else unset CUDA_HOME; fi
if [ -n "${_HOLA_OLD_TORCH_CUDA_ARCH_LIST-}" ]; then export TORCH_CUDA_ARCH_LIST="$_HOLA_OLD_TORCH_CUDA_ARCH_LIST"; else unset TORCH_CUDA_ARCH_LIST; fi
unset _HOLA_OLD_CUDA_HOME _HOLA_OLD_TORCH_CUDA_ARCH_LIST
EOF
done
