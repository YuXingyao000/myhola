#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_DEFAULT_TIMEOUT=120
export PIP_RETRIES=10
export PIP_RESUME_RETRIES=10
export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
export PIP_CACHE_DIR=/mnt/d/migration/pip-cache
mkdir -p /mnt/d/migration/logs /mnt/d/migration/installers
exec > >(tee -a /mnt/d/migration/logs/environment-install.log) 2>&1
trap 'printf "Environment installation failed at line %s\n" "$LINENO"' ERR
apt-get update
apt-get install -y --no-install-recommends build-essential cmake ninja-build git rsync tmux zsh curl wget ca-certificates unzip zip p7zip-full libgl1 libegl1 libglib2.0-0 libxrender1 libxext6 libsm6 libxi6 libxkbcommon0 libxkbcommon-x11-0 libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 ffmpeg libgomp1 patchelf
if [[ ! -x /mnt/d/anaconda3/bin/conda ]]; then
  if ! printf '%s  %s\n' 57b2b48cc5b8665e25fce7011f0389d47c1288288007844b3b1ba482d4f39029 /mnt/d/migration/installers/Anaconda3-2025.12-2-Linux-x86_64.sh | sha256sum -c -; then
    flock /mnt/d/migration/installers/anaconda.lock curl -fL --retry 5 --retry-delay 3 -C - https://repo.anaconda.com/archive/Anaconda3-2025.12-2-Linux-x86_64.sh -o /mnt/d/migration/installers/Anaconda3-2025.12-2-Linux-x86_64.sh
  fi
  printf '%s  %s\n' 57b2b48cc5b8665e25fce7011f0389d47c1288288007844b3b1ba482d4f39029 /mnt/d/migration/installers/Anaconda3-2025.12-2-Linux-x86_64.sh | sha256sum -c -
  bash /mnt/d/migration/installers/Anaconda3-2025.12-2-Linux-x86_64.sh -b -p /mnt/d/anaconda3
fi
/mnt/d/anaconda3/bin/conda config --set auto_activate_base false
/mnt/d/anaconda3/bin/conda init bash zsh
if [[ ! -x /mnt/d/anaconda3/envs/torch/bin/python ]]; then
  /mnt/d/anaconda3/bin/conda create -y --override-channels -c conda-forge -n torch python=3.10 pythonocc-core=7.8.1.1 occt=7.8.1 numpy=1.26.4 pip
fi
printf 'pythonocc-core ==7.8.1.1\nocct ==7.8.1\n' > /mnt/d/anaconda3/envs/torch/conda-meta/pinned
torch_wheel='https://download.pytorch.org/whl/cu128/torch-2.8.0%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl'
if [[ -f /mnt/d/migration/wheels/torch-2.8.0+cu128-cp310-cp310-manylinux_2_28_x86_64.whl ]]; then
  printf '%s  %s\n' 0c96999d15cf1f13dd7c913e0b21a9a355538e6cfc10861a17158320292f5954 /mnt/d/migration/wheels/torch-2.8.0+cu128-cp310-cp310-manylinux_2_28_x86_64.whl | sha256sum -c -
  torch_wheel=/mnt/d/migration/wheels/torch-2.8.0+cu128-cp310-cp310-manylinux_2_28_x86_64.whl
fi
torchvision_wheel='https://download.pytorch.org/whl/cu128/torchvision-0.23.0%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl'
torchaudio_wheel='https://download.pytorch.org/whl/cu128/torchaudio-2.8.0%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl'
for component in torchvision torchaudio; do
  if [[ $component == torchvision && -f /mnt/d/migration/wheels/torchvision-0.23.0+cu128-cp310-cp310-manylinux_2_28_x86_64.whl ]]; then torchvision_wheel=/mnt/d/migration/wheels/torchvision-0.23.0+cu128-cp310-cp310-manylinux_2_28_x86_64.whl; fi
  if [[ $component == torchaudio && -f /mnt/d/migration/wheels/torchaudio-2.8.0+cu128-cp310-cp310-manylinux_2_28_x86_64.whl ]]; then torchaudio_wheel=/mnt/d/migration/wheels/torchaudio-2.8.0+cu128-cp310-cp310-manylinux_2_28_x86_64.whl; fi
done
runtime_wheels=()
if [[ -f /mnt/d/migration/wheels/runtime-download.complete ]]; then
  runtime_wheels=(/mnt/d/migration/wheels/nvidia*.whl /mnt/d/migration/wheels/triton-3.4.0*.whl)
fi
/mnt/d/anaconda3/envs/torch/bin/python -m pip install "$torch_wheel" "$torchvision_wheel" "$torchaudio_wheel" "${runtime_wheels[@]}"
/mnt/d/anaconda3/envs/torch/bin/python -m pip install --no-index --no-deps pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
open3d_wheel='open3d==0.19.0'
if [[ -f /mnt/d/migration/wheels/open3d-download.complete ]]; then
  open3d_wheel=$(find /mnt/d/migration/wheels -maxdepth 1 -name 'open3d-0.19.0-cp310-*.whl' -print -quit)
fi
/mnt/d/anaconda3/envs/torch/bin/python -m pip install 'numpy==1.26.4' 'torch-geometric==2.6.1' 'pytorch-lightning==2.5.0.post0' 'hydra-core==1.3.2' 'omegaconf==2.3.0' 'hydra-colorlog==1.2.0' 'diffusers==0.37.1' 'transformers==4.57.6' 'sentence-transformers==3.3.1' 'accelerate==1.13.0' 'gguf>=0.10' sentencepiece protobuf einops trimesh "$open3d_wheel" wandb tensorboard scipy 'plyfile==1.1' h5py tqdm rich 'opencv-python==4.10.0.84' matplotlib 'ray==2.40.0' geomdl psutil 'scikit-image==0.25.0' 'scikit-learn==1.7.0' point-cloud-utils rtree lightning-utilities pytest ninja setuptools wheel packaging imageio iopath
/mnt/d/anaconda3/envs/torch/bin/python -m pip check
if [[ ! -x /mnt/d/anaconda3/envs/generate/bin/python ]]; then
  /mnt/d/anaconda3/bin/conda create -y --override-channels -c conda-forge -n generate --clone torch
fi
/mnt/d/anaconda3/bin/conda list -n torch --explicit > /mnt/d/migration/logs/torch-conda-explicit.txt
/mnt/d/anaconda3/envs/torch/bin/python -m pip freeze > /mnt/d/migration/logs/torch-pip-freeze.txt
printf 'Initial environment installation completed.\n'
touch /mnt/d/migration/environment-bootstrap.complete
