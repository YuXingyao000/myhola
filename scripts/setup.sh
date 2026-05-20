#!/bin/bash
# ============================================================
# Environment Setup
# ============================================================
# Run once on a new machine.

set -e

echo "[1/4] Creating conda environment..."
conda env create -f environment.yml
conda activate hola-brep

echo "[2/4] Installing project in editable mode..."
pip install -e .

echo "[3/4] Building PointNet++ CUDA ops..."
cd thirdparty/Pointnet2_PyTorch/pointnet2_ops_lib
pip install -e .
cd ../../..

echo "[4/4] Verifying imports..."
python -c "
from src.brepnet.models import build_model, build_strategy, build_condition_extractor
from src.brepnet.models.vae import AutoEncoder_1119
from src.brepnet.models.diffusion import DiffusionCrossAttn
from src.brepnet.models.strategies import KnowledgeDistillation, FeatureDomainMapper
from src.brepnet.models.condition_encoders import ConditionExtractor
print('All imports OK!')
"

echo "Setup complete. Edit scripts/train.sh paths, then run:"
echo "  bash scripts/train.sh feature_mapper"
