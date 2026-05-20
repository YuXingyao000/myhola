#!/bin/bash
# ============================================================
# Strategy B: Feature Domain Mapper
# ============================================================
# 最快验证的方案（几小时出结果）
# 训一个小transformer，把真实照片的DINOv2特征映射到白模特征空间
# 训完后直接插入原有pipeline推理，不需要重训diffusion
# ============================================================

set -e

# ─── 路径配置（改成你的） ─────────────────────────────────
DATA_ROOT="/path/to/your/data"              # 数据根目录
COND_ROOT="/path/to/condition_data"         # 条件数据（包含 single_view.npz 和 imgs.npz）
OUTPUT_DIR="./outputs/feature_mapper"       # 输出目录

# ─── 训练参数 ─────────────────────────────────────────────
GPUS=1                 # 这个模型小，1卡够了
BATCH_SIZE=64          # 可以开大，模型小
LR=1e-4
MAX_EPOCHS=100         # 通常50-100 epoch就收敛了
EXP_NAME="feature_mapper_v1"

# ─── 开始训练 ─────────────────────────────────────────────
echo "============================================"
echo " Strategy B: Feature Domain Mapper"
echo " 训练 DINOv2(real_photo) → DINOv2(white_model) 的映射"
echo "============================================"

python -m src.brepnet.train \
    experiment=train_feature_mapper \
    dataset.data_root="$DATA_ROOT" \
    dataset.cond_root="$COND_ROOT" \
    trainer.gpus=$GPUS \
    trainer.batch_size=$BATCH_SIZE \
    trainer.learning_rate=$LR \
    trainer.max_epochs=$MAX_EPOCHS \
    trainer.exp_name="$EXP_NAME" \
    trainer.output_dir="$OUTPUT_DIR"

echo ""
echo "============================================"
echo " 训练完成！"
echo " Checkpoint 在: $OUTPUT_DIR/$EXP_NAME/checkpoints/"
echo ""
echo " 下一步：用训好的mapper做推理"
echo "   python -c \""
echo "   from src.brepnet.models.strategies import FeatureDomainMapper"
echo "   mapper = FeatureDomainMapper.load_trained('$OUTPUT_DIR/$EXP_NAME/checkpoints/best.pt')"
echo "   mapper.plug_into_pipeline(your_diffusion_model)"
echo "   \""
echo "============================================"
