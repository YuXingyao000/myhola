#!/bin/bash
# ============================================================
# Strategy A: Knowledge Distillation
# ============================================================
# Teacher(白模conditioned, frozen) 教 Student(真实照片conditioned)
# Student 必须在每个 timestep 产出和 Teacher 一样的去噪方向
# 需要1-2天，但效果比 Feature Mapper 强
# ============================================================

set -e

# ─── 路径配置（改成你的） ─────────────────────────────────
DATA_ROOT="/path/to/your/data"
FACE_Z_DIR="/path/to/cached_latents"        # 预缓存的 VAE latent features
COND_ROOT="/path/to/condition_data"         # 条件数据
VAE_CKPT="/path/to/vae_checkpoint.ckpt"     # 预训练的 VAE 权重
TEACHER_CKPT="/path/to/white_model_diffusion.ckpt"  # ← 关键：白模conditioned的diffusion权重
OUTPUT_DIR="./outputs/distillation"

# ─── 训练参数 ─────────────────────────────────────────────
GPUS=4                 # 蒸馏比较重，建议多卡
BATCH_SIZE=16          # 需要同时跑 teacher + student，显存吃紧
LR=5e-5               # 比从头训低一些（因为student从teacher初始化）
MAX_EPOCHS=500
EXP_NAME="distill_real_photo_v1"

# ─── Loss 权重 ────────────────────────────────────────────
KD_WEIGHT=1.0          # MSE(student_pred, teacher_pred) 的权重
DIFFUSION_WEIGHT=1.0   # MSE(student_pred, noise) 的权重
ALIGN_WEIGHT=0.5       # CLIP-style alignment loss 的权重

# ─── 开始训练 ─────────────────────────────────────────────
echo "============================================"
echo " Strategy A: Knowledge Distillation"
echo " Teacher: $TEACHER_CKPT"
echo " Student: 从Teacher初始化，只重置cross-attn层"
echo " 数据: 100% 真实照片"
echo "============================================"

python -m src.brepnet.train \
    experiment=train_distill \
    dataset.data_root="$DATA_ROOT" \
    dataset.face_z_dir="$FACE_Z_DIR" \
    dataset.cond_root="$COND_ROOT" \
    dataset.real_photo_ratio=1.0 \
    model.autoencoder.weights="$VAE_CKPT" \
    strategy.teacher.checkpoint="$TEACHER_CKPT" \
    strategy.loss.kd_weight=$KD_WEIGHT \
    strategy.loss.diffusion_weight=$DIFFUSION_WEIGHT \
    strategy.loss.align_weight=$ALIGN_WEIGHT \
    trainer.gpus=$GPUS \
    trainer.batch_size=$BATCH_SIZE \
    trainer.learning_rate=$LR \
    trainer.max_epochs=$MAX_EPOCHS \
    trainer.exp_name="$EXP_NAME" \
    trainer.output_dir="$OUTPUT_DIR"

echo ""
echo "============================================"
echo " 训练完成！"
echo " Student checkpoint 在: $OUTPUT_DIR/$EXP_NAME/checkpoints/"
echo ""
echo " 推理时直接用 Student 模型即可（它已经学会了处理真实照片）"
echo "============================================"
