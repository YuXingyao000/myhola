#!/bin/bash
# ============================================================
# Two-Stage Pipeline (Image → Point Cloud → BRep)
# ============================================================
# 兜底方案：用外部模型先把图片变成点云，再用已有的PC-conditioned diffusion
# 不需要训练，只需要推理
# ============================================================

set -e

# ─── 路径配置（改成你的） ─────────────────────────────────
INPUT_IMAGES="/path/to/real_photos"             # 输入的真实照片目录
PC_OUTPUT="/path/to/generated_point_clouds"     # Stage 1 输出的点云
BREP_OUTPUT="./outputs/two_stage"               # 最终 BRep 输出

PC_DIFFUSION_CKPT="/path/to/pc_conditioned_diffusion.ckpt"  # 点云conditioned的diffusion权重
VAE_CKPT="/path/to/vae_checkpoint.ckpt"

# ============================================================
# Stage 1: Image → Point Cloud
# ============================================================
# 用 TRELLIS 或 InstantMesh 从真实照片生成粗糙3D
# 这一步需要你手动安装对应的模型
# ============================================================

echo "============================================"
echo " Stage 1: Image → Point Cloud"
echo " 使用外部 Image-to-3D 模型"
echo "============================================"

# --- 选一个你安装好的 ---

# 方式1: TRELLIS (Microsoft, 推荐)
# pip install trellis3d
# python -m trellis.generate \
#     --input "$INPUT_IMAGES" \
#     --output "$PC_OUTPUT" \
#     --format pointcloud \
#     --num_points 10000

# 方式2: InstantMesh
# python third_party/InstantMesh/run.py \
#     --input_dir "$INPUT_IMAGES" \
#     --output-dir "$PC_OUTPUT" \
#     --export_pointcloud

# 方式3: 如果你已经有其他方式生成的点云，直接放到 PC_OUTPUT 目录下
# 格式: 每个文件 .ply 或 .npy, 包含 [N, 6] (xyz + normals)

echo ""
echo "请确认点云已生成在: $PC_OUTPUT"
echo "按 Enter 继续 Stage 2..."
read -r

# ============================================================
# Stage 2: Point Cloud → BRep
# ============================================================
# 用 HoLa-BRep 已训好的 PC-conditioned diffusion 生成 BRep
# ============================================================

echo "============================================"
echo " Stage 2: Point Cloud → BRep"
echo " 使用 PC-conditioned Diffusion"
echo "============================================"

python -m src.brepnet.inference \
    --diffusion-weights "$PC_DIFFUSION_CKPT" \
    --autoencoder-weights "$VAE_CKPT" \
    --condition point_cloud \
    --input "$PC_OUTPUT" \
    --output-dir "$BREP_OUTPUT/raw" \
    --num-samples 1

# ============================================================
# Post-processing: 控制点 → STEP 文件
# ============================================================

echo "============================================"
echo " Post-processing: 生成 STEP 文件"
echo "============================================"

python -m src.brepnet.post.construct_brep \
    --input "$BREP_OUTPUT/raw" \
    --output "$BREP_OUTPUT/step"

echo ""
echo "============================================"
echo " 完成！"
echo " 生成的 BRep (npz): $BREP_OUTPUT/raw/"
echo " 生成的 STEP 文件:  $BREP_OUTPUT/step/"
echo ""
echo " 评估："
echo "   python -m src.brepnet.eval.run \\"
echo "       --pred_dir $BREP_OUTPUT/step \\"
echo "       --gt_dir /path/to/ground_truth_step"
echo "============================================"
