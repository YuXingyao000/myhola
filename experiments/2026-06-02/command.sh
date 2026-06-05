# --- 数据修复 ---
cd /mnt/d/python && python tools/fix_corrupted_npz.py --scan-only
cd /mnt/d/python && python tools/fix_corrupted_npz.py --fix

# --- DETR-style topology predictor (方向 A) ---
# 和 svr_single (CLS+MLP) 对照: patch tokens + face queries + cross-attention
cd /mnt/d/python && python experiments/2026-06-02/topology_from_svr_single_detr.py \
    --output-dir experiments/2026-06-02/outputs_svr_single_detr \
    --epochs 20 --batch-size 32

# --- Autoregressive topology transformer (方向 C) ---
# image -> edge token sequence (cross-entropy LM training)
cd /mnt/d/python && python experiments/2026-06-02/topology_ar_train.py \
    --output-dir experiments/2026-06-02/outputs_ar \
    --epochs 30 --batch-size 32

# --- Learned topology diffusion (用 toy svr_single 嫁接) ---
cd /mnt/d/python && python -m src.brepnet.train \
    --config-name train_diffusion_learned_topology


