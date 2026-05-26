#!/usr/bin/env bash
# 2026-05-26 数据清洗与迁移
cd /mnt/d/python

# ═══════════════════════════════════════════════════════════════
# 1. Model List 去重 + 完整性检查
# ═══════════════════════════════════════════════════════════════

# Dry-run: 查看重复数量和文件缺失情况
python tools/generate_model_lists.py --dry-run

# 正式输出新 list (train.txt, val.txt, test.txt, excluded.txt)
python tools/generate_model_lists.py

# ═══════════════════════════════════════════════════════════════
# 2. 清理 deepcad_v6 → 复制有用数据到 deepcad_v2 (原数据不动)
# ═══════════════════════════════════════════════════════════════

# Dry-run
python tools/clean_deepcad_v6.py --dry-run

# 小规模测试
python tools/clean_deepcad_v6.py --max-models 10

# 正式 (复制 normalized_shape.step + mesh.ply + data.npz(去imgs) → deepcad_v2/)
python tools/clean_deepcad_v6.py

# ═══════════════════════════════════════════════════════════════
# 3. Debug 验证旋转映射 (人眼检查)
# ═══════════════════════════════════════════════════════════════

python tools/debug_rotation_migration.py --model-id 00261287

# ═══════════════════════════════════════════════════════════════
# 4. 迁移到 deepcad_cond_v2 + ae_cache_24
# ═══════════════════════════════════════════════════════════════

# Dry-run
python tools/migrate_64_to_24.py --dry-run

# 小规模测试
python tools/migrate_64_to_24.py --max-models 10

# 正式全量迁移
python tools/migrate_64_to_24.py
