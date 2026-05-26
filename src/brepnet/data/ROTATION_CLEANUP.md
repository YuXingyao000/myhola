# 旋转相关代码清单 (待数据迁移后清理)

## 现状总结

- **Blender/dataset/rotations.py** 已统一为 cube24 顺序（index 0 = 180° Y-rot, identity at 18）
- **eval `run.py`** 已改为 `--rotation-id 0`（对齐 cube24_id=0）
- **数据迁移后**：ae_cache 和 imgs 都按 cube24_id 命名，不再需要 euler64 映射

## 需要清理的文件（数据迁移完成后）

### 1. `src/brepnet/dataset.py`

| 行号 | 内容 | 改动 |
|------|------|------|
| 136-153 | 大段注释解释 euler64 → cube24 映射历史 | 删除，换成简短注释 |
| 164-180 | `_build_cube24_rotation_matrices()` | 删除，改为 `from .data.rotations import OCTAHEDRAL_ROTATIONS` |
| 183-188 | `_build_euler64_rotation_matrices()` | 删除 |
| 191-204 | `_build_cube24_to_euler64_mapping()` | 删除 |
| 207 | `CUBE24_TO_EULER64` 常量 | 删除 |
| 208 | `NUM_CUBE24_VIEWS = 24` | 改为从 rotations 导入 |
| 209-211 | `CUBE24_ROTATION_MATRICES` | 删除，用 `OCTAHEDRAL_ROTATIONS` |
| 214-216 | `cube24_to_euler64()` 函数 | 删除 |
| 645 | `id_aug = cube24_to_euler64(cube_id)` | 改为 `id_aug = cube_id`（直接用 cube24_id 加载） |
| 646 | `latent_root / (folder + f"_{id_aug}")` | 改为 `latent_root / (folder + f"_{cube_id}")` |
| 290 | `svr_imgs[v_id_aug]` 错误索引（用 cube_id 索引 euler64 数组） | 迁移后读 `svr.npz["images"][cube_id]` → 自动正确 |
| 426 | `legacy_aug_id = cube24_to_euler64(cube_id)` | 直接用 cube_id |
| 429 | output prefix `f"{folder}_{legacy_aug_id}"` | 改为 `f"{folder}_{cube_id}"` |
| 761 | Diffusion_dataset_mm 同样的映射 | 同上 |

### 2. `src/brepnet/models/condition_encoders.py`

| 行号 | 内容 | 改动 |
|------|------|------|
| 449-452 | docstring 写 euler64 分解公式 | 改为 cube24 说明 |
| 459-468 | `euler_id % 4, euler_id // 4 % 4, euler_id // 16` bit-pattern 分解 | 改为从 `OCTAHEDRAL_ROTATIONS[cube_id]` 查表取旋转矩阵 |
| 709-710 | `euler_id = data.get("id_aug")` | key 可以保持，但值含义从 euler64 变为 cube24 |

### 3. `src/brepnet/data/rotations.py`

| 内容 | 改动 |
|------|------|
| `legacy_64_rotation_matrix()` | 保留但标记为 deprecated（迁移工具还要用） |
| `cube24_to_euler64_mapping()` | 同上 |
| `euler64_to_cube24_mapping()` | 同上 |

迁移完毕且旧数据删除后，这些 legacy 函数可以完全删除。

### 4. `src/brepnet/data/extract_imgs.py`

| 行号 | 内容 | 改动 |
|------|------|------|
| 106-127 | `for i in range(64)` + euler 分解 | 整个文件废弃（不再需要生成 64 视角图片） |
| 135-155 | sketch 同上 | 废弃 |
| 184-205 | mvr 同上 | 废弃 |

这个文件在新 pipeline 中不再使用（Blender 已经替代了 OCC 渲染图生成）。

### 5. `src/brepnet/test/viz_mvr_data.py`

| 行号 | 内容 | 改动 |
|------|------|------|
| 37-41 | euler64 分解 + Rotation.from_euler | 改为用 `OCTAHEDRAL_ROTATIONS[id]` |

### 6. `experiments/2026-05-22/noise_sensitivity_retest/latent_sensitivity.py`

| 行号 | 内容 | 改动 |
|------|------|------|
| 101, 199 | `f"{prefix}_{euler_id}/features.npy"` | 迁移后改为 `f"{prefix}_{cube24_id}/features.npy"` |

### 7. `src/brepnet/eval/metrics/condition.py`

| 行号 | 内容 | 改动 |
|------|------|------|
| 26 | `from src.brepnet.data.rotations import OCTAHEDRAL_ROTATIONS` | 不需要改（已对齐） |
| 350, 390, 433 | 使用 `OCTAHEDRAL_ROTATIONS[rotation_index]` | 不需要改（顺序已对齐） |

### 8. `src/brepnet/eval/run.py`

| 行号 | 内容 | 改动 |
|------|------|------|
| 143-144 | `--rotation-id default=0` | 已改好 ✓ |

## 清理顺序

```
1. 数据迁移完成 (svr.npz/ae_cache_24 按 cube24_id)
2. 修改 dataset.py 读新格式 (直接用 cube24_id, 删除映射代码)
3. 修改 condition_encoders.py (查表替代 bit-pattern)
4. 删除 extract_imgs.py (或移到 legacy/)
5. 清理 rotations.py 里的 legacy 函数
6. 跑一遍训练 sanity check 确认不 break
```
