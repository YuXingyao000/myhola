# 2026-05-26 实验笔记

## 今日目标

数据清洗与迁移：将混乱的旧数据整理到新目录结构，64-rotation → 24-rotation。

## 背景

旧数据问题：
- `deepcad_v6/{model_id}/` 里有无用的 pc.ply、post_processed_shape.step、data.npz 含冗余 imgs
- `deepcad_v6_cond/{model_id}/imgs.npz` 存了 64 张图（只有 24 张不重复）
- `ae_cache/1119_deepcad_aug1_11k/{model}_{euler64_id}/` 用 euler64 命名
- train/val/test list 之间有重复的 model_id

新数据结构（迁移后）：
```
/mnt/d/data/deepcad_v2/{model_id}/           ← GT 数据 (只保留有用的)
  normalized_shape.step, mesh.ply, data.npz(无imgs)

/mnt/d/data/deepcad_cond_v2/{model_id}/      ← 条件数据 (24旋转, 清晰命名)
  svr.npz, sketch.npz, mvr.npz, real_photo.npz, pc.ply, text.npz

/mnt/d/data/ae_cache/1119_deepcad_aug1_11k_24/{model}_{cube24_id}/  ← Latent cache
  features.npy, data.npz
```

---

## 运行顺序与命令

### Step 1: Model List 去重 + 完整性检查

**目的**：发现 train/val/test 之间的重复 ID，按 training > validation > testing 优先级去重；同时检查哪些模型的文件不完整（缺 data.npz 或 ae_cache）。

```bash
cd /mnt/d/python

# 先 dry-run 看看重复数量和缺失情况
python tools/generate_model_lists.py --dry-run

# 确认无误后输出新 list
python tools/generate_model_lists.py
# 输出: src/brepnet/data/list/{train.txt, val.txt, test.txt, excluded.txt}
```

### Step 2: Debug 验证旋转映射

**目的**：对一个模型输出 debug 目录，人眼确认 cube24 ↔ euler64 映射正确，确保迁移逻辑没错。

```bash
python tools/debug_rotation_migration.py --model-id 00261287
# 输出: debug_output/00261287/
#   cube24_images/         ← 迁移后的 24 张图
#   euler64_mapping/       ← 每个 cube24 对应哪些 euler64 重复图(应该一模一样)
#   result_sample/         ← 最终 npz 解压后的样例
#   mapping_table.txt      ← 映射表
```

检查要点：
- `euler64_mapping/cube24_00/` 下的多张图应该**完全一样**
- `cube24_images/XX.png` 和 `euler64_mapping/cube24_XX/` 中的图应该一致
- `mapping_table.txt` 确认 cube24_id=0 → euler64_id=4（不是 0！）

### Step 3: 清理 deepcad_v6 → deepcad_v2

**目的**：从旧 GT 数据中只复制有用的文件（normalized_shape.step + mesh.ply + data.npz 去掉 imgs）到新目录。原数据不动。

```bash
# dry-run
python tools/clean_deepcad_v6.py --dry-run

# 小规模测试 (10 个模型, 串行)
python tools/clean_deepcad_v6.py --max-models 10 --no-ray

# 正式全量 (Ray 并行, 32 CPU)
python tools/clean_deepcad_v6.py --num-cpus 32
```

### Step 4: 迁移条件数据 + ae_cache (64→24)

**目的**：
- 从 `imgs.npz[64]` 提取 24 张独立图 → 拆分为 `svr.npz` + `sketch.npz` + `mvr.npz`
- `single_view.npz` → `real_photo.npz`
- `text.txt` + `text_feat.npy` → `text.npz`
- 复制 `pc.ply`
- `ae_cache/{model}_{euler64_id}/` → `ae_cache_24/{model}_{cube24_id}/`
- **如果某模型的 24 个旋转中任何一个缺 features.npy，整个模型跳过**

```bash
# dry-run
python tools/migrate_64_to_24.py --dry-run

# 小规模测试 (10 个模型, 串行)
python tools/migrate_64_to_24.py --max-models 10 --no-ray

# 正式全量 (Ray 并行, 32 CPU)
python tools/migrate_64_to_24.py --num-cpus 32
```

---

## 依赖

| 条件 | 路径 |
|------|------|
| 旧 GT 数据 | `/mnt/d/data/deepcad_v6/{model_id}/` |
| 旧条件数据 | `/mnt/d/data/deepcad_v6_cond/{model_id}/` |
| 旧 ae_cache | `/mnt/d/data/ae_cache/1119_deepcad_aug1_11k/{model_id}_{euler64_id}/` |
| Model lists | `src/brepnet/data/list/deduplicated_deepcad_*.txt` |
| scipy | `pip install scipy` |
| ray | `pip install ray` |

## 输出

| 新数据 | 路径 |
|--------|------|
| GT 数据 | `/mnt/d/data/deepcad_v2/{model_id}/` |
| 条件数据 | `/mnt/d/data/deepcad_cond_v2/{model_id}/` |
| Latent cache | `/mnt/d/data/ae_cache/1119_deepcad_aug1_11k_24/{model_id}_{cube24_id}/` |
| 新 model lists | `src/brepnet/data/list/{train.txt, val.txt, test.txt}` |
| 排除模型 | `src/brepnet/data/list/excluded.txt` |

## 关键映射事实

- `cube24_id=0` → `euler64_id=4`
- `euler64_id=0` → `cube24_id=18`
- 每个 cube24 有 2~4 个 euler64 重复

## ⚠️ 发现的严重 Bug：白模训练时图片-Latent 不匹配

**问题**：`dataset.py` 中 `prepare_condition` 用 `svr_imgs[cube_id]` 索引图片，但 `imgs.npz` 里的 `svr_imgs` 是按 euler64 顺序存储的。这导致：

```
cube_id = 0
latent  = ae_cache/{model}_4/features.npy     ← euler64=4 = cube24_id=0 的旋转 ✓
image   = imgs.npz["svr_imgs"][0]             ← euler64=0 = cube24_id=18 的旋转 ✗
```

图片是一个视角，latent 是另一个视角，两者不匹配。

**影响范围**：
- 所有 `real_photo_ratio=0.0`（纯白模）的训练 — 图片和 latent 视角错配
- 所有 `is_aug=1`（随机旋转增强）的训练 — 同样错配
- `real_photo_ratio=1.0`（FLUX）— **不受影响**（FLUX 是单张图，来自 Blender view 0 = cube24_id=0，和 latent 对得上）

**为什么之前没发现**：DINOv2 对旋转有一定容忍度（同一物体不同视角的 CLS token 仍然接近），loss 仍然下降，只是不是最优。

**修复**：迁移到 `svr.npz["images"][cube24_id]` 后自动修复——新数据按正确的 cube24 顺序存储。

同时修改 eval 的 `OCTAHEDRAL_ROTATIONS` 对齐 dataset/Blender 顺序，彻底消除 rotation_id hidden knowledge。

## 后续（今天不做）

- [ ] 修改 `dataset.py` 让代码读新目录
- [ ] VAE 补充缺失的 ae_cache（对 excluded.txt 中 `missing_ae_cache` 的模型）
- [ ] 验证通过后删除旧数据释放磁盘空间

## 结论
发现blender和svr cube24 的0号旋转直接还是有偏差，现在怀疑是blender的坐标系，和OCC的坐标系不一致导致的，似乎把OCC图片向顺时针方向旋转90度就对的上了。不对并没有这么简单。
blender的0号旋转对应的是cube24的18号旋转
这就比较难办，可能必须考虑从新生成svr，blender和FLUX数据集太重了，没法从新生成，svr可以在1小时内从新生成完成，不过应该还是迁移会更方便