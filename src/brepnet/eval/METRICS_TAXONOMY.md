# Evaluation 指标分类与接口设计

## 四类指标（对应论文章节）

### Category A: 几何重建质量 (Conditioned Reconstruction)

**论文用途**: Main table, 和 baseline 对比的核心数据

| 指标 | 定义 | 单位 | 越X越好 |
|------|------|------|---------|
| Face CD | pred faces 和 GT faces 之间的 Chamfer Distance（Hungarian 匹配后） | ×10⁻³ | 低 |
| Edge CD | pred edges 和 GT edges 之间的 Chamfer Distance | ×10⁻³ | 低 |
| Vertex CD | pred vertices 和 GT vertices 之间的 Chamfer Distance | ×10⁻³ | 低 |
| Face F1 | Face precision × recall 的调和平均（CD < τ 算 match） | % | 高 |
| Edge F1 | 同上，edge 级别 | % | 高 |
| Vertex F1 | 同上，vertex 级别 | % | 高 |
| Validity | 生成的 STEP 是否是合法 solid（BRepCheck 通过） | % | 高 |
| Face Count Δ | |pred_faces| - |gt_faces| 的绝对误差 | 个 | 低 |

**输入**: pred `.step` + GT `.step`（或 GT `data.npz`）
**依赖**: PythonOCC, chamferdist
**Rotation 敏感**: ⚠️ YES — 必须对齐 pred 和 GT 的坐标系

---

### Category B: 生成分布质量 (Unconditional / Distribution)

**论文用途**: 证明生成的多样性和分布覆盖，通常用于 unconditional 生成

| 指标 | 定义 | 越X越好 |
|------|------|---------|
| MMD-CD | 最小匹配距离（生成集 → 参考集的最近邻 CD 均值） | 低 |
| COV-CD | 参考集中被生成集"覆盖"的比例（CD < τ） | 高 |
| JSD | 占据体素的 Jensen-Shannon 散度 | 低 |

**输入**: 两组点云 PLY（各 2000 点/模型）
**依赖**: chamfer_distance, sklearn
**Rotation 敏感**: 弱 — PCA 自动对齐

---

### Category C: 数据集质量 (Dataset / FLUX Generation)

**论文用途**: 数据集贡献章节，证明生成的真实照片质量足够好

| 指标 | 定义 | 比较对象 | 越X越好 |
|------|------|---------|---------|
| Silhouette IoU | SAM2(FLUX) mask vs OCC mask 的 IoU | FLUX ↔ OCC | 高 (>0.7) |
| DINO Cosine | DINOv2 CLS token 余弦相似度 | FLUX ↔ OCC | 高 (>0.5) |
| CLIP Realism | CLIP(FLUX, "a real photograph...") | FLUX ↔ 文字 | 高 (>0.25) |

**输入**: `imgs.npz` (OCC renders) + `real_photo.npz` (FLUX outputs)
**依赖**: SAM2, DINOv2, CLIP
**Rotation 敏感**: 否（2D 图像指标）

---

### Category D: 多样性与新颖性 (Diversity / Novelty)

**论文用途**: 补充 table，证明不是 mode collapse

| 指标 | 定义 | 越X越好 |
|------|------|---------|
| Uniqueness | 生成结果中不重复的比例（VF2 图同构去重） | 高 |
| Novelty | 生成结果中和训练集不重复的比例 | 高 |
| LFD | Light Field Descriptor 距离（旋转不变） | — (描述性) |

**输入**: 所有 pred `.step` + 训练集 `.step`
**依赖**: PythonOCC, NetworkX
**Rotation 敏感**: 否（图同构和 LFD 都旋转不变）

---

## 当前 Hidden Knowledge 清单

以下是会坑人但没有文档的隐式假设：

| Hidden Knowledge | 在哪里 | 坑点 |
|-----------------|--------|------|
| `rotation_id=0` 是 identity | `data/rotations.py` | 运行时统一使用 identity-first cube24 |
| `imgs.npz` 运行时只应有 24 个视角 | `tools/migrate_64_to_24.py` | 旧 Euler64 数据只能在迁移/调试工具里出现 |
| cached latent 文件名包含 rotation_id | `ae_cache/{model}_{rotation_id}/` | 必须使用迁移后的 24-id cache |
| `eval_brep.py` 读 `recon_face/*.stl`，当前 pipeline 不输出这个 | `post/legacy/eval_brep.py` | 用了就得到全 0 结果 |
| `condition.py` 的 F1 阈值是硬编码的 | `metrics/condition.py` | 不同论文用不同 τ |

---

## 理想接口设计（Rotation 定了之后实现）

```python
# 未来目标: 一条命令跑完所有需要的指标
python -m src.brepnet.eval.run \
    --pred-root /path/to/post_output \
    --gt-root /mnt/d/data/deepcad_v6 \
    --split-list src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt \
    --metrics condition,validity,complexity,unique \
    --output eval_results/

# Rotation 固定为 identity-first cube24:
#   rotation_id=0 is identity
#   eval condition metric always uses identity GT
```

---

## Rotation 整理的依赖关系

```
数据集重排 (64→24)
    ↓
迁移 cached latent (以 24 旋转命名)
    ↓
修改 dataset.py (cube_id 直接对应 latent 文件)
    ↓
inference 默认 identity 条件
    ↓
eval 固定 identity GT
```
