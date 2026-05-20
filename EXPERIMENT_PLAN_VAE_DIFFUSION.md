# VAE & Diffusion 实验计划

## 核心发现回顾

| 实验 | 结论 |
|------|------|
| TokenVAE (enc+dec全替换) | 拓扑loss↓10x, 几何coord loss↑10x → attention不适合精确空间重建 |
| TokenEnc + ConvDec | 同上趋势 → 问题在encoder破坏了latent的空间结构 |
| Addition Tag | 彻底失败 → diffusion不能生成离散值 |
| ControlNet条件注入 | valid rate↑但Chamfer不动 → 条件特征缺几何信息 |
| KL=1e-6 | latent space可能不regular, diffusion偏离时decode崩溃 |
| pad_method=random + dedup | 面数错误时valid rate从0.89暴跌到0.24 |

## 结论

- **Conv encoder/decoder 不要动** — 它做局部空间重建的能力是对的
- **Attention/Graph 中间层可以加强** — 拓扑/结构已经受益于attention
- **KL weight 需要调** — 可能是 diffusion 生成偏离的根因之一
- **Padding 策略要改** — zero-pad + classifier 比 random-pad + dedup 更原则性
- **离散信号不混入diffusion** — 面数预测/mask 用单独模块

---

## Phase 1: VAE Latent Space 诊断 (1-2天)

### Exp V0: Latent 分布统计

**目标**：确认 latent space 有多不 regular。

```python
# 遍历训练集，收集所有 face_z
all_z = collect_all_latents(vae, train_dataloader)  # [total_faces, 32]

# 统计
print(f"Global mean: {all_z.mean():.4f}")       # 理想: ~0
print(f"Global std:  {all_z.std():.4f}")        # 理想: ~1
print(f"Per-dim std: min={all_z.std(0).min():.4f}, max={all_z.std(0).max():.4f}")
print(f"Kurtosis: {kurtosis(all_z)}")           # 理想: ~3 (正态)

# 可视化: t-SNE / PCA 看 latent 分布形状
# 可视化: 每个维度的直方图（是否接近高斯）
```

**预期结果**：
- 如果 std << 1 → latent 挤在很小区域，diffusion从N(0,1)出发太远
- 如果某些维度std接近0 → 那些维度是"死的"，信息容量浪费
- 如果分布多峰 → latent space有"洞"，diffusion可能采到洞里

**产出**：一张图 + 一组数字，决定 KL sweep 的必要性。

---

## Phase 2: KL Weight Sweep (3-5天训练)

### Exp V1: 训练多个不同 KL weight 的 VAE

| 编号 | gaussian_weights | 预期 |
|------|-----------------|------|
| V1-a | 1e-6 (现在) | baseline，重建好但latent不regular |
| V1-b | 1e-4 | 轻微正则，重建可能微降 |
| V1-c | 1e-3 | 中度正则，重建有损失但latent更regular |
| V1-d | 1e-2 | 强正则，重建明显变差但latent接近N(0,1) |

**每个 VAE 都评估**：
1. 重建 Chamfer (GT → encode → decode → CD)
2. Latent 分布统计 (mean, std, kurtosis)
3. Latent space 插值质量 (两个物体间插值是否平滑)

**决策逻辑**：
```
找到 KL sweet spot:
  重建 CD 仍在 0.01 以内 且 latent std 最接近 1.0 的那个
  → 这就是最适合 diffusion 的 VAE
```

---

## Phase 3: Padding 策略改进 (与 Phase 2 可并行)

### Exp V2: Zero-Padding + Classifier

**不需要重训 VAE。** 只需要改 diffusion 训练时的数据准备方式。

| 对比项 | random-pad + dedup | zero-pad + classifier |
|--------|-------------------|----------------------|
| 训练 | 复制真实face填满30 | 真实face放前面，后面填0 |
| 推理 | 生成30 → distance去重 | 生成30 → classifier判断keep/drop |
| 面数依赖 | 阈值1e-2经验值 | classifier sigmoid>0.5 |

```bash
# 用 zero-pad 重训 diffusion（VAE不变）
python -m src.brepnet.train \
    model=diffusion_cross_attn \
    model.pad_method=zero \
    ...  # 其他和之前一样
```

**评估对比**：
- Valid rate (尤其是面数 over/under 分布)
- 是否消除了去重阈值敏感性

### Exp V3: Face Count Predictor

**额外训一个小模型**：从 image features 预测面数。

```python
class FaceCountPredictor(nn.Module):
    # image_features [B, 257, 1024] → pooling → MLP → softmax(30)
    # 训练: CE loss with GT face count
    # 推理: 预测面数 → 只生成/保留那么多 token
```

这个可以和 Exp V2 的 classifier 结合——双重保险：
1. 先预测面数 N
2. 生成 30 个 → classifier 筛选 → 取 top-N 个 confidence 最高的

---

## Phase 4: Diffusion + Best VAE 联合验证 (1-2天)

### Exp D1: 在最佳 KL 的 VAE 上训 Diffusion

选 Phase 2 中最佳的 VAE（比如 V1-c），重新：
1. 提取所有训练样本的 cached latent (face_z)
2. 用这些新 latent 训 diffusion
3. 对比原来 V1-a 的 diffusion

**关键对比**：

| 指标 | Diffusion on V1-a (KL=1e-6) | Diffusion on V1-c (KL=1e-3) |
|------|------------------------------|------------------------------|
| Valid Rate | ? | ? |
| Face CD | ? | ? |
| F1 | ? | ? |

如果 V1-c 明显更好 → 证明 latent regularity 确实是瓶颈。

### Exp D2: Zero-pad Diffusion on Best VAE

Exp D1 + Exp V2 叠加：
- 最佳 KL 的 VAE
- Zero-padding + classifier
- 完整 eval

---

## Phase 5: Condition Fusion on Best VAE+Diffusion

**前提**：Phase 4 确定了最佳的 VAE + padding 组合。

在这个基础上，再叠加 condition fusion 改进：

### Exp C1: Feature Mapper on Best Pipeline

```
Best VAE (KL=1e-3) + Zero-pad Diffusion + Feature Mapper
```

### Exp C2: Knowledge Distillation on Best Pipeline

```
Best VAE (KL=1e-3) + Zero-pad Diffusion + KD
```

### Exp C3: Topology Attention Bias on Best Pipeline

```
Best VAE (KL=1e-3) + Zero-pad Diffusion + Topology Bias + Feature Mapper
```

---

## Phase 6: VAE 中间层加强 (如果前面不够)

只有当 Phase 1-5 的结果显示"VAE 重建质量是瓶颈"时才做。

### Exp V4: 加强 face_attn / GAT

| 配置 | face_attn layers | GAT layers | bottleneck_dim |
|------|-----------------|------------|----------------|
| baseline | 12 | 10 | 768 |
| deeper_attn | 16 | 10 | 768 |
| deeper_gat | 12 | 15 | 768 |
| wider | 12 | 10 | 1024 |

### Exp V5: Auxiliary Topology Loss

给 encoder 加一个辅助 loss：让 face_z 直接就能预测 adjacency。

```python
# 在 encode 之后、sample 之前
# 用 face_features 预测 adjacency（不需要走完 decode）
aux_adj_pred = aux_classifier(face_features[pairs])
aux_loss = BCE(aux_adj_pred, gt_adjacency)
total_loss += 0.1 * aux_loss
```

目的：鼓励 face_z 在 sample 之前就已经包含足够的拓扑信息。

---

## 完整时间线

```
Week 1:
  ├── Day 1-2: Exp V0 (诊断) + Exp V2 (zero-pad diffusion, 不需要新VAE)
  └── Day 3-7: Exp V1 (KL sweep, 4个VAE同时训)

Week 2:
  ├── Day 1-3: Exp D1 (best VAE → 重新提取latent → 训diffusion)
  └── Day 4-7: Exp D2 (叠加zero-pad) + Feature Mapper训练

Week 3:
  ├── Day 1-3: Exp C1/C2/C3 (condition fusion on best pipeline)
  └── Day 4-7: Ablation + 可视化 + 跑 select_cases.py 挑选展示用例

Week 4:
  ├── 如果够了: 写论文/准备presentation
  └── 如果不够: Exp V4/V5 (VAE中间层)
```

---

## 每个实验的成功/失败判断标准

| 实验 | 成功信号 | 失败信号 → 下一步 |
|------|---------|-------------------|
| V0 (诊断) | std远离1.0, 分布不正态 | std≈1.0 → KL不是问题, 跳过V1 |
| V1 (KL sweep) | 某个KL下重建CD<0.01且std更接近1 | 所有KL都重建差 → VAE容量不够 |
| V2 (zero-pad) | Valid rate提升, 面数over减少 | 没变化 → padding不是主因 |
| D1 (new VAE+diffusion) | CD明显下降 | 没变化 → latent regularity不是瓶颈 |
| C1 (Feature Mapper) | CD降到0.01级 | 没下来 → 需要KD |
| C2 (KD) | CD降到0.005级, F1>0.7 | 没下来 → 需要改VAE中间层 |

---

## 可视化 Checkpoint

每个 Phase 结束时做一次可视化分析：

```bash
# 1. 跑 eval
python -m src.brepnet.eval.run --pred-root <results> --gt-root <gt> --metrics condition,validity

# 2. 挑选 case
python -m src.brepnet.eval.select_cases --pred-root <results> --num 5

# 3. 对比上一轮
# 重点看：worst_valid category 的 failure mode 变了吗？
#   如果之前是 "面多了" → V2 之后是否解决？
#   如果之前是 "面对了但位置偏" → D1 之后是否改善？
#   如果之前是 "形状和GT完全不像" → C1 之后是否接近GT？
```
