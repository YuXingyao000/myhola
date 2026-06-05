# 2026-06-05 Topology Predictor 实验汇报与下一步规划

## 1. 已完成实验结果汇总（来自 2026-06-01）

四个 toy topology predictor 已训练完毕。架构均为：
**frozen DINOv2 → image feature → face_head (30 face tokens) → valid_head + edge_head → face_adj [30×30]**。

差异只在 image feature 来源。

### Best validation metrics

| 模型 | edge_ap | edge_auc | precision | recall | edge_f1 | edge_iou | recall@k | face_mae | exact% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `svr24` (24 灰模) | 0.613 | 0.735 | 0.527 | 0.635 | 0.576 | 0.404 | 0.596 | **0.88** | **2.97%** |
| `sketch24` (24 线框) | 0.609 | 0.731 | 0.524 | 0.633 | 0.573 | 0.402 | 0.593 | 1.00 | 2.40% |
| `real_photo` (1 张 FLUX) | 0.598 | 0.727 | 0.516 | 0.642 | 0.572 | 0.401 | 0.583 | 1.60 | 0.26% |
| `svr_single` (1 张灰模) | 0.583 | 0.720 | 0.502 | 0.653 | 0.568 | 0.396 | 0.576 | 1.74 | 0.29% |

## 2. 关键发现

### 2.1 DINOv2 feature 确实编码了 topology 信息

- `edge_auc = 0.72-0.73` 显著高于 0.5（随机基线）
- `edge_ap = 0.58-0.61` 远高于 sparse baseline（~0.03）
- **核心 hypothesis 验证成功**：冻结的 DINOv2 feature 里有可被解码的拓扑信号

### 2.2 视角数量主要影响"结构复杂度"判断，不影响"局部邻接"

- 单图 → 24 图，**edge prediction 几乎没提升**（F1: 0.57 → 0.58）
- 但 **face_count_mae 从 ~1.7 降到 ~0.9**（2x 提升）
- **exact_match 从 0.003 提升到 0.030**（10x 提升）

**解读**：多视角告诉你"物体多复杂"，单视角足以推断"局部哪些面相邻"。这给单图路线留了空间。

### 2.3 svr vs sketch 几乎一样

两者差距 < 1%。说明 DINOv2 在白模上已经隐式提取了 edge 结构，**显式线框 input 并不带来显著增益**。

### 2.4 real_photo 略优于 svr_single（!）

`edge_ap: 0.598 vs 0.583`，real photo 反而更好。说明 FLUX 真实照片对 DINOv2 不是噪声，反而 in-distribution。这对最终的 single real-photo input 是正面信号。

### 2.5 Precision/Recall 不对称

所有模型 **recall (0.63-0.65) > precision (0.50-0.53)**。模型倾向"多报不漏报"。
对 topology bias 是好事——soft mask 里有少量 false positive 不致命，但漏掉真实邻接会严重影响 attention。

### 2.6 当前架构已饱和

train/val gap 很小（svr24: 0.78 → 0.80），并非严重过拟合，更像是模型容量已经饱和。继续训练或加 epoch 不会提升。**瓶颈在 representation 和 output formulation**。

## 3. 当前架构的固有问题

### 问题 1：信息瓶颈 — CLS 一个 vector 编码整个物体

CLS 是 1024 维 global summary，但 face_adj 包含 30 个面的精细结构。**这是 sparse 信号在 dense matrix 里的问题**。

### 问题 2：固定 face slot 顺序

模型必须学"第 0 个 slot 应该放什么 face"——这是一个**任意约定**，不该由模型学。

### 问题 3：1-to-many ambiguity

单图 → 拓扑天然是不确定的（背面、内部腔体看不到）。Binary matrix 只能给一个答案，模型会学到所有合法拓扑的"平均"，平均后的概率矩阵不对应任何真实拓扑。

## 4. 改进方向

### 方向 A：DETR-style face queries（最低成本）

```
旧 (svr_single):
  DINOv2 → CLS [1024] → face_head MLP → 30 face_tokens
                          (每个 face_token 由 global feature 决定)

新 (DETR):
  DINOv2 → patches [256, 1024] → patch_projection → [256, 128]

  30 个可学习 face_queries [30, 128]
       ↓ cross-attention (TransformerDecoder × 2 layers)
  30 face_tokens [30, 128]
       (每个 face_token 自适应 attend 到自己感兴趣的图像区域)

  ↓ valid_head + edge_head（保留，与 svr_single 同构）
```

**优点**：
- 解决问题 1（patch tokens 保留空间结构）
- 解决问题 2 部分（face_query 学的是 attend pattern，不再是 slot 顺序）
- 改动小，head 输出端不变 → 可直接公平对比 F1

**参考**：DETR (Carion et al., ECCV 2020), MaskFormer (Cheng et al., NeurIPS 2021)

### 方向 C：Autoregressive topology transformer（novelty 最强）

把 face_adj **重新表示成 token 序列**，用 GPT 风格 transformer 生成。

#### 序列设计（已简化）

HoLa-BRep 假设所有面是 B-Spline，**不需要 face type token**。Face ordering 用数据原始顺序（topology 作为先验，简单假设即可）。Rotation 暂不引入。

```
[BOS] (0,1) (0,3) (1,2) ... [EOS] [PAD] [PAD] ...
```

每个 token 表示一条 edge `(i, j)` with `i < j`。

#### Vocabulary

```python
PAD=0, BOS=1, EOS=2
edge_token(i, j) = 3 + i * max_faces + j   # i < j
# vocab_size = 3 + 30*30 = 903
```

#### 架构：标准 Transformer Decoder

`Attention is all you need` 的标准架构 + 几个适配：

| 组件 | 经典 | 我们的 |
| --- | --- | --- |
| Encoder | text encoder | DINOv2 (frozen) → patches → projection |
| Memory for cross-attn | encoder output | image patch tokens [256, d_model] |
| Decoder | 标准 | **完全不变**（self-attn + cross-attn + FFN + LN） |
| Causal mask | triu | **完全不变** |
| Token embedding | word/BPE | 自定义 vocab (edge tokens) |
| Positional | sin/cos 或 learned | learned positional embedding |
| Output head | LM head | LM head (vocab_size = 903) |
| Loss | cross-entropy | cross-entropy (ignore PAD) |

#### 训练 vs 推理

| | Training | Inference |
| --- | --- | --- |
| 输入 | image + GT sequence (teacher forcing) | image + partial sequence |
| Loss | next-token cross-entropy | 无 |
| 输出 | logits [B, L, V] | autoregressive sample |
| 多样性 | 无 | **temperature sampling 自然产生多样拓扑** |

#### 单独训练，不和 diffusion 联合

- AR 用 cross-entropy loss（离散 token 分类）
- Diffusion 用 MSE/L1 loss（连续 latent 回归）
- 两者梯度量纲不同，混合训练梯度打架
- AR 训练范式（GPT-style）成熟稳定，独立训能跑出最好结果
- **训完一次的 AR predictor 可以服务多个 diffusion 实验**

推理流程：
```python
ar_predictor.generate(image, temperature=0.8)  # 采样得到 face_adj
diffusion.inference(image, face_adj=...)        # 用作 self-attention bias

# 多样性采样
brep_samples = [generate_brep(image) for _ in range(K)]
```

### 方向 A vs 方向 C

| | A (DETR queries) | C (Autoregressive) |
| --- | --- | --- |
| 解决 信息瓶颈 | ✓ | ✓ |
| 解决 face slot 顺序 | 部分 | ✓（顺序由生成决定） |
| 解决 1-to-many | ✗ | ✓（采样多样拓扑） |
| 工程量 | 小 | 中 |
| 与 oracle setup 兼容 | 直接 | 直接 |
| Novelty | 中 | 强 |

## 5. 嫁接到现有 diffusion model 的方案

### 已实现：`LearnedTopologySelfAttentionMask`

放在 `src/brepnet/models/diffusion_denoiser.py`，和 `OracleTopologySelfAttentionMask` 共享 factory `build_topology_self_attention`。

数据流：

```
batch["conditions"]["imgs"] [B, 3, 224, 224]
  → frozen DINOv2 → face_head → edge_head → sigmoid
  → predicted_adj [B, 30, 30]
  → soft_bias 公式（同 oracle）：adj * timestep_weight * scale
  → self-attention bias
```

**关键设计**：
- `freeze: true`（默认）：predictor 参数不更新，反向传播只走 diffusion，实验严谨
- 与 oracle 实验唯一变化是 mask 来源
- 可以将来 `freeze: false` 做 stage C 联合 fine-tune

Config: `configs/train_diffusion_learned_topology.yaml`

## 6. 运行命令

所有命令记录在 `experiments/2026-06-02/command.sh` 与本目录。

### 数据修复（迁移过程中发现少量损坏的 npz）

```bash
cd /mnt/d/python && python tools/fix_corrupted_npz.py --scan-only
cd /mnt/d/python && python tools/fix_corrupted_npz.py --fix
```

### 方向 A：DETR-style topology predictor

```bash
cd /mnt/d/python && python experiments/2026-06-02/topology_from_svr_single_detr.py \
    --output-dir experiments/2026-06-02/outputs_svr_single_detr \
    --epochs 20 --batch-size 32
```

### 方向 C：Autoregressive topology transformer

```bash
cd /mnt/d/python && python experiments/2026-06-02/topology_ar_train.py \
    --output-dir experiments/2026-06-02/outputs_ar \
    --epochs 30 --batch-size 32
```

### Learned topology diffusion（嫁接 toy svr_single）

```bash
cd /mnt/d/python && python -m src.brepnet.train \
    --config-name train_diffusion_learned_topology
```

## 7. 下周计划

| 优先级 | 任务 | 产出 |
| --- | --- | --- |
| 1 | 跑方向 A（DETR predictor）| F1 vs MLP baseline 对照 |
| 2 | 跑方向 C（AR transformer）| F1 + sample diversity 数据 |
| 3 | 嫁接 svr_single 到 diffusion 看 FID gap | 验证 0.57 F1 mask 是否够用 |
| 4 | 如果 A 或 C 显著优于 MLP，更新 LearnedTopologyPredictor 的实现 | |

## 8. 论文叙事方向

如果 A + C 都做出来：

> 1. 我们发现 single-image B-Rep generation 的瓶颈在 topology prediction（oracle 实验已验证）
> 2. 我们提出 DETR-style face query topology extractor（A）
> 3. 我们进一步提出 autoregressive topology generation（C），把 topology 表示成 sequence，自然处理 variable-length 和 multi-modal 输出
> 4. (C) 解决 single-image → 多个合法拓扑 的 ambiguity，可以采样得到多样合理输出

如果只做 A：B 刊够用。
如果做了 C：可以冲 SIGGRAPH Asia / TOG。

## 9. 评估指标补充建议

AR 模型的真正优势在 sampling 多样性。除了 greedy F1，建议加：

- **Top-K oracle accuracy**：sample K 次，取最接近 GT 的那次。这才是 AR 相对 MLP/DETR 的真正优势体现。
- **Diversity score**：sample 10 次计算 pairwise diff，量化输出多样性。直接对应 novelty 故事。
