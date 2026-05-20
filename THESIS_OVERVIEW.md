# 论文整体思路总结

## 一句话定位

HoLa-BRep 的 holistic latent 在 Encoder 阶段通过 GNN 融合了拓扑，但在之后的整个 pipeline（Diffusion 生成 + Decoder 解码）中，拓扑信息都被"遗忘"了——只是隐式存在于 latent 里，没有被显式利用。

## 问题诊断

```
Encoder:  Conv2D + Transformer + GAT(真实边) + Global
              ↓
          face_z [N, 32] ← 同时包含几何+拓扑 ✓
              ↓
Diffusion: 24层 TransformerEncoder (self-attention, 所有face平等)
              ↓                                    ✗ 不知道谁和谁相邻
          生成的 face_z [30, 32]
              ↓
Decoder:  face_attn2 (all-to-all, 无拓扑引导)      ✗ 不知道谁和谁相邻
              ↓
          face_points_decoder (独立decode每个面)    ✗ 不知道自己的边界在哪
          edge_points_decoder (独立decode每条边)    ✗ 不知道对应面长什么样
              ↓
          AttnIntersection → 预测拓扑             只是输出，没有反馈
```

**拓扑在 Encoder 之后就"断流"了。**

## 解决方案

### Diffusion 侧：Topology Attention Bias

```
Image → TopologyPredictor → adj_matrix [30, 30]
                                 ↓
Diffusion self-attention += topology_bias
  → 相邻面互相关注，生成的 latent 结构性更一致
  → 主要提升: F1, Valid Rate
```

### Decoder 侧：拓扑约束几何解码

```
预测拓扑 (AttnIntersection) → 不只是输出，还要反馈回几何解码
  方法1: Boundary Consistency Loss (面的边界 ≈ 对应边)
  方法2: Topology-Aware face_attn2 (用预测邻接做attention mask)
  方法3: Edge→Face Refinement (边信息回传指导面decode)
  → 主要提升: Valid Rate, 后处理成功率
```

### Condition 侧：特征域对齐

```
DINOv2(real_photo) 不含精确几何 → Feature Mapper / KD 解决
  → 主要提升: Chamfer Distance
```

## 论文 Contribution 框架

```
"Real-Photo to CAD via Topology-Aware Latent Diffusion"

Contribution 1: Condition Fusion
  - Feature Domain Mapper: 解决真实照片的特征域gap
  - (可选) Knowledge Distillation: 更强的条件引导

Contribution 2: Topology-Aware Generation
  - Diffusion阶段: Topology Attention Bias (让生成过程感知结构)
  - Decoder阶段: Boundary Consistency + Topo-Aware Decoding (让解码尊重拓扑)

Contribution 3: Dataset
  - FLUX.1-Kontext 生成的 real-photo benchmark
  - 质量评估指标 (Silhouette IoU, DINOv2 cosine, etc.)
```

## 实验总计划 (按执行顺序)

```
Phase 0: 诊断 (今天)
  ├── Decoder 鲁棒性测试 (5分钟)
  └── 确认 diffusion latent 偏离多少

Phase 1: Condition Fusion (本周)
  ├── Exp 1: Feature Mapper (几小时)
  └── 评估: Chamfer 能降到多少

Phase 2: Topology-Aware Diffusion (下周)
  ├── 训 TopologyPredictor (几分钟)
  ├── Exp 3: Topology Bias + 重训 Diffusion
  └── 评估: F1 和 Valid Rate 提升

Phase 3: Topology-Aware Decoder (第3周)
  ├── Boundary Consistency Loss (加 loss, 重训 VAE)
  ├── Topo-Aware face_attn2 (推理时改 mask)
  └── 评估: Valid Rate + 后处理成功率

Phase 4: 组合最佳方案 + Ablation (第4周)
  └── Feature Mapper + Topology Bias (Diff) + Boundary Loss (Dec)

Phase 5: 如果需要更强
  ├── Knowledge Distillation
  ├── Edge→Face Refinement
  └── Zero-padding + Classifier
```

## 每个改进点的独立性

```
                    Condition Fusion    Topo Diffusion    Topo Decoder
影响 Chamfer:           ★★★★★              ★☆☆☆☆            ★☆☆☆☆
影响 F1:                ★★☆☆☆              ★★★★☆            ★★★☆☆
影响 Valid Rate:        ★☆☆☆☆              ★★★☆☆            ★★★★★
互相独立?:              ✓ 独立             ✓ 独立           ✓ 独立

三者可以自由组合，互不干扰，每个都有独立贡献。
```
