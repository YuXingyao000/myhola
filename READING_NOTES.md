# 论文阅读笔记

## 1. DTGBrepGen (CVPR 2025)
- **Paper**: https://arxiv.org/abs/2503.13110
- **Core Idea**: 拓扑-几何解耦生成 B-Rep

### 架构
```
Stage 1: 拓扑生成 (Topology)
  ├── Step 1: Edge-Face Adjacency Matrix
  │   └── Transformer VAE: 上三角序列 → 潜码 → 重建
  │       Loss = CrossEntropy + KL
  └── Step 2: Edge-Vertex Adjacency  
      └── Transformer + Pointer Network (自回归)

Stage 2: 几何生成 (Geometry, conditioned on topology)
  ├── Face Bounding Boxes (6D per face)
  ├── Vertex Coordinates (3D points)
  ├── Edge Geometries (cubic B-spline, 4 control points = 12D)
  └── Face Geometries (bi-cubic B-spline, 4×4 grid = 48D)
  
  每个阶段: 8-layer Transformer Diffusion, 512-dim, 8 heads
  Topology 信息通过 attention bias 注入（shared-edge counts → attention weights）
```

### 关键结果
| Dataset | BrepGen Valid↑ | DTGBrepGen Valid↑ | HoLa-BRep Valid↑ |
|---------|----------------|-------------------|-------------------|
| DeepCAD | 68.23% | **79.80%** | ~82% |
| ABC | 47.11% | **62.08%** | - |

### 关键发现
- 用 GT topology 做几何生成 → DeepCAD valid rate 90.24%
- 说明 topology generation 的质量是整体 valid rate 的瓶颈
- 先有正确拓扑，再填几何，valid rate 大幅提升

---

## 2. BrepARG (2026)
- **Paper**: https://arxiv.org/abs/2601.16771
- **Core Idea**: 把 B-Rep 的几何+拓扑统一编码为 token sequence，用 decoder-only Transformer 自回归生成

### Token 序列化
```
一个 B-Rep → 一个 token 序列:

[face_1_geom_tokens] [face_1_pos_tokens] [face_1_index_tokens]
[edge_1_geom_tokens] [edge_1_pos_tokens]
[face_2_geom_tokens] ...
```

三种 token:
- Geometry tokens: NURBS 控制点的离散化
- Position tokens: Bounding box 位置
- Face index tokens: 拓扑连接（哪些面共享这条边）

### 架构
- Decoder-only Transformer + causal masking
- 训练: next-token prediction
- 推理: 逐 token 采样

### 优势
- 不需要 VAE，不需要 padding/deduplication
- 一个统一框架处理 geometry + topology
- 可以做 conditional generation / autocompletion

### 劣势
- 序列很长（一个 CAD 模型可能 1000+ tokens）
- 自回归推理慢
- 错误累积

---

## 3. BrepGPT (2025)
- **Paper**: https://arxiv.org/abs/2511.22171
- **Core Idea**: Voronoi Half-Patch 分解 + 自回归

### 方法
- 把 B-Rep 分解为 Voronoi Half-Patch (VHP) 局部单元
- 每个 VHP = 一个面的一部分 + 对应的边
- 用 GPT 风格逐 patch 生成

---

## 对我们工作的启发

### 从 DTGBrepGen 借鉴
1. **拓扑作为先验/条件注入 diffusion**（而不是后预测）
2. **Attention bias**: 已知/预测的拓扑关系 → 修改 self-attention 的权重
3. **分阶段生成**: 先粗(topo) 后细(geom)

### 从 BrepARG 借鉴
1. **序列化的可能性**: 如果 face 数量少(≤30)，序列不会太长
2. **统一表示**: 避免分开预测 topology 和 geometry 的 error 累积

### 最小改动方案（在 HoLa-BRep 上）
→ 见 models/topology_guided.py 的实现
