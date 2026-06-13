# 周报 PPT 参考：Topology VAE 阶段总结（2026-06-06 至 2026-06-12）

> 这份文档用于手动写 5-6 页周报 PPT。重点放在 **Topology VAE 为什么要做、怎么演化、目前得到什么结论**。CVAE 结果不放进本周 PPT，留到下周作为 image-conditioned topology generation 的主线。

## PPT 主线

**一句话主线**：本周完成了 topology-only VAE 的系统 ablation，确认 corruption-only VAE 是当前最强 topology decoder；同时明确了后续 CVAE 不应以 exact recovery 为唯一目标，而应以 topology mask 的 structural validity 和 sample@K 可用性为核心。

建议 6 页：

1. 背景与本周目标
2. Topology VAE 表示与模型
3. 实验演化路线
4. 结果总览
5. 关键发现与取舍
6. 下周计划

如果只能 5 页，可以把第 2 页和第 3 页合并。

---

## Slide 1. 背景与本周目标

**标题建议**

Topology VAE: 为 real-photo → B-Rep topology mask 建立 decoder 基础

**要讲的点**

- 最终目标：从单张真实照片生成 B-Rep topology mask，作为 HoLa-BRep diffusion 的拓扑约束。
- 已有 diffusion 实验证明：如果给 diffusion 正确/可用的 topology mask，合法性和几何/拓扑指标会明显提升。
- 本周只解决第一步：先训练一个能稳定 autoencode face-adjacency topology 的 VAE decoder。
- CVAE / image-conditioned topology generation 留到下一阶段。

**可放图**

```text
real photo -> topology generator -> topology mask -> HoLa diffusion -> B-Rep
                     ^
              本周先训练 topology VAE decoder
```

**一句话结论**

先把 topology 本身学好，再把图像条件接进来；否则 image→topology 失败时无法判断是图像信号弱，还是 topology decoder 本身不行。

---

## Slide 2. Topology VAE 表示与模型

**标题建议**

把 face-adjacency graph 转成可自回归生成的 token sequence

**要讲的点**

- 输入 topology：B-Rep face adjacency matrix。
- Canonical ordering：使用 WL-style ordering 降低同构图排序不稳定性。
- Sequence contract：
  - `[N_FACE]`
  - `[N_EDGE]`
  - `[PAIR_0 ... PAIR_434]`
- `PAIR_i` 是 upper-triangle face pair 的 `EDGE / NO_EDGE` token。
- VAE：
  - Transformer encoder 得到 `mu/logvar`
  - latent `z`
  - causal Transformer decoder 自回归生成 topology sequence

**可放图**

```text
face_adj -> WL ordering -> [N_FACE, N_EDGE, PAIR tokens]
             |
             v
      Transformer VAE
             |
             v
      generated topology sequence -> adjacency matrix
```

**指标解释**

- `ar_f1`: 用 posterior latent 自回归重建 topology 的 edge F1，衡量 decoder capacity。
- `ar_valid`: generated graph 是否满足 `connected ∧ no_isolated`，衡量 structural validity。
- `prior_valid`: 从 unconditional prior 采样时的 structural validity，仅作为 VAE prior 质量参考。
- `exact`: 整张 adjacency 完全匹配 GT；有参考价值，但不是最终 CVAE 的主指标。

---

## Slide 3. 实验演化路线

**标题建议**

从 baseline 到 corruption-only：逐步解决 decoder capacity 和 exposure bias

**建议用时间线**

| 日期 | 改动 | 目的 | 结论 |
|------|------|------|------|
| 06-06 | KL=0.1 baseline | 验证 topology VAE 可行性 | KL 太强，decoder 信息不足 |
| 06-06 | KL=0.001 | 放弱 KL，优先学 reconstruction | `ar_f1` 大幅提升 |
| 06-06 | WL ordering | 减少 face ordering 不稳定 | 小幅稳定提升 |
| 06-06 | +N_EDGE token | 显式给 edge count 约束 | 小幅提升 |
| 06-08 | prefix corruption | 缩小 teacher forcing 和 AR inference gap | 当前最强 decoder |
| 06-08 | degree token | 尝试强 structural constraint | validity 有帮助，但 F1 下降 |

**关键讲法**

- 最大提升来自 `KL=0.001`：autoencoding 阶段不能过早把 latent 压成标准正态。
- WL 和 edge count 是稳定的“小增益”。
- Corruption 是最值得保留的训练策略：不改生成 contract，只让 decoder 训练时见到错误 prefix。
- Degree hard constraint 没有成为主路线，因为它牺牲了 pair-level reconstruction。

---

## Slide 4. 结果总览

**标题建议**

Corruption-only 得到最强 decoder，但 reconstruction 和 prior validity 存在 tradeoff

**推荐图表**

- 主图：`experiments/2026-06-12/model_table.png`
- 备选图：`experiments/2026-06-12/model_bars.png`

**核心数据**

| 模型 | ar_f1 | ar_valid | prior_valid | exact |
|------|------:|---------:|------------:|------:|
| KL=0.1 | 0.7811 | 0.9950 | **0.9961** | 0.4187 |
| KL=0.001 | 0.9185 | **0.9959** | 0.9648 | 0.7463 |
| WL + KL=0.001 | 0.9295 | 0.9955 | 0.9473 | 0.7706 |
| WL + edge_count | 0.9340 | 0.9942 | 0.8047 | 0.7797 |
| **corruption-only** | **0.9479** | 0.9922 | 0.7070 | **0.7826** |
| degree-only | 0.9147 | 0.9942 | 0.7891 | 0.7496 |
| corruption + degree | 0.9265 | 0.9942 | 0.7871 | 0.7562 |

**讲图时的重点**

- 如果目标是 decoder reconstruction，corruption-only 最好：`ar_f1=0.9479`。
- 如果目标是 unconditional prior validity，强 KL 最好：`prior_valid=0.9961`，但 decoder 太弱。
- `ar_valid` 整体都很高，约 `0.992-0.996`；posterior reconstruction 的合法性不是当前最大瓶颈。
- `prior_valid` 和 `ar_f1` 明显冲突，说明 unconditional VAE prior 不是最终目标。

---

## Slide 5. 关键发现与评估视角调整

**标题建议**

从 exact recovery 转向 valid topology mask

**要讲的点**

- 单张照片到 topology 是 one-to-many：同一个视觉外观可能对应多个合理 B-Rep 拓扑。
- 所以 `exact_adj_acc` 不能作为最终目标，只能作为 decoder sanity check。
- 更重要的是 topology mask 是否：
  - connected
  - no isolated face
  - 面数/边数大体合理
  - 能帮助 diffusion 生成合法 B-Rep

**本周形成的新指标框架**

| 阶段 | 主指标 | 用途 |
|------|--------|------|
| VAE decoder | `ar_f1`, `ar_valid` | 判断 topology decoder 是否可靠 |
| VAE prior | `prior_valid` | 观察 latent space 是否容易采到合法图 |
| 后续 CVAE | `sample@K_valid_any` | 同一张图采 K 次，至少一次合法 |
| 后续 diffusion | valid STEP rate | 最终下游指标 |

**一句话结论**

本周最重要的不是 exact 提升到 0.78，而是确定了：后续 image-conditioned topology generator 应该围绕 `sample@K_valid` 和 downstream diffusion utility 来评估。

---

## Slide 6. 本周结论与下周计划

**标题建议**

Topology VAE decoder 已可用，下一步进入 image-conditioned generation

**本周结论**

- 完成了 7 个 topology VAE 变体的同口径 test eval。
- 当前最强 decoder 是 `corruption-only`：
  - `ar_f1=0.9479`
  - `ar_valid=0.9922`
  - `exact=0.7826`
- `degree` 路线没有成为主路线：validity 有帮助，但 reconstruction 明显下降。
- `prior_valid` 与 reconstruction 存在 tradeoff，不应直接用 unconditional prior 指标决定最终路线。

**下周计划**

1. 进入 image-conditioned CVAE：用 DINOv2 real-photo features 条件化 topology latent。
2. 使用 `sample@K_valid_any` / `sample@K_best_f1` 做 CVAE 主评估。
3. 尝试 decode-time connectivity repair，让 sampled topology 更稳定合法。
4. 最终把 predicted topology mask 接入 HoLa diffusion，比较 no-mask / predicted-mask / oracle-mask。

---

## 备用材料

**主要文件**

| 日期 | 文件 | 说明 |
|------|------|------|
| 06-06 | `experiments/2026-06-06/topology_faceadj_vae.py` | 基础 topology VAE |
| 06-06 | `experiments/2026-06-06/topology_faceadj_vae_edgecount.py` | +N_EDGE token |
| 06-08 | `experiments/2026-06-08/topology_faceadj_vae_corrupt.py` | corruption-only，当前最优 |
| 06-08 | `experiments/2026-06-08/topology_faceadj_vae_degree.py` | degree token |
| 06-08 | `experiments/2026-06-08/topology_faceadj_vae_both.py` | corruption + degree |
| 06-12 | `experiments/2026-06-12/model_table.png` | PPT 表格图 |
| 06-12 | `experiments/2026-06-12/model_bars.png` | PPT 柱状图 |

**PPT 备注**

- 不建议把所有 7 个模型的全部指标逐字讲完；重点讲 3 个对照：
  - `KL=0.1` vs `KL=0.001`: KL 强度决定 decoder capacity。
  - `WL+edge_count` vs `corruption-only`: corruption 提升 AR generation。
  - `corruption-only` vs `degree-only`: reconstruction 与 hard validity constraint 的取舍。
- CVAE 结果留到下周，不放入本周主线。
