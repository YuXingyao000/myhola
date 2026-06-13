# 2026-06-12 实验笔记

## 今日目标

整理 06-06 到 06-12 的 topology VAE 阶段周报材料，并补充 CVAE full val/test sample@8 的结果解读。

## 实验记录

### 1. CVAE full val/test sample@8 结果解读

**为什么这么做**

- 06-11 DINO in-context CVAE 训练期只用 32 个 val samples 做轻量生成验证，不足以判断 topology validity。
- 对 image-conditioned topology generation 来说，`exact` 不是主指标；更关键的是 sample@K 是否能得到 strict valid topology mask。

**具体实施**

- 读取：
  - `experiments/2026-06-11/outputs_cvae_incontext_corrupt/val_metrics.json`
  - `experiments/2026-06-11/outputs_cvae_incontext_corrupt/test_metrics.json`
- 同步把结果补写到 `experiments/2026-06-11/NOTE.md`。

**结果**

- test `sample8_valid_any=1.0000`，说明每张测试图采 8 次时至少有一个 strict valid topology。
- test `sample8_valid_mean=0.9239`，说明平均 sampled topology 合法率也较高。
- test `cond_prior_mu_valid_strict_ratio=0.9530`。
- test `sample8_best_edge_f1=0.7242`，topology fidelity 仍不足。
- 当前结论：第一版 CVAE 的 structural validity 比训练期小样本指标更乐观，但 topology fidelity / count accuracy 仍需改进。

### 2. 周报 PPT 参考大纲重写

**为什么这么做**

- DeepSeek 草稿内容完整，但更像长文汇总，不太像 5-6 页 PPT 的施工单。
- 草稿中 `exact` 权重偏高，容易和当前评估视角冲突。对本周 VAE 汇报来说，应该突出 decoder capacity、structural validity、以及 reconstruction-vs-prior-validity tradeoff。

**具体实施**

- 重写：
  - `experiments/2026-06-12/weekly_report_0606_0612.md`
- 新结构按 6 页 PPT 组织：
  1. 背景与本周目标
  2. Topology VAE 表示与模型
  3. 实验演化路线
  4. 结果总览
  5. 关键发现与评估视角调整
  6. 本周结论与下周计划
- 保留 CVAE 为下周方向，不把 06-11/06-12 CVAE 结果写进本周周报主体。

**结果**

- 周报现在更适合作为 PPT 参考：每页都有标题、讲述要点、可放图表和关键数据。
- `model_table.png` 可作为结果总览主图，`model_bars.png` 可作为备用图。

## 今日结论

- CVAE 口头结论：validity 已经可看，test `sample8_valid_any=1.0`，但 topology fidelity 还不够。
- 周报 PPT 仍聚焦 VAE 阶段：corruption-only 是当前最强 topology decoder，后续 CVAE 应围绕 sample@K validity 和 downstream mask utility 评估。

## 待办

- [ ] 手动制作 5-6 页周报 PPT。
- [ ] 下周继续 CVAE：优先考虑冻结 06-08 topology decoder，只训练 image-conditioned prior / adapter。
