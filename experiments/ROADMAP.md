# Experiment Roadmap

> 这是长期目标与近期方向的单一事实来源（single source of truth）。
> 每天规划实验前先读这里；产生新结论后回写这里。详细每日记录仍写在 `experiments/YYYY-MM-DD/NOTE.md`。
> 维护规则见 [brepnet-experiment-planning](../skills/brepnet-experiment-planning/SKILL.md)，报告规则见 [brepnet-experiment-notes](../skills/brepnet-experiment-notes/SKILL.md)。

## 长期目标（North Star，不轻易改动）

**Image-Conditioned Diffusion Model for B-Rep model generation.**

- 输入：单张图像（real photo / sketch / SVR 渲染）。
- 输出：合法、可用的 B-Rep 模型。
- 评估的最终下游指标：valid STEP rate 以及条件一致性，而不是任何单一中间指标。
- 论文叙事主线：`image -> (topology mask) -> HoLa diffusion -> B-Rep`。

任何一天的实验都必须能说清它如何服务这条主线。如果说不清，今天的方向就要重新选。

## 当前阶段（Current Phase）

> 一句话描述现在主攻哪一步，以及它在主线中的位置。

- 阶段：differentiable topology-guided diffusion interface validation。
- 在主线中的位置：先验证 topology 作为连续 attention prior 是否真正改善 HoLa diffusion，再训练 image-conditioned topology head；暂不继续独立 two-step CVAE。

## 近期进展时间线（最近优先，保留约 2~3 周）

| 日期 | 方向 | 关键结果 | 链接 |
|------|------|----------|------|
| 2026-06-28 | 重新审计 topology attention 并固定新路线 | 旧 `scale=2` 是 attention logit `+2`，不是预期的 `2x`；已实现 `log(1+P)` oracle 实验配置并通过公式、梯度、Hydra、完整模型与真实样本 forward smoke test，待正式训练 | [NOTE](./2026-06-28/NOTE.md) |
| 2026-06-19 | two-step CVAE topology + diffusion validation | 计划补齐 real-photo no-mask baseline diffusion；新增 visible-only condition metric；新增 06-18 CVAE topology + baseline diffusion two-step 推理脚本，待 smoke/full run | [NOTE](./2026-06-19/NOTE.md) |
| 2026-06-18 | visibility-aware CVAE supervision | 基于已验证的 24-view visibility cache，实现 06-11 in-context CVAE 的 soft visibility-weighted pair loss；test `sample8_valid_mean=0.9256`，`sample8_visible_best_edge_f1=0.7649`，但 old full-GT `sample8_best_edge_f1=0.7012` 未超过 06-11，因此下一步转 downstream 检验 | [NOTE](./2026-06-18/NOTE.md) |
| 2026-06-17 | 任务定义反思 | 识别根本性问题：image→topology 是 1-to-many，但 CVAE 当前 loss 假设 1-to-1（fit 整张 GT face_adj），强制 model 在不可见 face 上瞎猜，是过去一周 fidelity 上不去的真因 | [NOTE](./2026-06-17/NOTE.md) |
| 2026-06-14 | 06-13 结果审计与 zero-init gated image adapter | frozen-prior 被判为负结果；已实现 Level-1 zero-init gated delta 脚本并通过 smoke test，下一步跑正式训练比较 06-11 | [NOTE](./2026-06-14/NOTE.md) |
| 2026-06-13 | 冻结 06-08 topology decoder，只训练 DINO image-conditioned prior | 证明单个 image-conditioned latent prior 不足：test `sample8_best_f1=0.6462`、`sample8_valid_mean=0.8058`，均低于 06-11；但保住了 06-08 decoder reconstruction | [NOTE](./2026-06-13/NOTE.md) |
| 2026-06-12 | Topology VAE 阶段总结 | corruption-only 为当前最强 decoder：`ar_f1=0.9479`、`ar_valid=0.9922`、`exact=0.7826`；确认评估应从 exact 转向 `sample@K_valid` 与下游 utility | [周报](./2026-06-12/weekly_report_0606_0612.md) |
| 2026-06-11 | image-conditioned topology CVAE（DINOv2 in-context，corruption-only） | 证明 `image -> topology` 有信号：`cond_prior_mu_edge_f1` 0.34→0.60；test `sample8_valid_any=1.0`、`sample8_valid_mean≈0.924`，但 fidelity 仍弱（`sample8_best_f1=0.7242`），且 in-context 结构稀释了原 decoder（`ar_recon_f1` 0.948→0.78） | [NOTE](./2026-06-11/NOTE.md) |
| 2026-06-08 | Topology VAE prefix corruption / degree token | corruption-only 成为最强 decoder；degree hard constraint 牺牲 reconstruction，未成主路线 | [NOTE](./2026-06-08/NOTE.md) |
| 2026-06-06 | Topology VAE 基线与 KL/WL/edge-count ablation | `KL=0.001` 带来最大提升；WL ordering、N_EDGE token 为稳定小增益 | [NOTE](./2026-06-06/NOTE.md) |

## 当前最优 / 基线（Known Best）

| 组件 | 当前最优 | 指标 | checkpoint |
|------|----------|------|-----------|
| Topology VAE decoder | 06-08 corruption-only | test `ar_f1=0.9479`、`exact=0.7826` | [best.pt](/mnt/d/python/experiments/2026-06-08/outputs_corrupt_only/best.pt) |
| Image-conditioned topology | 06-11 DINOv2 in-context CVAE | test `sample8_valid_any=1.0`、`best_f1=0.7242` | [best.pt](/mnt/d/python/experiments/2026-06-11/outputs_cvae_incontext_corrupt/best.pt) |
| Visibility-aware topology | 06-18 DINOv2 in-context CVAE + visibility-weighted pair loss | test `sample8_visible_best_edge_f1=0.7649`、`sample8_valid_mean=0.9256`；注意缺少 06-11 同口径 visible baseline，不能直接称为全局最优 | [best.pt](/mnt/d/python/experiments/2026-06-18/outputs_cvae_visibility/best.pt) |
| End-to-end diffusion | 旧结果仅作历史参考，待统一协议后重建 | 06-28 确认旧 topology `+2` 的真实强度约为 `7.39x`；此前结果不能回答新的 `2x` multiplicative prior 是否有效 | [NOTE](./2026-06-28/NOTE.md) |

## 开放问题（Open Questions，按优先级）

1. **（new, 06-28）topology-to-diffusion 接口是否有效**：把 GT adjacency 从 additive `+2` 改为 `log(1+P)`、即最大 `2x` attention mass 后，是否能稳定提高 valid STEP 且不损害 image-condition consistency？
2. **（new, 06-28）联合训练是否必要**：exact-supervised topology head 的 `P` 进入 attention 时，允许 diffusion loss 回传是否比 `P.detach()` 更有下游收益？
3. **（new, 06-28）visibility weighting 何时有价值**：在 exact predictor 与联合梯度链路成立后，提高 visible-visible pair loss、降低 visible-hidden / hidden-hidden loss，是否改善 single-view 条件一致性？
4. topology head 应继续复用 CVAE，还是改为 `30 face-valid + 435 symmetric pair affinity` 的确定性或随机生成模块？
5. real photo 与 sketch / SVR 渲染之间的 domain gap 对 image encoder 的影响。

## 下一步候选（Next Candidates，会被每日规划消费）

- [ ] **（new, 06-28, P0）实验 0：GT oracle + log-multiplier**：保持数据、image condition 和 diffusion 架构不变，将 topology prior 改为 `bias=timestep_weight*log(1+P)`，验证最大 `2x` 的 GT topology guidance 是否有下游收益。
- [ ] **（new, 06-28, P1）实验 1：exact topology predictor + detached prior**：输出 30 个 face-valid 与 435 个对称 edge probability，以 exact BCE 训练，diffusion 使用 `log(1+P.detach())`。
- [ ] **（new, 06-28, P2）实验 2：joint gradient**：相对实验 1 只移除 `detach()`，验证 diffusion loss 回传 topology head 的价值。
- [ ] **（new, 06-28, P3）实验 3：visibility-weighted exact loss**：模型不变，只按 continuous face visibility 调整 pair BCE 权重。
- [x] **（06-17）visibility-aware CVAE supervision**：06-18 已完成；作为 visibility loss 与数据基础保留，但独立 two-step CVAE 不再是当前主线。
- [x] **（06-19）two-step CVAE topology + diffusion validation**：已完成且下游退化；06-28 进一步发现旧 additive `+2` 实际过强，因此该结果用于暴露接口问题，不再用于继续调 CVAE。

## 反指标 / 不要陷进去的局部坑（Anti-goals）

- 不要为了把 `exact_adj_acc` 再抬高几个点而连续多天死磕；这是 one-to-many 问题，exact 不是终点。
- 不要只盯单张样本或单一 split 的指标波动做大量微调。
- 不要在 topology 模块上无限打磨而迟迟不验证它对下游 diffusion 的真实收益。
- **（new, 06-17）不要在错误的任务定义下继续优化结构**：在确认 supervision target（loss / labels / training data）匹配实际任务之前，结构变种（adapter / latent diffusion prior / spanning tree 等）和 downstream ablation 都可能在解决错的问题。Task validity 优先级高于 structural search。
- **（new, 06-17）不要默认 GT face_adj = ground truth supervision**：GT 是众多合法 topology 中的一个，和单视角 image 的可观测性无关。把它整张当 1-to-1 label 喂 CVAE，会强制 model 在不可见区域瞎学。
- **（new, 06-28）不要把 additive attention bias `+2` 解释为 attention `2x`**：softmax 下它对应约 `exp(2)=7.39x` 的未归一化 mass。
- **（new, 06-28）不要把 legacy evaluator 的 `N/N are valid` 当 valid solid rate**：它表示 condition evaluation coverage，真实 validity 以 STEP solid check / `success.txt` 为准。
