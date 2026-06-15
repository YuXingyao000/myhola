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

- 阶段：image-conditioned topology generation（image -> topology mask）。
- 在主线中的位置：为 HoLa diffusion 提供拓扑约束的前置模块。

## 近期进展时间线（最近优先，保留约 2~3 周）

| 日期 | 方向 | 关键结果 | 链接 |
|------|------|----------|------|
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
| End-to-end diffusion | （待补：no-mask / predicted-mask / oracle-mask 对照） | - | - |

## 开放问题（Open Questions，按优先级）

1. 如何在保留 06-08 强 topology decoder 的同时接入 image 条件（冻结 decoder + 只训 image prior/adapter）？
2. CVAE topology fidelity（`sample@K_best_f1`）能否在不牺牲 validity 的前提下提升？
3. predicted topology mask 接入 HoLa diffusion 后，相比 no-mask / oracle-mask 的下游收益有多大？
4. real photo 与 sketch / SVR 渲染之间的 domain gap 对 image encoder 的影响。

## 下一步候选（Next Candidates，会被每日规划消费）

- [x] 冻结 06-08 topology decoder，仅训练 image-conditioned prior / adapter（06-13 已写脚本并通过 smoke test，正式训练待运行）。
- [ ] CVAE 加 decode-time connectivity repair / constrained decoding，提升 `sample@K_valid`。
- [ ] `--image-token-dropout 0.1` 或 real image augmentation，缓解过拟合。
- [ ] 把 predicted topology mask 接入 HoLa diffusion，跑 no-mask / predicted-mask / oracle-mask 三方对照。
- [x] 实现 gated / adapter decoder conditioning：保留 06-08 decoder 能力，同时让 DINO image tokens 通过 zero-init / gate / adapter 进入 decoder（06-14 已完成脚本和 smoke test）。
- [ ] 跑 06-14 gated adapter 正式训练与 full val/test sample@8。
- [ ] 如果 gated adapter 仍不能超过 06-11，考虑 image-conditioned latent diffusion over `z_topology` 或先做 downstream utility check。

## 反指标 / 不要陷进去的局部坑（Anti-goals）

- 不要为了把 `exact_adj_acc` 再抬高几个点而连续多天死磕；这是 one-to-many 问题，exact 不是终点。
- 不要只盯单张样本或单一 split 的指标波动做大量微调。
- 不要在 topology 模块上无限打磨而迟迟不验证它对下游 diffusion 的真实收益。
