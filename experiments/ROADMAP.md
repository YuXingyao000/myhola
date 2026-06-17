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
| End-to-end diffusion | （待补：no-mask / predicted-mask / oracle-mask 对照） | - | - |

## 开放问题（Open Questions，按优先级）

1. **（new, 06-17）任务定义合理性**：CVAE 当前用全图 face pair token CE loss，相当于把 1-to-many 任务（一张图对应多种合法 topology）退化成 1-to-1 fit。是否应该重新定义 supervision，让 CVAE 只在「图像可见」的 face pair 上贴合 GT，不可见 face 区域只要满足 connectivity / no-isolated 等结构 validity？
2. predicted topology mask 接入 HoLa diffusion 后，相比 no-mask / oracle-mask 的下游收益有多大？（Oracle 路径已知非常强：100 测试集 validity=100%，几何指标也很好；瓶颈在 image→mask 一段。）
3. CVAE topology fidelity（`sample@K_best_f1`）能否在不牺牲 validity 的前提下提升？**注意**：在 OQ 1 解决前，提升它本身可能就是错误目标。
4. real photo 与 sketch / SVR 渲染之间的 domain gap 对 image encoder 的影响。
5. 如何在保留 06-08 强 topology decoder 的同时接入 image 条件？（06-13/06-14 已部分回答：frozen-prior 信息瓶颈太窄；gated adapter 比 frozen-prior 好但仍输 06-11；OQ 1 可能让这条问题部分作废，因为新 loss 下 decoder 行为会变。）

## 下一步候选（Next Candidates，会被每日规划消费）

- [ ] **（new, 06-17, P0）visibility-aware CVAE supervision**：从 GT STEP + 相机外参构造 face-level visibility mask，CVAE pair-token loss 按可见性加权（fully_visible=1.0，partial=0.3，hidden=0.0）+ structural validity penalty。这是 OQ 1 的直接答案。
- [ ] **（new, 06-17, P0 快速验证版）topology mask oracle-cap experiment**：用 06-08 corruption-only VAE posterior 出的 GT-encoded topology mask 喂 HoLa diffusion，复现你之前 oracle 100% validity 的结果，确认在 deepcad_v7 + real_photo 协议下 oracle 通路仍然 SOTA。这是 visibility CVAE 的上界基准，1 天内能跑出来。
- [ ] CVAE 加 decode-time connectivity repair / constrained decoding，提升 `sample@K_valid`。
- [ ] `--image-token-dropout 0.1` 或 real image augmentation，缓解过拟合。
- [ ] 把 predicted topology mask 接入 HoLa diffusion，跑 no-mask / predicted-mask / oracle-mask 三方对照（**等 visibility-aware CVAE v1 出来再做，避免在错误任务下做 ablation**）。
- [x] 实现 gated / adapter decoder conditioning：保留 06-08 decoder 能力，同时让 DINO image tokens 通过 zero-init / gate / adapter 进入 decoder（06-14 已完成脚本和 smoke test）。
- [ ] ~~跑 06-14 gated adapter 正式训练与 full val/test sample@8~~（06-14 已跑完，结果在 [06-16 NOTE](./2026-06-16/NOTE.md)；不再继续这条 adapter 调参方向）。
- [ ] 如果 visibility-aware CVAE v1 仍不能超过 06-11 fidelity，考虑 image-conditioned latent diffusion over `z_topology` 或 explicit count head。

## 反指标 / 不要陷进去的局部坑（Anti-goals）

- 不要为了把 `exact_adj_acc` 再抬高几个点而连续多天死磕；这是 one-to-many 问题，exact 不是终点。
- 不要只盯单张样本或单一 split 的指标波动做大量微调。
- 不要在 topology 模块上无限打磨而迟迟不验证它对下游 diffusion 的真实收益。
- **（new, 06-17）不要在错误的任务定义下继续优化结构**：在确认 supervision target（loss / labels / training data）匹配实际任务之前，结构变种（adapter / latent diffusion prior / spanning tree 等）和 downstream ablation 都可能在解决错的问题。Task validity 优先级高于 structural search。
- **（new, 06-17）不要默认 GT face_adj = ground truth supervision**：GT 是众多合法 topology 中的一个，和单视角 image 的可观测性无关。把它整张当 1-to-1 label 喂 CVAE，会强制 model 在不可见区域瞎学。
