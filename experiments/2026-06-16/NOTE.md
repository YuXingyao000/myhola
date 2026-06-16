# 2026-06-16 实验笔记

## 今日目标

汇总 [2026-06-14 gated adapter](../2026-06-14/NOTE.md#3-level-1-zero-init-gated-logit-delta-实施) 跑完后的完整 val/test 结果，判断它是否值得继续沿 adapter / decoder conditioning 方向推进。

## 实验记录

### 1. Level-1 zero-init gated logit delta 完整评估

**为什么这么做**

- [2026-06-11 in-context CVAE](../2026-06-11/NOTE.md#5-full-valtest-sample8-评估) 的 sample@8 fidelity / validity 最好，但 posterior reconstruction 被削弱。
- [2026-06-13 frozen-prior](../2026-06-13/NOTE.md#1-frozen-topology-decoder--dino-image-prior) 保住了 frozen topology decoder，但 image condition 只通过单个 latent 进入 decoder，sample@8 fidelity 和 validity 都不够。
- [2026-06-14 gated adapter](../2026-06-14/NOTE.md#3-level-1-zero-init-gated-logit-delta-实施) 试图走中间路线：保留 [06-08 corruption-only topology VAE checkpoint](../2026-06-08/outputs_corrupt_only/best.pt)，同时让 DINO image tokens 通过 zero-init gated logit residual 影响 decoder 输出。

**具体实施**

- 脚本：[topology_faceadj_cvae_gated_adapter.py](../2026-06-14/topology_faceadj_cvae_gated_adapter.py)。
- 命令：[command.sh](../2026-06-14/command.sh)。
- 输出目录：[outputs_cvae_gated_adapter](../2026-06-14/outputs_cvae_gated_adapter/)。
- best checkpoint：[best.pt](../2026-06-14/outputs_cvae_gated_adapter/best.pt)。
- 关键配置：
  - `epochs=100`
  - `batch_size=16`
  - `order_mode=wl`
  - `wl_rounds=3`
  - `kl_beta=0.01`
  - `latent_mse_weight=0.1`
  - `prior_loss_weight=1.0`
  - `posterior_loss_weight=1.0`
  - `image_backbone=dinov2`
  - `image_source=real_flux`
  - condition data：[deepcad_v7_cond](/mnt/d/data/deepcad_v7_cond)
- 训练按 `best_metric=cond_prior_mu_edge_f1` 保存；[best.pt](../2026-06-14/outputs_cvae_gated_adapter/best.pt) 来自 epoch 90。
- full eval 结果来自 [val_metrics.json](../2026-06-14/outputs_cvae_gated_adapter/val_metrics.json) 和 [test_metrics.json](../2026-06-14/outputs_cvae_gated_adapter/test_metrics.json)。

**结果**

Full val:

| metric | value |
|--------|------:|
| `num_samples` | 3505 |
| `cond_prior_mu_edge_f1` | 0.5861 |
| `cond_prior_mu_valid_strict_ratio` | 0.8439 |
| `sample8_valid_any` | 0.9937 |
| `sample8_valid_mean` | 0.8826 |
| `sample8_best_edge_f1` | 0.6823 |
| `sample8_mean_edge_f1` | 0.5680 |
| `sample8_best_exact` | 0.2254 |
| `ar_recon_edge_f1` | 0.9237 |
| `ar_recon_valid_strict_ratio` | 0.9692 |

Full test:

| metric | value |
|--------|------:|
| `num_samples` | 2424 |
| `cond_prior_mu_edge_f1` | 0.5451 |
| `cond_prior_mu_valid_strict_ratio` | 0.8337 |
| `sample8_valid_any` | 0.9963 |
| `sample8_valid_mean` | 0.8801 |
| `sample8_best_edge_f1` | 0.6523 |
| `sample8_mean_edge_f1` | 0.5336 |
| `sample8_best_exact` | 0.1976 |
| `ar_recon_edge_f1` | 0.9135 |
| `ar_recon_valid_strict_ratio` | 0.9773 |

Test 对照：

| model | `cond_mu_f1` | `cond_mu_valid` | `sample8_valid_any` | `sample8_valid_mean` | `sample8_best_f1` | `sample8_mean_f1` | `sample8_best_exact` | `ar_recon_f1` | `ar_valid` |
|-------|-------------:|----------------:|--------------------:|---------------------:|------------------:|------------------:|---------------------:|---------------:|-----------:|
| 06-11 in-context | 0.5126 | 0.9530 | 1.0000 | 0.9239 | 0.7242 | 0.4828 | 0.3016 | 0.7839 | 0.9435 |
| 06-13 frozen-prior | 0.5088 | 0.8882 | 0.9983 | 0.8058 | 0.6462 | 0.4469 | 0.1205 | 0.9478 | 0.9922 |
| 06-14 gated-adapter | 0.5451 | 0.8337 | 0.9963 | 0.8801 | 0.6523 | 0.5336 | 0.1976 | 0.9135 | 0.9773 |

结论：

- 06-14 没有超过 06-11 的主要目标：test `sample8_best_f1=0.6523`，低于 06-11 的 `0.7242`。
- 06-14 相对 06-13 有实质恢复：test `sample8_valid_mean` 从 `0.8058` 提升到 `0.8801`，`sample8_mean_f1` 从 `0.4469` 提升到 `0.5336`，`sample8_best_exact` 从 `0.1205` 提升到 `0.1976`。
- 06-14 的 posterior reconstruction 也没有像 06-11 那样被严重破坏：test `ar_recon_f1=0.9135`、`ar_valid=0.9773`，明显高于 06-11 的 `ar_recon_f1=0.7839`。
- 但 deterministic prior 的 strict validity 下降明显：test `cond_prior_mu_valid=0.8337`，低于 06-11 的 `0.9530` 和 06-13 的 `0.8882`。这说明 adapter 增强了平均 sample fidelity，但 `p_mu(image)` 单点生成还不够稳。
- 当前判断是 **iterate，不是 win，也不是 kill**：adapter 方向比 frozen-prior 更合理，但 Level-1 logit delta 太弱或控制位置不够好，还不足以超过 06-11 in-context。

**更精确的能力诊断**

- 一句话结论：06-14 学会了生成“合法、平均还行”的 topology，但没有学会从图像里稳定读出目标模型的具体拓扑结构，尤其是 face / edge count 和具体 edge placement。
- 这不是 decoder 不会生成 topology。06-14 的 test `ar_recon_f1=0.9135`、`ar_recon_exact_adj_acc=0.7446`、`ar_recon_valid_strict_ratio=0.9773` 说明只要给 topology posterior latent，frozen decoder 路径仍然有较强生成能力。
- 真正瓶颈在 image-conditioned prior / adapter：test `cond_prior_mu_edge_f1=0.5451`、`cond_prior_mu_exact_adj_acc=0.1894`、`cond_prior_mu_valid_strict_ratio=0.8337`。也就是说 image condition 没有把正确的 topology information 传进去。
- scale 预测是当前最直接的问题：test `cond_prior_mu_face_count_acc=0.4810`、`cond_prior_mu_edge_count_acc=0.4637`、`cond_prior_mu_edge_count_mae=6.7050`。face / edge count 错了以后，`sample8_best_exact` 和 edge F1 都会被系统性拖低。
- 06-14 的采样分布偏保守：相对 06-11，`sample8_mean_edge_f1` 从 `0.4828` 提升到 `0.5336`，但 `sample8_best_edge_f1` 从 `0.7242` 降到 `0.6523`，`sample8_best_exact` 从 `0.3016` 降到 `0.1976`，`sample8_diversity` 从 `0.2687` 降到 `0.2166`。这说明 06-14 更稳定地产生中等质量样本，但 sample@8 不太能撞到接近 GT 的 topology mode。
- 因此 06-14 有一点走偏：它更像一个 valid topology regularizer，而不是足够强的 image-conditioned topology generator。它学到了合理拓扑分布里的常见形状，却没有充分利用 real photo 中的 target-specific structure。
- 后续目标应从“提高平均合法性”转为“让图像条件控制拓扑规模和局部连接”。优先加强 `image -> face_count / edge_count`，再把 image condition 从 final logits 前移到 decoder hidden / layer-wise adapter；否则最终 logits delta 很难修正早期 autoregressive planning 错误。

**参考 / 关联**

- 关联实验：[2026-06-14 gated adapter](../2026-06-14/NOTE.md#3-level-1-zero-init-gated-logit-delta-实施)
- 对照实验：[2026-06-11 in-context CVAE](../2026-06-11/NOTE.md#5-full-valtest-sample8-评估)
- 对照实验：[2026-06-13 frozen-prior](../2026-06-13/NOTE.md#1-frozen-topology-decoder--dino-image-prior)
- 训练输出：[outputs_cvae_gated_adapter](../2026-06-14/outputs_cvae_gated_adapter/)
- full test 指标：[test_metrics.json](../2026-06-14/outputs_cvae_gated_adapter/test_metrics.json)
- full val 指标：[val_metrics.json](../2026-06-14/outputs_cvae_gated_adapter/val_metrics.json)

## 今日结论

- 06-14 gated adapter 证明“保 decoder + 加 image token control”比 06-13 frozen-prior 更有希望，尤其是 `sample8_mean_f1` 和 `sample8_valid_mean`。
- 但目前还没有击败 06-11 in-context；更准确地说，06-14 变成了更保守、更平均的 topology generator，缺少 target-specific topology mode 覆盖能力。
- 当前最明确的短板是 image-conditioned scale prediction：face count / edge count 错得多，导致 exact 和 edge placement 上限被压住。
- 优先方向：显式加强 `image -> face_count / edge_count`，再从 logit-level delta 升级到 decoder-hidden / layer-wise adapter，或在 decoder memory 侧保留 `[z]` 主路径的同时增加少量 gated image tokens，而不是只在最终 logits 加 residual。
- 评估时继续以 `sample8_best_f1`、`sample8_valid_mean`、`cond_prior_mu_valid` 和 `ar_recon_f1` 一起看；单看 `cond_prior_mu_f1` 会误导。

## 待办

- [ ] 设计 06-16/06-17 下一版 image-conditioned adapter：更强条件注入，但避免 06-11 那种直接洗掉 decoder。
- [ ] 如果继续用单卡训练，优先减少每轮 full generation 开销，把 full sample@8 留到 checkpoint 后统一评估。
- [ ] 后续若需要 8 卡，加速方案应单独整理 DDP 版本，不再使用 `nn.DataParallel`。
