# 2026-06-14 实验笔记

## 今日目标

审计 [2026-06-13 frozen topology decoder + image prior](../2026-06-13/NOTE.md) 的完整 val/test 结果，判断 image-conditioned topology generation 下一步应该继续调 prior，还是切换到 decoder adapter / downstream 检查。

## 方向反思

- 长期目标仍是 `image -> (topology mask) -> HoLa diffusion -> B-Rep`，当前阶段是 image-conditioned topology generation，目标是给 HoLa diffusion 提供可用 topology mask。
- 06-11 和 06-13 已经分别验证了两个极端：
  - [06-11 in-context CVAE](../2026-06-11/NOTE.md#5-full-valtest-sample8-评估)：image tokens 直接进 decoder，sample@8 fidelity/validity 最好，但 posterior reconstruction 被稀释。
  - [06-13 frozen-prior](../2026-06-13/NOTE.md#1-frozen-topology-decoder--dino-image-prior)：保住 06-08 decoder，但单个 image-conditioned latent prior 无法提供足够 topology fidelity。
- 因此现在不是继续对同一个 `cond_prior_mu_f1` 做小调参。06-13 已经说明 `kl_beta` / `latent_mse_weight` 不是主瓶颈；主瓶颈是 image condition 的信息通道太窄。
- 今天选择“继续推进上一条，但换结构瓶颈”：保留 06-08 decoder 能力，同时让 image condition 以 gated adapter / lightweight control 的形式进入 decoder。

## 实验记录

### 1. 06-13 frozen-prior 结果审计

**为什么这么做**

- 06-13 的判定规则是：如果 frozen-prior full test 的 `sample8_best_f1` 明显超过 06-11 的 `0.7242` 且 validity 维持住，则继续该路线；否则说明单 latent prior 不足，应切换结构。

**具体实施**

- 读取 [06-13 test_metrics.json](../2026-06-13/outputs_cvae_frozen_prior/test_metrics.json) 和 [06-13 val_metrics.json](../2026-06-13/outputs_cvae_frozen_prior/val_metrics.json)。
- 与 [06-11 test_metrics.json](../2026-06-11/outputs_cvae_incontext_corrupt/test_metrics.json) 同口径比较。

**结果**

| model | split | cond_mu_f1 | cond_mu_valid | sample8_valid_any | sample8_valid_mean | sample8_best_f1 | sample8_mean_f1 | sample8_best_exact | ar_recon_f1 |
|-------|-------|------------:|--------------:|------------------:|-------------------:|----------------:|----------------:|-------------------:|------------:|
| 06-11 in-context CVAE | test | 0.5126 | 0.9530 | 1.0000 | 0.9239 | 0.7242 | 0.4828 | 0.3016 | 0.7839 |
| 06-13 frozen-prior | test | 0.5088 | 0.8882 | 0.9983 | 0.8058 | 0.6462 | 0.4469 | 0.1205 | 0.9478 |

结论：

- 06-13 的 `ar_recon_f1=0.9478` 证明 frozen 06-08 decoder 本身没有问题。
- 但 `sample8_best_f1=0.6462` 明显低于 06-11 的 `0.7242`，`sample8_valid_mean=0.8058` 也明显低于 06-11 的 `0.9239`。
- frozen-prior 路线按原判定规则应判为 **kill**：单个 256-d Gaussian image prior 不能充分承载 real photo 到 topology 的条件信息。

**参考 / 关联**

- 关联实验：[2026-06-13 frozen-prior](../2026-06-13/NOTE.md#1-frozen-topology-decoder--dino-image-prior)
- 对照实验：[2026-06-11 in-context CVAE](../2026-06-11/NOTE.md#5-full-valtest-sample8-评估)
- 结果文件：[06-13 test_metrics.json](../2026-06-13/outputs_cvae_frozen_prior/test_metrics.json)

### 2. 下一步：gated / adapter decoder conditioning

**为什么这么做**

- 06-13 说明“只让 image 预测一个 latent”太窄；06-11 说明“让 image tokens 直接进 decoder”有效但会破坏原 decoder。下一步应该走中间路线：保留 06-08 topology decoder 作为强 backbone，通过 zero-init / gated adapter 逐步引入 image condition，使初始模型等价于原 topology decoder，训练中只学习必要的 image control。
- 这个方向和 [ControlNet](https://arxiv.org/abs/2302.05543) / [T2I-Adapter](https://arxiv.org/abs/2302.08453) 的思路一致：锁住强预训练 backbone，用轻量可训练控制分支对齐外部条件，避免直接 fine-tune 把原能力洗掉。

**具体实施**

- 脚本：[topology_faceadj_cvae_gated_adapter.py](./topology_faceadj_cvae_gated_adapter.py)。
- 建议结构：
  - 从 [06-08 corruption-only best.pt](/mnt/d/python/experiments/2026-06-08/outputs_corrupt_only/best.pt) 加载 topology encoder/decoder。
  - topology 主干初始化时保持 06-08 能力；优先冻结 embedding / encoder / decoder 的大部分参数。
  - DINOv2 image tokens 经过 trainable compressor，压成少量 image control tokens。
  - 用 zero-init gate 或 adapter residual 把 image control 注入 decoder 输出，初始 residual 为 0，避免一开始破坏 topology decoder。
  - 训练目标沿用 06-11 的 full CVAE 评估路径，但 best metric 优先看 `sample@8_best_f1` 和 `sample@8_valid_mean`，不是单纯 `cond_prior_mu_f1`。

**判定规则**

- win：test `sample8_best_f1 > 0.7242`，`sample8_valid_mean >= 0.90`，同时 posterior `ar_recon_f1` 不低于约 `0.90`。
- iterate：`sample8_best_f1` 接近 06-11，但 `ar_recon_f1` 明显恢复；说明 adapter 保住了 decoder，但 image control 还需加强。
- kill：`sample8_best_f1` 仍低于 06-11 且 `ar_recon_f1` 也被破坏；说明 adapter 设计没有隔离好，应考虑 image-conditioned latent diffusion over `z_topology` 或先做 downstream utility check。

**参考 / 关联**

- 论文：[ControlNet](https://arxiv.org/abs/2302.05543)
- 论文：[T2I-Adapter](https://arxiv.org/abs/2302.08453)
- 关联实验：[2026-06-11 in-context CVAE](../2026-06-11/NOTE.md#5-full-valtest-sample8-评估)
- 关联实验：[2026-06-13 frozen-prior](../2026-06-13/NOTE.md#1-frozen-topology-decoder--dino-image-prior)

### 3. Level-1 zero-init gated logit delta 实施

**为什么这么做**

- 今天讨论后进一步明确：06-11 的 naive in-context 不是没有用，而是 image condition 入口太粗，直接把 decoder cross-attn memory 从 `[z]` 改成 `[z, image_tokens]`，导致 06-08 decoder 能力被洗掉；06-13 又说明只让 image 预测单个 `z` 太窄。
- gate 的思想不是要回到 LSTM 结构，而是借用“控制新信息写入多少”的机制。它和 [Gated Linear Attention](https://arxiv.org/abs/2312.06635)、Kimi / Gated DeltaNet 这类近期 gated memory update 思路相通，但当前实现不替换 attention，只在 frozen decoder logits 上加一个可控 image residual。
- 不直接上完整 [ControlNet](https://arxiv.org/abs/2302.05543) 结构。第一版只做最小 Level-1 adapter：验证 image tokens 以 zero-init residual 的方式进入 decoder，是否能超过 06-11，同时保住 06-08 decoder。

**具体实施**

- 新脚本：[topology_faceadj_cvae_gated_adapter.py](./topology_faceadj_cvae_gated_adapter.py)。
- 运行命令：[command.sh](./command.sh)。
- 从 [06-08 corruption-only best.pt](/mnt/d/python/experiments/2026-06-08/outputs_corrupt_only/best.pt) 加载 topology VAE。
- 默认冻结 topology encoder / decoder / output head；可用 `--train-topology` 显式放开，但第一版不建议。
- 原始 decoder 路径保持不变：

```text
base_logits
  = frozen TransformerDecoder(tgt=shifted topology prefix, memory=z_token)
```

- 新增 zero-init gated logit delta：

```text
Delta = CrossAttn(Q=LayerNorm(prefix_embedding), K=image_tokens, V=image_tokens)
Delta_logits = zero_init_linear(Delta)
logits = base_logits + image_gate * Delta_logits
```

- 这里没有同时把 `image_gate` 初始化成 0，因为 `zero_init_linear` 已经保证初始 `Delta=0`，模型初始严格等价于 06-08 decoder；`image_gate=1.0` 可以让 zero-init projection 第一轮就收到梯度，避免 image branch 死掉。
- loss 使用 prior/posterior 双路径：

```text
loss =
  prior_loss_weight * topology_loss(D(p_mu(image), image), y)
+ posterior_loss_weight * topology_loss(D(q_mu(topology), image), y)
+ kl_beta * KL(q(z|topology) || p(z|image))
+ latent_mse_weight * MSE(p_mu, q_mu)
```

- `posterior` 路径用于监控 adapter 是否破坏 06-08 decoder；`prior` 路径对应真实推理时的 `image -> topology`。
- checkpoint 仍只保存 trainable 参数，避免保存 frozen DINO backbone / frozen topology VAE。

**结果**

- 语法检查通过：
  - `conda run -n torch python -m py_compile experiments/2026-06-14/topology_faceadj_cvae_gated_adapter.py`
- 1 train sample + 1 val sample smoke test 通过，输出：[smoke output](/tmp/brepnet_0614_gated_adapter_smoke)。
- smoke 只验证代码路径，不代表指标；关键日志：
  - train step 成功。
  - val posterior greedy / conditional prior greedy / sample@1 都成功。
  - `ar_f1=1.0000`，说明 posterior adapter generate 路径没有破坏最小样本上的 frozen decoder。
  - `cond_f1=0.3077`、`sample1_valid=1.0000` 是随机初始化 / 单样本 smoke 的现象，不作为结论。
- 正式训练尚未运行，输出目录预定为：[outputs_cvae_gated_adapter](./outputs_cvae_gated_adapter)。
- 8 卡 `nn.DataParallel` 路径会在 frozen topology decoder 的 CUDA attention 内触发 `misaligned address` / `CUBLAS_STATUS_INTERNAL_ERROR`，该问题不是旧模型 checkpoint 损坏；同一脚本单卡可以正常训练。
- 由于今天目标是推进实验而不是继续排查 PyTorch 多卡底层问题，正式命令改为单卡运行：[command.sh](./command.sh)。当前训练命令使用 `CUDA_VISIBLE_DEVICES=0`、`--batch-size 16`、`--no-data-parallel`，并保留 `posterior_loss_weight=1.0` 的双路径训练设计。
- 单卡最终 smoke 通过，输出：[single final smoke](/tmp/brepnet_0614_gated_adapter_single_final_smoke)。关键日志：`train_loss=19.9235`，`val_loss=19.9796`，`ar_f1=1.0000`，`cond_valid=1.0000`。这只说明单卡代码路径可跑，不作为指标结论。

**参考 / 关联**

- 关联实验：[2026-06-11 in-context CVAE](../2026-06-11/NOTE.md#5-full-valtest-sample8-评估)
- 关联实验：[2026-06-13 frozen-prior](../2026-06-13/NOTE.md#1-frozen-topology-decoder--dino-image-prior)
- 论文：[ControlNet](https://arxiv.org/abs/2302.05543)
- 论文：[T2I-Adapter](https://arxiv.org/abs/2302.08453)
- 论文：[Gated Linear Attention](https://arxiv.org/abs/2312.06635)
- 脚本：[topology_faceadj_cvae_gated_adapter.py](./topology_faceadj_cvae_gated_adapter.py)

## 今日结论

- 当前 image-conditioned topology 最强仍是 06-11 in-context CVAE，而不是 06-13 frozen-prior。
- 06-13 的价值是定位瓶颈：image condition 不能只通过单个 latent 进入 decoder。
- 下一步应做 gated / adapter decoder conditioning，而不是继续调 frozen prior 的 KL/MSE；Level-1 zero-init gated logit delta 脚本已经实现并通过单卡 smoke test。
- 06-14 训练暂时采用单卡路径；多卡 `DataParallel` 已确认不稳定，后续如果确实需要加速，再单独整理 DDP 版本。
- 暂不建议直接把 06-11 或 06-13 接入 HoLa diffusion：06-11 fidelity 仍弱，06-13 已被 kill；先做一个能同时保住 decoder 和利用 image tokens 的 adapter 版本更合理。

## 待办

- [x] 实现 [topology_faceadj_cvae_gated_adapter.py](./topology_faceadj_cvae_gated_adapter.py)。
- [x] 做 1 train / 1 val smoke test。
- [x] 给出单卡训练命令和 full val/test sample@8 命令：[command.sh](./command.sh)。
- [ ] 跑 [command.sh](./command.sh) 第 1 条训练命令。
- [ ] 训练完成后跑 full val/test sample@8。
