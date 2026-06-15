# 2026-06-13 实验笔记

## 今日目标

从 06-11 的 in-context CVAE 切到更保守的结构：冻结 06-08 corruption-only topology VAE，只训练 `image -> p(z_topology | image)`，判断图像条件接入时是否必须避免破坏强 topology decoder。

## 方向反思

- 长期目标仍是 `image -> (topology mask) -> HoLa diffusion -> B-Rep`，当前阶段是 image-conditioned topology generation，直接对应 [ROADMAP](../ROADMAP.md) 的开放问题 1：如何在保留 06-08 强 topology decoder 的同时接入 image 条件。
- 06-06 到 06-10 已经基本回答了纯 topology VAE 的局部问题：当前强 decoder 是 [06-08 corruption-only](../2026-06-08/NOTE.md#1-prefix-corruption训练方法侧)，test `ar_f1=0.9479`、`exact=0.7826`、`ar_valid=0.9922`。继续微调 `exact_adj_acc` 不符合 one-to-many 的最终任务。
- [06-11 in-context CVAE](../2026-06-11/NOTE.md#5-full-valtest-sample8-评估) 证明 DINO real photo condition 有信号：test `sample8_valid_any=1.0000`、`sample8_valid_mean=0.9239`，但 fidelity 弱，test `sample8_best_f1=0.7242`，而且 posterior reconstruction 从 06-08 的 `0.9479` 掉到 `0.7839`。这说明最大疑点不是“图片完全没有拓扑信号”，而是 in-context 结构稀释/破坏了原来的 topology decoder。
- 因此今天选择继续推进 CVAE，但跳出“继续修 in-context”的局部循环，改为隔离变量：固定 topology encoder/decoder，只训练 image-conditioned prior/adapter。

## 实验记录

### 1. Frozen topology decoder + DINO image prior

**为什么这么做**

- 06-11 第一版 CVAE 把 image tokens 拼进 decoder memory，虽然让 `p(z | image)` 学到了一些 topology signal，但同时把 06-08 的强 topology decoder 拉弱。今天的实验直接回答一个更干净的问题：如果完全保留 06-08 corruption-only 的 topology latent space 和 decoder，只训练 DINO real photo 到 `z_topology` 的 conditional prior，能否提升 `cond_prior_mu_f1` / `sample@K_best_f1`，同时保持高 strict validity。
- 这个设计和 [CVAE structured output](https://papers.nips.cc/paper/5775-learning-structured-output-representation-using-deep-conditional-generative-models) 的拆分一致：训练时有 `q(z | x, y)`，推理时用 `p(z | x)`；也和 [Probabilistic U-Net](https://arxiv.org/abs/1806.05034) 的一对多 structured prediction 视角一致。这里 `x` 是 real photo，`y` 是 topology adjacency，sample@K 比 exact recovery 更接近最终使用方式。

**具体实施**

- 新脚本：[topology_faceadj_cvae_frozen_prior.py](./topology_faceadj_cvae_frozen_prior.py)。
- 命令记录：[command.sh](./command.sh)。
- 冻结的 topology checkpoint：[06-08 corruption-only best.pt](/mnt/d/python/experiments/2026-06-08/outputs_corrupt_only/best.pt)。
- 使用数据：
  - raw root: [/mnt/d/data/deepcad_v7](/mnt/d/data/deepcad_v7)
  - condition root: [/mnt/d/data/deepcad_v7_cond](/mnt/d/data/deepcad_v7_cond)
  - train list: [deduplicated_deepcad_training_7_30.txt](../../src/brepnet/data/list/deduplicated_deepcad_training_7_30.txt)
  - val list: [deduplicated_deepcad_validation_7_30.txt](../../src/brepnet/data/list/deduplicated_deepcad_validation_7_30.txt)
  - test list: [deduplicated_deepcad_testing_7_30.txt](../../src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt)
- 模型结构：
  - `topology = FaceAdjTransformerVAE`，从 06-08 checkpoint 加载后全部冻结。
  - frozen topology encoder 给出 `q(z | topology)`，作为 image prior 的训练目标。
  - DINOv2 ViT-L frozen backbone + trainable projection，输出 image global token。
  - trainable prior head 输出 `p_mu, p_logvar = p(z | image)`。
  - decoder 使用 frozen topology decoder，训练时用 `p_mu` teacher-forcing decode adjacency。
- 模型数据流：

```text
训练时：

GT topology sequence y
  -> frozen topology encoder E_topo
  -> q_mu, q_logvar = q(z | y)
       |
       | 只作为监督目标，不反传到 E_topo
       v

real photo x
  -> frozen DINOv2 backbone
  -> trainable DINO projection
  -> trainable prior MLP
  -> p_mu, p_logvar = p(z | x)
  -> frozen topology decoder D_topo(p_mu, teacher-forced prefix)
  -> predicted topology tokens

推理时：

real photo x
  -> p_mu, p_logvar = p(z | x)
  -> z = p_mu 或 z ~ N(p_mu, exp(p_logvar))
  -> frozen topology decoder autoregressive generate
  -> topology adjacency mask
```

- 和 06-11 in-context CVAE 的区别：
  - 06-11：`q(z | topology, image)`，`p(z | image)`，decoder 每一步都看 `concat([z_token, image_tokens])`，因此 image 直接进入 decoder memory；优点是容量大，缺点是会改变/稀释原来的 topology decoder。
  - 06-13：`q(z | topology)` 和 decoder 都来自 06-08 并冻结，image 只负责预测 latent prior `p(z | image)`；优点是隔离变量、保住 06-08 decoder，缺点是所有图像信息都必须压进一个 256-d latent。
- 这版严格来说不是重新端到端训练一个 full CVAE，而是 **frozen topology VAE + image-conditioned latent prior**。它要回答的问题是：DINO real photo 能不能预测到 06-08 topology latent space 里的可解码区域。
- loss：
  - frozen decoder 上的 face count / edge count / pair token reconstruction loss。
  - `KL(q(z | topology) || p(z | image))`，默认 `kl_beta=0.01`。
  - `MSE(p_mu, q_mu)`，默认 `latent_mse_weight=0.1`，用于稳定 latent mean 对齐。
- checkpoint 保存策略：
  - `best.pt` 只保存 trainable 的 DINO projection + prior head 参数。
  - frozen DINO backbone 和 frozen topology VAE 不重复保存，避免 1GB 级 checkpoint。

**结果**

- 语法检查通过：
  - `conda run -n torch python -m py_compile experiments/2026-06-13/topology_faceadj_cvae_frozen_prior.py`
- 1 train sample + 1 val sample smoke test 通过，输出：[smoke output](/tmp/brepnet_0613_frozen_prior_smoke)。
- 正式训练与 full val/test `sample@8` 已完成，输出目录：[outputs_cvae_frozen_prior](./outputs_cvae_frozen_prior)。
- 结果文件：
  - [metrics.json](./outputs_cvae_frozen_prior/metrics.json)
  - [history.json](./outputs_cvae_frozen_prior/history.json)
  - [val_metrics.json](./outputs_cvae_frozen_prior/val_metrics.json)
  - [test_metrics.json](./outputs_cvae_frozen_prior/test_metrics.json)
  - [best.pt](./outputs_cvae_frozen_prior/best.pt)

Full split 结果：

| split | samples | ar_recon_f1 | ar_recon_valid | cond_mu_f1 | cond_mu_valid | sample8_valid_any | sample8_valid_mean | sample8_best_f1 | sample8_mean_f1 | sample8_diversity |
|-------|--------:|------------:|---------------:|------------:|--------------:|------------------:|-------------------:|----------------:|----------------:|------------------:|
| val | 3505 | 0.9515 | 0.9892 | 0.5474 | 0.8950 | 0.9994 | 0.8122 | 0.6696 | 0.4762 | 0.3216 |
| test | 2424 | 0.9478 | 0.9922 | 0.5088 | 0.8882 | 0.9983 | 0.8058 | 0.6462 | 0.4469 | 0.3221 |

和 [2026-06-11 in-context CVAE](../2026-06-11/NOTE.md#5-full-valtest-sample8-评估) 的 test 对比：

| model | cond_mu_f1 | cond_mu_valid | sample8_valid_any | sample8_valid_mean | sample8_best_f1 | sample8_mean_f1 | sample8_best_exact | ar_recon_f1 |
|-------|------------:|--------------:|------------------:|-------------------:|----------------:|----------------:|-------------------:|------------:|
| 06-11 in-context CVAE | 0.5126 | 0.9530 | 1.0000 | 0.9239 | 0.7242 | 0.4828 | 0.3016 | 0.7839 |
| 06-13 frozen-prior | 0.5088 | 0.8882 | 0.9983 | 0.8058 | 0.6462 | 0.4469 | 0.1205 | 0.9478 |

观察：

- 06-13 成功保住了 frozen topology decoder：test `ar_recon_f1=0.9478`，几乎回到 06-08 corruption-only 的 `0.9479`。这说明 06-11 的 `ar_recon_f1=0.7839` 下降确实来自 in-context CVAE 改动/训练稀释了原 decoder。
- 但 image-conditioned prior 没有变强：test `cond_mu_f1=0.5088`，和 06-11 的 `0.5126` 基本持平；`cond_mu_valid=0.8882` 明显低于 06-11 的 `0.9530`。
- `sample@8` 更差：test `sample8_best_f1=0.6462`，低于 06-11 的 `0.7242`；`sample8_valid_mean=0.8058`，低于 06-11 的 `0.9239`。
- `sample8_diversity=0.3221` 高于 06-11 的 `0.2687`，但这是“更散”而不是“更好”：更多样的样本没有带来更高 best F1 或 validity。

结论：

- 按预设判定规则，这条 frozen-prior 路线应判为 **kill / 不继续作为主线**。
- 它回答了一个关键问题：**单个 256-d image-conditioned Gaussian latent 不足以把 real photo 中的 topology 信息传给 frozen topology decoder**。
- 下一步不应继续只调 `kl_beta` / `latent_mse_weight`。更合理的方向是保留 06-08 decoder 能力，但让 image condition 以轻量 adapter / gated in-context 的方式进入 decoder，而不是只压进一个 latent。

**判定规则**

- win：full test 上 `sample8_best_f1` 明显超过 06-11 的 `0.7242`，同时 `sample8_valid_any` 接近 `1.0`、`sample8_valid_mean` 不低于约 `0.90`。
- iterate：validity 保持高，但 `sample8_best_f1` 只小幅提升；下一步调 `kl_beta` / `latent_mse_weight` 或加 image augmentation。
- kill：`cond_prior_mu_f1` 和 `sample8_best_f1` 低于 06-11，说明仅做 latent prior regression 不足以把图片信息转成可用 topology，需要回到 decoder conditioning 或 latent diffusion over `z_topology`。

**参考 / 关联**

- 关联实验：[2026-06-08 corruption-only](../2026-06-08/NOTE.md#1-prefix-corruption训练方法侧)
- 关联实验：[2026-06-11 in-context CVAE](../2026-06-11/NOTE.md#1-in-context-real-photo-topology-cvae--corruption-only-训练)
- 论文：[CVAE structured output](https://papers.nips.cc/paper/5775-learning-structured-output-representation-using-deep-conditional-generative-models)
- 论文：[Probabilistic U-Net](https://arxiv.org/abs/1806.05034)
- 脚本：[topology_faceadj_cvae_frozen_prior.py](./topology_faceadj_cvae_frozen_prior.py)

## 今日结论

- 今天不应该继续做 topology VAE exact 的小修小补，也不应该直接把 06-11 CVAE 接 diffusion。
- “冻结强 topology decoder + 只训 image prior”恢复了 06-08 decoder 能力，但没有改善 image-conditioned fidelity；06-13 是一个有价值的负结果。
- 当前最强 image-conditioned topology 仍是 06-11 in-context CVAE：test `sample8_valid_any=1.0000`、`sample8_best_f1=0.7242`。
- 下一步应该尝试 adapter / gated in-context decoder conditioning：让 image tokens 能影响 decoder，但用 zero-init / gate / freeze discipline 避免再次破坏 06-08 topology decoder。

## 待办

- [x] 跑 [command.sh](./command.sh) 第 1 条训练命令。
- [x] 训练完成后跑 full val sample@8。
- [x] full val 正常后跑 full test sample@8。
- [x] 和 [2026-06-11 #5](../2026-06-11/NOTE.md#5-full-valtest-sample8-评估) 对齐汇报结果。
- [ ] 06-14 规划 adapter / gated in-context decoder conditioning。
