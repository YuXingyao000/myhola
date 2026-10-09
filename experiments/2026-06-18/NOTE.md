# 2026-06-18 实验笔记

## 今日目标

基于 [06-17 visibility cache](../2026-06-17/NOTE.md#1-deepcad-v7-24-view-per-face-visibility-cache-脚本)，把 06-11 image-conditioned topology CVAE 的 full adjacency supervision 改成 visibility-aware soft pair supervision，验证任务定义修正是否带来更合理的 image→topology 信号。

## 方向反思

- North Star 仍然是 `image -> topology mask -> HoLa diffusion -> B-Rep`，当前阶段是 image-conditioned topology generation。
- [06-17 反思](../2026-06-17/NOTE.md#反思-3任务定义合理性new这是今天的核心) 已确认旧 CVAE loss 把单视图 one-to-many 任务强行变成 full GT adjacency 1-to-1 reconstruction，这是过去一周调 adapter / prior 仍然卡住的根因。
- 昨晚生成的 [deepcad_v7_visibility](/mnt/d/data/deepcad_v7_visibility) 已通过人工可视化检查，说明现在可以先修 supervision，再讨论结构。
- 今天不改模型结构、不改 image encoder、不改 count token 设计；只改 pair-token loss。`N_FACE` / `N_EDGE` 仍有 1-to-1 假设，用户已指出后续可能要改成 `N_VISIBLE_FACE` / visible edge count，但第一版先保持不动以隔离变量。

## 实验记录

### 1. Visibility-aware DINO in-context topology CVAE

**为什么这么做**

- [06-11 DINO in-context CVAE](../2026-06-11/NOTE.md#5-full-valtest-sample8-评估) 已证明 image condition 有 topology 信号，但旧 full-GT `pair_loss` 会强迫模型在不可见区域拟合唯一 GT adjacency。
- [Probabilistic U-Net](https://arxiv.org/abs/1806.05034) 和 [CVAE](https://papers.nips.cc/paper/5775-learning-structured-output-representation-using-deep-conditional-generative-models) 都支持把单输入多解的 structured output 看作 conditional distribution；对我们的任务而言，单图可观测区域应该强监督，不可见区域应该弱监督。
- 因此今天只把 pair BCE 改为连续 visibility soft weight：`pair_weight[i,j] = min(visible_fraction[i], visible_fraction[j])`，避免二值 threshold，也避免继续 full-map hard supervision。

**具体实施**

- 新脚本：[topology_faceadj_cvae_visibility.py](./topology_faceadj_cvae_visibility.py)。
- 训练 / 评估命令：[command.sh](./command.sh)。
- 继承 [06-11 脚本](../2026-06-11/topology_faceadj_cvae_incontext_corrupt.py) 的结构：
  - DINOv2 image encoder；
  - in-context decoder memory；
  - posterior `q(z | topology, image)`；
  - conditional prior `p(z | image)`；
  - topology prefix corruption；
  - 06-08 corruption-only VAE 初始化：[best.pt](/mnt/d/python/experiments/2026-06-08/outputs_corrupt_only/best.pt)。
- 新增 visibility 数据：
  - cache root：[deepcad_v7_visibility](/mnt/d/data/deepcad_v7_visibility)
  - real photo 使用 `view_000/visibility.npz`
  - training split 中缺 visibility 的 3 个模型会在 dataset filtering 阶段跳过，不维护第二份 list。
- loss 改动：
  - 原 `pair_loss = BCE * pair_mask`
  - 新 `pair_loss = BCE * pair_mask * pair_weight`
  - 同时保留 `full_pair_loss` 日志，方便观察新旧监督差异。
- metrics 改动：
  - 保留旧 full-GT 指标：`tf_edge_f1`、`cond_prior_mu_edge_f1`、`sample8_best_edge_f1`。
  - 新增主指标：`tf_visible_edge_f1`、`cond_prior_mu_visible_edge_f1`、`sample8_visible_best_edge_f1`、`sample8_visible_mean_edge_f1`。
- Face order：
  - visibility cache 本身保持 `data.npz` 原始 face order，不排序。
  - 训练命令先沿用 06-11 的 `--order-mode wl`，以便和 06-11 做最可比实验；脚本内部会用同一个 WL permutation 重排 `visible_fraction`，保证 pair weight 和 sequence pair 对齐。
  - 如果本实验有效，下一步再单独跑 `--order-mode none` 对齐 diffusion latent face order。

**结果**

- 语法检查通过：
  - `conda run -n torch python -m py_compile experiments/2026-06-18/topology_faceadj_cvae_visibility.py`
- 1 train / 1 val smoke 通过：
  - output: [/tmp/brepnet_0618_cvae_visibility_smoke](/tmp/brepnet_0618_cvae_visibility_smoke)
  - DINOv2 权重加载、visibility loader、weighted loss、posterior generation、conditional-prior generation、visible metrics 均跑通。
  - smoke 输出示例：`tf_visible_f1=0.7638`、`cond_visible_f1=0.2873`、`cond_valid=1.0000`。样本数为 1，只说明流程可运行，不代表实验结论。
- 06-19 正式训练初次启动失败：
  - 失败点：PyTorch `DataLoader` worker 在 batch collate 阶段报 `RuntimeError: Trying to resize storage that is not resizable`。
  - 原因：dataset 返回了变长的 `visible_fraction` tensor；训练实际只需要固定长度 `pair_weight`，多 worker collate 变长字段会失败。
  - 修复：[topology_faceadj_cvae_visibility.py](./topology_faceadj_cvae_visibility.py) 不再返回 `visible_fraction`，只返回固定长度 `pair_weight`；[command.sh](./command.sh) 增加 `set -euo pipefail`，避免训练失败后继续运行 eval 并触发 `best.pt` 不存在的连锁报错。
  - 追加验证：`batch_size=4`、`num_workers=2`、8 train / 4 val smoke 通过，输出 [/tmp/brepnet_0618_cvae_visibility_workers_smoke](/tmp/brepnet_0618_cvae_visibility_workers_smoke)。
- 正式训练已完成，输出目录：[outputs_cvae_visibility](./outputs_cvae_visibility/)。
- full val 结果来自 [val_metrics.json](./outputs_cvae_visibility/val_metrics.json)：
  - `cond_prior_mu_visible_edge_f1=0.5914`，`cond_prior_mu_edge_f1=0.5185`
  - `sample8_visible_best_edge_f1=0.7894`，`sample8_best_edge_f1=0.7226`
  - `sample8_valid_any=0.9997`，`sample8_valid_mean=0.9239`
  - `cond_prior_mu_valid_strict_ratio=0.9572`
- full test 结果来自 [test_metrics.json](./outputs_cvae_visibility/test_metrics.json)：
  - `cond_prior_mu_visible_edge_f1=0.5557`，`cond_prior_mu_edge_f1=0.4867`
  - `sample8_visible_best_edge_f1=0.7649`，`sample8_best_edge_f1=0.7012`
  - `sample8_valid_any=1.0000`，`sample8_valid_mean=0.9256`
  - `cond_prior_mu_valid_strict_ratio=0.9579`
  - `cond_prior_mu_face_count_acc=0.4047`，`cond_prior_mu_edge_count_acc=0.4002`
- 和 [06-11 DINO in-context CVAE](../2026-06-11/NOTE.md#5-full-valtest-sample8-评估) 对照：06-18 的 old full-GT `sample8_best_edge_f1=0.7012` 低于 06-11 的 `0.7242`，但 strict validity 保持在同一水平（`sample8_valid_mean=0.9256` vs `0.9239`）。这说明 visibility-aware loss 没有破坏合法性，但单看旧 full-GT F1 不能证明它更好。
- 关键解释：06-18 的 visible 指标更高，但目前缺少 06-11 同口径 visible 指标；更重要的是，它仍然只是 topology-CVAE 局部指标。下一步应验证这种“可见拓扑约束”接入 HoLa diffusion 后是否真的提高 STEP validity / condition consistency。

**参考 / 关联**

- 任务定义反思：[2026-06-17 NOTE](../2026-06-17/NOTE.md#反思-3任务定义合理性new这是今天的核心)
- visibility 数据脚本：[extract_visibility.py](../../src/brepnet/data/extract_visibility.py)
- visibility cache：[deepcad_v7_visibility](/mnt/d/data/deepcad_v7_visibility)
- 基线 CVAE：[2026-06-11 NOTE](../2026-06-11/NOTE.md)
- CVAE 论文：[Learning Structured Output Representation using Deep Conditional Generative Models](https://papers.nips.cc/paper/5775-learning-structured-output-representation-using-deep-conditional-generative-models)
- Probabilistic U-Net：[arXiv:1806.05034](https://arxiv.org/abs/1806.05034)

## 今日结论

- visibility-aware CVAE v1 已完成正式训练与 full val/test。
- 它把监督目标从 full GT adjacency 推向 visible region，但还不能用旧 full-GT F1 宣称击败 06-11；当前更合理的判断是：**validity 稳，可见区域 fidelity 有信号，但 count token / hidden topology 仍未解决**。
- 下一步不应继续局部调 CVAE，而应做 downstream 检验：用 GT adjacency 乘 continuous visibility 构造 oracle visible topology bias，训练 HoLa diffusion，看“只给可见拓扑约束”是否足够支撑 B-Rep 生成。

## 待办

- [x] 运行 [command.sh](./command.sh) 的训练命令。
- [x] 训练完成后运行 [command.sh](./command.sh) 的 full val/test sample@8。
- [ ] 06-19 开始 downstream visible-oracle topology bias diffusion 实验，判断 visibility supervision 是否服务最终 `image -> B-Rep` 目标。
