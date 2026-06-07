# 2026-06-06 Learned Topology Diffusion 失败记录

## 今日结论

`Learned topology diffusion` 这条路线暂时判定失败。

失败现象不是单纯的 predictor 指标不够好，而是它接入 HoLa-BRep diffusion 后，在过拟合实验中也无法达到可生成合法 STEP 的 loss 区间。根据已有实验经验，diffusion validation loss 至少需要下降到约 `0.035` 才开始有机会生成合法 STEP；当前 learned-topology 版本最低只能到约 `0.05`，且比不加 learned topology 的 baseline 更差。因此继续沿这个配置堆训练没有意义。

更准确的结论是：

> 在 HoLa 的 random-padded latent diffusion 协议下，外部 image-to-adjacency predictor 生成的固定 slot topology bias 不能有效帮助 diffusion，反而会伤害 denoising。

这不是简单证明 “image -> topology 不可能”，但它强烈说明：简单 MLP / DETR-style fixed-slot predictor 不适合作为 HoLa diffusion 的外部拓扑注入模块。

## HoLa-BRep 背景补充

HoLa-BRep 的核心不是显式先生成拓扑图，而是把 B-Rep 的几何和拓扑关系压进 holistic surface latent 中。论文的关键观察是：两个 surface 的拓扑连接与它们的交线几何天然绑定，因此 topology learning 可以被转化为 pairwise surface intersection / curve reconstruction 问题。

HoLa 的 LDM 训练使用 fixed-length surface latent sequence。由于每个 B-Rep 的 surface 数量不同，论文采用 random padding / repeated padding 把 surface latent 扩展到固定最大长度。论文中也明确提到 random padding 显著优于 zero padding；虽然 random padding 仍会引入噪声和歧义，但它是 HoLa-BRep 正式生成质量成立的关键协议之一。

因此，不应该为了让 learned topology predictor 对齐 slot 而改用 zero padding。zero padding 本身会伤害 HoLa diffusion 的表现，这个方向不符合已有 HoLa 实验结论。

参考：

- HoLa arXiv: https://arxiv.org/abs/2504.14257
- HoLa HTML: https://ar5iv.labs.arxiv.org/html/2504.14257v3

## 为什么 Learned Topology Diffusion 会失败

### 1. Image-to-exact-topology gap

单图或渲染图可以提供整体形状、复杂度和局部可见边界，但很难稳定决定精确的 B-Rep face adjacency。

前面的 topology predictor 实验已经支持这一点：

- DINOv2 feature 确实包含非随机拓扑信号，`edge_auc` 约 `0.72`。
- 但是 edge prediction 卡在 `edge_f1 ~= 0.57`。
- `exact_matrix_acc` 极低。
- 24-view 和 DETR-style patch query 主要改善 face count / 复杂度判断，对具体 adjacency 的改善很小。

这说明 image feature 可以预测 “这个物体大概有多复杂”，但不能可靠预测 “第 i 个 face 和第 j 个 face 是否相邻”。

### 2. Fixed-slot topology 与 HoLa random padding 不兼容

HoLa diffusion 中的 latent sequence 是 random-padded / repeated surface latent sequence。也就是说，同一个真实 face 在训练时可能出现在不同 slot，padding 区域也可能重复已有 face。

Oracle topology 可以成立，是因为 dataset 在构造 batch 时知道 random padding 的 `index`，可以把 GT adjacency 按同一个 index 重排到当前 latent slot 顺序。

但 learned topology predictor 只看 image。它输出的是 toy predictor 学到的某种 “原始 face slot 顺序” adjacency。它并不知道当前 diffusion batch 中 latent sequence 的 random padding index，也不知道哪些 slot 是重复 face。

因此 learned topology bias 很可能在 diffusion 中变成：

```text
predicted_adj: predictor 原始 slot 顺序
latent_sequence: HoLa random padding 后的 slot 顺序
```

这会把错误的 face pair 加权到 self-attention 里。它不是弱监督，而是系统性噪声。

### 3. Noisy topology bias 会污染 denoising

不加 learned topology 时，HoLa diffusion 可以自己在 holistic latent space 里学习 geometry/topology coupling。

加 learned topology 后，一个只有约 `0.57 F1`、并且 slot 可能错配的 adjacency 被作为 soft self-attention bias 注入 denoiser。这样会强迫 denoiser 在错误的 face relation 上交换信息。

因此 observed failure 是合理的：

- validation diffusion loss 卡在约 `0.05`
- 低于合法 STEP 生成所需的经验阈值
- 比不加 learned topology 更差

这说明 noisy external topology bias 对 HoLa latent diffusion 是有害的。

## 对 DETR-style Topology Predictor 的定位

DETR-style predictor 并不是完全无效。它使用 DINOv2 patch tokens 和 learnable face queries，确实比 plain CLS+MLP 稍微改善了 face count。

但是它没有引入真正 DETR 的 set matching / Hungarian assignment。训练 loss 仍然要求 query 0 对齐 GT face 0，query 1 对齐 GT face 1。CAD face order 本身不是视觉可观测的稳定顺序，因此 learnable queries 仍然退化为 fixed-slot decoder。

所以 DETR-style 这次实验的负结果不是 “patch token 没用”，而是：

> 只替换 image feature extractor，保留 fixed-slot adjacency supervision，无法解决 topology prediction 的核心问题。

## 当前更稳妥的论文表述

建议表述为：

> The learned-topology bias route fails under HoLa's random-padded latent diffusion protocol. This failure indicates that a standalone image-to-adjacency predictor, especially a simple MLP or DETR-style fixed-slot predictor, is not sufficient as a topology bias for HoLa diffusion. The core issue is not only predictor capacity, but the mismatch between explicit fixed-slot topology prediction and HoLa's set-like random-padded holistic latent generation protocol.

中文表述：

> Learned topology diffusion 在 HoLa 的 random-padded latent diffusion 协议下失败。这个失败说明，独立的 image-to-adjacency predictor，尤其是简单 MLP 或 DETR-style fixed-slot predictor，不足以作为 HoLa diffusion 的 topology bias。核心问题不仅是 predictor 容量不够，而是显式 fixed-slot topology prediction 与 HoLa 的 set-like random-padded holistic latent generation 协议不兼容。

## 后续方向

短期内不继续做 zero-padding learned topology 实验，因为 zero padding 已知会伤害 HoLa diffusion，不符合正式实验协议。

后续如果继续研究 topology，应避免把 image predictor 输出的 fixed-slot adjacency 直接注入 diffusion。更合理的方向包括：

1. 让 topology 留在 HoLa latent / intersection decoder 内部，由 diffusion 自己学习 geometry-topology coupling。
2. 如果要显式预测 topology，应改成 set / graph / autoregressive formulation，而不是 fixed-slot adjacency BCE。
3. 如果要注入 diffusion，应设计和 random padding 协议兼容的 permutation-aware / padding-aware topology representation。
4. AR topology predictor 可以继续作为独立研究方向，但不应直接假设它能无缝作为 HoLa self-attention bias。

## AR Topology Generator 的后续严格版本

当前 AR v1 仍然使用 face-adjacency graph 作为拓扑表达：

```text
[BOS] [N_FACE=n] edge(i,j) edge(a,b) ... [EOS]
```

其中 `edge(i,j)` 表示 face `i` 和 face `j` 相邻。这个表达可以通过 grammar / legal decoding 保证每个 token 都是合法 face pair：

- `0 <= i < j < n_face`
- 不生成 padding face
- 不重复生成同一条 face pair
- 可选要求 edge token 按字典序递增
- 可选在生成少于 `n_face - 1` 条边时禁止过早 `EOS`

这些约束能让生成结果更像合法 face graph，但它们不能严格表达 B-Rep manifold topology。原因是 face-adjacency edge 只是 “两个 face 是否相邻” 的二值关系，不等价于真实 B-Rep topological edge。真实 B-Rep 中还存在 edge entity、vertex entity、loop / coedge / half-edge incidence 等结构；仅靠 binary face adjacency 无法验证完整 Euler 约束或 manifold incidence。

更严格的后续版本应考虑直接从 `data.npz["edge_face_connectivity"]` 建模 edge-face incidence list：

```text
[BOS] [N_FACE=n] [N_EDGE=m]
topo_edge_0_faces(i,j)
topo_edge_1_faces(a,b)
...
[EOS]
```

这样每个 generated topological edge 显式 incident to exactly two faces，更接近 B-Rep manifold 的 “一条边连接两个面” 语义。它仍然不是完整 STEP topology，因为还没有 vertex / loop / coedge 信息，但比 binary face adjacency 更接近真实 B-Rep 拓扑。

今天先不切换到这个严格版本。当前优先级仍是把 AR v1 做成一个干净的 image-conditioned face-graph generator：`count token + compact edge vocab + legal decoding + greedy/sample@K metrics`，先验证单张灰模图是否能生成比 MLP/DETR 更可靠的拓扑图。

## 后续工程整理：`src/brepnet/data`

今天讨论 DTGBrepGen-style topology generator 时，又发现一个工程问题：`src/brepnet/data` 下历史脚本较多，包含数据提取、condition 生成、list 过滤、debug、demo、fidelity sketch 等不同阶段遗留代码。当前实验还能直接使用 `deepcad_v7/data.npz["face_adj"]`、`edge_face_connectivity`、`deepcad_v7_cond/svr.npz` 和 cached HoLa face latent，但数据脚本本身已经不够清晰。

后续应单独整理 `src/brepnet/data`：

- 区分正式 pipeline、一次性迁移脚本、debug / visualization helper。
- 标注 v6/v7/v7_cond、24-view rotation、SVR/sketch/real-photo condition 的输入输出约定。
- 明确 `data.npz` 中 `face_adj`、`edge_face_connectivity`、`zero_positions` 的语义，尤其是 HoLa 提取脚本会把同一对 face 之间的多段 curve merge 成一条，因此当前数据更适合 binary face-adjacency，而不是严格 DTGBrepGen-style shared-edge-count `FeF`。
- 避免后续 topology 实验继续依赖隐式路径和历史脚本约定。

## 后续方向：Image-conditioned Topology VAE

DTGBrepGen-style binary face-adjacency Transformer VAE 的第一步目标，是先验证纯 topology prior：

```text
GT face_adj -> z_topo -> reconstructed / sampled face_adj
```

如果这个 prior 能够稳定生成合理 face-adjacency，下一步不应再回到简单 `image -> adjacency` MLP，而应该借鉴 CVAE / Probabilistic U-Net 式 structured output prediction：

```text
image = x
face_adj = y
topology latent = z
```

参考思想：

- CVAE structured prediction: 训练 `q(z | x, y)`、`p(z | x)` 和 `p(y | x, z)`，推理时从 `p(z | x)` sample，再 decode 出 structured output。
- Probabilistic U-Net: 面向 ambiguous segmentation，用 image-conditioned prior 生成多个 plausible segmentation hypotheses。我们的单图像到 topology 也是一对多问题，可以类比成 `p(face_adj | image)`。
- 3D latent generation / Michelangelo 类路线：先学习目标模态 latent space，再学习 image/text 到该 latent space 的条件生成模型。

对当前项目，建议分阶段：

1. 先冻结 topology VAE，训练 `image -> p(z_topo | image)`，拟合 topology encoder 的 posterior latent。
2. 再做完整 CVAE：posterior `q(z | image, face_adj)`，prior `p(z | image)`，decoder `p(face_adj | image, z)`。
3. 如果单 Gaussian prior 不够表达多模态，再考虑 image-conditioned latent diffusion over `z_topo`。

评估也应相应从单一 greedy F1 改成：

- posterior reconstruction F1
- image prior sample@K oracle F1
- face count MAE
- sample diversity
- predicted topology 接入 HoLa diffusion 后的 loss / valid STEP ratio

这条路线的前提是：纯 topology VAE 自身要足够强。如果 `GT face_adj -> z_topo -> face_adj` 都不能稳定恢复合理拓扑，那么 image-conditioned 版本没有可靠 decoder/prior 可用。

## DTGBrepGen-style Binary Face-adjacency VAE 初步结果

新增脚本：

```text
experiments/2026-06-06/topology_faceadj_vae.py
```

目标是先验证纯 topology prior，而不是图像条件预测：

```text
GT face_adj -> canonical order -> upper-triangle binary sequence -> z_topo -> face_adj
```

数据需求：

- 只需要 `deepcad_v7/{model_id}/data.npz["face_adj"]`
- `num_faces` 由 `face_adj.shape[0]` 得到
- 不需要 SVR image、HoLa latent、STEP、edge geometry

默认训练参数：

```bash
cd /mnt/d/python && python experiments/2026-06-06/topology_faceadj_vae.py \
  --output-dir experiments/2026-06-06/outputs_faceadj_vae \
  --epochs 100 \
  --batch-size 256 \
  --num-workers 8 \
  --order-mode degree \
  --kl-beta 0.1 \
  --kl-warmup-epochs 10 \
  --eval-generate-limit 128 \
  --prior-samples 128
```

训练到中途的观测：

```text
epoch 035
train_loss=0.1994
val_loss=0.1824
tf_f1=0.8847
ar_f1=0.7399
ar_count_mae=0.00
prior_conn=0.992
```

降低 KL 权重后，模型更接近 autoencoder，posterior autoregressive reconstruction 明显提升：

```bash
cd /mnt/d/python && python experiments/2026-06-06/topology_faceadj_vae.py \
  --output-dir experiments/2026-06-06/outputs_faceadj_vae_kl001 \
  --epochs 80 \
  --batch-size 256 \
  --num-workers 8 \
  --order-mode degree \
  --kl-beta 0.001 \
  --kl-warmup-epochs 10
```

观测结果：

```text
epoch 013
train_loss=0.1992
val_loss=0.1725
tf_f1=0.8837
ar_f1=0.8002
ar_count_mae=0.00
prior_conn=0.961
```

结论：

- `kl-beta=0.001` 明显优于 `kl-beta=0.1` 的 posterior AR reconstruction，说明当前瓶颈之一是 VAE regularization 过强，latent 被压得太接近标准正态后损失了 topology reconstruction 信息。
- 对后续 image-conditioned topology prior 来说，当前更重要的是先获得强 posterior topology decoder / latent space；prior sampling 质量可以作为第二阶段再优化。
- 需要继续关注 `ar_recon_exact_adj_acc`、`prior_density_mean` 和 KL 是否过低导致 prior 不可用。

测试集评估可以直接加载 checkpoint，不需要重新训练：

```bash
cd /mnt/d/python && python experiments/2026-06-06/topology_faceadj_vae.py \
  --eval-only \
  --eval-split test \
  --checkpoint experiments/2026-06-06/outputs_faceadj_vae_kl001/best.pt \
  --output-dir experiments/2026-06-06/outputs_faceadj_vae_kl001 \
  --batch-size 256 \
  --num-workers 8 \
  --eval-generate-limit 2424 \
  --prior-samples 512
```

其中 `eval-generate-limit` 控制 posterior autoregressive reconstruction 的样本数。周报应重点记录：

- `tf_edge_f1`
- `ar_recon_edge_f1`
- `ar_recon_exact_adj_acc`
- `ar_recon_count_mae`
- `prior_connected_ratio`
- `prior_density_mean`

## WL Canonical Ordering 结果

在 `kl-beta=0.001` 已经显著优于默认 KL 的前提下，进一步尝试把 canonical face order 从 `degree` 改成 `wl`：

```bash
cd /mnt/d/python && python experiments/2026-06-06/topology_faceadj_vae.py \
  --output-dir experiments/2026-06-06/outputs_faceadj_vae_wl_kl001 \
  --epochs 80 \
  --batch-size 256 \
  --num-workers 8 \
  --order-mode wl \
  --wl-rounds 3 \
  --kl-beta 0.001 \
  --kl-warmup-epochs 10 \
  --eval-generate-limit 256 \
  --prior-samples 256
```

对比 `degree + kl-beta=0.001`：

```text
degree, epoch 80:
val_loss=0.0597
tf_f1=0.9630
ar_f1=0.9229
exact_adj_acc=0.7305
ar_count_mae=0.00
prior_conn=0.9570
prior_density=0.2843
kl=2.2830

degree, best:
best ar_f1=0.9339 @ epoch 79
best exact_adj_acc=0.7578 @ epoch 77
```

```text
wl, epoch 80:
val_loss=0.0471
tf_f1=0.9723
ar_f1=0.9326
exact_adj_acc=0.7734
ar_count_mae=0.00
prior_conn=0.9727
prior_density=0.2960
kl=1.8002

wl, best:
best ar_f1=0.9326 @ epoch 76 / 80
best exact_adj_acc=0.7734 @ epoch 80
```

结论：

- WL ordering 相比 degree ordering 有稳定但不巨大的提升，尤其是 `exact_adj_acc` 从约 `0.73-0.76` 提升到 `0.7734`。
- `tf_f1` 和 validation loss 也有改善，说明更强的 topology-only canonical order 确实减少了 face index 噪声。
- `ar_f1` 没有突破新的大台阶，但后 10 个 epoch 的趋势仍在缓慢改善，epoch 80 也是 best loss / best exact，说明训练可能尚未完全收敛。
- 后续保留 `--order-mode wl --wl-rounds 3` 作为 topology VAE 默认设置。

## Edge-count Topology VAE v2

新增脚本：

```text
experiments/2026-06-06/topology_faceadj_vae_edgecount.py
```

相比 `topology_faceadj_vae.py`，这个版本在 sequence 中显式加入 graph edge count：

```text
[N_FACE=n] [N_EDGE=m] pair(0,1) pair(0,2) ... pair(max_faces-2,max_faces-1)
```

其中：

- `N_FACE` 是 face 数量。
- `N_EDGE` 是 binary face graph 中的邻接 face-pair 数量，即 `sum(face_adj upper triangle)`。
- 这个 edge count 不是真实 B-Rep topological edge count，也不是 `edge_face_connectivity` 行数。

Loss：

```text
face-count CE
+ edge-count CE
+ pair BCE
+ beta * KL
```

默认 `edge-count-loss-weight=0.2`，避免 edge-count 大类别分类在训练早期压过 pair reconstruction。生成时会用 `N_EDGE` 作为硬预算约束：

- 已生成 edge 数达到预测 `N_EDGE` 后，后续 pair 强制 `NO_EDGE`。
- 如果剩余有效 pair 数必须全部为 edge 才能达到 `N_EDGE`，则强制 `EDGE`。

训练命令：

```bash
cd /mnt/d/python && python experiments/2026-06-06/topology_faceadj_vae_edgecount.py \
  --output-dir experiments/2026-06-06/outputs_faceadj_vae_edgecount_wl_kl001 \
  --epochs 100 \
  --batch-size 256 \
  --num-workers 8 \
  --order-mode wl \
  --wl-rounds 3 \
  --kl-beta 0.001 \
  --kl-warmup-epochs 10 \
  --edge-count-loss-weight 0.2 \
  --eval-generate-limit 256 \
  --prior-samples 256
```

新增重点指标：

- `tf_edge_count_acc`
- `tf_edge_count_mae`
- `ar_recon_edge_count_acc`
- `ar_recon_edge_count_mae`
- `ar_recon_edge_count_token_mae`

这个版本的目标是提升 exact adjacency accuracy，而不是单纯提升 pair-level F1。

### Edge-count 版 sequence layout

`topology_faceadj_vae_edgecount.py` 的 target sequence 固定长度为：

```text
[N_FACE] [N_EDGE] [PAIR_0] [PAIR_1] ... [PAIR_434]
```

在默认 `max_faces=30` 时：

```text
num_pair_tokens = 30 * 29 / 2 = 435
sequence_length = 2 + 435 = 437
```

位置语义：

```text
position 0:
  face count token

position 1:
  graph edge count token

position >= 2:
  fixed upper-triangle face-pair adjacency token
```

pair token 顺序是固定 upper triangle：

```text
(0,1), (0,2), ..., (0,29),
(1,2), (1,3), ..., (1,29),
...
(28,29)
```

Token id：

```text
PAD     = 0
NO_EDGE = 1
EDGE    = 2
```

Face count token：

```text
N_FACE=n -> token = 3 + n
```

所以：

```text
N_FACE=7  -> token 10
N_FACE=30 -> token 33
```

Edge count token：

```text
edge_count_offset = 3 + max_faces + 1
                  = 34

N_EDGE=m -> token = 34 + m
```

例如：

```text
N_EDGE=12  -> token 46
N_EDGE=100 -> token 134
```

BOS token 位于所有 count token 之后：

```text
max_graph_edges = 30 * 29 / 2 = 435
BOS = edge_count_offset + max_graph_edges + 1
    = 34 + 435 + 1
    = 470
```

训练时 decoder 使用 teacher forcing 的 shifted input：

```text
decoder input:
[BOS] [N_FACE] [N_EDGE] [PAIR_0] ... [PAIR_433]

target:
[N_FACE] [N_EDGE] [PAIR_0] ... [PAIR_434]
```

因此 decoder 第一步预测 face count，第二步预测 edge count，之后按固定 upper-triangle 顺序逐个预测 pair adjacency。
