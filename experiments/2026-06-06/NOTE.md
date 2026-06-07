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

### Edge-count v2 结果

训练输出：

```text
experiments/2026-06-06/outputs_faceadj_vae_edgecount_wl_kl001
```

配置：

```text
order_mode=wl
wl_rounds=3
kl_beta=0.001
edge_count_loss_weight=0.2
epochs=100
batch_size=256
d_model=256
encoder_layers=4
decoder_layers=4
eval_generate_limit=256
prior_samples=256
```

与不带 edge count 的 WL baseline 对比：

```text
WL baseline, epoch 80:
val_loss=0.0471
tf_f1=0.9723
ar_f1=0.9326
exact_adj_acc=0.7734
ar_count_mae=0.00
prior_conn=0.9727
prior_density=0.2960
kl=1.8002
```

```text
WL + edge count, epoch 100:
val_loss=0.0444
tf_f1=0.9747
ar_f1=0.9409
exact_adj_acc=0.7852
ar_face_count_mae=0.00
ar_edge_count_mae=0.00
ar_edge_count_acc=1.0000
tf_edge_count_mae=0.00
tf_edge_count_acc=1.0000
prior_conn=0.7969
prior_density=0.2918
prior_edge_count_actual_mean=42.9141
prior_edge_count_token_mean=42.9141
kl=1.6987
```

后 12 个 epoch 中，edge-count 版本在 epoch 100 同时达到 best validation loss、best `ar_f1` 和 best `exact_adj_acc`：

```text
best ar_f1=0.9409 @ epoch 100
best exact_adj_acc=0.7852 @ epoch 100
best val_loss=0.0444 @ epoch 100
```

结论：

- 显式 `N_EDGE` 对 posterior autoregressive reconstruction 有帮助，但提升幅度不大：`ar_f1` 从 `0.9326` 提到 `0.9409`，`exact_adj_acc` 从 `0.7734` 提到 `0.7852`。
- edge-count 约束确实生效：posterior reconstruction 中 `ar_edge_count_mae=0.00`、`ar_edge_count_acc=1.0`，生成出的 actual graph edge count 与预测 edge-count token 一致。
- 有一个明显副作用：prior sampling 的 `prior_connected_ratio` 从 baseline 的 `0.9727` 降到 `0.7969`。这说明 edge-count 预算约束让无条件 prior 采样更严格匹配边数，但可能破坏连通性；或者当前 Gaussian prior 还没有学好 `N_FACE/N_EDGE/adjacency` 的联合分布。
- 因为当前下一步重点是 `GT face_adj -> z_topo -> face_adj` 的 posterior topology autoencoding，edge-count 版本仍然值得保留；但如果后续要做 unconditional topology prior 或 image-conditioned sampling，需要同时关注 `prior_connected_ratio`，不能只看 exact adjacency。

补充：同样看 epoch 80 时，edge-count 版本还没有超过 WL baseline：

```text
WL + edge count, epoch 80:
val_loss=0.0484
tf_f1=0.9714
ar_f1=0.9258
exact_adj_acc=0.7578
ar_edge_count_mae=0.00
prior_conn=0.8242
prior_density=0.3062
kl=1.6669
```

因此 edge-count 的收益需要结合更长训练看待。它在 epoch 90/100 继续改善，并在 epoch 100 超过 WL baseline；这说明该改动可能让模型收敛更慢，但后期上限略高。

## 四个 Topology VAE 的 Test Split 结果

测试集：

```text
src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt
num_samples=2424
```

四个实验均使用对应 `best.pt` 在 test split 上 eval-only，结果写在各自目录：

```text
experiments/2026-06-06/outputs_faceadj_vae/test_eval/test_metrics.json
experiments/2026-06-06/outputs_faceadj_vae_kl001/test_eval/test_metrics.json
experiments/2026-06-06/outputs_faceadj_vae_wl_kl001/test_eval/test_metrics.json
experiments/2026-06-06/outputs_faceadj_vae_edgecount_wl_kl001/test_eval/test_metrics.json
```

### 1. Degree ordering + `kl_beta=0.1`

```text
loss=0.1428
count_loss=0.0000
pair_loss=0.1313
kl=0.1151
kl_beta=0.1

tf_f1=0.9143
ar_f1=0.7811
exact_adj_acc=0.4187
ar_count_mae=0.00
prior_conn=0.9961
prior_density=0.4249
prior_count_mean=12.7090
```

这个最原始 VAE 的 test loss 最高，但不是 count loss 高。`count_loss=0.0`、`tf_count_acc=1.0`、`ar_count_acc=1.0`，说明 face count 已经学会。真正拖累的是 pair reconstruction：`pair_loss=0.1313`，以及较强 KL 正则导致 posterior latent 信息不足，因此 `ar_f1` 和 exact adjacency 明显偏低。

### 2. Degree ordering + `kl_beta=0.001`

```text
loss=0.0550
count_loss=0.0000
pair_loss=0.0529
kl=2.1209
kl_beta=0.001

tf_f1=0.9663
ar_f1=0.9185
exact_adj_acc=0.7463
ar_count_mae=0.00
prior_conn=0.9648
prior_density=0.2900
prior_count_mean=18.2773
```

降低 KL 权重后，test posterior reconstruction 大幅提升：`ar_f1` 从 `0.7811` 到 `0.9185`，exact 从 `0.4187` 到 `0.7463`。这验证了之前 validation 上的判断：当前 topology autoencoding 阶段需要弱 KL，过强 KL 会压掉拓扑重建信息。

### 3. WL ordering + `kl_beta=0.001`

```text
loss=0.0508
count_loss=0.0000
pair_loss=0.0490
kl=1.7691
kl_beta=0.001

tf_f1=0.9693
ar_f1=0.9295
exact_adj_acc=0.7706
ar_count_mae=0.00
prior_conn=0.9473
prior_density=0.2915
prior_count_mean=17.7012
```

WL ordering 在 test 上继续有效：相比 degree + low KL，`ar_f1` 从 `0.9185` 到 `0.9295`，exact 从 `0.7463` 到 `0.7706`。提升不大，但稳定，说明更强的 topology-only canonical ordering 确实减少了 face index 噪声。

### 4. WL ordering + edge count + `kl_beta=0.001`

```text
loss=0.0482
face_count_loss=0.0000
edge_count_loss=0.0001
pair_loss=0.0465
kl=1.7064
kl_beta=0.001

tf_f1=0.9723
ar_f1=0.9340
exact_adj_acc=0.7797
ar_face_count_mae=0.00
ar_edge_count_mae=0.00
ar_edge_count_acc=1.0000
prior_conn=0.8047
prior_density=0.3003
prior_count_mean=18.5156
prior_edge_count_actual_mean=41.9668
prior_edge_count_token_mean=41.9668
```

Edge-count 版本在 test 上也有小幅收益：相比 WL baseline，`ar_f1` 从 `0.9295` 到 `0.9340`，exact 从 `0.7706` 到 `0.7797`。同时 `ar_edge_count_mae=0.00`，说明显式 `N_EDGE` 约束在 posterior reconstruction 中确实生效。

但副作用同样存在：`prior_conn` 从 WL baseline 的 `0.9473` 降到 `0.8047`。因此 edge-count 版本更适合作为 posterior topology autoencoder / image-conditioned reconstruction decoder 的候选；如果目标是 unconditional prior sampling，则需要额外处理连通性或联合 prior 分布。

### Test 结论

- 四个实验在 test split 上的趋势与 validation 一致。
- 最重要的改动是降低 KL：`kl_beta=0.1 -> 0.001` 带来最大提升。
- WL ordering 带来稳定小幅提升，可以保留为默认 canonical ordering。
- Edge count 带来更小但真实的提升，主要体现在 exact adjacency accuracy 和 edge-count consistency。
- 最原始 VAE 的问题不是 count loss，而是 pair reconstruction 在强 KL 下不足。
- 当前最佳 test posterior reconstruction 是 `WL + edge count + kl_beta=0.001`：`ar_f1=0.9340`，`exact_adj_acc=0.7797`。
- 当前最佳 prior connected ratio 反而是最原始强 KL 版：`prior_conn=0.9961`，但它的 posterior reconstruction 太弱，不适合作为后续 image-conditioned topology decoder 的首选。



## Edge Count 后的下一步：把边放成连通且合理的图

今天观察到 `WL + edge count + kl_beta=0.001` 已经能把 posterior reconstruction 推到 `ar_f1=0.9340`、`exact_adj_acc=0.7797`，但 prior connected ratio 降到 `0.8047`。这说明显式 `N_EDGE` token 已经让模型学会了“应该生成多少条边”，但没有保证这些边被放在正确位置，也没有保证它们组成一个连通、局部合理的 face-adjacency graph。

直觉上，如果能修复 edge-count 版本的连通性和局部 degree 分布问题，exact adjacency accuracy 有希望突破 `0.8`。这不是简单调参问题，而是需要给 topology generator 更强的结构约束。

### 方向 1：加入 per-face degree 约束

当前序列是：

```text
+[N_FACE] [N_EDGE] [PAIR_0] [PAIR_1] ... [PAIR_K]
```

可以扩展为：

```text
+[N_FACE] [N_EDGE] [DEG_0] [DEG_1] ... [DEG_{N-1}] [PAIR_0] [PAIR_1] ... [PAIR_K]
```

或者把 degree sequence 作为 decoder condition embedding，而不是显式 token。对应 loss 可以加入 degree CE / L1：

```text
loss = pair_bce + face_count_ce + edge_count_ce + degree_loss + beta * kl
```

这个约束比全局 edge count 更细，因为 exact adjacency 的错误往往体现在某些 face row/column 的边数错了。edge count 只约束总数，degree sequence 则约束每个 face 的局部连接预算。

推理时还可以做 degree-budget constrained decoding：如果某个 face 的剩余 degree budget 已经为 0，则后续所有包含该 face 的 pair 强制为 no-edge；如果剩余 pair 数已经不足以满足某个 face 的 degree budget，则必须选择相关 edge。这个约束不需要反向传播，属于 decoding-time hard constraint。

### 方向 2：AR decoding 加连通性硬约束

Edge-count 版本 prior connected ratio 下降，说明模型会生成“边数正确但图断开”的 adjacency。可以在 AR sampling 中维护 union-find / DSU：

```text
current_edges -> connected components
remaining_edge_budget = N_EDGE - selected_edges
remaining_possible_pairs -> future edges
```

然后禁止会导致未来不可能连通的动作。例如，如果剩余 edge budget 只够把当前 components 接起来，那么后续就必须优先选择跨 component 的 pair。这个约束可以直接修 prior connected ratio，也能减少明显非法的 topology。

### 方向 3：两阶段生成：先 spanning tree，再 extra edges

更结构化的做法是把 face-adjacency graph 拆成：

```text
Stage 1: 生成一个 connected spanning tree，保证 N 个 face 连通
Stage 2: 在剩余 candidate pairs 中生成 N_EDGE - (N_FACE - 1) 条 extra edges
```

这样连通性天然成立，模型不再需要从 435 个 pair 的自由 0/1 序列里自己发现连通性规则。这个方向比直接输出完整 adjacency matrix 更符合拓扑生成的结构，但实现改动较大，可以放在 degree 约束和 constrained decoding 之后。

### 方向 4：soft graph loss 作为辅助项

在 teacher forcing logits 上可以构造 soft adjacency `P`，再加入一些可微图约束：

```text
edge_count_loss = |sum(P_ij) - E_gt|
degree_loss = sum_i |sum_j P_ij - deg_gt_i|
isolated_loss = sum_i relu(1 - sum_j P_ij)
```

更激进的做法是用 graph Laplacian 的第二小特征值 `lambda_2` 约束连通性：

```text
L = D - P
connectivity_loss = relu(tau - lambda_2(L))
```

但这个方向需要谨慎。`eigvalsh` 可以反传，但数值可能敏感，尤其是在图接近断开时。因此它适合作为后续辅助实验，不应作为第一优先级。

### 方向 5：缓解 teacher-forcing 和 AR inference 的分布偏移

现在 `tf_f1` 明显高于 `ar_f1`，说明 decoder 在 teacher forcing 下看到 GT prefix 时很强，但推理时吃自己生成的 prefix 会出现误差级联。可以尝试 prefix corruption / denoising training：

```text
训练时随机把一部分历史 prefix token 替换成错误 token 或模型采样 token，
让 decoder 学会从非完美 prefix 中恢复。
```

这比普通 scheduled sampling 更容易控制，也更贴合 topology sequence 的错误累积问题。

### 明天优先级

优先做 `per-face degree token / degree loss / degree-budget constrained decoding`。这是 edge count 的自然升级，改动量适中，而且直接针对 exact adjacency 的 row/column 结构。第二步再加 AR decoding 的连通性 hard constraint。若仍然卡在 `0.8` 附近，再考虑 spanning-tree-first topology VAE。


## Topology VAE 可视化工具计划

导师比较重视可视化，因此 topology VAE 不能只在周报里放 `ar_f1`、`exact_adj_acc`、`prior_conn` 这些数值。当前实验非常适合做 topology-level visualization：把模型学到的 adjacency structure、错误类型、edge-count 约束的收益和连通性问题都画出来。

建议明天开发一个独立脚本：

```bash
python experiments/2026-06-06/visualize_faceadj_vae.py \
    --checkpoint experiments/2026-06-06/outputs_faceadj_vae_edgecount_wl_kl001/best.pt \
    --output-dir experiments/2026-06-06/outputs_faceadj_vae_edgecount_wl_kl001/visuals_test \
    --split test \
    --num-samples 32
```

如果脚本需要同时支持原始 `topology_faceadj_vae.py` 和 edge-count 版本，可以加 `--model-type base/edgecount`，或者自动从 checkpoint/config 里判断。

### 输入

- checkpoint：当前 topology face-adjacency VAE 的 `best.pt`。
- split：`train/val/test`，优先支持 `val` 和 `test`。
- dataset list：沿用训练脚本里的 DeepCAD split list。
- num samples：默认 32，可选随机种子。
- output dir：输出 PNG 和 HTML gallery。

### 输出

建议输出：

```text
visuals_test/index.html
visuals_test/case_000.png
visuals_test/case_001.png
...
visuals_test/summary_degree.png
visuals_test/summary_connectivity.png
visuals_test/summary_edge_count.png
```

`index.html` 用于周报截图或直接给导师看，里面按 case 展示图和指标。

### 必做可视化 1：GT / Pred / Error adjacency matrix

每个样本画三张矩阵：

```text
GT face-adjacency | Pred face-adjacency | Error matrix
```

Error matrix 颜色建议：

```text
white = correct
red = false positive edge
blue = false negative edge
gray = padding / invalid face
```

这张图用于解释 exact adjacency accuracy。它能直观看到模型是否只是错了少数 edge，还是整体结构错了。

### 必做可视化 2：Face graph 对比

把 face-adjacency matrix 转成 graph：

```text
node = face
edge = face adjacency
```

每个样本并排画：

```text
GT graph | Pred graph | Difference graph
```

Difference graph 颜色建议：

```text
black edge = true positive
red edge = false positive
blue edge = false negative
node color = per-face degree error
component color / outline = connected component
```

这张图对当前 edge-count 实验尤其重要，因为它可以展示：

```text
edge count 对了，但边可能放错位置，甚至生成 disconnected graph。
```

这能自然支撑下一步加入 `degree constraint` 和 `connectivity constraint`。

### 必做可视化 3：Case gallery

不要只随机采样，应该按错误类型挑一些 representative cases：

```text
Exact reconstruction
1-edge error / near miss
multi-edge error
Disconnected prediction
high-face-count hard case
```

每个 case 标注：

```text
model_id
faces / edges
F1
exact_adj
edge_count_gt / edge_count_pred
connected_gt / connected_pred
degree_mae
```

这样周报里能说明：模型不是完全失败，而是在接近 exact 的地方出现结构性错误。

### 必做可视化 4：Edge count 和 degree 分布

Summary 图建议包含：

```text
GT edge count vs Pred edge count scatter
per-face GT degree vs Pred degree scatter
per-sample degree MAE histogram
edge-count error histogram
```

当前 edge-count 版本已经做到 `edge_count_acc=1.0`，但 exact adjacency 还没突破 `0.8`。如果 degree 图显示 degree distribution 仍然有误，就能直接证明：下一步应该加入 per-face degree token / degree loss。

### 必做可视化 5：Prior sample gallery

从 prior 采样一些 topology graph，分成：

```text
Connected prior samples
Disconnected prior samples
```

这张图用于解释 `prior_conn`。尤其是 edge-count 版本 test 上 `prior_conn=0.8047`，图像会比单个数字更直观：模型会生成边数正确但断开的 graph。

### 可选可视化 6：Training evolution

固定几个 validation samples，展示不同 epoch 的 reconstruction：

```text
epoch 10 | epoch 30 | epoch 60 | epoch 100
```

这个可以展示 topology VAE 逐步学会结构的过程。实现上需要多个 checkpoint，优先级低于 case gallery。

### 可选可视化 7：Latent interpolation

选两个 topology：

```text
z = (1 - t) * z_A + t * z_B
```

然后 decode 出一排 graph：

```text
A | t=0.25 | t=0.5 | t=0.75 | B
```

这个适合展示 VAE latent space 是否有连续拓扑语义，但不保证效果稳定。可以作为展示性实验，不作为主指标。

### 开发优先级

第一版只需要完成：

```text
1. matrix triplet: GT / Pred / Error
2. graph triplet: GT / Pred / Diff
3. case gallery index.html
4. summary degree / edge count / connectivity plots
```

这套可视化足够用于周报，并且能把当前结论讲清楚：`edge count` 改进了全局边数一致性，但 exact adjacency 仍受 degree distribution 和 connectivity 影响，所以下一步应加入 per-face degree 和 constrained decoding。
