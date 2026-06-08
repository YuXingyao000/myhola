# 2026-06-08 实验笔记

## 今日目标

在 06-06 最优 topology VAE（`WL + edge_count + KL=0.001`，test `ar_f1=0.934`、`exact_adj_acc=0.780`）的基础上，独立验证两条优化方向，看哪条能把 exact adjacency 推过 0.8：
方向 A 是训练时的 prefix corruption（解决 teacher-forcing vs autoregressive 的 exposure bias），方向 B 是架构里的 per-face degree token + degree-budget constrained decoding（给 row-sum 加局部硬约束）。再加一个 A+B 的组合。

## 上下文：06-06 当前瓶颈

- `WL + edge_count + KL=0.001` 是 06-06 最优 posterior reconstruction：`tf_f1=0.972, ar_f1=0.934, exact_adj_acc=0.780, prior_conn=0.805`。
- 两个 gap 共存：
  - `tf_f1 - ar_f1 ≈ 4 个点`：teacher forcing 上限和 AR 推理实际表现的差距，属于 exposure bias，prefix 累积错误。
  - `ar_f1 = 0.93` 但 `exact_adj_acc = 0.78`：AR 已经 pair-level 很准，但 22% 的 graph 整张错。这些错误更像结构性错位（某些 face 的 row-sum 不对），而不是均匀的随机错误。
- CVAE 视角下 `prior_conn` 不是优化目标：image 会作为强 condition 约束 z 的分布，所以 unconditional prior 的连通率不会直接影响下游 image-conditioned 重建。
- 因此今天三个变种都不去管 prior connectivity，只看 `ar_recon_*` 指标。

## 实验记录

### 1. Prefix corruption（训练方法侧）

**为什么这么做**

- 06-06 的 `tf_f1 - ar_f1 ≈ 4 个点` gap 是免费午餐：模型容量已经够，只是训练时只见过完美 prefix，inference 用自己生成的 prefix 时就崩了。
- Prefix corruption 是 scheduled sampling 的简化版：训练时按一定概率随机把 decoder input 的 pair token 翻转（NO_EDGE ↔ EDGE），target 不变。让 decoder 学会从含错 prefix 中恢复。
- 这是训练时改动，不动架构、不动 loss、不动 vocab，是最低成本的 ablation。

**具体实施**

- 脚本：`experiments/2026-06-08/topology_faceadj_vae_corrupt.py`
- Token / 序列 / 模型 / loss / generate 全部沿用 06-06 edge-count baseline。唯一新增：
  - `corrupt_pair_prefix(decoder_input, decoder_mask, corrupt_prob)`: 只翻转 NO_EDGE/EDGE 位置，特殊 token（BOS/N_FACE/N_EDGE/PAD）天然不在这两个 id 上，因此不会被破坏。
  - 训练循环改成 `encode → reparameterize → corrupt → decode`，eval 不腐蚀。
  - `--corrupt-prob 0.15`、`--corrupt-warmup-epochs 10` 线性 ramp，前 10 epoch 让模型先学到稳定 prefix 再加噪声。
- 训练命令在 `experiments/2026-06-08/command.sh` 第 1 段，`CUDA_VISIBLE_DEVICES=0`。
- 关键配置：
  - data root: `/mnt/d/data/deepcad_v7`
  - order_mode=wl, wl_rounds=3, kl_beta=0.001, kl_warmup=10
  - edge_count_loss_weight=0.2, corrupt_prob=0.15, epochs=100, batch_size=256

**结果**

- 还没跑，回家直接 `bash experiments/2026-06-08/command.sh`。
- 预期：`ar_f1` 0.93 → 0.95，`exact_adj_acc` 0.78 → 0.82+。
- 输出：`experiments/2026-06-08/outputs_corrupt_only/best.pt`、`metrics.json`、`history.json`、`outputs_corrupt_only.log`。

### 2. Per-face degree token + degree-budget constrained decoding（架构侧）

**为什么这么做**

- 06-06 的 `edge_count` token 解决了"总边数对不对"，但没解决"边放在哪"。`exact_adj_acc` 仍卡在 0.78 大概率是因为某些 face 的 row-sum 错了：边数总数对了，但被分配到错误的 face 上。
- per-face degree 是 row-sum 的精确控制信号，比 edge_count 更细粒度。给每个 face 一个 degree token，模型必须先决定每张面要连几条边，再生成具体 pair。
- Degree 的精确语义：`face_adj.sum(axis=1)` 在 canonical ordered adjacency 上算得到的 row-sum，不去碰 `edge_face_connectivity` 的半边记录（HoLa 已经把同一对 face 间多段 curve merge 成一条，所以两边数值应该一致）。
- 推理时同时用 edge_count budget 和 per-face degree budget 做硬约束：`remaining_deg[i]==0` 或 `remaining_deg[j]==0` 强制 NO_EDGE，`remaining_deg[i] >= 剩余可包含 face i 的 pair 数` 强制 EDGE。

**具体实施**

- 脚本：`experiments/2026-06-08/topology_faceadj_vae_degree.py`
- 序列布局变化：从 06-06 的 `[N_FACE][N_EDGE][PAIR_0..PAIR_434]`（长度 437）扩展为 `[N_FACE][N_EDGE][DEG_0..DEG_29][PAIR_0..PAIR_434]`（长度 467）。
  - 新 token 范围：`degree_offset = edge_count_offset + max_graph_edges + 1`，每个 degree token 是 `degree_offset + degree_value`，degree 取值范围 `[0, max_faces - 1]`。
  - BOS 也相应往后挪。
- Loss：`face_count_ce + edge_count_loss_weight * edge_count_ce + degree_loss_weight * degree_ce + pair_bce + kl_beta * kl`。`--degree-loss-weight 0.2`。
- Generation 加入 4 个 step：face count → edge count → 每个 face 的 degree token → 每个 pair 的 NO_EDGE/EDGE。pair 阶段同时维护 `edge_count` 和 `per-face degree` 两个 budget，互相不冲突时按 logits 选，冲突时优先 NO_EDGE 保安全。
- 新增评估指标：`tf_degree_acc`、`tf_degree_mae`、`ar_recon_degree_acc`、`ar_recon_degree_mae`、`ar_recon_degree_self_consistent`（生成出的 adjacency 的 row-sum 和 degree token 是否一致）。
- 训练命令 `CUDA_VISIBLE_DEVICES=1`。

**结果**

- 还没跑。预期：`exact_adj_acc` 0.78 → 0.85+；degree-budget hard constraint 直接修复 row-sum 错位，理论上能消掉一大批 1-2 edge error。
- 输出：`experiments/2026-06-08/outputs_degree_only/best.pt`、相关 metrics 和 log。

### 3. Corruption + degree（组合实验，2 的迭代）

**为什么这么做**

- 1 和 2 解决的是两个独立问题：1 解决 prefix 累积错误，2 解决 row-sum 错位。理论上可以叠加。
- 但 corruption 必须只腐蚀 NO_EDGE/EDGE pair token；degree token 是分类预算控制信号，腐蚀会破坏 hard constraint 的语义。脚本里的 `corrupt_pair_prefix` 用 `(token == NO_EDGE) | (token == EDGE)` 自然只命中 pair 位置，degree token 落在更高 id 范围，不受影响。

**具体实施**

- 脚本：`experiments/2026-06-08/topology_faceadj_vae_both.py`，在 degree 版基础上加入 corruption 训练逻辑。
- 训练循环：`encode → reparameterize → shifted_decoder_input → corrupt pair tokens (degree 不变) → decode`。
- 训练命令 `CUDA_VISIBLE_DEVICES=2`，参数同时包含 `--degree-loss-weight 0.2 --corrupt-prob 0.15`。

**结果**

- 还没跑。预期：如果 1 和 2 都各自有效，组合应该比单独的更好或至少持平。如果差于单独的某个，说明两者有 interaction，需要单独调参。
- 输出：`experiments/2026-06-08/outputs_both/best.pt`、metrics、log。

## 评估方法

所有三个实验的 `metrics.json` 都和 06-06 兼容，可以直接和 `experiments/2026-06-06/outputs_faceadj_vae_edgecount_wl_kl001/test_eval/test_metrics.json` 对比。

要看的核心指标：

- `tf_edge_f1` / `ar_recon_edge_f1`: pair-level 基本指标
- `ar_recon_exact_adj_acc`: 整张 adjacency matrix 全对的比例（关键指标，今天的优化目标）
- `ar_recon_degree_acc` / `ar_recon_degree_self_consistent`（仅 degree / both）: degree token 是否准、生成的 adjacency 是否和 degree token 自洽
- `tf_edge_f1 - ar_recon_edge_f1`: 看 corruption 是否真的缩小了 exposure bias gap

不再优化 `prior_connected_ratio`，CVAE 阶段会被 image condition 替代。

## 今日结论

- 三个脚本都已经写好并通过 syntax check：`topology_faceadj_vae_corrupt.py`、`topology_faceadj_vae_degree.py`、`topology_faceadj_vae_both.py`。
- 三个跑批的 shell 命令在 `command.sh`，三张卡同时跑。
- 训练目标统一是 `ar_recon_exact_adj_acc`（best checkpoint 用这个 score 选）。

## 4. Discussion: 当前优化方向的合理性反思

写完三个脚本后再讨论时浮现的怀疑，记录在这里。

### 4.1 AR v1（06-02 image-conditioned topology AR）失败的真实原因

`experiments/2026-06-02/topology_ar_train.py` 的失败主要不是 AR 架构问题，而是**输入信号不够**：单图 → topology 的 edge_f1 上限大概在 0.57（DINOv2 frozen feature 的能力上限）。今天的 topology VAE 是 `GT face_adj → z → face_adj`，输入端有完整 GT，不存在那个信息瓶颈，因此 AR v1 的失败模式不会直接搬过来。

### 4.2 Degree token 方法的潜在风险

写完后回头看，degree 这条路可能有几个隐患：

- **Degree token 在 GT 上太"准"**：encoder 看到整张 adj，degree 信息几乎是 latent 的免费 leak。decoder 在 teacher forcing 下能轻松预测 face_count / edge_count / 30 个 degree token，但这些预测拉长了 sequence、加大了模型容量负担，但不一定贡献新信息。最坏情况是模型学到 shortcut "degree 对了之后 pair 用 budget 填进去"，pair_loss 反而不再下降。
- **Constrained decoding 制造新的训练-推理分布偏移**：degree-budget hard constraint 在 inference 才启用，训练时不启用。这本身就是一种新 exposure bias。
- **CVAE 阶段可能反过来变成负担**：CVAE 推理时 z 来自 `p(z | image)`，不再来自 GT 的 posterior。如果 z 不能稳定预测 degree token，pair 阶段的 hard constraint 会把模型锁在错误的 budget 里，**主动放大 degree token 错误**。06-06 baseline 没有 degree 约束，错就错了，不会被人为放大。

结论：degree 方法在 unconditional posterior reconstruction 阶段可能能挤一点指标，但搬到 CVAE 时很可能反而不如 baseline。

### 4.3 真正的核心问题：`exact_adj_acc` 不是正确的优化目标

今天三个实验的 best checkpoint 都用 `ar_recon_exact_adj_acc` 选。但这个指标和最终目标不一致：

```
1-to-many：       一张图片可以对应多种合法 topology
exact_adj_acc：   要求和 GT 完全一致
```

我们真正要给 diffusion 的不是"GT 那个 topology"，而是**任意一个合法且与图片几何兼容的 topology**。把 `exact_adj_acc` 拉到 0.9+ 既不可能（1-to-many 限制了 ceiling），也不必要（diffusion 不需要 GT，需要 valid）。

这一节的实际后果：

- 06-06 的 `degree+KL=0.1` checkpoint，`exact_adj_acc=0.42` 看起来很差，但 `prior_conn=0.996`：它生成的几乎全部是连通图，只是不等于 GT。如果用 validity 视角，它可能比 `exact_adj_acc=0.78` 的 baseline 更可用。
- 真正的瓶颈不是"如何更精确复现 GT"，而是：
  - **生成的拓扑必须 valid**（connected, no isolated face, reasonable degree, reasonable density）
  - **CVAE 推理时多次采样 + validity check + retry**，保证至少有一个 valid 输出能喂给 diffusion

### 4.4 重新定位 corruption / degree

按 validity 视角重排：

| 方法 | 真实贡献 | 在 CVAE 阶段的兼容性 |
|------|---------|---------|
| `corruption` | 缩小 `tf_f1 - ar_f1` gap，让 AR 推理更稳定 → 间接提升 valid_rate | 完全兼容，无副作用 |
| `degree` | row-sum 强制一致 → 直接提升 no_isolated_ratio | 风险：缩窄 1-to-many 的多样性，z 错的时候放大错误 |
| `connectivity hard constraint`（DSU，未实现） | 直接保证 connected | 只在 inference 加，不影响训练 |

`corruption` 的价值不是"提升 exact"，而是 AR inference 稳定性。**留下，跑**。

`degree` 在 unconditional 阶段大概率有用，在 CVAE 阶段可能负面。**仍然跑**（已写好，不浪费），但它的输出更适合做 unconditional baseline，不一定是 CVAE decoder 的最佳起点。

### 4.5 未来 metrics 改造方向

应该把"以 GT 为目标"的指标和"以 validity 为目标"的指标分开看：

- **以 GT 为目标（保留）**：`tf_edge_f1`、`ar_recon_edge_f1`、`ar_recon_exact_adj_acc`、`ar_recon_face_count_mae`、`ar_recon_edge_count_mae`
- **以 validity 为目标（应该加）**：见下一节

### 4.6 关于 VQ-VAE：现阶段不适合

讨论了 VQ-VAE / VQ-VAE-2 / VQGAN / MeshGPT 的核心思路（encoder → 离散 codebook → AR prior over codes）。结论：

- 当前 VAE 的瓶颈不是 latent 表达力（`tf_f1=0.972` 说明 z 信息够），而是结构性约束 + validity
- VQ-VAE 的核心优势是离散化方便后续 AR prior 建模，但你的 decoder 输出端**已经在做 AR**（next-token prediction over pair tokens），不需要再加一层 AR
- VQ-VAE 训练有 codebook collapse、commitment loss 调参、code reset 等大量工程坑（参考 MeshGPT 论文的 codebook utilization 一节）
- CVAE 视角下 Gaussian prior 天然兼容（image encoder 输出 μ, σ 直接喂 reparameterize），离散 prior 反而需要 image → token AR，回到 06-02 的 0.57 上限问题

折中：如果 unconditional VAE 训完发现 z 容量不够，可以先尝试 **multi-token continuous VAE**（encoder 输出多个 z 而不是 mean pool 成一个），这是 VQ-VAE 的连续版，工程量小很多。VQ-VAE 留到未来真正出现"多模态 prior 表达不出来"或"要联合 tokenize geometry+topology"时再考虑。

## 5. Metrics 现状 vs 应该补齐的

### 5.1 当前已有的 metrics

记录在所有 06-06 / 06-08 脚本的 `evaluate()` 输出里。所有"ar_recon_*" 是用 posterior `mu` 直接 decode 的；"tf_*" 是 teacher forcing 一次 forward 算的；"prior_*" 是从 N(0, I) 采样 z 再 decode 的。

**A. Loss 部分**

- `loss`: 总 loss
- `face_count_loss`: face count CE
- `edge_count_loss`: edge count CE（仅 edge_count / both 版）
- `degree_loss`: per-face degree CE（仅 degree / both 版）
- `pair_loss`: pair BCE（pair_mask 加权）
- `kl`: KL(q(z|x) || N(0, I))
- `kl_beta`: 当前 epoch 的 KL warmup 系数

**B. Teacher-forcing pair-level（一次 forward 拿到的 logits 上算）**

- `tf_edge_precision`, `tf_edge_recall`, `tf_edge_f1`, `tf_edge_iou`
- `tf_face_count_acc`, `tf_face_count_mae`
- `tf_edge_count_acc`, `tf_edge_count_mae`（edge_count / both 版）
- `tf_degree_acc`, `tf_degree_mae`（degree / both 版）

**C. AR posterior reconstruction（用 mu 当 z，autoregressive generate 之后算）**

- `ar_recon_edge_precision`, `ar_recon_edge_recall`, `ar_recon_edge_f1`, `ar_recon_edge_iou`
- `ar_recon_exact_adj_acc`: 整张 adj matrix 完全对的比例（best checkpoint 选这个）
- `ar_recon_face_count_acc`, `ar_recon_face_count_mae`
- `ar_recon_edge_count_acc`, `ar_recon_edge_count_mae`, `ar_recon_edge_count_token_mae`
- `ar_recon_degree_acc`, `ar_recon_degree_mae`, `ar_recon_degree_self_consistent`（degree / both 版）
- `ar_recon_connected_ratio`, `ar_recon_no_isolated_ratio`

**D. Unconditional prior（z ~ N(0, I)，generate 后算）**

- `prior_connected_ratio`: 生成图连通的比例
- `prior_no_isolated_ratio`: 没有孤立 face 的比例
- `prior_density_mean`, `prior_edge_count_mean`
- `prior_count_mean`, `prior_count_std`: face count 分布
- `prior_edge_count_token_mean`, `prior_edge_count_actual_mean`: 预测 edge_count token 和实际生成图的 edge 数

### 5.2 当前 metrics 的不足

针对 1-to-many + 给下游 diffusion 提供 valid topology 这两个目标，现在缺：

1. **Sample@K 指标**：CVAE 推理是多次采样取一个 valid 的，但当前 generate 全部用 greedy 或 single sample。应该有 `sample@K_valid_rate`（采样 K 次至少一个 valid）和 `sample@K_oracle_f1`（K 次里最接近 GT 的那次的 f1）。
2. **Composite validity rate**：当前只有 `connected_ratio` 和 `no_isolated_ratio` 单独指标。应该有一个综合 `valid_rate`：connected ∧ no_isolated ∧ degree 合理 ∧ density 合理。
3. **k-edge-error 分布**：现在只有 `exact_adj_acc`（0 错），没区分 1 错 / 2 错 / 5 错。06-06 NOTE 也提到这个，可以从 `edge_f1` + `exact_adj_acc` 反推但不直观。
4. **Diversity**：1-to-many 的核心是输出多样性。从同一个 z 采 K 次，量化 K 个输出之间的 pairwise edit distance。如果 diversity 太低说明模型实质上是 deterministic predictor。
5. **接 diffusion 后的真实指标**：predicted topology 喂给 frozen diffusion model 后，diffusion val_loss / valid STEP rate 是什么。这才是最终指标，但需要走完 CVAE pipeline 才能算。

### 5.3 推荐的 metrics 优先级

短期（unconditional VAE 阶段）：

```
+ valid_rate        = (connected) ∧ (no_isolated) ∧ (3 <= median_degree <= 8)
                                   ∧ (0.05 <= density <= 0.6)
+ k_edge_error_dist = histogram of "differ from GT by k edges"
+ sample@K_valid    = sample K=5 from posterior, at least one valid
+ sample@K_oracle   = sample K=5, take the closest to GT (best-of-K f1)
```

第一个是单数字，其他可以是分布或 list。

中期（image-conditioned CVAE 阶段）：

```
+ image_consistency = 把 K 次采样的 topology 各自跑 diffusion，validity rate
+ topology_diversity = K 次采样之间的 pairwise edit distance mean / std
+ prior_kl_to_posterior = KL(p(z|image) || q(z|image, face_adj)) on validation
```

### 5.4 改造工程量

仅 5.3 第一段（unconditional 阶段的四个新指标）需要改：

- 加一个 `compute_validity_metrics(adj, count)` 函数：单图 valid 判定 + 综合 valid_rate
- 加一个 `compute_k_edge_error_dist(pred_pairs, gt_pairs, mask)` 函数：返回 0..N 错的 histogram
- 改 `evaluate()`：从 posterior `mu` 多次 sample（reparameterize 时 sample_posterior=True，跑 K 次 generate）
- prior 部分本来就是 sample，只需改 generate 函数允许多次

改动量大概 100-150 行，可以集中在一个 `validity_utils.py` 模块里，三个 06-08 脚本共用。今晚训练完后再做这部分。

## 6. 第一步落地：strict structural validity 加进所有 5 个脚本

### 6.1 决策

完整的 `valid_rate`（含 degree 中位数 / max degree / density 三个经验阈值）需要先在训练集上做 calibration，工程量稍大。今天只先做最小可落地的版本：

```
valid_strict = connected ∧ no_isolated
```

理由：

- 这两个判定本来就在 `adjacency_stats` 函数里**已经分别在算**（`connected_ratio` 和 `no_isolated_ratio`），加交集只多一行 `bool(conn_flag and no_iso_flag)`，不增加任何计算量。
- 不引入任何经验阈值，所有结论都来自纯结构判定，论文里也好交代。
- 06-06 / 06-08 五个脚本统一加，所有已有 metrics 都向后兼容。
- 后续要做完整 `valid_rate`（degree + density 阈值版）时，calibration 流程独立，不影响这一版数字。

### 6.2 修改内容

修改了 5 个脚本的 `adjacency_stats` 函数：

- `experiments/2026-06-06/topology_faceadj_vae.py`
- `experiments/2026-06-06/topology_faceadj_vae_edgecount.py`
- `experiments/2026-06-08/topology_faceadj_vae_corrupt.py`
- `experiments/2026-06-08/topology_faceadj_vae_degree.py`
- `experiments/2026-06-08/topology_faceadj_vae_both.py`

每个函数都新增一个 `valid_strict` 列表，循环里新增一行：

```python
valid_strict.append(float(conn_flag and no_iso_flag))
```

返回 dict 多一个 key `valid_strict_ratio`。`n <= 1` 的退化情况按 trivially valid 算（和 connected/no_isolated 的处理一致）。

### 6.3 自动传递到下游 metrics

`adjacency_stats` 在两处被调用：

- `generated_metrics(...)` 内部：得到的 `connected_ratio` / `valid_strict_ratio` 会以 `ar_recon_*` 前缀出现在 metrics 输出里。所以**自动得到 `ar_recon_valid_strict_ratio`**。
- `evaluate(...)` 末尾的 prior sampling 块：以 `prior_*` 前缀输出。所以**自动得到 `prior_valid_strict_ratio`**。

不需要再改 `generated_metrics`、`evaluate`、`pair_metrics_from_logits`、`compute_loss` 等下游函数。

### 6.4 用法

回去训练完后，跑 test split eval-only 命令（在 `command.sh` 末尾，已写但注释掉），新指标会出现在每个实验目录的 `test_metrics.json` 里：

- `ar_recon_valid_strict_ratio`：从 posterior `mu` 解码出的 adjacency 满足 connected + no_isolated 的比例
- `prior_valid_strict_ratio`：从 N(0, I) 采样 z 解码出的 adjacency 满足 connected + no_isolated 的比例

也可以直接用同样的 `--eval-only --eval-split test` 把 06-06 已有的四个 baseline checkpoint 重新跑一遍：

```bash
# 例：06-06 strong baseline
cd /mnt/d/python && python experiments/2026-06-06/topology_faceadj_vae_edgecount.py \
    --eval-only --eval-split test \
    --checkpoint experiments/2026-06-06/outputs_faceadj_vae_edgecount_wl_kl001/best.pt \
    --output-dir experiments/2026-06-06/outputs_faceadj_vae_edgecount_wl_kl001 \
    --batch-size 256 --num-workers 8 \
    --eval-generate-limit 2424 --prior-samples 512
```

跑完直接看 `test_metrics.json` 里的 `ar_recon_valid_strict_ratio` 和 `prior_valid_strict_ratio` 即可，不需要重训。

### 6.5 预期对比

按 06-06 已知的 prior_connected_ratio + prior_no_isolated_ratio（NOTE 06-06 没单独报 no_isolated 的 prior 数字，但训练日志里有），可以预测：

| 模型 | prior_conn | 预期 valid_strict |
|------|---|---|
| degree+KL=0.1 | 0.996 | 接近 0.99（强 KL 让模型只学 trivial connected graph） |
| WL+edge_count+KL=0.001 | 0.805 | 略低于 0.80（一些 connected graph 仍会有 isolated padding 错误？需要实测） |

如果 `valid_strict` 拉开差距，那"哪个 checkpoint 是下一步 CVAE decoder 最佳起点"的答案可能反转。

## 待办

- [ ] 回家执行 `bash experiments/2026-06-08/command.sh`，三张卡并行跑 100 epoch。
- [ ] 训练完成后，分别在 test split 上做 eval-only（命令在 `command.sh` 末尾，已注释，去掉注释直接跑），把 `test_metrics.json` 和 06-06 baseline 比。
- [ ] 用 06-06 两个改过 `adjacency_stats` 的脚本对四个旧 checkpoint 跑 eval-only（不重训），拿到 `ar_recon_valid_strict_ratio` / `prior_valid_strict_ratio`。看 `degree+KL=0.1` 是否在 valid_strict 上反而最高。
- [ ] 后续工程：写完整的 `validity_utils.py`（degree/density 经验阈值版 + sample@K），先跑 train-set calibration 拿阈值，再做严格 valid_rate 评估。
- [ ] 决定 CVAE decoder 起点：在 ar_recon_exact_adj_acc 高 vs valid_strict_ratio 高之间根据数据选。
- [ ] 把 best 的那个变种作为下一步 image-conditioned CVAE 的 decoder 起点。
