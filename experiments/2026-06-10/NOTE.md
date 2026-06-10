# 2026-06-10 实验笔记

## 今日目标

汇总当前 7 个 topology face-adjacency VAE 相关模型的结果，重点看 posterior reconstruction、strict validity 和 prior connectivity 的权衡。

## 实验记录

### 1. 7 个 topology VAE 模型结果审计

**为什么这么做**

- 06-06 和 06-08 累积了 7 个 topology VAE 变种，需要先把当前已有指标放到同一张表里，判断下一步应该优先提升 exact adjacency、posterior strict validity，还是 prior connectivity。
- 06-06 的 4 个模型已经跑了 test split eval；06-08 的 3 个模型当前只有训练结束后保存的 validation metrics / best checkpoint metrics，还没有 test split eval，因此不能直接做最终同口径排名。

**具体实施**

- 读取以下结果文件：
  - `experiments/2026-06-06/outputs_faceadj_vae/test_metrics.json`
  - `experiments/2026-06-06/outputs_faceadj_vae_kl001/test_metrics.json`
  - `experiments/2026-06-06/outputs_faceadj_vae_wl_kl001/test_metrics.json`
  - `experiments/2026-06-06/outputs_faceadj_vae_edgecount_wl_kl001/test_metrics.json`
  - `experiments/2026-06-08/outputs_corrupt_only/best.pt`
  - `experiments/2026-06-08/outputs_degree_only/best.pt`
  - `experiments/2026-06-08/outputs_both/best.pt`
- 指标含义：
  - `ar_f1`: posterior `mu` autoregressive reconstruction 的 edge F1。
  - `exact`: 整张 face adjacency 完全正确的比例。
  - `ar_valid`: posterior reconstruction 是否满足 `connected ∧ no_isolated`。
  - `prior_valid`: 从 `N(0, I)` prior 采样时是否满足 `connected ∧ no_isolated`。

**结果**

| 模型 | 指标来源 | epoch | loss | tf_f1 | ar_f1 | exact | ar_valid | prior_valid |
|------|----------|------:|-----:|------:|------:|------:|---------:|------------:|
| 06-06 base, KL=0.1, degree order | test | 86 | 0.1428 | 0.9143 | 0.7811 | 0.4187 | 0.9950 | 0.9961 |
| 06-06 KL=0.001, degree order | test | 80 | 0.0550 | 0.9663 | 0.9185 | 0.7463 | 0.9959 | 0.9648 |
| 06-06 WL + KL=0.001 | test | 80 | 0.0508 | 0.9693 | 0.9295 | 0.7706 | 0.9955 | 0.9473 |
| 06-06 WL + edge_count + KL=0.001 | test | 100 | 0.0482 | 0.9723 | 0.9340 | 0.7797 | 0.9942 | 0.8047 |
| 06-08 corruption-only | val best | 99 | 0.0577 | 0.9644 | 0.9467 | 0.8047 | 0.9883 | 0.6914 |
| 06-08 degree-only | val best | 100 | 0.0491 | 0.9729 | 0.9092 | 0.7539 | 1.0000 | 0.8320 |
| 06-08 corruption + degree | val best | 97 | 0.0667 | 0.9608 | 0.9290 | 0.7813 | 0.9922 | 0.8203 |

补充观察：

- 06-06 test 上最强的 reconstruction 仍是 `WL + edge_count + KL=0.001`：`ar_f1=0.9340`、`exact=0.7797`。
- 06-08 `corruption-only` 在 validation best 上把 `exact` 推到 `0.8047`，并且 `ar_f1=0.9467`，说明 prefix corruption 对 autoregressive reconstruction 是有效的。但它的 `prior_valid=0.6914` 很低，unconditional prior 质量变差。
- 06-08 `degree-only` 把 posterior `ar_valid` 推到 `1.0000`，说明 per-face degree hard budget 的确能强行修 connectivity / isolated-face 问题；但 `ar_f1=0.9092`、`exact=0.7539`，比 06-06 edge_count baseline 更差，说明 degree budget 会牺牲 pair-level exactness。
- 06-08 `corruption + degree` 介于二者之间：`ar_f1=0.9290`、`exact=0.7813`、`ar_valid=0.9922`。组合没有明显超过 corruption-only 的 exact，也没有保住 degree-only 的 1.0 strict validity。
- 对 posterior reconstruction 来说，strict validity 不是当前最大瓶颈：06-06 四个 test 模型的 `ar_valid` 都约为 `0.994+`。真正拉开差距的是 `ar_f1` 和 `exact`。
- 对 unconditional prior 来说，connectivity 差异很大：强 KL 的 base 模型 `prior_valid=0.9961`，但 exact 很差；edge_count / corruption 路线 exact 更好，但 `prior_valid` 明显下降。这说明存在明显的 reconstruction-vs-prior-validity tradeoff。

## 今日结论

- 如果只看 06-06 test split，同口径最强模型是 `WL + edge_count + KL=0.001`。
- 如果 06-08 validation 能迁移到 test，`corruption-only` 可能是最值得保留的改动，因为它最直接提升 AR reconstruction 和 exact adjacency。
- `degree-only` 证明了 hard budget 可以提升 strict validity，但它对 exact adjacency 有副作用。degree token 不应直接作为下一步默认 decoder，除非最终目标明确优先 connected / no-isolated over exact match。
- 下一步要先跑 06-08 三个 checkpoint 的 test split eval，不能只拿 validation best 和 06-06 test 混合做最终判断。

### 2. 重新定位：为什么要训练 topology VAE

**为什么这么做**

- 最终目标不是单独生成一个和 GT 完全一致的 topology adjacency，而是从一张真实照片生成一个**合法且对 HoLa diffusion 有帮助的 topology mask**。
- 已有 diffusion 实验说明：如果 diffusion model 能拿到 topology mask，几何、拓扑和合法性指标都会明显提升。因此 topology VAE / CVAE 的价值应当按“能否提供可用 mask”来评估，而不只是按 VAE posterior reconstruction exact accuracy 排名。

**具体实施**

- 把当前指标重新分成三类：
  - **Decoder capacity / posterior sanity**：`ar_f1`、`exact`、`ar_valid`。这类指标说明 VAE decoder 是否有能力从 topology latent 还原合理图。
  - **Sampling prior quality**：`prior_valid`。这不是最终 CVAE 指标，但能粗略反映 latent space / decoder 在无条件采样时是否容易落到合法拓扑区域。
  - **Downstream mask utility**：还没跑，是最终应该看的指标。包括 predicted topology mask 接入 diffusion 后的 valid STEP rate、diffusion val loss、geometry/topology F1，以及和 no-mask / oracle-mask 的差距。
- 对 CVAE 来说，`exact` 不是最高优先级，因为真实照片到 topology 是 one-to-many。更关键的是：
  - sample 出来的 topology 是否 legal / connected / no isolated。
  - topology 是否和图像几何复杂度大体一致。
  - 多次采样里是否至少有一个可用 mask。
  - 这个 mask 喂给 diffusion 后是否真的提升 STEP 合法性和几何/拓扑质量。

**结果**

按 CVAE / diffusion mask 视角重新读当前 7 个结果：

| 模型 | 主要优势 | 主要问题 | CVAE 视角判断 |
|------|----------|----------|---------------|
| 06-06 base, KL=0.1 | `prior_valid=0.9961`，无条件采样几乎都 connected | `exact=0.4187`、`ar_f1=0.7811`，decoder reconstruction 太弱 | prior 看起来合法，但可能是过强 KL 造成的平庸/简单 connected graph；不适合作为强 decoder |
| 06-06 KL=0.001 | test `ar_f1=0.9185`、`exact=0.7463`，`prior_valid=0.9648` | 不如 WL/edge_count 的 reconstruction | 可作为较平衡 baseline |
| 06-06 WL + KL=0.001 | test `ar_f1=0.9295`、`exact=0.7706` | `prior_valid=0.9473` 比 degree-order 略低 | WL 排序改善 reconstruction，仍较稳 |
| 06-06 WL + edge_count | test `ar_f1=0.9340`、`exact=0.7797`，06-06 test 最强 | `prior_valid=0.8047`，prior connected 掉很多 | decoder capacity 好，但 latent prior / sampling 合法性不够稳 |
| 06-08 corruption-only | val best `ar_f1=0.9467`、`exact=0.8047`，AR exposure bias 改善明显 | `prior_valid=0.6914` 很差 | 最有希望做 CVAE decoder，但必须依赖 image-conditioned prior 或 decode-time repair，不能直接信无条件 prior |
| 06-08 degree-only | val `ar_valid=1.0000`，degree hard budget 能强制 strict validity | `ar_f1=0.9092`、`exact=0.7539`，pair-level reconstruction 下降 | 证明 hard constraint 有效，但 degree token 不一定适合作为主路线；更像 decoding constraint 的启发 |
| 06-08 corruption + degree | val `exact=0.7813`、`ar_valid=0.9922`、`prior_valid=0.8203` | 没有超过 corruption-only 的 exact，也没保住 degree-only 的 1.0 validity | 组合不够干净，说明 degree constraint 和 corruption 有 interaction |

重新排序后的结论：

1. **VAE decoder 起点**：优先看 `corruption-only`（如果 test 也成立），其次是 `WL + edge_count`。原因是它们最能重建接近 GT 的 topology，说明 decoder capacity 更强。
2. **合法性保证**：不要把希望完全放在 VAE/CVAE 学出来。`degree-only` 说明 hard constraint 能提升 validity，但会伤 reconstruction。更合理的是在 decoder 后加 lightweight repair / constrained decoding，例如 `edge_count >= n_face - 1`、DSU connectivity repair、no-isolated repair。
3. **CVAE 评估指标**：下一阶段应从单一 `exact` 转成：
   - `sample@K_valid`: 同一张图采 K 次，至少一次得到 connected/no-isolated topology 的比例。
   - `sample@K_oracle_f1`: K 次里和 GT 最接近的 topology F1。
   - `sample@K_diffusion_valid`: K 个 topology mask 分别喂 diffusion，至少一个生成合法 STEP 的比例。
   - `downstream_gap`: predicted-mask diffusion 和 oracle-mask / no-mask diffusion 的差距。
   - `diversity`: K 次采样之间的 topology edit distance，避免 CVAE 退化成单一 deterministic topology。
4. **当前不能用 prior_valid 一票否决 corruption-only**：无条件 `z ~ N(0,I)` 的 prior 很差，但 CVAE 推理会用 `p(z | image)`，不是直接用标准正态。`prior_valid` 是风险信号，不是最终判决。
5. **最终选择标准**：哪个 topology generator 能让 diffusion 的 legal STEP rate、geometry/topology F1 接近 oracle-mask，而不是哪个 VAE 的 exact adjacency 最高。

### 3. 06-08 三个模型 test split 补评估与 CVAE 指标重排

**为什么这么做**

- 06-08 三个模型已经补跑 test split，需要用同一个 test set 重新比较 7 个模型，避免继续混用 06-06 test 和 06-08 validation best。
- 目标也需要从“哪个 VAE exact adjacency 最高”调整成“哪个模型更适合作为后续 image-conditioned CVAE / diffusion topology mask generator 的起点”。
- 单张真实照片到 topology mask 是 one-to-many 问题，因此 exact GT reconstruction 不是唯一目标；但 strict validity 会直接影响后续 mask 能否作为 diffusion 的合法拓扑约束。

**具体实施**

- 读取以下新结果：
  - `experiments/2026-06-08/outputs_corrupt_only/test_metrics.json`
  - `experiments/2026-06-08/outputs_degree_only/test_metrics.json`
  - `experiments/2026-06-08/outputs_both/test_metrics.json`
- 与 06-06 四个已有 test 结果合并比较。
- test set size 为 `2424`；prior 采样数为 `512`。
- 这里的 `strict validity` 定义为 `connected ∧ no_isolated`。当前所有模型里 `ar_recon_valid_strict_ratio == ar_recon_connected_ratio`，说明 posterior reconstruction 的 strict invalid 主要由 disconnected components 造成，而不是 isolated faces。

**结果**

| 模型 | loss | tf_f1 | ar_f1 | exact | ar_valid | ar_connected | ar_no_iso | prior_valid |
|------|-----:|------:|------:|------:|---------:|-------------:|----------:|------------:|
| 06-06 base, KL=0.1 | 0.1428 | 0.9143 | 0.7811 | 0.4187 | 0.9950 | 0.9950 | 0.9996 | 0.9961 |
| 06-06 KL=0.001 | 0.0550 | 0.9663 | 0.9185 | 0.7463 | 0.9959 | 0.9959 | 0.9992 | 0.9648 |
| 06-06 WL + KL=0.001 | 0.0508 | 0.9693 | 0.9295 | 0.7706 | 0.9955 | 0.9955 | 0.9992 | 0.9473 |
| 06-06 WL + edge_count | 0.0482 | 0.9723 | 0.9340 | 0.7797 | 0.9942 | 0.9942 | 0.9975 | 0.8047 |
| 06-08 corruption-only | 0.0610 | 0.9618 | 0.9479 | 0.7826 | 0.9922 | 0.9922 | 0.9950 | 0.7070 |
| 06-08 degree-only | 0.0532 | 0.9699 | 0.9147 | 0.7496 | 0.9942 | 0.9942 | 0.9992 | 0.7891 |
| 06-08 corruption + degree | 0.0719 | 0.9565 | 0.9265 | 0.7562 | 0.9942 | 0.9942 | 0.9988 | 0.7871 |

按 test split 重新读：

- **AR reconstruction 最强**：`06-08 corruption-only`，`ar_f1=0.9479`，超过 06-06 edge_count 的 `0.9340`；exact 也是最高的 `0.7826`，但只比 edge_count 的 `0.7797` 高一点。
- **posterior strict validity 最强**：`06-06 KL=0.001`，`ar_valid=0.9959`，约 `2414/2424` 个 posterior reconstruction strict valid。
- **corruption-only 的 validity 风险**：`ar_valid=0.9922`，约 `2405/2424` valid，约 `19` 个 disconnected / invalid；它 reconstruction 最强，但 strict validity 比 06-06 KL/WL/edge_count 路线略差。
- **degree-only 没有保住 validation 上的 1.0 validity**：test `ar_valid=0.9942`，与 edge_count / both 相同；同时 `ar_f1=0.9147`、`exact=0.7496`，说明 degree token 在 test 上没有带来足够收益。
- **prior validity 排名和 reconstruction 排名相反**：强 KL base `prior_valid=0.9961`，但 `ar_f1=0.7811`、`exact=0.4187`；corruption-only `ar_f1` 最强，但 `prior_valid=0.7070`。这说明 unconditional prior validity 不能单独决定 CVAE 路线，但它暴露了 latent space 里 invalid 区域很多。

CVAE / diffusion mask 视角下的指标优先级：

1. `sample@K_valid`: 单张图片采样 K 个 topology，至少一个 strict valid 的比例。one-to-many 场景下这比 one-shot exact 更合理。
2. `sample@K_best_f1`: K 个 topology 里和 GT adjacency 最接近的 best F1。用于衡量“候选集合里有没有合理拓扑”。
3. `sample@K_diffusion_valid`: K 个 predicted masks 分别喂 diffusion，至少一个生成合法 STEP 的比例。这应该成为最终指标。
4. `downstream_gap`: predicted-mask diffusion 与 oracle-mask / no-mask diffusion 在 STEP validity、geometry F1、topology F1 上的差距。
5. `diversity`: K 个 samples 之间的 topology edit distance，避免 CVAE 退化成只输出单一平均拓扑。

当前结论：

- `corruption-only` 是最值得作为 CVAE decoder 起点的模型，因为它的 posterior AR reconstruction 最强。
- 但它不能直接裸用 one-shot sample，因为 `ar_valid=0.9922` 已经略低，`prior_valid=0.7070` 更说明随机 latent 区域里 disconnected graph 很多。
- 进入 CVAE 后必须把 validity 作为 first-class 指标，而不是只看 exact。更合理路线是 `corruption-only decoder + image-conditioned prior + sample@K + connectedness repair / constrained decoding`。
- 如果只允许单次采样且不能 repair，06-06 KL=0.001 / WL 系列更稳；如果允许 sample@K 和 repair，corruption-only 更可能给 diffusion 提供更接近 oracle topology 的 mask。

## 待办

- [x] 对 `outputs_corrupt_only/best.pt` 跑 test split eval，补 `test_metrics.json`。
- [x] 对 `outputs_degree_only/best.pt` 跑 test split eval，补 `test_metrics.json`。
- [x] 对 `outputs_both/best.pt` 跑 test split eval，补 `test_metrics.json`。
- [x] 用同一张 test 表重新排名 7 个模型。
- [ ] 在 corruption-only 上加 decode-time connectivity repair，而不是继续强化 degree token。
- [ ] 设计 CVAE 阶段的 `sample@K_valid`、`sample@K_oracle_f1`、`sample@K_diffusion_valid` 评估脚本。
- [ ] 在进入 image-conditioned CVAE 前，先做一个 generation-only repair 实验：对 corruption-only / edge_count 输出做 DSU connectivity repair，测试 repair 后的 strict validity、F1、edge edit distance。
