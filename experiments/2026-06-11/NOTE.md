# 2026-06-11 实验笔记

## 今日目标

开始把 topology VAE 推进到 image-conditioned topology CVAE，先做最小的 in-context concatenate 版本，验证真实照片条件是否能引导 topology latent / topology decoder。

## 实验记录

### 1. In-context real-photo topology CVAE + corruption-only 训练

**为什么这么做**

- 06-08 `corruption-only` 是当前 test split 上 AR reconstruction 最强的 topology VAE：`ar_f1=0.9479`、`exact=0.7826`。
- 但最终任务不是无条件生成 topology，而是从一张真实照片生成一个合法且可用于 diffusion topology mask 的 adjacency。
- 第一版 CVAE 需要尽量少改代码，先回答一个更具体的问题：图片条件进入模型后，`p(z | image)` 能否生成比无条件 prior 更合理的 topology。
- random corruption 仍然只作用在 topology decoder prefix 上。原因是它本质是在缓解 topology 自回归 teacher forcing 和 inference 的 exposure bias；图像条件是同一个样本的观测，不应该在第一版里同时扰动。图像噪声或 patch dropout 后面可以作为独立实验。

**具体实施**

- 新脚本：
  - `experiments/2026-06-11/topology_faceadj_cvae_incontext_corrupt.py`
- 复用 06-08 corruption-only 脚本中的：
  - topology token contract: `[N_FACE, N_EDGE, PAIR_0...PAIR_434]`
  - WL ordering: `--order-mode wl --wl-rounds 3`
  - KL: `--kl-beta 0.001 --kl-warmup-epochs 10`
  - edge-count loss: `--edge-count-loss-weight 0.2`
  - random topology prefix corruption: `--corrupt-prob 0.15 --corrupt-warmup-epochs 10`
- 新增 CVAE 结构：
  - image: `deepcad_v7_cond/{model_id}/real_photo.npz["flux"]`
  - image encoder: DINOv2 ViT-L，复用 `src/brepnet/models/condition_encoders.py` 的 `DINOv2ImageEncoder`。
  - DINO CLS token 用作 image global，给 posterior / conditional prior 使用。
  - DINO full tokens（CLS + patch tokens，`257` tokens）作为 decoder in-context memory。
  - posterior: `q(z | topology, image)`。
  - conditional prior: `p(z | image)`。
  - decoder memory: `concat([z_token, image_context_tokens])`，即 in-context concatenate。
  - KL: `KL(q(z | topology, image) || p(z | image))`。
- 兼容初始化：
  - 支持 `--init-vae-checkpoint experiments/2026-06-08/outputs_corrupt_only/best.pt`。
  - 会加载 shape 兼容的 topology embedding / encoder / decoder / output layer；posterior/prior/image encoder 新参数随机初始化。
- 第一条正式训练命令：

```bash
cd /mnt/d/python
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python experiments/2026-06-11/topology_faceadj_cvae_incontext_corrupt.py \
  --epochs 100 \
  --batch-size 256 \
  --num-workers 8 \
  --order-mode wl \
  --wl-rounds 3 \
  --kl-beta 0.001 \
  --kl-warmup-epochs 10 \
  --edge-count-loss-weight 0.2 \
  --corrupt-prob 0.15 \
  --corrupt-warmup-epochs 10 \
  --condition-root /mnt/d/data/deepcad_v7_cond \
  --image-source real_flux \
  --image-backbone dinov2 \
  --image-token-dropout 0.0 \
  --eval-generate-limit 32 \
  --eval-sample-k 0 \
  --prior-temperature 1.0 \
  --init-vae-checkpoint experiments/2026-06-08/outputs_corrupt_only/best.pt \
  --output-dir experiments/2026-06-11/outputs_cvae_incontext_corrupt
```

**结果**

- 语法检查通过：
  - `conda run -n torch python -m py_compile experiments/2026-06-11/topology_faceadj_cvae_incontext_corrupt.py`
- 当前有效 smoke test 见下一条 DINOv2 记录。
- 尚未跑正式训练。

### 2. 第一版 image encoder 改为 DINOv2

**为什么这么做**

- 之前的 diffusion / condition 实验里 DINO 对 CAD 图像有一定 topology semantics，虽然信号不强，但明显比从零训练的小 CNN 更适合作为第一版 CVAE condition encoder。
- 如果第一版用轻量 CNN，失败时很难判断是 CVAE 路线不对，还是 image encoder 太弱。

**具体实施**

- 修改 `experiments/2026-06-11/topology_faceadj_cvae_incontext_corrupt.py`：
  - 删除轻量 CNN image encoder。
  - 使用 `DINOv2ImageEncoder(projection_dim=d_model, backbone="dinov2")`。
  - `encode_image()` 返回：
    - `global_token = dino_tokens[:, 0]`
    - `context = dino_tokens`，即 CLS + 256 patch tokens。
  - `--image-token-dropout` 仍保留，但默认 `0.0`；如果开启，只 dropout patch tokens，不 dropout CLS。
- 训练命令同步改成 `--image-backbone dinov2`。

**结果**

- 语法检查通过：
  - `conda run -n torch python -m py_compile experiments/2026-06-11/topology_faceadj_cvae_incontext_corrupt.py`
- DINO smoke test 通过：
  - train samples: 1
  - val samples: 1
  - batch size: 1
  - output: `/tmp/brepnet_0611_cvae_dino_smoke`
  - DINOv2 权重可加载，CVAE forward、train step、posterior generation、conditional-prior generation、sample@1 metrics 都跑通。

### 3. 修复正式训练启动 / 验证阶段看起来卡住的问题

**为什么这么做**

- 正式训练启动后看起来卡住，主要有两个原因：
  - dataset 初始化阶段逐个样本打开 `real_photo.npz` 做 key 检查，训练集约 4.6 万个样本，会非常慢。
  - 每个 epoch 的验证一开始会对 `eval_generate_limit` 个样本做 posterior greedy、conditional-prior greedy、sample@K 三组 AR generate；DINO memory 有 `257` 个 image tokens，AR 长度约 `437`，所以在 `val` 进度条前段会停很久。

**具体实施**

- 修改 `experiments/2026-06-11/topology_faceadj_cvae_incontext_corrupt.py`：
  - dataset filtering 只检查 raw `data.npz`、condition file 是否存在、face 数是否超过 `max_faces`。
  - 不再在 filtering 阶段全量 `np.load(real_photo.npz)`。
- 修改正式训练命令：
  - 训练期使用 `--eval-generate-limit 32`。
  - 训练期使用 `--eval-sample-k 0`，避免每个 epoch 做随机 sample@K。
  - sample@K 评估留到训练完成后单独跑 eval。

**结果**

- 当前状态：已修脚本和 NOTE 命令，待重新启动正式训练。

### 4. DINO in-context CVAE 训练完成后的第一轮结果

**为什么这么做**

- 检查第一版 `image -> conditional prior -> topology` 是否已经有可用信号。
- 这次训练期验证使用轻量设置：`--eval-generate-limit 32 --eval-sample-k 0`，所以生成类指标只基于 32 个 val samples；当前结论只能作为方向判断，不能替代完整 val/test eval。

**具体实施**

- 输出目录：
  - `experiments/2026-06-11/outputs_cvae_incontext_corrupt`
- 训练配置：
  - epochs: `100`
  - image source: `real_flux`
  - image backbone: `dinov2`
  - init: `experiments/2026-06-08/outputs_corrupt_only/best.pt`
  - best metric: `cond_prior_mu_edge_f1`
- 关键结果文件：
  - `args.json`
  - `history.json`
  - `metrics.json`
  - `best.pt`

**结果**

Best checkpoint by `cond_prior_mu_edge_f1`:

| metric | value |
|--------|------:|
| epoch | 64 |
| val loss | 0.2414 |
| train loss | 0.1531 |
| tf_edge_f1 | 0.8791 |
| ar_recon_edge_f1 | 0.7719 |
| ar_recon_exact_adj_acc | 0.5938 |
| ar_recon_valid_strict_ratio | 0.9375 |
| cond_prior_mu_edge_f1 | 0.5964 |
| cond_prior_mu_exact_adj_acc | 0.3750 |
| cond_prior_mu_valid_strict_ratio | 0.8750 |
| cond_prior_mu_face_count_acc | 0.5938 |
| cond_prior_mu_face_count_mae | 1.50 |
| cond_prior_mu_edge_count_acc | 0.5938 |
| cond_prior_mu_edge_count_mae | 4.44 |
| KL | 14.5627 |

Last epoch:

| metric | value |
|--------|------:|
| epoch | 100 |
| train loss | 0.1124 |
| val loss | 0.3522 |
| tf_edge_f1 | 0.8597 |
| ar_recon_edge_f1 | 0.7730 |
| ar_recon_exact_adj_acc | 0.5312 |
| ar_recon_valid_strict_ratio | 0.9375 |
| cond_prior_mu_edge_f1 | 0.5572 |
| cond_prior_mu_exact_adj_acc | 0.3438 |
| cond_prior_mu_valid_strict_ratio | 0.9375 |

观察：

- `cond_prior_mu_edge_f1` 从 epoch 1 的 `0.3438` 上升到 best epoch 64 的 `0.5964`，说明 DINO condition 确实给 `p(z | image)` 提供了一些 topology 信号。
- 但是 conditional prior 仍然很弱：best `cond_prior_mu_edge_f1=0.5964`、`exact=0.3750`，而且 face count / edge count accuracy 只有 `0.5938`。
- posterior reconstruction 也明显低于 06-08 corruption-only topology VAE：本次 best `ar_recon_edge_f1=0.7719`，而 06-08 corruption-only test 是 `0.9479`。这说明第一版 CVAE in-context 结构虽然跑通，但它破坏/稀释了原来的强 topology decoder 能力。
- validity 不是当前最大亮点：best conditional prior `valid_strict=0.8750`，last epoch 到 `0.9375`，但样本数只有 32。它没有比原 VAE posterior validity 更稳。
- 训练后期有过拟合迹象：train loss 持续下降到 `0.1124`，但 val loss 从 epoch 28 的 `0.1856` 后上升到 epoch 100 的 `0.3522`。

结论：

- 第一版 DINO in-context CVAE 已经证明 `image -> topology` 有信号，但结果还不到能接 diffusion mask 的程度。
- 下一步不应直接接 diffusion；应该先做完整 val/test eval，并考虑更保守的结构：冻结 06-08 topology decoder，仅训练 image prior / adapter，避免破坏已训练好的 topology decoder。

### 5. Full val/test sample@8 评估

**为什么这么做**

- 训练期只用 `eval_generate_limit=32` 做轻量验证，不能代表完整 split。
- 对 CVAE 来说，`exact` 不是主指标；更重要的是同一张图多次采样时能否至少得到一个 strict valid topology mask。
- 因此补跑完整 val/test，并打开 `sample@8`。

**具体实施**

- checkpoint:
  - `experiments/2026-06-11/outputs_cvae_incontext_corrupt/best.pt`
- 输出：
  - `experiments/2026-06-11/outputs_cvae_incontext_corrupt/val_metrics.json`
  - `experiments/2026-06-11/outputs_cvae_incontext_corrupt/test_metrics.json`
- 评估设置：
  - `--eval-generate-limit 999999`
  - `--eval-sample-k 8`
  - `--batch-size 16`
  - `--prior-temperature 1.0`

**结果**

| split | samples | cond_mu_valid | sample8_valid_any | sample8_valid_mean | sample8_best_f1 | sample8_mean_f1 | sample8_diversity |
|-------|--------:|--------------:|------------------:|-------------------:|----------------:|----------------:|------------------:|
| val | 3505 | 0.9541 | 0.9997 | 0.9258 | 0.7456 | 0.5150 | 0.2592 |
| test | 2424 | 0.9530 | 1.0000 | 0.9239 | 0.7242 | 0.4828 | 0.2687 |

补充指标：

| split | cond_mu_f1 | cond_mu_face_acc | cond_mu_edge_acc | ar_recon_f1 | ar_recon_valid |
|-------|-----------:|-----------------:|----------------:|------------:|---------------:|
| val | 0.5492 | 0.4916 | 0.4790 | 0.8030 | 0.9469 |
| test | 0.5126 | 0.4385 | 0.4175 | 0.7839 | 0.9435 |

观察：

- 从 validity 角度看，结果比训练期 32-sample 指标乐观得多：test `sample8_valid_any=1.0000`，说明每张测试图采 8 次时至少能得到一个 strict valid topology。
- `sample8_valid_mean≈0.924`，说明随机采样本身大部分也是合法的，不只是偶然命中。
- `cond_prior_mu_valid≈0.953`，deterministic `p_mu` decode 的合法率也不低。
- 但 topology 和 GT 的相似度仍弱：test `sample8_best_f1=0.7242`、`sample8_mean_f1=0.4828`；face count / edge count accuracy 也偏低。
- 结论应调整为：**第一版 CVAE 的 structural validity 已经可看，但 topology fidelity 仍不足**。这支持继续做 sample/rerank/repair，而不是因为 exact 低就否定路线。

## 今日结论

- 第一版 CVAE 用 frozen DINOv2 ViT-L 做 image encoder，不从零训练小 CNN。
- 先不做图像噪声或 patch dropout，避免同时改变两个变量。
- random corruption 继续保持 topology-only：corrupt decoder prefix，target topology 和 image condition 都保持干净。
- 训练期主要观察指标应该从 `ar_recon_*` 转向 `cond_prior_mu_*`；训练后再单独跑 sample@K eval，因为 `sampleK_*` 更接近推理时的 `image -> topology` 路径，但不适合每个 epoch 都跑。

## 待办

- [x] 用修正后的轻量验证命令跑正式 8 GPU 训练。
- [x] 训练完成后跑完整 val/test split eval，尤其打开 `sample@K`。
- [ ] 如果 conditional prior validity 不稳，再开第二条实验：`--image-token-dropout 0.1` 或 real image augmentation。
- [ ] 如果 conditional prior connected ratio 仍然低，再加 connectedness repair / constrained decoding。
- [ ] 尝试冻结 06-08 topology decoder，只训练 image-conditioned prior / adapter。

## 评估命令

完整 val/test + sample@K 评估先用 `K=8`。这会比较慢，因为每个样本都要跑 posterior greedy、conditional-prior greedy 和 8 次 stochastic prior AR decode；但这是目前最接近 CVAE 推理形态的 topology validity 评估。

### Full val sample@8

```bash
cd /mnt/d/python
CUDA_VISIBLE_DEVICES=0 python experiments/2026-06-11/topology_faceadj_cvae_incontext_corrupt.py \
  --eval-only \
  --eval-split val \
  --checkpoint experiments/2026-06-11/outputs_cvae_incontext_corrupt/best.pt \
  --output-dir experiments/2026-06-11/outputs_cvae_incontext_corrupt \
  --batch-size 16 \
  --num-workers 8 \
  --eval-generate-limit 999999 \
  --eval-sample-k 8 \
  --prior-temperature 1.0
```

### Full test sample@8

```bash
cd /mnt/d/python
CUDA_VISIBLE_DEVICES=0 python experiments/2026-06-11/topology_faceadj_cvae_incontext_corrupt.py \
  --eval-only \
  --eval-split test \
  --checkpoint experiments/2026-06-11/outputs_cvae_incontext_corrupt/best.pt \
  --output-dir experiments/2026-06-11/outputs_cvae_incontext_corrupt \
  --batch-size 16 \
  --num-workers 8 \
  --eval-generate-limit 999999 \
  --eval-sample-k 8 \
  --prior-temperature 1.0
```

重点看：

- `cond_prior_mu_valid_strict_ratio`: deterministic `p_mu` decode 的合法率。
- `sample8_valid_any`: 每张图采 8 次，至少一次 strict valid 的比例；这是当前最重要的 CVAE topology mask 指标。
- `sample8_valid_mean`: 所有 sampled topology 的平均合法率。
- `sample8_connected_any` / `sample8_no_isolated_any`: 判断 invalid 是 disconnected 还是 isolated face。
- `sample8_best_edge_f1`: K 个样本里与 GT 最接近的 F1，只作为合理性参考，不优先于 validity。
