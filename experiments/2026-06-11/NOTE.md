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

## 今日结论

- 第一版 CVAE 用 frozen DINOv2 ViT-L 做 image encoder，不从零训练小 CNN。
- 先不做图像噪声或 patch dropout，避免同时改变两个变量。
- random corruption 继续保持 topology-only：corrupt decoder prefix，target topology 和 image condition 都保持干净。
- 训练期主要观察指标应该从 `ar_recon_*` 转向 `cond_prior_mu_*`；训练后再单独跑 sample@K eval，因为 `sampleK_*` 更接近推理时的 `image -> topology` 路径，但不适合每个 epoch 都跑。

## 待办

- [ ] 用修正后的轻量验证命令跑正式 8 GPU 训练。
- [ ] 训练完成后跑 test split eval。
- [ ] 如果 conditional prior validity 不稳，再开第二条实验：`--image-token-dropout 0.1` 或 real image augmentation。
- [ ] 如果 conditional prior connected ratio 仍然低，再加 connectedness repair / constrained decoding。
