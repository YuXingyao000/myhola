# 2026-06-19 实验笔记

## 今日目标

验证真实两阶段系统：`real photo -> 06-18 CVAE topology matrix -> diffusion model -> B-Rep`，并与不使用 topology mask 的 real-photo baseline diffusion 在 full test set 上对比，主指标集中在图像可见面。

## 方向反思

- North Star 仍然是 `image -> topology mask -> HoLa diffusion -> B-Rep`，最终评价应该落在生成 B-Rep 的 validity 和 visible-region condition consistency。
- [06-18 visibility-aware CVAE](../2026-06-18/NOTE.md#1-visibility-aware-dino-in-context-topology-cvae) 已完成第一步 topology predictor 训练；现在不应继续只看 CVAE 的局部 `sample8_*` 指标。
- 第二步“给 diffusion topology matrix 可以提升合法性”之前已经验证过，但这台机器缺少当前 deepcad_v7 + real-photo + cube24 协议下的 no-topology baseline checkpoint。因此今天先补 baseline，再把 06-18 CVAE 和 baseline diffusion 拼成真实 two-step pipeline。
- 评估口径要修正：单视图不可见面不应该进入几何 Chamfer / face F1 主指标；需要新增 visible-only evaluation，并与现有 `src/brepnet/eval` 的 identity-first cube24 新协议保持一致。

## 文献 / 依据检查

- [CVAE](https://papers.nips.cc/paper/5775-learning-structured-output-representation-using-deep-conditional-generative-models) 和 [Probabilistic U-Net](https://arxiv.org/abs/1806.05034) 支持把单图到结构输出看成 one-to-many conditional generation；因此最终比较应允许不可见区域多解。
- [DDPM](https://arxiv.org/abs/2006.11239) / [LDM](https://arxiv.org/abs/2112.10752) 支撑当前 latent diffusion 主线；今天不改 diffusion 主体，只比较 no-mask baseline 和 predicted-mask two-step。
- 现有 [REFERENCES.md](../REFERENCES.md) 已覆盖今天的理论依据，没有新增外部论文。

## 实验记录

### 1. Real-photo baseline diffusion checkpoint

**为什么这么做**

- 这台机器上没有当前协议下的 no-topology baseline diffusion checkpoint；后续所有 predicted topology mask 对比都需要一个固定 baseline。
- baseline 必须和 two-step 使用同一 raw root、latent cache、condition root、test list、real-photo view 0 和 post/eval 协议，否则比较没有意义。

**具体实施**

- 配置：[train_diffusion_baseline_real.yaml](../../configs/train_diffusion_baseline_real.yaml)。
- 命令清单：[command.sh](./command.sh) 的第 1-4 段。
- 输出目录：
  - training/checkpoints：[diffusion_baseline_real_v7_cube24](../baseline/diffusion_baseline_real_v7_cube24/)
  - test predictions：[test_outputs](../baseline/diffusion_baseline_real_v7_cube24/test_outputs/)
  - post/eval：[test_post](../baseline/diffusion_baseline_real_v7_cube24/test_post/)
- 数据：
  - raw B-Rep：[deepcad_v7](/mnt/d/data/deepcad_v7)
  - condition：[deepcad_v7_cond](/mnt/d/data/deepcad_v7_cond)
  - latent cache：[1119_deepcad_aug1_11k_24](/mnt/d/data/ae_cache/1119_deepcad_aug1_11k_24)
  - AE checkpoint：[1119_deepcad_aug1_11k.ckpt](/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt)
  - test list：[deduplicated_deepcad_testing_7_30.txt](../../src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt)
- 关键设置：
  - `model.topology_bias.enabled=false`
  - `dataset.real_photo_ratio=1.0`
  - `dataset.validation_real_photo_ratio=1.0`
  - `dataset.is_aug=false`
  - `dataset.load_topology=false`

**结果**

- 已给出 full train / full test / post/eval 命令，待运行。
- 配置 compose 检查通过：
  - `dataset.real_photo_ratio=1.0`
  - `dataset.validation_real_photo_ratio=1.0`
  - `dataset.cached_latent_root=/mnt/d/data/ae_cache/1119_deepcad_aug1_11k_24`
  - `model.topology_bias.enabled=False`

**参考 / 关联**

- 配置：[train_diffusion_baseline_real.yaml](../../configs/train_diffusion_baseline_real.yaml)
- 命令：[command.sh](./command.sh)
- 评估入口：[eval/run.py](../../src/brepnet/eval/run.py)

### 2. Visible-only condition evaluation

**为什么这么做**

- 单张 real photo 只约束可见面；不可见面的几何差异不应该进入主 Chamfer / face F1 结论。
- full condition metrics 仍可保留为辅助，但 today's comparison 应重点看 visible face CD / F1 / recall 和 valid STEP rate。

**具体实施**

- 新增 metric：[visible_condition.py](../../src/brepnet/eval/metrics/visible_condition.py)。
- 注册到新评估入口：[eval/run.py](../../src/brepnet/eval/run.py)，用法：
  - `--metrics visible_condition`
  - `--visibility-root /mnt/d/data/deepcad_v7_visibility`
  - `--visibility-view-id 0`
  - `--visible-threshold 0.05`
- metric 逻辑：
  - 读取 [deepcad_v7_visibility](/mnt/d/data/deepcad_v7_visibility) 的 `visible_fraction`。
  - `visible_fraction >= threshold` 的 GT faces 作为 visible GT set。
  - 输出 `visible_face_gt_to_recon_cd`、`visible_face_recon_to_gt_cd`、`visible_face_cd`、`visible_face_fscore`、`visible_face_precision`、`visible_face_recall`。
  - 主解释优先看 `visible_face_gt_to_recon_cd` 和 `visible_face_recall/F1`；`visible_face_recon_to_gt_cd` 会惩罚额外生成的 hidden faces，作为 stricter 辅助指标。
- 命令清单：[command.sh](./command.sh) 的第 4、7 段。

**结果**

- 语法和 CLI 检查通过：
  - `conda run -n torch python -m py_compile src/brepnet/eval/metrics/visible_condition.py src/brepnet/eval/run.py`
  - `python -m src.brepnet.eval.run --help` 已显示 `visible_condition`、`--visibility-root`、`--visibility-view-id`、`--visible-threshold`。
- 当前没有现成的、folder name 直接匹配 [deepcad_v7_visibility](/mnt/d/data/deepcad_v7_visibility) 的 post-processed outputs，因此数值 smoke 等 baseline/two-step post 生成后再跑。

**参考 / 关联**

- visibility cache：[deepcad_v7_visibility](/mnt/d/data/deepcad_v7_visibility)
- 新 metric：[visible_condition.py](../../src/brepnet/eval/metrics/visible_condition.py)
- eval 入口：[eval/run.py](../../src/brepnet/eval/run.py)

### 3. 06-18 CVAE + baseline diffusion two-step pipeline

**为什么这么做**

- [06-18 CVAE](../2026-06-18/NOTE.md#1-visibility-aware-dino-in-context-topology-cvae) 已经完成 topology matrix 预测训练；现在应该直接验证它接入 diffusion 后是否提升 B-Rep 生成。
- 对比对象是 [#1](#1-real-photo-baseline-diffusion-checkpoint) 的 no-mask baseline diffusion，变量是“是否使用 CVAE predicted topology mask”。

**具体实施**

- 新增 two-step 推理脚本：[cvae_diffusion_two_step.py](./cvae_diffusion_two_step.py)。
- 命令清单：[command.sh](./command.sh) 的第 5-7 段。
- 输入：
  - CVAE checkpoint：[06-18 best.pt](../2026-06-18/outputs_cvae_visibility/best.pt)
  - baseline diffusion checkpoint：`experiments/baseline/diffusion_baseline_real_v7_cube24/checkpoints/last.ckpt`（训练完成后生成）
- 输出：
  - CVAE topology cache：[cvae_visibility_topology_cache](../baseline/cvae_visibility_topology_cache/)
  - two-step diffusion predictions：[cvae_visibility_diffusion_outputs](../baseline/cvae_visibility_diffusion_outputs/)
  - two-step post/eval：[cvae_visibility_diffusion_post](../baseline/cvae_visibility_diffusion_post/)
- 关键接口处理：
  - 06-18 CVAE 使用 `order_mode=wl` 训练，输出 adjacency 是 WL face order。
  - 脚本会用同一个 `canonical_face_order()` 对每个 GT model 计算 `face_order`，然后把 predicted adjacency 反排回原始 face order，再作为 diffusion `face_adj`。
  - 如果这一步不做，topology mask 和 diffusion latent face slots 会错位。

**结果**

- 语法和 CLI 检查通过：
  - `conda run -n torch python -m py_compile experiments/2026-06-19/cvae_diffusion_two_step.py`
  - `python experiments/2026-06-19/cvae_diffusion_two_step.py --help` 正常显示参数。
- 首次用 [baseline last.ckpt](../baseline/diffusion_baseline_real_v7_cube24/checkpoints/last.ckpt) + [06-18 CVAE best.pt](../2026-06-18/outputs_cvae_visibility/best.pt) 运行 two-step 时，单卡 batch size `4` 在第一批 diffusion cross-attention 处 OOM：同一进程同时驻留 CVAE/DINO 和 diffusion/DINO，GPU 0 约 `23.48 / 23.53 GiB` 已占用，只差 `20 MiB` 分配失败。
- 已修复 [cvae_diffusion_two_step.py](./cvae_diffusion_two_step.py)：推理流程改成两段式，先完整生成并保存 [cvae_visibility_topology_cache](../baseline/cvae_visibility_topology_cache/)；随后删除 CVAE、清理 CUDA cache，再加载 baseline diffusion 做 topology-guided inference。这样同一时刻 GPU 上只保留一套 image backbone / diffusion stack。
- [command.sh](./command.sh) 第 5 段同步改成 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + `batch-size=2`，优先保证 full test 跑完；如果后续确认显存余量充足，可手动调回 `4`。
- two-step 数值结果待重跑第 5-7 段后补充。

**参考 / 关联**

- CVAE 实验：[2026-06-18 NOTE](../2026-06-18/NOTE.md)
- CVAE checkpoint：[best.pt](../2026-06-18/outputs_cvae_visibility/best.pt)
- two-step 脚本：[cvae_diffusion_two_step.py](./cvae_diffusion_two_step.py)
- 命令：[command.sh](./command.sh)

## 今日结论

- 今天主线从“topology predictor 局部指标”推进到真实 two-step 系统验证。
- baseline diffusion、visible-only metric、CVAE+diffusion 组合推理是同一套实验的三个必要部分。
- 最终比较表应集中在：`valid STEP rate`、`visible_face_gt_to_recon_cd`、`visible_face_fscore/recall`，full face CD/F1 只作为辅助。

## 待办

- [x] 对新增脚本跑 py_compile / CLI smoke。
- [ ] 运行 [command.sh](./command.sh) 第 1 段，训练 baseline diffusion。
- [ ] baseline 训练完成后运行第 2-7 段，得到 baseline 与 two-step 的 full test 对比。
