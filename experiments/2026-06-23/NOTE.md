# 2026-06-23 实验笔记

## 今日目标

汇总 [2026-06-19 two-step 计划](../2026-06-19/NOTE.md) 的完整 test set 结果，判断 `real photo -> CVAE topology -> diffusion -> B-Rep` 是否相对 no-topology baseline 有提升。

## 实验记录

### 1. Two-step CVAE topology mask 接入 diffusion 的完整测试结果

**为什么这么做**

- 长期目标是 `real photo -> topology mask -> HoLa diffusion -> B-Rep`；[06-18 visibility-aware CVAE](../2026-06-18/NOTE.md) 已经完成第一步 topology predictor 训练，所以需要验证它接入 diffusion 后是否真正提升最终 B-Rep。
- 对比对象必须是同一协议下的 no-topology real-photo diffusion baseline，即 [06-19 baseline diffusion](../2026-06-19/NOTE.md#1-real-photo-baseline-diffusion-checkpoint)。
- 主指标应该优先看 `valid solid rate` 和 single-view 可见面的 `visible_condition`，full-face 指标只作为辅助。

**具体实施**

- Baseline diffusion checkpoint：[last.ckpt](../baseline/diffusion_baseline_real_v7_cube24/checkpoints/last.ckpt)。
- Two-step 脚本：[cvae_diffusion_two_step.py](../2026-06-19/cvae_diffusion_two_step.py)。
- CVAE checkpoint：[06-18 best.pt](../2026-06-18/outputs_cvae_visibility/best.pt)。
- Baseline 输出：
  - raw predictions：[test_outputs](../baseline/diffusion_baseline_real_v7_cube24/test_outputs/)
  - post/eval：[test_post](../baseline/diffusion_baseline_real_v7_cube24/test_post/)
- Two-step 输出：
  - predicted topology cache：[cvae_visibility_topology_cache](../baseline/cvae_visibility_topology_cache/)
  - raw predictions：[cvae_visibility_diffusion_outputs](../baseline/cvae_visibility_diffusion_outputs/)
  - post/eval：[cvae_visibility_diffusion_post](../baseline/cvae_visibility_diffusion_post/)
- 评估数据：
  - raw B-Rep：[deepcad_v7](/mnt/d/data/deepcad_v7)
  - condition：[deepcad_v7_cond](/mnt/d/data/deepcad_v7_cond)
  - visibility cache：[deepcad_v7_visibility](/mnt/d/data/deepcad_v7_visibility)
  - test list：[deduplicated_deepcad_testing_7_30.txt](../../src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt)
- 聚合文件：
  - Baseline summary：[eval_summary.json](../baseline/diffusion_baseline_real_v7_cube24/test_post/eval_summary.json)
  - Two-step summary：[eval_summary.json](../baseline/cvae_visibility_diffusion_post/eval_summary.json)
  - Baseline visible summary：[eval_visible_condition_summary.json](../baseline/diffusion_baseline_real_v7_cube24/test_post/eval_visible_condition_summary.json)
  - Two-step visible summary：[eval_visible_condition_summary.json](../baseline/cvae_visibility_diffusion_post/eval_visible_condition_summary.json)

**结果**

| 指标 | Baseline diffusion | CVAE topology + diffusion | 变化 |
|---|---:|---:|---:|
| Valid solid rate | 1160 / 2424 = 47.85% | 700 / 2424 = 28.88% | -18.98 pp |
| `visible_face_cd` mean / median | 0.03218 / 0.01692 | 0.03508 / 0.01855 | 变差 |
| `visible_face_gt_to_recon_cd` mean / median | 0.00362 / 0.00070 | 0.00544 / 0.00104 | 变差 |
| `visible_face_fscore` mean / median | 0.6593 / 0.6667 | 0.6250 / 0.6429 | -0.0343 mean |
| `visible_face_precision` mean / median | 0.5285 / 0.5385 | 0.4923 / 0.5000 | 下降 |
| `visible_face_recall` mean / median | 0.9150 / 1.0000 | 0.9009 / 1.0000 | 小幅下降 |
| Full face F-score mean / median | 0.8659 / 0.9333 | 0.8209 / 0.8571 | 下降 |
| FE topology F-score mean / median | 0.7317 / 0.7622 | 0.6348 / 0.5882 | 明显下降 |
| EV topology F-score mean / median | 0.7223 / 0.7481 | 0.6184 / 0.5650 | 明显下降 |

- 覆盖情况：
  - Validity：baseline 和 two-step 都有 `2424 / 2424` 个 eval。
  - Visible condition：两者都是 `2423 / 2424` 个 eval，缺失样本为 `00547707`。
  - Full condition：baseline 为 `2420 / 2424`，two-step 为 `2418 / 2424`。
- paired validity：
  - 两者都 valid：589。
  - 两者都 invalid：1153。
  - baseline valid 但 two-step invalid：571。
  - two-step valid 但 baseline invalid：111。
- predicted topology cache 本身的 test set 统计：
  - cache 数量：2424。
  - GT face mean / median：12.70 / 11。
  - predicted face mean / median：12.20 / 10，face MAE = 2.11。
  - GT edge mean / median：29.21 / 24。
  - predicted edge token mean / median：28.21 / 24，但实际 restored adjacency edge mean / median 只有 22.71 / 21，edge MAE = 6.75。
  - restored adjacency F1 mean / median：0.533 / 0.500，precision mean = 0.589，recall mean = 0.507。
- 直接结论：这版 two-step 不是改善，而是明确退化。退化同时发生在 validity、visible geometry、full geometry 和 topology consistency 上。
- 主要原因判断：
  - diffusion 对 topology bias 很敏感；当 topology mask 不够准时，它不是“温和提示”，而是在 denoising 过程中强行改变 face interaction。
  - 06-18 CVAE 的 generated adjacency test F1 只有约 0.53，recall 约 0.51，说明它经常漏掉 GT 邻接；这种 mask 输入 diffusion 后会切断本来需要通信的 face pair。
  - two-step 输出的平均 face/edge/vertex 数量都高于 baseline，但 valid solid rate 更低，说明 predicted topology bias 更像是在制造结构噪声，而不是稳定约束。

**参考 / 关联**

- 计划与脚本：[2026-06-19 NOTE](../2026-06-19/NOTE.md)
- CVAE 训练：[2026-06-18 NOTE](../2026-06-18/NOTE.md)
- Two-step 脚本：[cvae_diffusion_two_step.py](../2026-06-19/cvae_diffusion_two_step.py)
- Baseline summary：[eval_summary.json](../baseline/diffusion_baseline_real_v7_cube24/test_post/eval_summary.json)
- Two-step summary：[eval_summary.json](../baseline/cvae_visibility_diffusion_post/eval_summary.json)

## 今日结论

- 当前 `CVAE topology -> diffusion topology bias` 路线在真实 two-step test 上失败：validity 从 47.85% 降到 28.88%，visible F-score 从 0.659 降到 0.625。
- 这不是“exact 不重要”的问题，而是 predicted topology mask 的错误会直接伤害 diffusion 的结构生成；当前 CVAE topology quality 还不足以作为 diffusion mask。
- 下一步不应继续盲目调 CVAE 结构，而应先补同一 baseline checkpoint 下的 oracle topology upper bound：如果 GT topology bias 能明显提高 valid rate，说明接口仍值得救；如果 oracle 也不提升，则说明当前 topology-bias 接入 diffusion 的方式需要重审。

## 待办

- [ ] 在同一 baseline diffusion checkpoint 上跑 GT oracle topology bias，建立 upper bound。
- [ ] 对 predicted topology 做阈值/softness/bias-scale ablation，确认是不是 `scale=2.0` 过强。
- [ ] 重新审视 topology predictor 目标：优先提高 adjacency recall / connectedness，而不是只看 CVAE sample validity。
