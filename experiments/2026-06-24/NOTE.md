# 2026-06-24 实验笔记

## 今日目标

汇总三组 full test 结果：no-topology real-photo baseline、06-18 CVAE predicted topology two-step、以及外部 checkpoint 的 GT oracle topology diffusion。

## 实验记录

### 1. Baseline / CVAE two-step / Oracle topology 三模型对比

**为什么这么做**

- [2026-06-23](../2026-06-23/NOTE.md) 的结论显示 predicted topology two-step 明确退化；今天补充 oracle topology checkpoint，判断 topology bias 接口是否仍有上界价值。
- 用户提供 oracle checkpoint：[diffusion_oracle_topology.ckpt](/mnt/d/data/new_cond_ckpt/diffusion_oracle_topology.ckpt)。

**具体实施**

- Baseline 输出：[test_post](../baseline/diffusion_baseline_real_v7_cube24/test_post/)。
- CVAE two-step 输出：[cvae_visibility_diffusion_post](../baseline/cvae_visibility_diffusion_post/)。
- Oracle topology 输出：[test_post](../baseline/diffusion_oracle_topology_real_v7_cube24/test_post/)。
- 评估协议：
  - raw B-Rep：[deepcad_v7](/mnt/d/data/deepcad_v7)
  - condition：[deepcad_v7_cond](/mnt/d/data/deepcad_v7_cond)
  - visibility cache：[deepcad_v7_visibility](/mnt/d/data/deepcad_v7_visibility)
  - test list：[deduplicated_deepcad_testing_7_30.txt](../../src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt)
  - visible threshold：`0.05`
- 聚合文件：
  - Baseline：[eval_summary.json](../baseline/diffusion_baseline_real_v7_cube24/test_post/eval_summary.json)
  - CVAE two-step：[eval_summary.json](../baseline/cvae_visibility_diffusion_post/eval_summary.json)
  - Oracle：[eval_summary.json](../baseline/diffusion_oracle_topology_real_v7_cube24/test_post/eval_summary.json)

**结果**

| 指标 | Baseline | CVAE predicted topology | GT oracle topology checkpoint |
|---|---:|---:|---:|
| Valid solid rate | 1160 / 2424 = 47.85% | 700 / 2424 = 28.88% | 1341 / 2424 = 55.32% |
| `visible_face_cd` mean | 0.03218 | 0.03508 | 0.25984 |
| `visible_face_fscore` mean | 0.6593 | 0.6250 | 0.1805 |
| `visible_face_precision` mean | 0.5285 | 0.4923 | 0.1403 |
| `visible_face_recall` mean | 0.9150 | 0.9009 | 0.2645 |
| Full face CD mean | 0.00898 | 0.01185 | 0.22823 |
| Full face F-score mean | 0.8659 | 0.8209 | 0.2231 |
| FE topology F-score mean | 0.7317 | 0.6348 | 0.5417 |
| EV topology F-score mean | 0.7223 | 0.6184 | 0.5548 |

- Paired validity：
  - Baseline vs Oracle：both valid = 893；baseline-only valid = 267；oracle-only valid = 448；both invalid = 816。
  - Baseline vs CVAE：both valid = 589；baseline-only valid = 571；CVAE-only valid = 111；both invalid = 1153。
  - CVAE vs Oracle：both valid = 581；CVAE-only valid = 119；oracle-only valid = 760；both invalid = 964。
- Oracle checkpoint 元信息：
  - `epoch=7`，`global_step=145704`。
  - checkpoint 内保存的训练配置显示 `dataset.real_photo_ratio=0.0`、`dataset.load_topology=True`、`model.topology_bias.mode=soft_bias`、`model.topology_bias.scale=2.0`。
- 解释：
  - Oracle topology checkpoint 在 validity 上最好，说明 GT topology bias 确实可以提高 solid validity。
  - 但 oracle 的 visible/full geometry 指标极差，这不是 topology oracle 本身的正常上界表现，而更像是 checkpoint 训练条件与当前 real-photo 测试条件不匹配：该 checkpoint 训练时 `real_photo_ratio=0.0`，今天测试命令用的是 `real_photo_ratio=1.0`。
  - 因此这次 oracle 结果只能说明“topology bias 对 validity 有潜在上界”，不能作为当前 real-photo task 的 geometry upper bound。

**参考 / 关联**

- 昨天总结：[2026-06-23 NOTE](../2026-06-23/NOTE.md)
- Oracle checkpoint：[diffusion_oracle_topology.ckpt](/mnt/d/data/new_cond_ckpt/diffusion_oracle_topology.ckpt)
- Oracle summary：[eval_summary.json](../baseline/diffusion_oracle_topology_real_v7_cube24/test_post/eval_summary.json)
- Baseline summary：[eval_summary.json](../baseline/diffusion_baseline_real_v7_cube24/test_post/eval_summary.json)
- CVAE two-step summary：[eval_summary.json](../baseline/cvae_visibility_diffusion_post/eval_summary.json)

## 今日结论

- 三者排序按 validity 是：Oracle GT topology `55.32%` > Baseline `47.85%` > CVAE predicted topology `28.88%`。
- 三者排序按 real-photo visible geometry 是：Baseline 最好，CVAE two-step 略差，Oracle checkpoint 极差。
- 关键判断：GT topology bias 接口可能仍然有价值，但当前这个 oracle checkpoint 不是 real-photo oracle upper bound；要做严谨上界，需要训练或取得 `real_photo_ratio=1.0` 的 oracle topology diffusion checkpoint。

## 待办

- [ ] 补一个真正的 real-photo oracle topology diffusion，或确认外部机器是否已有 `real_photo_ratio=1.0` 的 oracle checkpoint。
- [ ] 在当前 baseline checkpoint 上做 topology bias scale ablation，避免 predicted mask 以过强 bias 破坏 diffusion。
- [ ] 对 CVAE predicted adjacency 优先优化 recall / connectedness，再讨论接入 diffusion。
