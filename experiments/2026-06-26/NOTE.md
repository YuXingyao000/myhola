# 2026-06-26 实验笔记

## 今日目标

汇总当前 real-photo diffusion 主线的四组 full test 结果，确认 topology oracle 上界、CVAE two-step 失败原因，以及下一步是否应转向 robust topology-conditioned diffusion。

## 实验记录

### 1. Baseline / CVAE two-step / old oracle / real-photo oracle 四模型总表

**为什么这么做**

- [2026-06-23](../2026-06-23/NOTE.md) 已确认 [06-18 CVAE](../2026-06-18/NOTE.md) predicted topology 接入 diffusion 后明显退化。
- [2026-06-24](../2026-06-24/NOTE.md) 的 old oracle checkpoint validity 最高但 geometry 崩掉，原因是 checkpoint 训练时 `real_photo_ratio=0.0`，不是 real-photo oracle。
- 今天用户完成真正的 real-photo oracle topology diffusion 训练与测试，需要重新判断 topology bias 对当前 real-photo task 的真实上界。

**具体实施**

- Baseline no-topology 输出：[test_post](../baseline/diffusion_baseline_real_v7_cube24/test_post/)。
- CVAE predicted topology two-step 输出：[cvae_visibility_diffusion_post](../baseline/cvae_visibility_diffusion_post/)。
- Old oracle checkpoint 输出：[test_post](../baseline/diffusion_oracle_topology_real_v7_cube24/test_post/)。
- Real-photo oracle checkpoint：[realphoto_oracle_topology.ckpt](../baseline/ckpt/realphoto_oracle_topology.ckpt)。
- Real-photo oracle 输出：[test_post](../baseline/diffusion_oracle_topology_realphoto_v7_cube24/test_post/)。
- 评估协议：
  - raw B-Rep：[deepcad_v7](/mnt/d/data/deepcad_v7)
  - condition：[deepcad_v7_cond](/mnt/d/data/deepcad_v7_cond)
  - visibility cache：[deepcad_v7_visibility](/mnt/d/data/deepcad_v7_visibility)
  - test list：[deduplicated_deepcad_testing_7_30.txt](../../src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt)
  - visible threshold：`0.05`

**结果**

| 指标 | Baseline | CVAE predicted topology | Old oracle ckpt | Real-photo oracle |
|---|---:|---:|---:|---:|
| Valid solid rate | 1160 / 2424 = 47.85% | 700 / 2424 = 28.88% | 1341 / 2424 = 55.32% | 1270 / 2424 = 52.39% |
| `visible_face_cd` mean | 0.03218 | 0.03508 | 0.25984 | 0.03329 |
| `visible_face_fscore` mean | 0.6593 | 0.6250 | 0.1805 | 0.6443 |
| `visible_face_precision` mean | 0.5285 | 0.4923 | 0.1403 | 0.5090 |
| `visible_face_recall` mean | 0.9150 | 0.9009 | 0.2645 | 0.9117 |
| Full face CD mean | 0.00900 | 0.01185 | 0.22823 | 0.01044 |
| Full face F-score mean | 0.8658 | 0.8209 | 0.2231 | 0.8540 |
| FE topology F-score mean | 0.7315 | 0.6348 | 0.5417 | 0.7480 |
| EV topology F-score mean | 0.7221 | 0.6184 | 0.5548 | 0.7392 |

- 覆盖情况：
  - 四组 validity 都是 `2424 / 2424`。
  - 四组 visible-condition 都是 `2423 / 2424`，缺失样本仍为 `00547707`。
- Paired validity vs baseline：
  - CVAE two-step：baseline-only valid = 571，CVAE-only valid = 111。
  - Old oracle ckpt：baseline-only valid = 267，old-oracle-only valid = 448。
  - Real-photo oracle：baseline-only valid = 265，realphoto-oracle-only valid = 375。
- Real-photo oracle 相对 baseline：
  - valid solid 增加 `+110` 个样本，absolute +4.54 percentage points。
  - visible CD 从 `0.03218` 到 `0.03329`，略差。
  - visible F-score 从 `0.6593` 到 `0.6443`，略差。
  - full FE / EV topology F-score 反而略高：FE `0.7315 -> 0.7480`，EV `0.7221 -> 0.7392`。
- 直接解释：
  - 真正 real-photo oracle 证明 GT topology bias 对 validity 有正向上界，但幅度不是压倒性：`47.85% -> 52.39%`。
  - 它没有改善 visible geometry，说明 topology bias 更像在帮助 STEP 合法性 / 拓扑结构，不会自动增强图像条件一致性。
  - CVAE predicted topology two-step 仍然是明确失败；原因不是 topology path 没上界，而是 predicted topology 太不准、bias 接入太强，导致 diffusion 被错误结构约束带偏。
  - Old oracle ckpt 的 validity 虽高但 geometry 异常差，继续作为 mismatch sanity check 保留，不应作为 real-photo upper bound。

**参考 / 关联**

- 06-18 CVAE 训练：[2026-06-18 NOTE](../2026-06-18/NOTE.md)
- two-step 失败总结：[2026-06-23 NOTE](../2026-06-23/NOTE.md)
- old oracle 对比：[2026-06-24 NOTE](../2026-06-24/NOTE.md)
- Baseline summary：[eval_summary.json](../baseline/diffusion_baseline_real_v7_cube24/test_post/eval_summary.json)
- CVAE summary：[eval_summary.json](../baseline/cvae_visibility_diffusion_post/eval_summary.json)
- Old oracle summary：[eval_summary.json](../baseline/diffusion_oracle_topology_real_v7_cube24/test_post/eval_summary.json)
- Real-photo oracle summary：[eval_summary.json](../baseline/diffusion_oracle_topology_realphoto_v7_cube24/test_post/eval_summary.json)

### 2. Real-photo oracle 在 100-sample test list 上的复查

**为什么这么做**

- [#1](#1-baseline--cvae-two-step--old-oracle--real-photo-oracle-四模型总表) 中 real-photo oracle 的 full test valid rate 只有 `52.39%`，低于预期；用户希望用常用的 100-sample test list 复查，排除训练/测试配置明显错误。

**具体实施**

- 使用同一个 checkpoint：[realphoto_oracle_topology.ckpt](../baseline/ckpt/realphoto_oracle_topology.ckpt)。
- 只替换 test list：[deduplicated_deepcad_testing_7_30_sample_100.txt](../../src/brepnet/data/list/deduplicated_deepcad_testing_7_30_sample_100.txt)。
- Sample100 输出：
  - raw predictions：[test_outputs_sample100](../baseline/diffusion_oracle_topology_realphoto_v7_cube24/test_outputs_sample100/)
  - post/eval：[test_post_sample100](../baseline/diffusion_oracle_topology_realphoto_v7_cube24/test_post_sample100/)
  - summary：[eval_summary.json](../baseline/diffusion_oracle_topology_realphoto_v7_cube24/test_post_sample100/eval_summary.json)

**结果**

| 指标 | Baseline full run on same 100 IDs | Oracle full run on same 100 IDs | Oracle sample100 rerun |
|---|---:|---:|---:|
| Valid solid rate | 48 / 100 = 48.00% | 60 / 100 = 60.00% | 56 / 100 = 56.00% |
| `visible_face_cd` mean | 0.03052 | 0.03338 | 0.03315 |
| `visible_face_fscore` mean | 0.6651 | 0.6613 | 0.6579 |
| `visible_face_precision` mean | 0.5372 | 0.5295 | 0.5275 |
| `visible_face_recall` mean | 0.9141 | 0.9201 | 0.9178 |
| Full face F-score mean | 0.8684 | 0.8734 | 0.8592 |
| FE topology F-score mean | 0.7546 | 0.7847 | 0.7510 |
| EV topology F-score mean | 0.7512 | 0.7722 | 0.7553 |

- sample100 rerun 的 valid rate 是 `56%`，明显高于 full test 的 `52.39%`，但 full test 输出中同一 100 个 ID 的 oracle valid rate 是 `60%`。
- sample100 rerun 与 full-run 同 100 ID 的 paired validity：
  - both valid = 47
  - full-run only valid = 13
  - sample100-rerun only valid = 9
  - both invalid = 31
- 解释：
  - 这次复查没有发现明显测试配置错误；同一 checkpoint、同一 100 个 ID 在 full run 与 sample100 rerun 上指标接近。
  - 两次 oracle 推理不是逐样本完全一致，说明 diffusion inference 有采样随机性；validity 在 `56%~60%` 之间波动合理。
  - 100-sample 子集上 oracle 相对 baseline 的合法性提升更强：`48% -> 56%/60%`，但 visible geometry 仍没有明显优于 baseline。

**参考 / 关联**

- 100-sample list：[deduplicated_deepcad_testing_7_30_sample_100.txt](../../src/brepnet/data/list/deduplicated_deepcad_testing_7_30_sample_100.txt)
- Sample100 summary：[eval_summary.json](../baseline/diffusion_oracle_topology_realphoto_v7_cube24/test_post_sample100/eval_summary.json)
- Full real-photo oracle summary：[eval_summary.json](../baseline/diffusion_oracle_topology_realphoto_v7_cube24/test_post/eval_summary.json)

## 今日结论

- 当前最强 valid solid rate 是 real-photo oracle topology diffusion：`52.39%`，比 no-topology baseline `47.85%` 高 `+4.54 pp`。
- 但 real-photo oracle 的 visible geometry 略低于 baseline，因此 topology bias 不能被当成“图像一致性增强器”；它主要是合法性 / 拓扑结构约束。
- 100-sample 复查支持同一判断：oracle valid rate 为 `56%`（同 100 ID full-run 为 `60%`），比该子集 baseline `48%` 更好，但 visible geometry 仍只是接近 baseline。
- CVAE predicted topology 不能直接作为强 mask 接入 diffusion。当前核心瓶颈应从“继续提高 CVAE 局部指标”转到“训练 diffusion 对 noisy / soft / partial topology 鲁棒”。

## 待办

- [ ] 训练 robust topology-conditioned diffusion：训练时混合 clean GT adjacency、corrupted adjacency、near-zero/no topology，让 diffusion 学会在 topology 错误时不要被毁掉。
- [ ] topology predictor 的目标从 exact adjacency/F1 转向 high-recall、soft confidence、visible partial topology。
- [ ] 后续 two-step 评估仍以 valid solid rate + visible-condition 为主，full condition 作为辅助。
