# 三版固定 checkpoint 的历史参考

三版输入latent逐ID哈希一致，固定2417件；生产后处理及预算相同。训练曝光不同，不是等预算结构消融。直接配对失败分析仍以conditioned→point_feedback为准。

| 方法 | 仅拟合有效solid | 优化后有效solid | 优化后比例 |
|---|---:|---:|---:|
| 原 Intersection＋refinement | 1062/2417 | 2113/2417 | 87.42% |
| 曲线条件调制 | 1047/2417 | 2111/2417 | 87.34% |
| 本轮逐点几何反馈 | 1048/2417 | 2104/2417 | 87.05% |

| 输入 | 指标 | 三方共同数 | 方法 | 均值 | 中位数 |
|---|---|---:|---|---:|---:|
| posterior_mean | face_mae | 2417 | 原 Intersection＋refinement | 0.0020294875 | 0.0019122701 |
| posterior_mean | face_mae | 2417 | 曲线条件调制 | 0.0018679559 | 0.0017439367 |
| posterior_mean | face_mae | 2417 | 本轮逐点几何反馈 | 0.0017276705 | 0.0015779349 |
| posterior_mean | face_cd_sq | 2417 | 原 Intersection＋refinement | 4.9448311e-05 | 3.799893e-05 |
| posterior_mean | face_cd_sq | 2417 | 曲线条件调制 | 4.0611396e-05 | 3.0545228e-05 |
| posterior_mean | face_cd_sq | 2417 | 本轮逐点几何反馈 | 3.6502283e-05 | 2.5627293e-05 |
| posterior_mean | edge_cd_sq | 2417 | 原 Intersection＋refinement | 7.7720833e-05 | 3.0634885e-05 |
| posterior_mean | edge_cd_sq | 2417 | 曲线条件调制 | 7.768578e-05 | 3.4281256e-05 |
| posterior_mean | edge_cd_sq | 2417 | 本轮逐点几何反馈 | 8.7761533e-05 | 3.1694009e-05 |
| archived_latents | face_cd_sq | 2417 | 原 Intersection＋refinement | 0.0051292545 | 0.0022519103 |
| archived_latents | face_cd_sq | 2417 | 曲线条件调制 | 0.0051075391 | 0.0022368571 |
| archived_latents | face_cd_sq | 2417 | 本轮逐点几何反馈 | 0.0050994002 | 0.0022221496 |
| archived_latents | edge_cd_sq | 2417 | 原 Intersection＋refinement | 0.013054119 | 0.0046563183 |
| archived_latents | edge_cd_sq | 2417 | 曲线条件调制 | 0.013145726 | 0.0047183179 |
| archived_latents | edge_cd_sq | 2417 | 本轮逐点几何反馈 | 0.013175294 | 0.0047232599 |

对最早refinement的optimized直接配对：救回52件，退步61件。

raw CD为归一化坐标双向平方距离；不代表最大边面间隙。归档latent没有重新采样diffusion。7件合同不支持样本仍在原2424总体分母中单列。
