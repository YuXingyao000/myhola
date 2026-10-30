# 第一阶段代码验收

2026-10-23。工作树基于 `77620a73597147009fe0ec0911de59db198c86d5`，本次实现为未提交改动；具体源码摘要随验证 run 的 `metadata/protocol.json` 和源码快照保存。此处记录实现验收，不是 DPO 效果结论。

## 测试

使用 `/mnt/d/anaconda3/envs/torch/bin/python`，`CUDA_VISIBLE_DEVICES=''`，CPU 数值库单线程。以下测试合跑 **97 passed in 7.46s**：

```text
src/brepnet/test/test_preference_relabel.py
src/brepnet/test/test_preference_observation_selection.py
src/brepnet/test/test_preference_selection.py
src/brepnet/test/test_preference_observation_worker.py
src/brepnet/test/test_observation_metrics.py
src/brepnet/test/test_observation_preference_dataset.py
src/brepnet/test/test_diffusion_preference.py
```

覆盖观测阈值/冲突/零距离、缺测弃权、确定性配对、GT 共用渲染、源文件和协议变化、原始 latent 路径、partial 拒绝训练、真实 fork 的超时/崩溃隔离、进程退出清理及 manifest 续跑无重复。旧选择器与旧训练接口测试保持通过。`git diff --check` 通过。

## 真实候选入口检查

源 run：`checkpoints/diffusion/preference_data/20261020T134930Z`。
验证产物：`checkpoints/diffusion/preference_data/observation_cpu_acceptance_20261023`，明确标记 `partial=true`。
使用原 validation 清单前 2 件（`00938923`、`00433839`），共 16 个候选；2 个 CPU worker，每 CAD 120 秒预算。

| 项目 | 结果 |
|---|---:|
| 固定 CAD / 候选分母 | 2 / 16 |
| 原 valid / invalid | 12 / 4 |
| 可测观测 / 测量失败 | 10 / 6 |
| 新 pair | 2（validity 1、geometry 1） |
| 相对旧 pair | 保留 1、替换 1 |
| 初次评分调度耗时 | 3.32 秒 |
| 改为 1 worker 后续跑耗时 | 1.37 秒 |

6 个测量失败均是 STEP 面无法完整三角化，原合法性标签保留，不能当作 loser。没有丢弃这些候选分母，没有修改后处理来挽救本小样本。仅两件不能代表全库缺测率或吞吐，也不能据此预测全量完成时间。

续跑核验了全部源摘要；16 个逐候选评分文件和两个 manifest 共 **18 个文件字节完全一致**。过程重建了完整 manifest，没有重复追加 pair。初次与续跑状态及摘要对比保存在验证 run 的 `acceptance.json`。

## 边界

未调用 GPU、未重新采样 diffusion、未重编码 VAE、未重跑 OCC 后处理；未启动全量 train/validation 重选。新数据 DPO 与几何辅助训练均待数据完成后再开展。运行与续跑指令见 [README.md](README.md)。
