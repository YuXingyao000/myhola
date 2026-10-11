# 2026-10-11 性能诊断

用户报告 intersection refinement 训练慢。本次只检查执行成本并给出命令覆盖，没有改变网络、损失、采样规则或默认配置。源码版本 `199cb2f4`；测量为实现诊断，不是重建质量评估。

## 慢在哪里

原训练只解码 GT 正边的几何；新版为了从预测几何读取反馈，解码所有对象内有向非 self 候选，并生成初稿和最终两轮几何。对于 29/28 面的两件零件，候选总数为 1568。初始 candidate chunk=64 导致 inter 约 25 次调用、两轮 edge decoder 又各约 25 次，activation checkpoint 在反向时重算这些分块。

几何读取还需让每条候选的 16 个采样点对照两张面的各 450 个三角片；该批量初稿就有 22579200 个点—三角片组合。初始 geometry chunk=32 产生大量小块和独立 GPU 运算。更大的对象 batch 如果仍使用很小的内部块，主要会增加循环次数。

## 实测

单卡 RTX 4090、bf16、geometry fp32、4 CPU 线程。使用同一组前 16 个 accepted train 对象，关闭旋转，数据预载 GPU；一次 warmup 后计时一次遍历，计时包含 forward/backward/梯度裁剪/Adam，不含读取、DDP、验证、W&B 或保存。完整 ID、面数、候选数、各步时间保存在对应 JSON。

| 配置 | 每卡 batch | candidate/geometry chunk | 反向重算 | 秒/step | 件/秒 | 峰值 allocated GiB |
|---|---:|---:|---|---:|---:|---:|
| 初始 | 2 | 64/32 | 开 | 0.786 | 2.54 | 2.33 |
| 调大分块 | 2 | 256/128 | 开 | 0.360 | 5.55 | 2.34 |
| 再关闭重算 | 2 | 256/128 | 关 | 0.302 | 6.63 | 2.71 |
| 增大 batch | 8 | 256/128 | 关 | 0.521 | 15.37 | 4.10 |
| 增大 batch | 16 | 256/128 | 关 | 0.829 | 19.29 | 5.62 |

batch=16 只有一个计时 step，因此这些数字只用于选择执行配置，不作为稳定吞吐基准。不同 batch 的优化轨迹也不同，不能用于比较损失数值。

另将 29/28 面的两个对象重复组成 16 件压力批量，共 12544 条候选，完成两步计时训练，峰值 allocated 为 13.51 GiB、reserved 为 13.93 GiB，loss 有限。该检查不是所有输入的最大显存保证。

随后使用 512 个不同训练对象，8 卡 × batch16、num_workers=2、cube24 增强，实际 CLI 完成 4 步训练、1 批验证与保存。首步之后三个日志时间间隔为 1.376/1.193/1.061 秒，平均 1.210 秒，每步处理 128 件。该数据含 DDP/读取/日志，但不包含启动和末尾保存；尚未测完整 epoch。

用户原 run 日志为 `74/2905 [01:32, 0.80it/s]`。8 卡每卡 batch2 的全局 batch 为 16，46480 件恰为 2905 步；改为每卡 batch16 后因 drop_last 每轮为 363 步。不能仅比较每步耗时，需要同时看每步处理对象数。

## 命令覆盖

在原正式训练命令末尾追加：

```bash
trainer.batch_size=16 \
model.refinement.candidate_chunk_size=256 \
model.refinement.geometry_chunk_size=128 \
model.refinement.checkpoint_candidates=false \
model.loss_weights.geometry_chunk_size=128
```

保留用户已选择的 `trainer.max_steps=1000000 trainer.max_epochs=10000`。`checkpoint_candidates=false` 关闭的是反向计算时的激活重算，不会关闭模型权重保存。学习率仍为 2e-5；更大 batch 的优化轨迹需要结合训练/验证曲线判断。

## 记录与复现边界

[summary.json](summary.json) 记录源配置/脚本哈希、单卡协议和多卡日志时间戳。单卡各变体报告位于上一层 `performance_*.json`；原始脚本与配置保留在本机 `metadata/`，多卡 CLI 命令与配置由 run 管理保存到 `ddp8/20261011T123216.545853Z/metadata/`。DDP 输入列表为 [ddp_train.txt](ddp_train.txt)，验证列表为已有 `validation/smoke_val.txt`。短运行全部 seed=42、W&B 关闭。

本次未改变 GT 候选筛选、距离精度或几何监督定义，也未启动新的正式长训练。进一步性能优化可以检查 GPU 同步校验、GT 几何对应缓存和按对象分组 attention；这些尚未实施。
