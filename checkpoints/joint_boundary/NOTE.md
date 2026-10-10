# 2026-10-10 — 联合共享边界 Decoder 实现

## 为什么与实现范围

在分支 `codex/joint-boundary-decoder` 实现第一次核心 Decoder 改造。起点为 `8cda6ab3`，原始研究重启状态可在原分支 `rotation_correction` 查看。本次不追加 `experiments/`；目录和命令见 [README](README.md)，事先确定的合同见 [PLAN](PLAN.md)。

`AutoEncoder_joint` 保留 HoLa light 的 encoder、posterior 和输入，使用 128 个共享边界 queries、3 轮面边交互、存在性与两侧 face assignment、共享曲线及曲面 XYZ 输出。训练以几何和归属做集合匹配，支持曲线反向、两侧交换与闭合曲线的离散循环起点。Decoder 前向不读取 GT connectivity；GT 仅用于监督和评分。

本版 Decoder 使用新的 **global XYZ heads**，没有沿用旧 local-grid + bbox heads，模型规模约 79.5M，其中冻结 encoder/posterior 约 73.7M、可训练 Decoder 约 5.7M。未来新旧 Decoder 对比应说明整个解码架构与损失的差异；单向配置可以隔离一部分双向更新贡献，但不是旧模型的等容量对照。

存在性损失按固定 query 容量平均，负类权重 0.1；曲线、归属与一致性按对象平均，面 L1 按 batch 中的面平均。改变容量、batch 或复杂度分布时，需重新审视这些尺度。面边一致性是期望采样距离代理，不保证连续参数面上的 trim、loop 或 solid 合法性。

## 数据预检与明确支持范围

当前数据为“每对不同面之间至多一条合并边界”，缺少原生 multi-edge/seam/UV-loop 信息。原始 face/half-edge 数组继续输入 encoder，新 targets 在旁路构造，不覆盖数据集。

全量 query 容量审计：训练最多 108 条共享边界，1,242 件超过 64，所以选择 128；验证最多 90、测试最多 84。[报告](validation/boundary_count_audit.json)。

按 reciprocal 残差 0.01、闭合端点阈值 1e-5 审计全部三个原 split。闭合边允许起点不同，用双向采样点到折线的距离核查：

| split | 原始数量 | 接受 | 拒绝 |
|---|---:|---:|---:|
| training | 46,627 | 46,480 | 147 |
| validation | 3,505 | 3,495 | 10 |
| testing | 2,424 | 2,417 | 7 |

每个 split 的 `accepted.txt`、`rejected.csv`、`summary.json` 均随代码版本保存；完整 `samples.csv` 留在本地可重建。原 split 未修改。新旧模型比较必须使用同一接受清单；完整测试覆盖率仍以 2,424 为分母，7 个不满足合同的样本单列，不能称 2,417 件为原始测试全集。

训练拒绝样本全部是闭合边。部分只是稀疏采样造成无法验证的相位差；也存在明显不一致，例如 `00705673` 的同一 pair 两侧折线长 3.03 和 1.16，残差约 0.63。没有把阈值放宽到掩盖问题，也没有宣称全部拒绝样本是错误 CAD。[逐件调查](validation/reciprocity_failure_details.json)。

## 实际实现检查

环境：`/mnt/d/anaconda3/envs/torch/bin/python`，Python 3.10、PyTorch 2.4.1+cu124、Lightning 2.5.0.post0、RTX 4090。该环境原有 OCC/PyG/scatter/Hydra/W&B；本次增加 pytest 8.4.2 用于测试。

- 真实旧 checkpoint 迁移：`1119_deepcad_aug1_11k.ckpt` 的 243 个 encoder/posterior tensors 通过 key/shape 验证。合成迁移测试中旧、新 encoder 的 features 和 mean/std 逐 bit 一致；新完整 checkpoint 严格重载。
- GPU frozen CLI：真实数据、cube24 augmentation、bf16、batch 2，完成 2 optimizer steps 和验证，保存 best/last。只有 Decoder 参数可训练；encoder 子模块保持 eval。运行记录在 `validation/cli_frozen/`。
- GPU joint CLI：从上述新模型 checkpoint 初始化，解冻 encoder；使用审计后预先按边界数选出的 4 件复杂样本，每件 30 面，边界数为 108/96/93/90。bf16、batch 4，完成 forward/backward/validation 和 checkpoint 保存。清单见 [complex_batch.txt](validation/complex_batch.txt)，记录在 `validation/cli_joint_complex/`。
- 严格加载 frozen checkpoint 后测试两件真实对象，分别导出 12/18 个面、81/128 等可变预测边数；检查 NPZ 的逐对象形状、float32 latent stats、表示标签与有限值。GT 没有决定激活 query 数。
- 原 run 断点续训从 step 2 到 step 3，完整状态恢复成功；W&B **离线** SDK 将日志写入原 run 的 `wandb/`。在线上传与云端恢复未在实现检查中调用；在线记录已由配置启用。离线 SDK 会另起本地日志段，不保证追加到旧离线文件。
- OCC 接口往返：提前选择训练清单首件 `00000061`，将同一 GT 的 30 有向边转换为 15 共享边界。固定 CPU、无几何优化、无删面；legacy/shared 两条路径均产出 STEP，经现有独立读取/有效性检查均为 solid。[报告](validation/post_roundtrip/summary.json)。这只验证接口兼容，非学习模型结果。

针对性测试 **39 passed**（15.90 秒；仅已有 Transformer nested-tensor 性能提示），[JUnit 报告](validation/tests.xml)。包括：对象隔离、face 置换等变、E→F 的实际影响、匹配等价性、空目标/空边、容量溢出、有限梯度、冻结与解冻、bf16、严格迁移、GT-free raw export、数据拒绝规则、post adaptation、目录隔离/恢复和旧 topology bias 回归。

集成检查中修复了：VAE test 先随机采样算 loss、再均值导出造成的两次不同预测；现在同一次 posterior mean forward 同时计算 loss 和导出。另修复续训路径误把任意上级 `checkpoints/` 当作旧运行的情况，只识别明确的 weights/legacy 布局，其余要求显式 run_dir。按 accepted 清单重新运行真实测试入口，确认按 2,417 件装载并正确导出。最终续训 checkpoint 的 global_step=3，全部权重有限。

重跑针对性测试：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python -m pytest -q \
  src/brepnet/test/test_joint_decoder.py \
  src/brepnet/test/test_joint_boundary_dataset.py \
  src/brepnet/test/test_shared_boundaries.py \
  src/brepnet/test/test_shared_boundary_post.py \
  src/brepnet/test/test_run_management.py \
  src/brepnet/test/test_boundary_audit.py \
  src/brepnet/test/test_topology_attention_bias.py
```

## 研究状态

尚未启动 100,000-step 正式研究运行，没有下游质量提升结论。下一步先运行 frozen 配置，再按结果开展 joint fine-tuning；可用相同的旧生成 latent 比较冻结 encoder 版本的 Decoder。Encoder 联合训练后，旧 latent cache/生成器不能被自动视为兼容。

## 2026-10-10 — 多卡配置与吞吐修正

用户第一次运行 `frozen/20261010T082441.611797Z` 实际继承 `devices=1, batch_size=4`；W&B 不会自动启用多卡。该次约到 1,025 steps 后由用户中止，尚未到首轮验证，`weights/` 为空，不能从该 run 恢复 optimizer。以上“尚未启动”是最初实现检查时的状态；现在有这次未完成的训练尝试，仍无正式下游质量结果。

本次将 frozen／one-way 默认改为 8 卡、每卡 batch 32，joint 解冻 encoder 时保留每卡 batch 4。入口打印设备数与全局 batch；由 Lightning 启动 DDP 并按 rank 初始化数据 worker 随机种子。正式测试／逐对象 NPZ 导出限制单卡，避免 sampler 补齐造成重复导出；DDP 训练验证仍可运行。

CPU/GPU 混合损失中，每件对象的曲线／归属 cost 原来在 matching 与可微 loss 各计算一次，现复用同一组 cost，仅在 CPU Hungarian 匹配时 detach。闭合边的循环移位候选改用一次索引构造，保留原候选顺序和相同最小值时的梯度选择。损失、权重和模型结构没有改变；新增回归测试逐项验证 loss、metrics 及几何／归属／存在性梯度与旧实现完全一致。

真实 accepted 训练数据、冻结旧 encoder、bf16、每种 batch 3 步预热 + 10 步计时，单张 RTX 4090 的短基准如下，单位为 objects/s：

| 设置 | batch 4 | batch 16 | batch 32 |
|---|---:|---:|---:|
| 原损失，CPU 64 threads，cuDNN search 开 | 19.0 | 36.4 | 48.1 |
| 复用 cost，CPU 64 threads，cuDNN search 开 | 20.1 | 39.8 | 52.7 |
| 复用 cost，CPU 4 threads，cuDNN search 关 | 36.6 | 66.4 | 86.6 |

测量包括取 batch 到 optimizer step 的完整耗时，每步末 CUDA synchronize。加载等待约 0.2–1.1 ms；这里数据已热，不能外推所有存储场景。encoder 的 packed 面／边数量变化会引入新卷积形状，因此 joint-boundary 关闭 `trainer.cudnn_benchmark`，其他配置保留原默认行为；命令设置 `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4`。batch 32 的最后一组短基准峰值 allocated memory 约 0.99 GiB，只反映所测数据，不是所有复杂对象的内存上界。

这些是短期启动吞吐诊断，不是长期加速比，也没有测得可报告的平均 GPU utilization；部分独立诊断与前一进程的 CPU profiler 汇总重叠，不是严格隔离的重复性能研究。逐对象 Hungarian、动态索引与小 kernel 仍有同步／调度开销，因此不承诺满载。原用户运行进度条约 8.5 step/s 与这里的短基准不能直接等同比较。

实际 8 卡 CLI 检查：`runtime_ddp8_check/20261010T083556.907473Z`。rank 0–7 全部注册，global batch 256，完成 10 optimizer steps、每卡 1 个 validation batch（每卡 16 件），`val/loss=1.014304`，保存 best/last，退出码 0；只有一个 run、一个 invocation 和一个离线 W&B run。checkpoint 内 `global_step=10`，设备／batch／cuDNN 配置与记录一致。这不是完整 validation 评估，也未启动新的正式长训练。

本次相关测试 **31 passed**（14.01 秒），包含旧／新损失及梯度逐项等价、CUDA bf16 前向反向、worker seed 与导出设备合同、原 run 管理回归。配置解析确认 frozen／one-way 为 8×32，joint 为 8×4。紧凑数字记录见 [runtime_20261010.json](validation/runtime_20261010.json)，详细诊断保存在本地 `metadata/performance_20261010/`。

预算注意：全局 batch 从 4 变为 256，保留 100,000 steps 会将样本曝光扩大约 64 倍；`1,563 × 256 ≈ 100,000 × 4` 只对齐曝光量，不保证优化等价。学习率没有线性放大；尚需正式训练判断大 batch 收敛。后续与旧模型比较时固定并报告预算，而不能仅对齐 step 数。
