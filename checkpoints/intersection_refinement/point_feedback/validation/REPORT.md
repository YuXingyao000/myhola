# Task

验证逐点曲线反馈的初始化兼容性、梯度与真实小批量资源需求。**最终检查通过；这是实现验证，不代表重建精度提升。**

# Assumptions

正式训练从授权的 conditioned epoch 157 / step 57354 checkpoint 重新初始化。此处关闭 augmentation，仅为精确对照；Encoder/posterior 保持冻结。16 对象显存测量不能保证任意复杂度批次均能装入显存。

# Inputs

- 配置：`train_vae_intersection_point_feedback`；完整 resolved model/dataset 配置与来源 checkpoint 见 [result.json](result.json)。
- accepted train 共 46,480 件；取前两件 `00000061`、`00000070` 做等价性和两步梯度验证，取前 16 个不同对象测资源。
- 批 2：[7, 14] 面，224 个有向候选。批 16：235 面、每件 7–26 面、3,602 个候选。
- checkpoint、train split、16 份 NPZ、模型源码、配置及验证脚本 SHA256 均保存在结果 JSON。曾建议的 `00000797` / `00001926` 不在该 accepted train，未使用。

# Outputs

[validate_real.py](validate_real.py)、[result.json](result.json)、本报告。[scale_probe.json](scale_probe.json) 保留修正前的尺度诊断；两次前置失败记录保存在忽略目录 `logs/`，没有保存临时训练权重。

# Processing Protocol

固定 seed 20261014，RTX 4090 单卡、CPU threads 2、确定性 CUDA、TF32 关闭。严格迁移后核对全部旧 state；同一模型在 eval/posterior-mean 下分别启用与临时移除新模块，在 FP32 与 BF16 autocast 内各自逐 tensor 精确对照。随后 BF16 train 模式执行两步 Adam，learning rate 2e-5、norm clip 1.0；再对 16 对象执行一次训练步。耗时为已加载数据后的 CUDA 同步测量。

# Metrics

| 检查 | 结果 |
|---|---|
| 旧权重 | 全部 736 个 state tensor 逐值保留 |
| 新增量 | 6 个 tensor，27,139 参数；模型总计 181,993,120 参数 |
| 原 inter | 8 层、512 维保持 |
| 冻结范围 | Encoder/posterior 共 243 个参数 tensor，无梯度 |
| FP32 零初始化 | 74 个编码、预测、损失 tensor 数值完全相等 |
| BF16 零初始化 | 同样 74 个 tensor 数值完全相等；仅 refined local_points 提升为 FP32 |
| 两步梯度 | 每步全部 499 个梯度 tensor 有限，原 493 个可训练 tensor 均有非零梯度；旧 classifier 实际更新 |
| 新 delta 首步梯度 L1 | 11.401243 |
| 新 hidden 首步 / 第二步梯度 L1 | 0 / 0.20038956 |
| 两步后 local/global 最大差 | 5.9604645e-08，在 FP32 舍入范围内 |
| 批 16 | 45 个 loss 与全部 499 个梯度 tensor 有限 |
| 批 16 显存 | allocated 5.857 GiB；reserved 6.039 GiB |
| 批 16 耗时 | forward 0.948s，backward 1.030s，含 optimizer/审计共 2.047s |

# Decision Rule

旧权重和零初始化预测/损失要求精确相等，不放宽容差。新 delta 第一轮必须获得梯度，前面的 hidden 层第二轮必须获得梯度；Encoder/posterior 不得有梯度。local/global 非零修正后的关系允许 FP32 舍入（rtol 1e-6、atol 2e-7）。所有条件均已通过。

# Edge Cases

最初一次启动因验证脚本的 Hydra 路径引用含 `=` 而退出，修正为结构化写入配置；模型尚未运行。随后初版反馈在 16 对象批次遇到原 bbox 预测尺度精确为零，不能直接除以该值。诊断关闭新反馈，仅观察原几何头；全部八组均无 NaN/Inf：

| 状态 | 模式/精度 | zero | zero 中 GT 正例 | nonfinite | 最小非零绝对尺度 |
|---|---|---:|---:|---:|---:|
| 初始化 | eval_fp32 | 0 | 0 | 0 | 0.00020378083 |
| 初始化 | eval_bf16 | 0 | 0 | 0 | 0.00048828125 |
| 初始化 | train_fp32 | 0 | 0 | 0 | 0.0015122518 |
| 初始化 | train_bf16 | 1 | 0 | 0 | 0.0014648438 |
| 两步后 | eval_fp32 | 0 | 0 | 0 | 0.0004504621 |
| 两步后 | eval_bf16 | 2 | 0 | 0 | 0.0009765625 |
| 两步后 | train_fp32 | 0 | 0 | 0 | 0.00062910467 |
| 两步后 | train_bf16 | 1 | 1 | 0 | 0.00048828125 |

初始化时 train/BF16 的零尺度属于 `00000633` 的负候选 `[3,6]`；两步后 train/BF16 的零尺度属于同件正候选 `[2,3]`。所以不能假设零尺度只出现在负候选，也不能把该现象归因于非有限梯度。

最终实现仅在**预测尺度精确为 0** 时屏蔽新 residual，原几何保持，原 bbox/XYZ 监督继续工作；正常尺度的 global XYZ delta 与除法路径保持。没有 epsilon clamp，没有依据 GT 选择是否屏蔽，nonfinite 仍报错。最小非零尺度照常参与计算。

# Validation / Sanity Checks

独立源码检查确认：反馈读取最终预测曲线、两个预测所属面和 `[E,F_i,F_j]`，GT 不进入 forward；GT adjacency 只用于损失及此处诊断正负例。最近三角片索引不求导，但选定三角片的顶点和查询点保持梯度。新 MLP 在 FP32 中输出 global delta，local XYZ 同步加 delta/scale，bbox 与方向通道值保持原行为；不会仅修改导出数据。无 point-index/dropout，因此相同点和条件收到同一修正。既有 incidence/twin/geometry loss 继续消费最终几何，权重未改。

# Implementation Constraints

只写本验证目录；未改模型、损失或数据，未启动正式 DDP。两步及批 16 的 optimizer 更新仅存于进程内，正式训练不能复用它们。该检查没有测量下游 solid 成功率，也没有证明能达到 1e-4 几何精度。

# Deliverables

最终 `result.json` 的 `status=passed`。验证进程已退出并释放 GPU，后续正式入口/多卡检查由主任务执行。

## 自动回归与8卡入口

相关测试去重150项通过：Decoder/VAE 47项，投影/loss/local/joint相关95项，统一训练契约8项。零尺度行为覆盖同批0、正、负及1e-8非零scale，非零不截断；NaN/±Inf仍拒绝。通用训练契约实际验证resume清除新增初始化来源，以及Timer到期后验证和保存当前状态。

实际统一入口8卡×16、BF16、原cube24与accepted split短运行完成，设置1秒Timer与2步安全上限，在完成第1个训练step后停止，完成各卡1个验证batch并保存best/last，退出码0。last含global_step1、499份Adam参数状态及Timer；243冻结tensor仍与来源精确一致，493个旧可训练tensor和新delta均已更新。该运行与正式8小时run隔离，正式训练重新从授权来源初始化。[核对记录](ddp_check.json)
