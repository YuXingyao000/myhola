# 实现与兼容性验证

日期：2026-10-13。仅验证代码归并、权重迁移和新增调制的计算行为；没有正式微调、整套测试集重建或有效 solid 提升结论。

## 归并等价性

修改前工作副本保存为独立源码快照，包含当时尚未提交的 local 实现。与归并后的源码分别运行：真实已训练 refinement、小型原 refinement、小型两层 local、小型完整 Transformer local、小型 joint 共五种变体。

五种变体的 state 名称／形状／顺序、参数注册顺序、可训练范围，以及 encoder、posterior、decoder、loss 张量均完全一致。AST 对比确认迁入的原编码器、损失、refiner、Intersection 与 joint/local block 定义保持原样；计算变化集中在最终几何头调制及其配置／初始化。

最初未控制确定性的独立 CUDA 运行在 encoder 归约已有约 3.8e-6 差异；统一启用 deterministic algorithms 与 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 后，两次独立进程的结果精确一致，没有放宽比较容差。

## 新增调制

- 来源：原 refinement `epoch=0010-step=00003993-val_loss=0.039476.ckpt`，全部 730 个旧 state 保留。
- 真实对象：`00000797`、`00001926`，分别 12／7 个面，共 174 个有向候选。
- 开启零初始化调制后，74 个输出／loss 张量与原模型精确一致，最大绝对差 0。
- 仅新增 6 个参数 tensor、239,424 个参数，总参数从 181,726,557 增至 181,965,981。
- 原 Intersection 仍为 8 层、512 维、8 heads、FFN 2048。
- 真实 BF16 反向：45 项 loss 有限，493 组参数梯度有限；point／bbox affine 均有梯度；encoder／posterior 冻结。未执行 optimizer step。

完整数值与环境记录：[compatibility.json](compatibility.json)。训练命令的配置解析、权重／数据路径及与原基线模型／数据配置的差异：[config.json](config.json)。

## 自动检查

合计 136 项通过，分三组执行，未重复计数：

| 检查 | 通过数 |
|---|---:|
| decoder／VAE，包括新增 14 项条件调制、迁移与恢复检查 | 26 |
| 投影、损失、joint、local 及其权重迁移回归 | 95 |
| 两个评估推理入口、源码溯源与导出兼容性 | 15 |

新增检查覆盖：零初始化等价；非零调制仅直接改变 refined 边几何；实际使用更新后的 E／两个有序 owner face；两侧条件梯度；chunk checkpoint 梯度一致；空候选；旧参数完整保留；拒绝不完整／额外／形状不符／已带调制的迁移来源；已训练调制与 optimizer 状态恢复。

归并后发现并修正了评估脚本中指向已移除文件的源码哈希清单；两套评估入口改为追踪 `vae.py`／`blocks.py`。历史评估产物未改写。新旧源码哈希不同，不能把新运行伪装成旧协议的续跑。

运行环境：`/mnt/d/anaconda3/envs/torch/bin/python`。GPU 兼容性检查使用 CUDA 1，seed 20261013；常规 CPU 测试限制线程数为 2。原配置指定的正式训练 seed 仍为 42。

模型结构与损失没有其它改动。此处的零误差是初始化／归并等价性，不是相对 GT 的几何精度，也不是训练后的性能保证。
