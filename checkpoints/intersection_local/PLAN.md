# 局部注意力 intersection：实验协议

2026-10-12 修订：用户要求只改变信息交换方式，不压缩原inter容量。此前2层128维版本的容量缩减超出讨论范围；历史训练与失败评估保留在 `frozen/`、`evaluation_20261012/` 和 `failure_census_20261012/`。本协议训练规格恢复8层512维。

2026-10-13 状态更新：完整容量版已正式训练，按验证集选定epoch42/step15609；固定2417件测试、优化后的合法solid为1705（70.54%），上一版refinement为2113（87.42%）。测试冻结协议及结果见 [evaluation_20261013](evaluation_20261013/PLAN.md)，失败诊断见 [failure_census_20261013](failure_census_20261013/PLAN.md)。不将下列原研究问题改写为已证实的结论。

## 研究问题

检验“从所属两面的局部特征与预测三维位置提取边特征，能否改善初始边几何及边面一致性”。当前失败统计支持研究边界几何，但不能证明原 inter 是唯一原因，也不能证明 refinement 已达到上限。本实验不声称实现完整拓扑建模。

## 已确认的范围

- 保持 encoder/posterior、face latent 语义与有序非 self 面对集合。每个对象 N 张面产生 N(N−1) 个有向候选；不合并 reciprocal edge feature，不额外去重。
- 替换初始 inter：`MLP([F_i,F_j])` 产生一个 query；已有 face CNN 输出投影前的 `96×16×16` 激活与对应预测全局 XYZ 形成每面 256 个局部 token。两面合成 512 个 memory token。
- 八层 `nn.TransformerDecoderLayer`，每层 hidden=512、heads=8、FFN=2048、dropout=0.1、ReLU、norm_first=True、batch_first=True，与旧 `AttnIntersection` 层内结构及初始化堆叠方式相同。每层有 query self-attention、读取局部memory的cross-attention、FFN；只有query更新，memory不更新。输出仍为 `E_ij[64]`，`E_ji` 由反向pair单独计算。
- 两个 side embedding 标记当前 pair 中的第一个／第二个面，不是 coedge 特征。没有 H/P，没有 vertex/next/loop 分支，也没有 face-to-face 局部 self-attention。
- S0 在 inter 之前解码并复用，S1 在 refinement 后正常解码。局部激活和 XYZ 不 detach，边损失可反传到 face CNN/bbox。
- 原线性 existence classifier、仅 logits 双向平均、GeometryRead 投影、refiner、几何头、全部 loss 权重保持原协议。

## 初始化与训练

从已评估的 refinement checkpoint `epoch=0010-step=00003993-val_loss=0.039476.ckpt` 显式初始化。只丢弃 `decoder.inter.*`；其余键和 shape 必须完整匹配，不能 permissive load。新 inter 随机初始化，其他 decoder 模块继续微调。encoder/posterior 冻结且保持 eval。

建立新 run、新 Adam，学习率 2e-5，bf16；8 卡×每卡 batch16，全局128。延续已运行协议的 cube24、accepted split 和 100 万 steps／1 万 epochs 上限；新 run 从 step0 计数。这是配置上限，不是已完成的训练预算。局部 attention 单独分块64并启用反向重算；原 refiner/edge decoder 分块256、geometry分块128、原候选重算关闭。

主配置 `train_vae_intersection_local` 显式指定 `block_type=transformer_decoder`，局部hidden/layers引用 `model.intersection_dim`／`model.intersection_layers`，不再出现顶层8层而实际local只有2层的配置分歧。新run目录为 `checkpoints/intersection_local/full_capacity/<UTC时间>/`。完整inter参数34,015,296，旧inter为33,683,008；两者8层主干均33,632,256，额外332,288来自Query/memory等输入输出适配，不声称总参数逐个相等。

为读取与复现历史证据，缺省block_type仍解释为原cross_attention，保留原模块名和state键。历史2层run使用 `train_vae_intersection_local_small_legacy`。恢复容量不迁移失败的2层local权重，也不额外改变encoder、refiner、几何头、loss、学习率或batch。

旧 latent cache 兼容取决于实际 encoder/posterior 权重、输入预处理和统计量约定相同。本次迁移完整保留这些权重，但本轮不宣称 diffusion 效果改善。新的完整 checkpoint 必须使用 local 配置严格加载；旧 `train_vae_intersection` 默认仍为 legacy。

## 判断依据

固定 split、checkpoint 选择协议、后处理版本与预算，区分 coarse 和 refined 几何；报告边几何、边面贴合、adjacency、fit-only 和优化后 solid 成功率。失败保留在预定分母。下游 solid 与固定 diffusion latent 的测试单独执行；实现验证不能替代研究结果。

本轮从已训练 refiner 开始，不能与“从原 HoLa 初始化且曝光量不同”的运行直接归因比较；若要隔离 local inter 效应，对照应从同一 checkpoint 继续微调、保持额外训练曝光量与评估协议一致。

## 状态与依据

恢复容量版本已实现并正式训练；实现验证见 [NOTE.md](NOTE.md)，2026-10-13测试见 [完整报告](evaluation_20261013/REPORT.md)和[研究解读](evaluation_20261013/NOTE.md)。历史2层版本已完成训练和评估，不能再称为未训练，也不能把其结果归到当前版本。完整容量版恢复了大部分损失，但当前权重仍落后上一版refinement。

- [原论文 HoLa](https://arxiv.org/abs/2504.14257)
- [原 refinement 协议](../intersection_refinement/README.md)
- [已评估 checkpoint 与结果](../intersection_refinement/evaluation_20261011/REPORT.md)
- [失败统计与证据边界](../intersection_refinement/failure_census_20261011/NOTE.md)
