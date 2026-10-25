# 64D UV Intersection 全量训练

2026-10-25，用户明确要求全量训练，尽量使用之前64维全解冻checkpoint初始化。
本轮 Encoder、Posterior、全部 Decoder 都参与优化，保留 KL。旧冻结run已暂停。

## 来源与迁移

- 源：142 `/mnt/d/python/checkpoints/latent_expansion/d64_no_edge_feature/20261023T034042Z/weights/last.ckpt`。
- 实读 metadata：`freeze_encoder=false`、`trainable_scope=all`、64D、step30000、epoch82、LR2e-5、KL weight5e-7、edge_feature weight0。
- SHA256：`4262f4998fe047e19c3ca21a835d74c183b7430f7a656c709ec8ed045262ada4`。
- 保留 encoder/posterior、全局face attention、face CNN、face center/scale，继续逐tensor严格映射。
- 新增显式迁移：旧 `decoder.inter.layers` 的8层512D pre-LN TransformerDecoder，映射到新 `CurveAttentionStack`，包括self/cross-attention、FFN和3个LayerNorm。
- 新的curve query、面空间特征投影、侧别embedding、UV/存在性输出头仍初始化；不按同shape猜测这些不同语义的接口。
- attention算子可迁移不代表任务未改变：query由1个变成16个，memory由1个面token变成两个面的512个空间token。收益须由本轮结果判断。
- 旧refinement、conditioning、三维curve CNN/bbox及旧UV分支无对应位置，明确不加载。
- 这是新run、全新Adam状态，从原64D全解冻权重初始化；不恢复刚暂停的冻结run。

## 固定协议

入口 `python -m src.brepnet.train --config-name train_vae_uv_intersection_joint`，实际运行显式传来源与数据路径。
维持seed42、LR2e-5、30000 optimizer steps、BF16、FP32 matmul highest、clip norm1。
计划物理GPU1–7、每卡batch2、累积8，112 CAD/update；以全模型真实最大尺寸验收决定能否沿用。
数据/split/增广/几何目标沿用 [TRAINING.md](TRAINING.md)，唯一loss合同变化是冻结时的KL诊断现在参与优化。
训练用后验采样，验证用均值；验证使用validation，不使用test调整方案。

旧冻结run与本轮还存在curve attention初始化差异，因此两者不是严格的单变量冻结消融。
本记录描述主实验训练协议，不将loss或实现检查等同于OCC合法率收益。

## 启动状态

待完成实际源全模型梯度、最大尺寸显存和短多卡检查，再填写正式run信息。
