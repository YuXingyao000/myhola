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

正式run已启动，7个rank正常推进；2026-10-25 14:46:19 UTC确认151个microbatch、至少18次参数更新，无致命错误。

- 运行目录（142）：`/mnt/d/python/checkpoints/uv_intersection/joint64/20261025T144428Z`。
- W&B：[uv-direct64-joint-30k](https://wandb.ai/whatseven777/hola-brep/runs/745b0a12df12)。
- tmux：`uv64-joint-20261025T144428Z`；源代码commit `ca7954d1112187bd11fc89034238718c797ee9ee`，启动时工作树干净。
- 实际参数：170,699,647全部训练，冻结0；592个源tensor迁移170,229,690个参数，新接口469,957个参数。
- 本地28项检查通过；两个真实30面validation CAD（00025689/00072409）完成全模型BF16训练步，
  峰值15.15GiB，encoder/posterior/decoder探针均实际更新，KL参与优化，所有可训练梯度有限。
- 迁移后的8层attention在相同16-query/512-memory输入下与旧算子逐位一致。
- 2卡、batch2、累积2的短运行完成2次optimizer更新与validation，checkpoint中617组参数状态全部step2。
- 正式运行采用7卡×batch2×累积8、30,000步、LR2e-5。每415步完整DDP validation；
  validation sampler会补齐5个重复样本，最终精确评估用单卡。停止在非完整epoch时不把旧val值当末步实测。
- 精确命令、环境与初始化映射在run的metadata；紧凑验收记录见 [JOINT_LAUNCH_20261025.json](JOINT_LAUNCH_20261025.json)。

## 中途观察（2026-10-26，北京时间）

读取W&B记录时global_step约10329/30000。validation记录从step414到9959：
face_xyz L1 0.0022325→0.0019342；curve_xyz MSE 0.069607→0.00376686；
curve_incidence MSE 0.111281→0.00595031。最近一次曲线项小幅回升，整体仍下降。
两侧对应采样点的RMS间距按日志口径为sqrt(3×MSE)≈0.1336（数据坐标），并非最近曲线距离或毫米值。
这是中途loss观察，不是几何/OCC验收；保持现有训练不变。原始validation历史见 [JOINT_PROGRESS_20261026.json](JOINT_PROGRESS_20261026.json)。
