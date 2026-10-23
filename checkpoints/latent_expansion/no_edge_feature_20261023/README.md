# 64维全解冻：关闭edge_feature

2026-10-23，用户授权在142训练。只把 `model.loss_weights.feature_weight` 从1改为0，teacher的local/bbox几何监督及其余loss保持不变。当前状态：已启动，8个DDP进程进入训练且loss有限；训练尚未完成，尚无本轮几何改善结论。

- [固定协议](PLAN.md)、[架构与loss变更](ARCHITECTURE.md)、[启动命令](launch.sh)。
- [启动记录](launch_record.json)、[实际配置](resolved_config.yaml)、[真实batch检查](preflight.json)。
- [在线W&B](https://wandb.ai/whatseven777/hola-brep/runs/8fd1dc1d1023)，名称 `vae-latent64-no-edge-feature-30k`；tmux会话 `vae64-no-edge-feature-30k`。
- 启动时间：2026-10-23 03:50:06 UTC（北京时间11:50:06）。训练源码提交 `7948c1b695d1160218c9ae82f15630d663ce6dbc`，与旧64的网络源码相同。
- 独立配置：`configs/train_vae_latent_expansion_no_edge_feature.yaml`。
- 训练目录（142）：`/mnt/d/python/checkpoints/latent_expansion/d64_no_edge_feature/20261023T034042Z`。目录时间是预分配标识，实际开始时间以metadata/run.json为准。
- 预算：30,000 optimizer steps，8×RTX5090、每卡batch16、LR2e-5、BF16、seed42。按旧64的0.619 step/s估算约13.5小时，实际以新运行速度为准。
- 初始化：与旧64同一shared-curve 32维checkpoint，按原规则扩为face latent64/edge feature128；新Adam，全模型解冻。不是从旧64后期权重resume。
- 原始 `edge_feature` 仍会出现在日志中；`weighted_edge_feature` 必须为0，它不再参与total loss和参数更新。保留原始值用于诊断，不表示开关没有生效。
- 原有保存器仍维护top3和last。本轮total loss减少了一项，不能和旧运行直接比较高低；后续以固定预算终点和各项几何指标判断。
- 本轮没有修改模型、dataset、trainer或后处理，没有展开其他消融或训练读出探针。

本实验工作树不使用主工作区自动同步hook。仅同步本轮新增配置/记录；网络源码基准与旧64相同。已有未提交工作与历史产物保持原状，具体提交和部署核验写入启动记录。

启动前真实16件训练batch通过BF16前向/反向检查，未执行optimizer更新：182,982,439个参数全部可训练，encoder/posterior/decoder梯度均有限且非零；raw `edge_feature=0.0208975`，`weighted_edge_feature=0`，后者对coarse edge feature的梯度精确为0。teacher local/bbox与KL加权项仍为正。这些结果仅证明实现符合协议。

03:53:12 UTC核验：8个rank均已注册、8卡各有训练进程，控制台已完成epoch 0的83个batch，loss为0.0235；W&B在线history已收到 `trainer/global_step=49`、`train/loss_step=0.0245988`。正式run的完整配置与预检配置一致。各分项按epoch记录，因此启动时还没有完整epoch的feature曲线。

服务器Git接收钩子因已有未跟踪历史产物拒绝直接推送，亦拒绝非当前分支的临时ref。核对文件不重叠后，使用本次6个新增文件的Git bundle执行fetch及 `merge --ff-only`；未修改接收钩子、未覆盖或删除历史产物，启动前已验证服务器HEAD与上述提交一致。
