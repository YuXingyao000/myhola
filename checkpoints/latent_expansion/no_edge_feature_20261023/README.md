# 64维全解冻：关闭edge_feature

2026-10-23，用户授权在142训练。只把 `model.loss_weights.feature_weight` 从1改为0，teacher的local/bbox几何监督及其余loss保持不变。当前状态：启动前检查，尚未正式开始；实际状态将在启动核验后更新。

- [固定协议](PLAN.md)、[架构与loss变更](ARCHITECTURE.md)、[启动命令](launch.sh)。
- 独立配置：`configs/train_vae_latent_expansion_no_edge_feature.yaml`。
- 训练目录（142）：`/mnt/d/python/checkpoints/latent_expansion/d64_no_edge_feature/20261023T034042Z`。目录时间是预分配标识，实际开始时间以metadata/run.json为准。
- 预算：30,000 optimizer steps，8×RTX5090、每卡batch16、LR2e-5、BF16、seed42。按旧64的0.619 step/s估算约13.5小时，实际以新运行速度为准。
- 初始化：与旧64同一shared-curve 32维checkpoint，按原规则扩为face latent64/edge feature128；新Adam，全模型解冻。不是从旧64后期权重resume。
- 原始 `edge_feature` 仍会出现在日志中；`weighted_edge_feature` 必须为0，它不再参与total loss和参数更新。保留原始值用于诊断，不表示开关没有生效。
- 原有保存器仍维护top3和last。本轮total loss减少了一项，不能和旧运行直接比较高低；后续以固定预算终点和各项几何指标判断。
- 本轮没有修改模型、dataset、trainer或后处理，没有展开其他消融或训练读出探针。

本实验工作树不使用主工作区自动同步hook。仅同步本轮新增配置/记录；网络源码基准与旧64相同。已有未提交工作与历史产物保持原状，具体提交和部署核验写入启动记录。
