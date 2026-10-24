# 64维全解冻：关闭edge_feature

2026-10-23，用户授权在142训练。只把 `model.loss_weights.feature_weight` 从1改为0，teacher的local/bbox几何监督及其余loss保持不变。当前状态：已正常完成30,000步；2026-10-24完成3495件validation的固定mean评估、同期曲线检查及完整OCC验证，详见[mean与曲线报告](evaluation_20261024/REPORT.md)、[OCC报告与六张总体图](occ_evaluation_20261024/REPORT.md)。

同期25–30k validation相对原64：最终edge XYZ低20.3%、teacher bbox低52.6%。固定mean相对来源32维：raw face CD²低24.8%、raw edge低8.5%、teacher逐边低5.6%；最终边79.3%的CAD改善，但coarse逐边均值高22.7%。这支持取消特征匹配能明显缓解退化，未证明全部架构问题已解决。旧64未保留30k权重，同预算结论限曲线；本轮未使用test。

同一3495件validation、原recover_v1/180秒预算下，OCC原生合法 **3179/3495（90.96%）**，来源32维为3180（90.99%）：救回28件、退步29件，合法率基本持平。双方共同合法且双方几何可测2960件中，STEP边CD²均值下降5.48%、面下降0.46%；候选316件失败中307件仍停在裁剪面阶段，232件face-pair集合正确。结论为“几何改善，合法性瓶颈未解”；这是单seed32D来源与64D终点比较，不能将全部变化唯一归因于feature开关。

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

2026-10-23 03:53:12 UTC启动核验：8个rank均已注册、8卡各有训练进程，控制台已完成epoch 0的83个batch，loss为0.0235；W&B在线history已收到 `trainer/global_step=49`、`train/loss_step=0.0245988`。正式run的完整配置与预检配置一致。各分项按epoch记录，启动时尚无完整epoch的feature曲线；后续83个train/82个validation分项记录已确认weighted feature全为0。

训练终点：北京时间2026-10-24 01:18:50完成W&B记录，last实际step=30000/epoch82，全部模型tensor有限、Adam步数一致。末权重SHA256 `4262f4998fe047e19c3ca21a835d74c183b7430f7a656c709ec8ed045262ada4`。日志末NCCL/TCPStore警告位于max_steps正常结束之后，tmux退出码0；详见[审计](evaluation_20261024/training_audit.json)。

服务器Git接收钩子因已有未跟踪历史产物拒绝直接推送，亦拒绝非当前分支的临时ref。核对文件不重叠后，使用本次6个新增文件的Git bundle执行fetch及 `merge --ff-only`；未修改接收钩子、未覆盖或删除历史产物，启动前已验证服务器HEAD与上述提交一致。
