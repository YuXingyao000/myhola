# 64维全解冻：关闭边特征匹配（2026-10-23）

## Task
检验全解冻时 `edge_feature` 特征匹配是否推动边几何退步。用户已授权在142启动一组64维实验；唯一训练目标改动为 `model.loss_weights.feature_weight: 1.0 -> 0.0`，预算30,000步。

## Assumptions
既有32/64同期曲线都出现teacher位置/尺度退步；64后期feature loss下降而边几何恶化。它们支持这一消融，尚未证明因果。本轮保留teacher几何重建、KL、随机posterior、dropout及全部解冻，不删除teacher分支，也不修改网络结构。

## Inputs
- 服务器：142，`/mnt/d/python`，既有PyTorch 2.8/cu128环境。
- 代码基准：`97355cf1371b1a1b684c9409b9debeb8ee6f3e20`；新增文件仅为独立配置及本实验记录/启动检查。
- 历史64维对照：`latent_expansion/d64/20261020T121050.873811Z`，W&B `f45ab7aa6bab`。
- 同源32维checkpoint：`shared_curve_uv/candidate/20261019T064906.049992Z/weights/epoch=0116-step=00042471-val_loss=0.036517.ckpt`；SHA256 `ce749416c1635dda1563161b971bd4260d064cfdb0c31f1fa9d06ad0e70961ff`。使用原扩维迁移初始化新run和新Adam，不从退步的64维末权重续训。
- DeepCAD v7、UV v1；既定train 46,480 / validation 3,495。train SHA256 `14a00585500f1c7996411caa1b3fba796a1c39c2b935a1153ec98e30c69dc097`，validation SHA256 `ae607bfe5499e919d1e68a5110c0c211daf1158aedbe57a4756eda99f6f3f0ed`。不运行test。

## Outputs
独立 `checkpoints/latent_expansion/d64_no_edge_feature/<UTC标识>/`，沿用weights/logs/wandb/metadata结构。在线W&B、完整resolved config、Git来源、启动检查和实际启动记录。训练上限30k，不自动延长或追加实验。

## Processing Protocol
1. 独立入口 `train_vae_latent_expansion_no_edge_feature` 继承原64维配置。模型计算图保留raw `edge_feature`供观察，加权项恒为0，不参与参数更新。
2. 与旧64保持8 GPU × batch16、Adam LR2e-5、BF16、seed42、is_aug=1、KL sum×5e-7、全部VAE参数可训练、原数据和其他loss权重。只允许目标权重、预算和run标识的配置差异。
3. 启动前验证实际配置差异、来源权重/名单摘要和真实16件训练batch的前后向；检查取消项加权值及梯度均为0，其余几何/teacher/KL仍保留。不执行预检optimizer step。
4. 在独立tmux会话启动一次；确认8卡DDP、在线W&B、真实训练步推进和有限loss。保留已有工作、已有run与其他进程。
5. 保留现有top3+last保存行为。最终评估以实际保存的固定预算终点为主，先读global_step；不以两个实验各自总loss最优权重比较。

## Metrics
同实际步数对齐train/随机posterior validation的teacher local/bbox、coarse/refined local/bbox、最终face/edge XYZ、incidence、UV/anchor、分类与KL；raw feature仅为诊断，weighted feature应恒为0。后续固定validation评估各自Encoder的posterior mean，报告raw face/edge及对应曲线的均值、中位数、P90、逐CAD改善/退步。原始几何与OCC有效性分开；本次启动不宣称下游改善。

## Decision Rule
主问题：移除feature匹配后，25–30k阶段的位置/尺度及最终边是否仍重复恶化。若最终边与位置/尺度改善且面/拓扑没有明显代价，支持该目标对本协议有负面作用；仅teacher改善不算最终任务解决。若趋势仍恶化，则feature匹配不足以单独解释问题，保留共享读出等候选机制，不继续围绕其权重扫参。单seed结果不外推普遍结论。

## Edge Cases
旧64未保留30k权重，同步数比较限既有曲线；不得把其约50k的mean几何冒充30k对照。失败、NaN/Inf、缺数据和中断保留记录，不跳过CAD或自动重启。预算终点可能非完整epoch，核对实际checkpoint步数。移除一项loss后总loss天然减小，不能直接用于跨实验选优。已有目录拒绝覆盖。

## Validation / Sanity Checks
检查配置实际消费字段、训练源码与原64的一致性、严格扩维加载、全参数训练范围、真实batch有限loss与encoder/posterior/decoder梯度。用同一次前向的其他weighted项重算total，并检查被关闭项对coarse特征的梯度为零。预检只验证实现，不是研究结论。

## Implementation Constraints
不修改模型、loss、trainer、dataset或后处理源码。新增独立配置，不覆盖原64入口。只同步本轮新增文件，保留两端未提交工作。沿用完整架构快照与本目录的loss变更说明；不新增监控automation、后台追加实验或跨聊天消息。

## Deliverables
本PLAN、独立配置、ARCHITECTURE补充、launch.sh、preflight.py及报告、resolved config、README与launch_record.json。完成后另记录训练终点与评估结果。
