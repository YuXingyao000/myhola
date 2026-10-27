# Task

在当前64维UV VAE的30,000步权重上继续全量训练，增加当前参数域的直接UV定位、实际三维轨迹走向、按GT边尺度加权的相对精度监督。主干、面表示、输出头及参数形状不变。用户已授权实现和启动142训练；本阶段不执行OCC评估或test开发。

# Assumptions

- 数据：沿用DeepCAD v7已接受的train/validation列表；GT每条无向边由原双向记录选定代表，保持16点顺序。新UV必须针对当前非周期三次固定样条反求，不能复用旧OCC UV。
- 范围：仅修改曲线目标及标签链路；不增加refiner、不冻结Encoder、不改变固定曲面重建、不加入面法向分支。沿用两侧共同正/反方向匹配，本阶段不扩展闭曲线循环相位匹配。
- 实现：离线生成GT UV，训练只读缓存。UV mask只影响UV项，所有CAD及边仍参与XYZ/拓扑等原目标和报告分母。
- 续训：保留上一轮所有模型参数；优先保留optimizer及step，在明确记录的新监督阶段追加30,000更新。续训目录/状态方式须核对现有run管理后记录，不能把只加载权重称为完整resume。

# Inputs

- `/mnt/d/python/checkpoints/uv_intersection/joint64/20261025T144428Z/weights/last.ckpt`，step30000，SHA256 `c4f409b1a904d431cf82cf4754001ca590b59dc44abbaf987181819977a32318`。
- `checkpoints/joint_boundary/validation/train_contract_audit/accepted.txt`、`val_contract_audit/accepted.txt`；raw root通过配置/CLI传入。test不参与本阶段生成规则调整和训练验收。
- 每CAD的`sample_points_faces[F,16,16,6]`、`sample_points_lines[H,16,6]`、`edge_face_connectivity[H,3]`。
- 保留64D、8×512曲线attention、原Encoder/面decoder；既有训练为7卡、每卡batch2、累积8、Adam lr2e-5、seed42。

# Outputs

- 新主题的代码、PLAN/README/ARCHITECTURE、配置、验证及启动记录。
- 每CAD独立UV sidecar：`face_pairs[R,2]`、`source_edge_ids[R]`、`uv[R,2,16,2]`、`valid[R,2,16]`、`residual[R,2,16]`，以及源数据SHA、representation/schema、固定曲面规则和投影参数元数据。可附失败/退化/连续性诊断字段。
- train/validation逐件生成状态、UV有效点/有效边覆盖和残差分布；缓存不得覆盖源CAD或旧UV。
- 新训练阶段raw及weighted loss日志、源checkpoint/代码/配置溯源、权重及运行状态。

# Processing Protocol

1. 只使用GT面XYZ按当前FixedBSplineSurface构造连续曲面，将同一GT曲线的16点反求到两个owner的`[0,1]^2`参数域。固定粗网格多起点、确定性有界迭代、沿整条路径选择连续分支；不投到当前模型预测面，不读取训练权重生成标签。
2. UV标签须同时满足投影残差和参数有效性/连续性判断；细化投影数值设置仅用固定前32件validation做实现验收，随后冻结生成清单和参数，完整报告覆盖。具体设置写入生成manifest，不静默fallback。
3. UV缓存与CAD源摘要、owner/代表边和当前样条定义绑定。刚性旋转只改变XYZ及方向，不改变UV标签。缺失或不匹配缓存停止训练，不动态退回无UV训练。
4. 新绝对几何项使用双侧对GT的逐点L1；相对项是该每边L1除以GT尺度下限。GT尺度为16点到均值中心的最大欧氏距离，不能使用预测尺度。
5. 走向项比较预测与GT相邻采样点差分的单位方向；GT长度>1e-6的线段有效，预测归一化eps=1e-6，保留预测坍缩惩罚和有限梯度。不直接复用旧数据后3通道的有歧义符号。正反方向选择必须对XYZ、UV、走向共用；incidence比较两侧相同下标。
6. 本阶段初始固定系数：三个原面L1各1；curve XYZ L1=1；UV L1=0.1；走向余弦误差=0.01；relative XYZ L1=0.05，GT尺度下限0.01（相对附加权重至多5）；incidence MSE=0.1；balanced existence BCE=0.1；KL sum系数5e-7。它们是可审计的初始研究选择，不宣称最优。启动前向用户明确列出；如实现验收发现数值问题，先记录理由及修订，不能静默改权重。
7. 保留原curve XYZ MSE为诊断日志，面法向另列raw诊断但不重复加入总loss。每个实际目标同时记录raw/weighted值；KL避免重复加权。

# Metrics

- 标签：计划CAD数、成功/失败/缺失数、有效点及owner轨迹覆盖，反求残差分位数，退化/多解/跳变数量，生成耗时。
- 实现：checkpoint逐tensor保留、shape、loss有限、梯度有限且Encoder/UVHead有梯度、raw×weight总和等于总目标、正反对齐、mask和零边行为。
- 启动：参数总数/可训练数、实际恢复step/optimizer、设备与吞吐、训练及validation日志、UV有效覆盖非零。
- 研究效果留待固定validation完整比较；小批量通过只证明实现可用，不宣称改善。

# Decision Rule

通过合成几何和真实小批量验收、完整缓存清单无失败/缺失且UV可靠覆盖可解释、严格加载/恢复成立后启动正式全量训练。若投影标签整体无法可靠生成，先解决实现或明确不可表示问题，不以错误UV标签强行开训。

# Edge Cases

空边保留空形状和可反传零loss；原CAD损坏/缺失、非有限数、源摘要或owner不匹配明确失败并保留记录。投影不可达、Jacobian退化、参数分支不可信只屏蔽UV定位并记录原因，不删除边/CAD。极短边用显式尺度下限；零长GT差分不计算方向项；预测零差分不作为跳过理由。

# Validation / Sanity Checks

平面和已知参数曲面投影/回代、边界点、正反轨迹、闭合路径、退化轨迹、不可达点与旋转不变性；同参数SciPy/当前Torch曲面求值核对。真实前32件validation标签报告、2件及DDP小批量前后向、旧权重严格加载/参数保留、梯度与日志重构、续训状态验收。

# Implementation Constraints

保留已有未提交图件；网络及loss采用小module、清晰forward，现有UV网络文件继续可独立审阅。不在网络层反复堆shape验证；数据缓存边界做来源验证。离线生成允许按固定ID分片并行，不运行test或更改split。机器路径显式传入，产物与旧run分开保存，GPU0其他任务不动。

# Deliverables

可复查代码和配置、标签生成/覆盖报告、真实小批量验证、完整权重延续说明、实际loss系数表、142正式训练启动及首批健康状态。训练完成与效果提升只在有实际结果时报告。
