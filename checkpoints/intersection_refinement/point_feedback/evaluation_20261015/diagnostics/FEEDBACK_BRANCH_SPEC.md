# Task

定位两个已固定案例的异常发生在已训练网络的最终逐点反馈之前还是之后；仅做本次推理的末端分支前后比较，不是移除模块重新训练的因果消融。

# Assumptions

模型参数、唯一验证最佳 checkpoint 与输入 latent 均不变。GT 只用于推理后的距离测量，不进入 forward、存在性筛选或分支开关。

# Inputs

固定 `00335203` 的面对 `(0,6)` 和 `00326127` 的面对 `(3,11)`，来源见 `OUTLIER_PAIR_PROBE.md/json`。使用正式 posterior_mean 协议记录的唯一 best checkpoint、原归档 float32 latent、该两件原始 raw NPZ 和 GT data.npz；不扩大样本或更换权重。

# Outputs

`feedback_branch_probe.py`、`feedback_branch_probe.json`、`FEEDBACK_BRANCH_PROBE.md`；两件反馈前后全部候选曲线和必要几何存入被 Git 忽略的 `feedback_branch_arrays/*.npz`。记录来源、脚本、模型及输出 SHA。

# Processing Protocol

沿用正式 FP32、TF32 off、seed42、eval 模式，严格加载 checkpoint。每件完整解码一次，以 hook 捕获末端逐点反馈模块的共同输入和输出；用共同输入的 edge_geometry 表示未应用末端反馈的几何，不重新计算上游 prefix，不改权重或文件。额外 hook 保存末端之前的 classifier 与 refiner 输出，检查其与完整输出相同；检查模块输入张量未被原地修改。记录全部候选和 threshold=.5 的 active 集合。GT 距离使用既有 FP64 CPU squared_chamfer 与 closest_surface_points；各16点曲线对16×16预测面网格（450三角片）。

协议修订：首次脚本尝试两次完整 CUDA 解码（启用/旁路），上游不变量检查发现 7/288 个值不同、最大差异 2.384185791015625e-7，脚本按原规则停止，没有生成诊断结果。正式配置的 deterministic_algorithms=False 保持不变；改为同一次 prefix 的输入/输出比较，避免把重复 CUDA 解码的微小数值差异当成末端模块效应。首次失败记录保存在 feedback_branch_initial_check.json，不放宽精确不变量条件。

# Metrics

逐点 XYZ delta 欧氏范数的 mean/median/P90/max（全部候选及 active 分开）；两条指定面对双方向的存在概率、candidate index、raw edge ID、前后自身GT边双向平方CD、到两个预测 owner 面的平均/最大距离、互反CD。保留全部候选的前后几何、中心尺度、特征与概率供审计。核验正式raw重放的逐字段最大误差与连通性一致。

# Decision Rule

以实际前后误差和位移量回答错误是否在模块输入中已存在，不凭单个误差变化推导训练因果。存在性/候选集合、预测面、bbox、face/edge feature、refiner delta 必须在同 prefix 的末端前后检查中逐值一致。正式raw重放记录实际误差，连接必须相同；浮点字段要求 max abs <=1e-5（仅CUDA重复执行舍入核验，不修改主指标或放宽几何精度）。

# Edge Cases

缺失输入、错误 checkpoint/SHA、非有限坐标、GT facing pair 不存在或多义、非精确分支不变量、raw重放超过预设误差均停止并报告，不换案例、不重复采样挑选成功。未active的pair仍可测量其候选几何，但明确它没有导出。空集合显示不可用。

# Validation / Sanity Checks

核验 latent 哈希与正式raw相同、checkpoint与正式protocol相同、hook前后共同输入没有原地修改、同 prefix 的不变量精确一致；前后曲线都保持同一个owner和GT对应。主指标、后处理、失败标签及已冻结科学runner保持只读。

# Implementation Constraints

只增加本诊断的脚本/spec/JSON/说明与两件数组；不修改模型源码、checkpoint、科学runner、主报告指标。单GPU、两件固定ID，一次末端分支比较；记录所有执行差异。

# Deliverables

两个固定案例的测量表、直接推理阶段结论与解释边界，及可复核来源和数组。
