# Task

独立读取已完成的 raw 指标，描述本轮逐点几何反馈与上一版曲线条件调制 的逐件几何变化、分布尾部及两方邻接均正确时的几何表现。原始 2417 件主结果保持不变。

# Assumptions

已知上游按固定 accepted split、相同 latent 执行；本分析仍重新核验全部 ID、输入 hash 和 summary。候选名称为 point_feedback，baseline 为 conditioned。训练曝光不同，描述性分组不构成架构或因果消融。无待用户确认项。

# Inputs

`../raw_metrics/{posterior_mean,archived_latents}/{per_object.csv,summary.json}`；受版本管理的 accepted.txt，固定 SHA256 `2864849e8240134c6033cc6cd64efa19337f67fea94b89a493690ae84355b7b3`，2417 件。

# Outputs

`raw_distribution.json`、`raw_distribution_per_object.csv`、`raw_distribution_extremes.csv`、`adjacency_subset.json` 和中文 `RAW_DISTRIBUTION.md`。JSON 记录输入 CSV/summary/split 与代码/spec SHA、独立统计核验。

# Processing Protocol

每模式恰好两个方法各 2417 个唯一 ID；CSV 与冻结 split 完全相等，逐 ID latent hash 相等且 shared_input_status=verified。每指标先重算各自可计算集合，再重算双方共同可计算集合，分别与上游对应 summary 的 count、mean、median、P90 核对，不能混淆两种集合。读取实际 schema 后确认上游不含 P95，该项由本诊断独立补算，不伪称已有上游对照。

按候选误差降序、ID 升序固定取最大 1%/5%/10%（ceil），从双方同时排除相同 ID，仅作尾部敏感性；另记录正退化增量最大的 1% 对全部正增量的份额。按 GT 面数与预测面数预定分组；缺失数单列。

posterior_mean 的邻接子集要求两方 topology_status=ok 且 FP=FN=0。任一方不可计算时归 unknown，不归错连；可计算时分为双方邻接全对与至少一方不全对，三个集合覆盖全部 2417 件。

# Metrics

面 XYZ MAE、面/边双向平方 CD、边的两个方向平方 CD：均值、中位数、P90/P95/P99/max；同 ID 改善/退步/持平，容差 1e-12，全部几何指标越低越好。记录均值比、逐件比值、配对差值和尾部贡献。复杂度仅作 Spearman 关联与预定面数组描述。归档 latent 无 GT 面对应，不计算对应 MAE/GT 邻接正确性。

# Decision Rule

先看全体共同可计算均值、分位数与改善比例，再参考去尾及邻接子集。不根据结果改变 checkpoint、样本或主分母；自动报告仅打印事实和边界，不预填“尾部驱动”“明显改善”“拓扑无关”等结论。

# Edge Cases

合法空字段记缺测并报告数量；零分母倍数/份额为 null；空集合统计为 null；全常数相关为 null。重复/缺失 ID、来源 hash 冲突、非法负值、NaN/Inf 或 CSV/summary 不一致时拒绝输出，不能静默取成功子集。合法缺测仍在分组分母中。

# Validation / Sanity Checks

核对各自与配对统计、win/loss/tie 守恒、复杂度组与邻接组覆盖；JSON 禁止 NaN。独立重算使用标准库 math.fsum/statistics 对照上游 NumPy 均值/中位数，分位数使用显式线性插值。输入哈希在所有计算后再次核对，避免读取中数据变化。输出禁止覆盖。

# Implementation Constraints

只读 CSV/JSON 与固定 split；单进程 NumPy/SciPy 确定性统计，不重跑模型、raw 几何或 post，不改正在使用的 runner。复用此前诊断的统计定义，将分布、邻接子集和无预设结论的报告合并于本目录一个脚本。

# Deliverables

完整描述性统计、来源与独立核验、中文解释及运行命令。运行命令：从仓库根目录执行 `python checkpoints/intersection_refinement/point_feedback/evaluation_20261015/diagnostics/raw_distribution_analysis.py`；只在两种 raw 指标全部完成后执行。

本轮复用历史统计函数：薄包装仅显式设置新root、methods、labels，数学计算保持原实现。包装及历史实现SHA另存reuse_provenance.json；不修改历史源文件。
