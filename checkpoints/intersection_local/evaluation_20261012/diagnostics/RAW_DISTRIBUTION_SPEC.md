# Task

独立读取两种 raw per_object.csv，判断均值退化是否主要由少量极端值驱动，并分析与实体数量的统计关联。补充诊断不替代冻结主指标。

# Assumptions

相同模式、ID 和 input_latent_sha256 可配对；模型身份 intersection/local。面/边数只作复杂度代理，不建立因果归因。每个模式预计2417件。

# Inputs

../raw_metrics/{posterior_mean,archived_latents}/per_object.csv 与 summary.json；不重新推理或计算底层几何，不改 post。

# Outputs

raw_distribution.json：分布、配对变化、尾部贡献、复杂度相关与分组、来源哈希；raw_distribution_per_object.csv：逐ID几何/复杂度/比值/变化；raw_distribution_extremes.csv：每模式每几何指标local最大的20个ID；RAW_DISTRIBUTION.md：中文解释；本spec。

# Processing Protocol

按方法和ID唯一性校验，验证固定2417件与逐IDlatent hash。每指标只对双方有限非负数使用共同集合；汇报缺失。按(metric descending, model_id ascending)固定极值顺序。top1%=ceil(.01×共同数)，按local指标选择并对同一组ID同时排除两方法。另给正增量top1%对正增量总量贡献；不使用GT重新选择预测。

# Metrics

面MAE/CD、边CD与方向CD的mean/median/p90/p95/p99/max；candidate-baseline差值与比值；方向规定几何越低越好，win容差1e-12。对local最大的1%、5%、10%记录总误差份额与相同ID baseline份额。Spearman rho及有效数使用GT face/boundary count、预测face/edge count；预定GT面数组1-5、6-10、11-15、16-20、21-30、31+。archived模式额外用预测face count分组，GT复杂度仅表示对应目标资产。

# Decision Rule

先看主均值与中位数、逐件win，再看尾部集中和固定排除敏感性，判断是广泛变化还是尾部驱动。不因去尾改善改写主结论；复杂度相关不代表该复杂度引发失败。

# Edge Cases

空值/非有限保留缺失计数；baseline=0时ratio为null，不伪造倍数。全常数相关为null。零分母份额为null。重复/缺ID、latent mismatch、非法负值拒绝输出。

# Validation / Sanity Checks

重算均值与既有summary核对；ID计数、配对和分组守恒；所有JSON禁止NaN。输出哈希并保留input来源，禁止覆盖既有诊断。

# Implementation Constraints

单进程numpy/scipy确定性统计。代码为本目录raw_distribution_analysis.py，严格只读原始输入；不改正在运行的工具源码、模型、post与历史资产。

# Deliverables

紧凑JSON、完整逐件CSV、极端IDCSV和中文解释；此分析与最终report的指标身份和分母保持一致。
