# Task

只读检查候选边CD最高对象 `00335203` 和平均互反CD正增量最高对象 `00326127`，定位缺失面对和两条有向曲线，解释整体CD与互反CD可能不一致的实际案例。

# Assumptions

posterior mean 保留 GT face 顺序；曲线有16个点，面是16×16网格。检查采样表示而非原生CAD精确曲面；不建立结构因果归因。

# Inputs

两方法 posterior raw NPZ、DeepCAD v7 对应GT NPZ、已冻结 per_object.csv。目标只限上述两个ID。

# Outputs

`outlier_pair_probe.json`、`OUTLIER_PAIR_PROBE.md`，包含来源SHA、具体pair/edge身份、采样距离与有限结论。

# Processing Protocol

使用原squared_chamfer逐pair核算两件的互反mean/max，与已发布CSV匹配；各方法最差互反pair及两方法缺失GT pair的并集为诊断pair，不新增大范围评估。比较两条curve到自身GT pair代表曲线的CD及到整个GT边点集的单向均方距离；通过原closest_surface_points计算两owner预测面的逐点最近三角片距离。面网格各450个三角片，所有16个曲线采样点参加。

# Metrics

原双向平方CD；曲线到整个GT边集合的单向平均平方距离；所属面三角网格距离的均值/最大值；缺失／额外GT pair，所查曲线到自身GT pair的误差。

# Decision Rule

仅描述实际曲线落点和标量差异；不把最近另一GT曲线视为模型内部重新分配，不推断有序concat导致。不改主指标和失败标签。

# Edge Cases

选中pair一方未预测时记录missing；GT无此pair时own-pair CD为null；非有限、非法索引、CSV复核不一致或来源变化拒绝输出。

# Validation / Sanity Checks

逐对象mean/max与原CSV近似1e-10相对精度核对；input_latent hash一致；两owner面均报告。输出前复核来源SHA。

# Implementation Constraints

CPU单线程、只读两件、原几何helper、数分钟内完成；新增自身诊断文件，不改模型/post/冻结工具或旧产物。

# Deliverables

紧凑案例JSON与中文解释；不要求新增绘图，数字与pair身份足以回答时停止。

本轮固定选择：00335203为候选边CD最高对象，00326127为平均互反CD正增量最高对象。检查双方最差互反pair以及两方法缺失的全部GT pair，不只检查现存曲线。只读原始预测，统计与test选择不变。
