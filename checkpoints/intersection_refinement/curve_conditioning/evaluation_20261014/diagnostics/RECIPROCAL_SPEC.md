# Task

使用既有逐对象互反半边 CD，区分全体分布变化与少量大误差的贡献；在双方 GT 邻接均正确的同 ID 集合复核。

# Assumptions

互反 mean 是每个 CAD 内无向面对的平均，max 是每个 CAD 内最大；本分析再对 CAD 统计。指标不检查方向。只读既有 scalar，不重算曲线几何。CSV 没有完整 pair 集合，不能把 pair 数相同当作集合相同；仅双方 GT 邻接全对集合支持相同无向面 pair 的核验。

# Inputs

两种 raw 模式的 per_object.csv、summary.json，固定 accepted split，以及已审计 adjacency_subset.json 中的 2306 件双方邻接全对清单。所有来源记录 SHA。

# Outputs

`reciprocal_distribution.json` 与 `RECIPROCAL.md`；保存整体、双方邻接全对、其余对象的统计、正增量贡献和极端 ID。

# Processing Protocol

复核两方法全部2417个ID、latent hash、原 summary count/mean/median/P90 与邻接 cohort 条件；两种模式均分析全体，posterior_mean 另给双方邻接全对与其余集合。按每件候选误差及正增量分别排序，ID为确定性次排序；同一组 tail ID 同时从双方移除。

# Metrics

legacy_reciprocal_cd_sq_mean/max 的均值、中位数、P90/P95/P99、max、同ID改善/退步/持平、差值和比值；候选最大1%对误差及正增量的份额，去尾后均值；正增量最大1%的份额、前10 ID。posterior全对/其余集合各自对全体正增量的份额。

# Decision Rule

并列报告均值、中位数、逐件改善与尾部份额，不预填广泛恶化或只由异常值导致；GT邻接相同不构成几何/结构因果隔离，不归因于有序条件或新增模块。

# Edge Cases

空字段缺测并保留分母；空集合统计null；零分母比值null；重复/缺ID、非有限、负值、来源变化或上游汇总不一致拒绝输出；未知拓扑不能算全对。

# Validation / Sanity Checks

复用已审计独立 math.fsum/median/线性分位数 helper，核对原summary；三个posterior集合数量守恒、逐件变化守恒。输出前再次核对来源SHA，JSON禁止NaN且不覆盖已有结果。

# Implementation Constraints

仅在本diagnostics目录新增脚本/记录；不改已有来源、模型、推理、post或统计工具。单进程只读CSV。

# Deliverables

紧凑JSON、中文短报告和明确不能证明的范围。运行 `python checkpoints/intersection_refinement/curve_conditioning/evaluation_20261014/diagnostics/reciprocal_distribution_analysis.py`。
