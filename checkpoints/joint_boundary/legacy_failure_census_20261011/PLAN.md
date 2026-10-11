# 全部原版 VAE 失败的阶段溯源统计

## 1. Task

对 comparison_20261011 中 legacy / posterior_mean / optimized 的全部 369 个原生无效 solid 做上游到下游的溯源，回答哪些机制反复出现、边附近多条曲线从哪里来，以及几何优化是否消除或引入异常。只新增数值诊断、统计图表和可复核记录，复用旧 50 件可视化。

已知事实：2417 个 accepted identity，2048 valid、369 failed；旧 50 件是分层子集。用户要求统计全部失败，不以 50 件估计发生率。原输出只读，不修改模型、训练或后处理决策。

## 2. Assumptions

- 数据：posterior mean 输入保留 GT 面身份；原推理 v_deduplicated=False。面 latent 去重没有执行，不能记为此批错误来源；仍检查解码面的几何重合。双向半边是设计内的冗余，不能直接当重复错误。
- 范围：研究的是原版 VAE 重建链，不是照片 diffusion 的生成失败。原 test 中 7 个合同不支持对象单列，不并入这 369 个已运行失败。
- 实现：使用已有 Python/OCC 环境、CPU 单线程隔离进程；重放与原 seed、优化预算、容差一致。统计图使用标准科学绘图库。
- 待确认的研究推断：一个上游异常是否真正导致该件最终失败通常不能仅靠相关性确定；保留机制证据等级与 unresolved 类，不要求每件强行唯一归因。无执行所必需的待用户确认项。

## 3. Inputs

- comparison_20261011 的 raw_metrics/posterior_mean/per_object.csv、post/{optimized,fit_only}/samples.csv、raw NPZ、原 STEP 和逐件运行记录。
- /mnt/d/data/deepcad_v7 的 GT NPZ/STEP。
- 旧 legacy_failures_20261011 的冻结 50 件和既有诊断，作为核验参照。
- 当前源码 Git 提交、诊断脚本 hash、原推理/后处理协议。

## 4. Outputs

- manifest.json、failed_ids.txt：完整固定失败清单、来源指纹、逐件已有指标与路径。
- traces/、replay/、upstream/：可再生逐对象溯源和阶段几何，保持原输出不变。
- cases.json / cases.csv：每件阶段状态、多标签机制、证据与未解析项。
- summary.json、charts/、统计页面、REPORT.md、README.md：固定分母频次、交叉关系与复现命令。

## 5. Processing Protocol

1. 对三份表执行 ID 唯一性和全集一致性检查，以 optimized.native_valid_solid=False 冻结 369 件。
2. 审计 raw 合同：面数/身份、无向与有向 pair、缺失/额外连接、reciprocal 曲线差异、跨 pair 近重合曲线和近重合面；用 GT 相近关系排除正常薄壁/本来重合边界。
3. 同参数重放后处理，记录半边选择/拒绝的来源 ID、选择后的集合与几何、优化前后几何和 loss、拟合/周期处理、逐面逐容差 wire、wire 修复/跳过、face 修复/拒绝和 sewing。
4. 尽量以非侵入函数包装或仅针对目标函数的 Python tracing 观察。诊断异常隔离；不能改变量、额外消耗随机数或改变算法分支。新增 OCC 测量尽量在输出确定后或独立几何上执行。
5. 对重放与原结果比较有效性/类型/实体数及 STEP DATA 段指纹；不一致单列，不能冒充原失败的精确轨迹。
6. 跨 raw/selected/optimized/fitted 阶段比较残差，使用现成 fit-only 对照定位优化路径退化。只报告现有预算下的结果，不额外修复筛选最好一次。
7. 汇总对象频次、面/边实体频次、阶段交叉和阈值敏感性；所有图明确分母及标签可重叠。

## 6. Metrics

- 上游离散结构：面数偏差、self pair、重复有向 pair、额外/缺失无向 GT pair、选择移除 GT pair、unsupported multiplicity；正常 reciprocal 两份单独计数。
- 上游几何：reciprocal 和跨 pair 曲线方向不敏感的离散距离；近重合面与 GT 对照；全部距离使用归一化 XYZ 长度，不能混用平方 CD。
- 阶段几何：每面边端点最近邻距离是闭合的必要条件代理，不称为实际闭环；边面贴合残差、GT 几何误差、优化前后改善/恶化、拟合残差。
- 阈值：优先报告连续分布，并在 0.002、0.006、0.01、0.08（已有连接/修复容差）报告敏感性；近重合与 GT 基线条件在实现注释和输出 metadata 中明确。阈值标签不是经验证的实体失败判据。
- 原 STEP 空间近重合：只比较不同面上的不同拓扑 free edge，排除 seam/退化边；以 128 点折线的双向 p95 距离为筛查，并同时保留最大距离。此标签表示未共享拓扑边但空间接近，不单独断言它们应被焊合；与 raw 正常 reciprocal 和选择后跨 pair 接近分开。
- 下游机制：无 wire、所有 wire 未闭合、修复后仍未闭合而跳过、部分线环被丢但面返回、闭合候选仍裁剪失败、边/面 context 错误、周期相关状态、trim gate、sewing free/multiple edges、无 free edge 仍无法构造 solid、保存/重读有效性变化。
- 对照：fit-only 有效而 optimized 失败称“优化路径退化”，不据此断言某个梯度项是根因。全 2417 对照只使用已有指标，不重新扩大重放范围。

## 7. Decision Rule

- 不制作声称每件唯一根因的饼图。主图分为“直接观察到的阻断/丢弃机制”和“更早已出现的结构/几何异常”。
- 可给出互斥的最终阻断阶段，但不能将其称为根因。机制多标签允许同件计入多个条目，总和可以大于 369。
- 优先级基于全量对象发生频次、机制可定位性和与下游失败的相接证据；不只按最终 BRepCheck 错误码排序。
- 对无法从记录进一步区分的曲面参数域、方向、曲线拟合等情况显式记 unresolved，不根据截图猜测。

## 8. Edge Cases

缺文件、解析失败、异常形状、NaN/Inf、timeout、crash 均保留 ID 和原失败分母；该诊断字段标 unknown，不填零。空边/无环/全闭合单边、seam、退化边、不同 pair 对应几何相近的正常薄壁分别处理。GT 对应依赖本次固定身份合同，不能套用到最终 STEP 的重排面序号。

## 9. Validation / Sanity Checks

先验证少量不同路径的真实对象（含 00839921、00009683、00480996），核验输出不变、内部 trace 有效且非空，再跑全部 369。为计数逻辑、正常 reciprocal、真实多余曲线、GT 薄壁排除、端点代理、丢弃环分类写必要的合成测试。最终审查 369 ID 全覆盖、分类未知分母、图表与 JSON 数值一致，源资产 hash 不变。

## 10. Implementation Constraints

所有新记录放本目录，不写 experiments/。保持旧 50 件 atlas 和生产模型/post 代码不变。独立诊断工具与测试受 Git 管理；运行原始几何/逐面长记录忽略，紧凑摘要和说明提交当前 experiment branch。使用固定 seed 和 worker 上限，单对象硬超时；重试仅为修复诊断实现且记录不同版本，不能改变研究预算。

## 11. Deliverables

全部失败的可审计分类清单、若干汇总图、只含统计的本地页面、中文报告与复现命令、代码和结果 Git 提交。明确哪些结论已定位到机制，哪些仍是上游异常线索。
