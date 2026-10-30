# 完整几何 loss 对照图

- Task：将已经完成的继续DPO／DPO＋几何项评估画成中文数值对照图。
- Assumptions：沿用已冻结评估协议，不重新测量或选择checkpoint；差值为加几何项减控制组。
- Inputs：`geometry_v1_evaluation/comparison/report/summary.json` 与 `observation/report/summary.json`，以及两阶段complete状态。
- Outputs：PNG、SVG、全部性能指标CSV及包含输入摘要的紧凑JSON；独立保存，不覆盖评估产物。
- Protocol：主图展示合法率、raw/post几何与可见一致性。均值和配对95%区间均读取现有报告。比例类差值显示百分点，距离与面数误差显示相对控制组均值的百分比变化；后者是原差值区间除以固定控制组均值，不是重新估计均值比的区间。数值用实际数量级。选择这些指标后固定，不按有利结果筛选。
- Edge Cases：失败留在全量3505分母；几何与观测均值只显示双方共同可测集合及数量。缺失不填零。GT自检常量不作为模型性能；单向可见距离另作诊断，注明区间未经多重比较校正。状态未complete、来源标签错误或缺失字段则停止出图。
