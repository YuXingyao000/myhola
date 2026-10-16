# 最终曲线逐点反馈：完整评估

固定2417件accepted test、同一float32 latent、同一生产后处理和预算；直接比较 `conditioned` → `point_feedback`。协议见 [PLAN](PLAN.md)，权重与训练来源见 [TRAINING_AUDIT](TRAINING_AUDIT.md)。

唯一测试权重由验证集选择：epoch34／step12705，val/loss=0.03722964599728584，SHA256 `1a61c40de044f59131aac8e22faca3a434855c60d7e0e974566209b0ad8129aa`。原encoder/posterior的243个张量未改变；旧decoder和新增反馈一起训练。本轮不是等预算架构消融。

## 结果入口

- [评估报告](REPORT.md)：原始几何、两种后处理、有效性和固定历史diffusion latent重放。
- [三版历史参考](HISTORICAL_REFERENCE.md)：原refinement、conditioning、本轮反馈的可比指标。
- [原始误差分布](diagnostics/RAW_DISTRIBUTION.md)、[互反半边分布](diagnostics/RECIPROCAL.md)、[两个极端对象核验](diagnostics/OUTLIER_PAIR_PROBE.md)、[末端反馈前后](diagnostics/FEEDBACK_BRANCH_PROBE.md)。
- [全集失败分析](../failure_census_20261015/REPORT.md)、[离线四图与逐ID详情](../failure_census_20261015/page/standalone.html)：全部完成；普通与离线页面均已通过桌面、390px布局和四组四阈值交互检查。
- [独立复核](independent_review.json)：主评估与全集诊断通过，pending为空；363对案例的81,312项标签判定从原始轨迹独立复算一致，19件路径对照另2,128项一致，5,665项原来源资产未变化。
- [末端探针追加复核](validation/feedback_branch_review.json)：两件全部候选位移、指定面对GT曲线距离、互反差距和16组owner面距离均从保存数组独立重算通过。

## 可复现入口

运行于 `/mnt/d/python_curve_eval`，Python为 `/mnt/d/anaconda3/envs/torch/bin/python`。`run_evaluation.py` 的阶段为 `smoke`、`inference_posterior_mean`、`inference_archived_latents`、`metrics_posterior_mean`、`metrics_archived_latents`、`post`。脚本明确指定checkpoint、split、来源和新method；重复执行已有阶段会拒绝覆盖。新实验应使用新的输出目录，不能用旧结果混跑。

原始指标完成后执行 `diagnostics/raw_distribution_analysis.py` 与 `diagnostics/reciprocal_distribution_analysis.py`；它们通过薄包装复用上轮统计函数，显式设置本轮方法和路径。独立来源见 `diagnostics/reuse_provenance.json`，历史源文件未修改。

全部后处理终态完成后，使用 `python -m tools.local_evaluate_report --comparison-root checkpoints/intersection_refinement/point_feedback/evaluation_20261015 --baseline-method conditioned --candidate-method point_feedback --failure-census-link ../failure_census_20261015/REPORT.md` 生成中文比较图及报告。历史三级参考入口为 `build_historical_reference.py`。它们均拒绝覆盖已完成产物。

实际阶段命令、日志、退出码在 `metadata/orchestration/`；输入与算法冻结在 `metadata/source_freeze.json` 和各阶段protocol，复核见 `validation/method_contract_review.json`。新预测、STEP、日志、临时轨迹留在本地并由ignore排除，紧凑科学记录保留版本管理。
