# Encoder 边内容消息续训：完整失败分析

两份在查看 test 前固定的权重均完成2,417件测试及完整失败诊断。最后权重为28,677步；补充权重为本次续训窗口validation最佳27,588步，后者不是全程最佳。两者优化后均有2,115个合法solid（87.51%），原始几何有改善，实体有效性仍未超过早期1,089步的2,118件。详细数值及训练终止审计见[主评估](../evaluation_20261016/REPORT.md)。

| 权重 | 失败 | 对基线救回 / 退步 | 双侧诊断ID | 仅拟合合法、优化后失败 |
|---|---:|---:|---:|---:|
| 最后28,677步 | 302 | 53 / 49 | 355 | 10 |
| 续训窗口最佳27,588步 | 302 | 51 / 47 | 353 | 13 |

每一失败及救回ID都与原conditioning基线成对诊断。原数据、checkpoint、split、posterior来源及后处理协议均核验；两份模型各用自己的Encoder posterior mean，不使用同一latent缓存。原始测试集2,424件中的7件合同不支持另列。这是CAD输入的VAE重建，不代表照片或diffusion效果。

## 报告与离线展示

| 交付 | 最后28,677步 | 续训窗口最佳27,588步 |
|---|---|---|
| 中文统计报告 | [REPORT](last/REPORT.md) | [REPORT](resume_best/REPORT.md) |
| 研究判断 | [NOTE](last/NOTE.md) | [NOTE](resume_best/NOTE.md) |
| 自包含离线页面 | [standalone.html](last/page/standalone.html) | [standalone.html](resume_best/page/standalone.html) |
| 普通页面 | [index.html](last/page/index.html) | [index.html](resume_best/page/index.html) |
| 高频失败观察图 | [PNG](last/visualizations/main_failure_observations.png) | [PNG](resume_best/visualizations/main_failure_observations.png) |
| 同ID成对变化图 | [PNG](last/visualizations/paired_observation_changes.png) | [PNG](resume_best/visualizations/paired_observation_changes.png) |
| 拟合／周期替换贴面图 | [PNG](last/visualizations/owner_residual_before_after.png) | [PNG](resume_best/visualizations/owner_residual_before_after.png) |
| 最终失败面检查图 | [PNG](last/visualizations/persistent_face_context_errors.png) | [PNG](resume_best/visualizations/persistent_face_context_errors.png) |
| 独立数值复核 | [JSON](last/independent_review.json) | [JSON](resume_best/independent_review.json) |
| 完成交付核验 | [JSON](last/completion.json) | [JSON](resume_best/completion.json) |

离线页内嵌四张图、全部双侧逐件统计及报告下载，无需运行HTTP服务。大型逐件payload和离线HTML仅保留本地，不提交Git；公开展示代码、紧凑统计、PNG和核验记录保留版本管理。

## 可以支持的判断

最后权重302件失败中217件面邻接全对；周期替换后289件边面距离超0.006，但53件救回中也有45件超阈值。因此高频贴合告警不能单独解释成败。177件曾出现全部候选环闭合并接纳但面仍失败，118件构面时丢弃部分边界；同ID救回／退步常伴随这两类观察的消失／出现。几何的平均精度改善尚未稳定解决边界与所属曲面的联合可裁剪性。

两份新权重均有完整失败分析，不能直接比较不同失败集的百分比判断机制变化。[时间对照报告](temporal_REPORT.md)从全集重新划分旧1,089、14,520与新权重的成败；阶段观察只在共同核验的ID上比较，旧轨迹缺失明确计未知。详细四阈值转换及来源摘要见[时间对照JSON](temporal_comparison.json)。

## 验证与复现

独立脚本核验1,416条双侧轨迹，重算158,592项主观察和2,576项fit路径判断、配对转换及最终失败面统计；全部通过。两组普通／离线页面分别遍历4分组×4阈值，检查详情、四图、手机宽度及脚本错误；离线检查阻止HTTP/HTTPS，无外部请求。历史科学源码与既有审计SHA一致，本轮没有改生产模型或后处理。

入口为`run_census.py --group last|resume_best`，之后`review_census.py --root <group>`、`build_standalone.py --root <group>`、`check_standalone.cjs`和`compare_temporal.py`。续跑需遵守已冻结[PLAN](PLAN.md)和各组`metadata/source_freeze.json`。本轮没有再次启动训练。
