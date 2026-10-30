# 单视图 diffusion 完整评估与失败分析

## 研究问题

DPO提高合法率时，是否同时保留输入图像约束？失败主要停在裁剪、构造实体还是最终检查，
这些失败是否伴随raw几何内部不一致？完整GT误差与可见区域误差分开，避免把单图背面的多解
简单等同于生成错误。

## 完成结果

完整对照与7,010件失败分析均已完成。首先阅读[结果与分析](RESULTS.md)，精确汇总见[RESULTS.json](RESULTS.json)。

| 指标 | 原SVR | DPO 5,000步 | 比较集合 |
|---|---:|---:|---|
| 合法率 | 77.72% | 88.25% | 各3,505件 |
| 未完成裁剪 | 758 | 400 | 各3,505件 |
| 输入轮廓IoU | 90.08% | 91.00% | 共同可测3,226件 |
| 可见表面平方Chamfer | 0.008984 | 0.007228 | 共同可测3,226件 |

合法率及上述可见指标的配对95%区间支持改善；原完整raw/post Chamfer的配对区间仍跨0。
面数减少与收益同时出现，尚未证明减少的是冗余面还是缺失结构；DPO同图多样性尚未测量。

## 实施

- 协议见[PLAN.md](PLAN.md)，工具说明见[单视图评估文档](../../../../src/brepnet/eval/SINGLE_VIEW_EVALUATION.md)。
- 原SVR和DPO 5,000步的完整3505件对照位于`checkpoints/diffusion/preference_eval/20261023T090834Z`。
  两组生成、原指标与新增失败分析均完成。
  新分析输出为`checkpoints/diffusion/failure_analysis/runs/20261023T105637Z_full`，16个CPU worker。
- 新工具已经实现，41项行为测试通过；6件真实输出已走通整个分析／汇总流程。
  5件全部可测，1件invalid compound有一个面无法三角化，明确保留为partial。
- 上述实现检查与完整集研究结论分开保存，完整结果见RESULTS.md。
- 原模型100图×8候选附录已完成，见[DIVERSITY_VALIDATION.json](DIVERSITY_VALIDATION.json)。
  没有对应DPO K=8，不能比较两者的同图多样性。

## 记录

- [VALIDATION.json](VALIDATION.json)：集成验证、源文件hash与测试。
- [OBSERVATION_VALIDATION.json](OBSERVATION_VALIDATION.json)：相机、投影、自比较、GT轮廓及缓存验证。
- [DIVERSITY_VALIDATION.json](DIVERSITY_VALIDATION.json)：固定100图、800候选和多样性工具验证。
- [LAUNCH.json](LAUNCH.json)、[run.sh](run.sh)：实际命令、PID、日志及输出。
- [FULL_COMPARISON_CONTEXT.md](FULL_COMPARISON_CONTEXT.md)：原指标、面数、成功集合分解及指标计算错误。
- [TRAINING_CONTEXT.md](TRAINING_CONTEXT.md)：实际偏好标签与训练目标。

原始运行只读。新分析的运行产物位于`checkpoints/diffusion/failure_analysis/runs/`，不提交大体量数据。
