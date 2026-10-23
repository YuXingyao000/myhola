# 单视图 diffusion 完整评估与失败分析

## 研究问题

DPO提高合法率时，是否同时保留输入图像约束？失败主要停在裁剪、构造实体还是最终检查，
这些失败是否伴随raw几何内部不一致？完整GT误差与可见区域误差分开，避免把单图背面的多解
简单等同于生成错误。

## 实施与状态

- 协议见[PLAN.md](PLAN.md)，工具说明见[单视图评估文档](../../../../src/brepnet/eval/SINGLE_VIEW_EVALUATION.md)。
- 原SVR和DPO 5,000步的完整3505件对照位于`checkpoints/diffusion/preference_eval/20261023T090834Z`。
  两组生成均完成；新增分析将在原CPU后处理与已有指标完成后读取保存结果。
- 新工具已经实现，41项行为测试通过；6件真实输出已走通整个分析／汇总流程。
  5件全部可测，1件invalid compound有一个面无法三角化，明确保留为partial。
- 全量结果尚待运行，不能把上述实现检查当成模型效果结论。
- 原模型100图×8候选附录已完成，见[DIVERSITY_VALIDATION.json](DIVERSITY_VALIDATION.json)。
  没有对应DPO K=8，不能比较两者的同图多样性。

## 记录

- [VALIDATION.json](VALIDATION.json)：集成验证、源文件hash与测试。
- [OBSERVATION_VALIDATION.json](OBSERVATION_VALIDATION.json)：相机、投影、自比较、GT轮廓及缓存验证。
- [DIVERSITY_VALIDATION.json](DIVERSITY_VALIDATION.json)：固定100图、800候选和多样性工具验证。
- 完整分析的命令、PID和输出将保存到`LAUNCH.json`；完成后补充最终报告。

原始运行只读。新分析的运行产物位于`checkpoints/diffusion/failure_analysis/runs/`，不提交大体量数据。
