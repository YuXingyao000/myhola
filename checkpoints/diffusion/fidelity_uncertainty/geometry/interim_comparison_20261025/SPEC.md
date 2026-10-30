# 几何辅助 loss 对照：进行中评估的只读快照

## 1. Task
回答不加几何辅助 loss 是否更好；读取已有评估，不启动或改变运行。

## 2. Assumptions
用户所说的几何条件暂按最近 DPO 几何辅助 loss 理解。两组由相同 DPO5000 初始化，各额外训练5000步。评估尚未完成，完成顺序可能带来样本选择偏差。

## 3. Inputs
`checkpoints/diffusion/preference_eval/geometry_v1_evaluation/comparison/` 的冻结3505件清单、协议、inference.json和metrics.json。baseline代表继续DPO，dpo_5000代表DPO+geometry。

## 4. Outputs
独立快照：全量逐件精简状态及源文件摘要、共同完成集合指标、配对区间和中文对照表；不写原评估目录。

## 5. Processing Protocol
单进程读取，每个JSON只读一次并记录SHA256和起止时间。全量3505件保留pending状态。配对表只取双方均达到终态的CAD，保留其中失败及unknown；不挑选成功样本计算有效率。原始几何只使用双方有限数值；post须双方valid且repeat0/1/2三次测量完整，先CAD内均值。

## 6. Metrics
全量生成覆盖、评估完成数、valid/invalid/unknown/pending；共同完成集合有效率、raw面/边CD²、面F1、面数MAE；共同完整post的面/边CD²及面F1。CD²展示乘1000，F1展示百分比。没有观测指标时明确未完成。

## 7. Decision Rule
几何组减控制组，按CAD配对bootstrap10000次、seed42、95%区间。仅描述此快照的差异，区间不修正完成顺序偏差，也不用于宣布全量最终胜负或选择checkpoint。

## 8. Edge Cases
缺失/运行中记pending，不作失败；已终态unknown不作valid。缺失、非有限或损坏记录保留原因，不填零。空共同集不给均值。ID重复、非预期ID、协议不匹配、种子不一致须报错。

## 9. Validation / Sanity Checks
核对冻结清单与协议一致，状态数之和3505，双侧seed一致；post三轮集合及数值完整；有效性转移计数与均值差一致。原始报告保持不变。

## 10. Implementation Constraints
使用已有评估模块的finite_number、paired_statistics等统计函数；不导入模型、不使用GPU、不重做OCC。不修改用户未提交代码。

## 11. Deliverables
可追溯快照JSON、生成脚本及简短README；聊天显示明确标为中途结果的比较表，并指出全量及可见一致性仍待评估。
