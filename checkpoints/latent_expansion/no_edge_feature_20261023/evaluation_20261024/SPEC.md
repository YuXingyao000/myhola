# 关闭边特征匹配：30k终点评估（2026-10-24）

## Task
按用户授权检查142上64维全解冻、feature_weight=0的30k运行是否正常完成，以及取消特征匹配后最终边与位置/尺度是否改善。完成既定实验的评估，不自动续训或追加消融。

## Assumptions
原始网络、数据和指标未改变；已有source32、init64和旧64末权重的validation逐件结果在源码、名单与真实样本复算一致后复用。旧64没有30k权重，等预算因果比较限完整训练/随机posterior validation曲线，不能把不同终点的mean-CD称为同预算消融。

## Inputs
- 142，`/mnt/d/python/checkpoints/latent_expansion/d64_no_edge_feature/20261023T034042Z/weights/last.ckpt`，预期global_step=30000，以实际元数据和SHA256核验。
- 同源32维checkpoint：`shared_curve_uv/candidate/20261019T064906.049992Z/weights/epoch=0116-step=00042471-val_loss=0.036517.ckpt`，SHA256 `ce749416c1635dda1563161b971bd4260d064cfdb0c31f1fa9d06ad0e70961ff`。
- validation：`joint_boundary/validation/val_contract_audit/accepted.txt`，3495 CAD，SHA256 `ae607bfe5499e919d1e68a5110c0c211daf1158aedbe57a4756eda99f6f3f0ed`；DeepCAD v7与UV v1。
- 原诊断 `diagnosis_20261021/` 的完整逐CAD结果和 `shape_placement/`；32维15k对照 `unfreeze_control/evaluation_20261023/` 仅作标注预算的背景。
- W&B：本轮 `8fd1dc1d1023`，旧64 `f45ab7aa6bab`。网络源码对应训练提交7948c1b6；启动记录提交12724437不改变模型。

## Outputs
本目录保存训练完成审计、历史曲线、逐CAD结果、配对统计、中文报告、PNG/SVG图、执行/复算检查和命令。固定预算last为主，不按新总loss挑选best。大体量原始产物依既有规则不入Git。

## Processing Protocol
1. 核验last的step/epoch、参数有限性、配置、正常终止和W&B状态；区分训练错误与退出清理警告。
2. 沿用现有evaluate_control的前向与measure_prediction/path_decomposition，仅在本目录适配变体名和路径；不修改指标或网络。先source32三个真实样本复算与新权重smoke，通过后再跑全量。
3. 完整3495件，模型自身encoder、eval、posterior mean、batch1、FP32模型、关闭TF32、无增强，按原CAD identity规则seed42。8个独立GPU分片，不依赖GT决定raw预测保留项。
4. 同一次forward统计raw几何、拓扑、teacher/coarse/refined、局部形状/center/scale替换和实际点集对齐、特征统计。GT匹配/替换均标为诊断oracle。
5. 主要对比source32→本轮30k。旧64约50k及32维15k为不同预算背景。训练曲线以epoch和global_step严格对齐完整epoch；末partial epoch单列；本轮未做OCC/test，不宣称solid合法率变化。

## Metrics
沿用原指标：全输出点集双向最近点平方距离均值之和CD²，CAD等权，展示单位1e-5；GT逐边匹配CD²、XYZ/local/bbox/center/scale L1；实际质心/半径误差；宏观均值/中位数/P90和逐CAD改善/退步；拓扑micro precision/recall/F1；teacher/coarse特征RMS/中心化RMS/协方差participation ratio。同期曲线比较teacher、最终face/edge、coarse/refined、incidence、UV、分类、KL与feature；total loss跨实验不直接比较。

## Decision Rule
若取消匹配使同期最终边与位置/尺度恶化消退，且固定mean结果没有显著面/拓扑代价，支持该目标在本训练协议中有负面作用；teacher单独恢复不代表任务解决。若最终边仍恶化，说明feature匹配不能单独解释问题。单seed不外推普遍机制，预算不等的末权重比较不隔离单一因果。

## Edge Cases
固定名单保留缺失、失败、NaN/Inf、空预测与对齐失败，不改分母、不挑成功对象。对齐子集必须带计数和同子集基线。漏号/重复ID/错split/错checkpoint终止聚合；脚本错误修复后新目录重跑对应失败任务，保留旧错误记录，不自动排除CAD。已有输出不得覆盖。

## Validation / Sanity Checks
检查数据/源码/权重/名单hash；真实source复算与历史结果一致；新权重严格加载且feature_weight=0；全分片退出0、ID覆盖3495恰好一次、匹配GT边总数199092。重算核心均值/配对数；分解原预测与matched CD一致；几何compose/GT底噪保持旧容差。曲线完整读取，保留对齐键和原始历史，weighted feature全程0；其余weighted项之和与total一致。

## Implementation Constraints
新脚本只写本目录，复用已有实现；保留两端用户dirty文件和历史产物，不改模型/loss/dataset/post、不训练、不调容差、不追加test/OCC。允许只同步本轮文件和下载紧凑结果；不发送跨聊天消息，不新增automation。

## Deliverables
`REPORT.md`、`summary.json`、训练审计/历史、逐CAD压缩表、图、完整执行核验与复现脚本，并更新本次实验README到实际状态。所有研究结论区分观察与机制推断。
