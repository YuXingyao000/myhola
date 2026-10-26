# 完整照片 validation baseline

## Task
用户要求完整 validation 获得更稳定的照片 baseline。覆盖全部3505个既有去重validation CAD，固定原始SVR、照片step4553、照片step310000三组；不继续训练或挑新checkpoint。

## Assumptions
完整清单与train无ID重叠，全部照片/GT NPZ/GT STEP存在。已有100件是全集子集、不是全部validation。step4553只在原100件上按loss选出，不能称为3505件的最佳checkpoint；全集仍是开发validation而非保留test。

## Inputs
`src/brepnet/data/list/deduplicated_deepcad_validation_7_30.txt`（3505）；`deepcad_v7_cond/{id}/real_photo.npz[flux]`；`deepcad_v7/{id}/{data.npz,normalized_shape.step}`；固定三个checkpoint，SHA256沿用evaluation_20261026；原100件完成的评估及源代码/运行环境记录。

## Outputs
独立full run；3505件冻结清单、输入/源码/权重摘要；逐件终态与raw/STEP；完整分母CSV/JSON、中文对照表、原100与新增3405子集统计、复用审计与进度。旧100件目录保持不变。

## Processing Protocol
完全复用原FP32 batch1、1000-step DDPM、identity派生seed42、K=1、原AE和图像预处理。8个GPU worker按CAD分片；CPU32 workers异步接收已完成生成，后处理预算仍每件180秒、200次优化、recover_v1/drop_num0。新增原始几何不要求GT latent cache；GT仅供计量。

## Metrics
与100件相同的raw face/edge squared CD、face F1、face count error、native solid validity、STEP覆盖、超时/失败；有效solid的post几何3次采样均值，单列各自/共同成功集合。所有覆盖率以3505为分母。配对bootstrap以CAD为单位，10000次，分块以限制内存。额外报告原100/新增3405防止小样本选择影响被隐藏。

## Decision Rule
按冻结协议报告全量结果，不依据新结果改变容差、筛样本、重采候选或换checkpoint。更大validation提供更稳定的开发baseline，不等同于独立test/实拍结论。

## Edge Cases
空/重复清单、与train重叠、原100非子集、权重/源码/输入变化拒绝启动或复用；缺失/损坏新增输入记完整分母中的终态失败，不静默删件。失败不补零，未完成不能标complete；重启复核协议和已保存产物，已中断候选不reroll。

## Validation / Sanity Checks
复用前核对原协议指纹、全部原300条输入/生成输出/STEP/metrics摘要与计量版本，路径引用回原目录且不修改原产物。源码允许编排/报告变化，sampler/decoder/计量/post保持hash一致。测试动态分母、对子集报告和篡改缓存拒绝；确认新run实际推进、最终三组各3505终态及总数闭合。

## Implementation Constraints
新增全量编排复用已有gpu_worker/evaluate_one，不复制或改模型和OCC算法。保留其他任务工作区修改。固定CPU32并发；如GPU/CPU重叠影响超时，仍保留超时并披露运行调度，不对失败个例放宽预算。

## Deliverables
完整validation三组比较及图表、baseline协议与路径。全量通过后更新主题记录和提交；训练保持停止。
