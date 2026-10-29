# Conditional Intersection：关闭特征匹配后重新训练

2026-10-29，用户明确要求停止feature-on运行，关闭edge_feature后重新训练，并评估旧轮次的OCC合法率。

## 决定与初始化

新的正式入口为`configs/train_vae_conditional_intersection_no_edge_feature.yaml`。
旧`train_vae_conditional_intersection.yaml`仅保留为feature-on历史配置，避免旧run复现／resume时静默改目标。

从原始32维共享XYZ基线重新初始化Conditional Intersection，不继承feature-on的训练更新或Adam。
来源step42471，SHA256 `ce749416c1635dda1563161b971bd4260d064cfdb0c31f1fa9d06ad0e70961ff`；
路径`/mnt/d/python/checkpoints/shared_curve_uv/candidate/20261019T064906.049992Z/weights/epoch=0116-step=00042471-val_loss=0.036517.ckpt`。
全模型解冻、seed42、Adam lr2e-5、BF16、GPU1–7、每卡batch8／累积2，总批量112；预算仍为30,000更新或24小时。

## 唯一监督改动

`model.loss_weights.feature_weight=0.0`。原始`edge_feature`仍可观察，但`weighted_edge_feature=0`，不进入有效优化目标。
面／边local与bbox重建、XYZ、分类、incidence、twin、辅助UV／UV anchor以及KL的权重全部不变。
Encoder边几何重建保留；关闭的是Intersection表示对GT边Encoder特征的逐维匹配。

[CONFIG_CHECK.json](CONFIG_CHECK.json)确认两份resolved config只有实验名、W&B名和feature_weight变化；
相同前向的全部原始项一致，total减少的量等于原feature项，剩余各主要模块仍有有限非零梯度。
这只是实现验收，不证明研究效果提升。

## 旧轮次与评估

旧run `xyz32/20261028T164331.682169Z`已按用户要求停止。
保留best415和最后完整保存的epoch12／step5395；停止请求之后没有伪称保存尚未落盘的更新。
旧W&B的failed状态由主动SIGTERM产生，不作为数值崩溃证据。
记录见[STOP](../feature_on_evaluation_20261029/STOP.json)和[训练分项](../feature_on_evaluation_20261029/TRAINING_HISTORY.json)。

epoch0到12总validation loss增加4.39%，去掉feature项后其余加权目标下降2.44%。
这不等同于纯几何度量，也不是源checkpoint到终点的比较。
新旧total loss不能直接对比高低；应比较相同原始几何项和固定3495件validation的下游合法率。

正式启动信息将在142验收后记入本目录`LAUNCH.json`。旧两权重后处理结果单独归档到
`../feature_on_evaluation_20261029/`，不使用test调规则。
