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

## 新运行

- 2026-10-29 01:36:47 UTC在142启动，代码提交`d2fa70c32508095a9c8161800f8ad3f2856ba524`。
- Run：`/mnt/d/python/checkpoints/conditional_intersection/no_edge_feature_xyz32/20261029T013647.209854Z`。
- tmux：`conditional-no-feature-20261029t013647`。
- [W&B](https://wandb.ai/whatseven777/hola-brep/runs/d99031bca830)；完整命令及来源见[LAUNCH.json](LAUNCH.json)。
- 190,236,065个参数全部可训练，7个DDP rank已注册并开始更新。首次启动检查没有Traceback；尚无本轮完整validation结论。
- [REAL_BATCH_CHECK.json](REAL_BATCH_CHECK.json)：142真实两件CAD、BF16前后向通过；weighted_edge_feature严格为0，各主要模块梯度有限。检查模型未用于正式初始化。
- [实际run配置核对](RESOLVED_RUN_CHECK.json)确认`feature_weight=0`、`freeze_encoder=false`、`trainable_scope=all`和`resume_from_checkpoint=null`；[启动状态](STARTUP_HEALTH.json)保存7卡注册及早期更新证据。启动时仅有未跟踪的评估脚本／验收记录，无已跟踪模型代码改动。

02:00 UTC复核时，新run已记录239个optimizer step，仍在首个epoch，未出现Traceback；见[LATEST_HEALTH.json](LATEST_HEALTH.json)。分项train loss按epoch写入，首轮结束前W&B尚无分项值是既有日志行为；feature关闭由实际resolved config及真实前后向共同核验。

旧两权重后处理[完整报告](../feature_on_evaluation_20261029/REPORT.md)：best415为3171/3495（90.73%），last5395为3172/3495（90.76%），来源3180/3495（90.99%）。last原始几何更准，但合法率净少8件。旧结果不是本次feature关闭新run的结果，不使用test调规则。
