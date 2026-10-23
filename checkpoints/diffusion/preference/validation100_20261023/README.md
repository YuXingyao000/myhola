# 原 SVR 与 DPO 的固定100件比较

本轮由用户要求检查：训练中GT去噪验证loss上升，是否对应实际重建退步。
只比较原SVR和最终5,000步权重，不选中间checkpoint，不启动新训练。
协议在运行前写入[PLAN.md](PLAN.md)。

## 运行与实现检查

- Run：`/mnt/d/python/checkpoints/diffusion/preference_eval/20261023T062400Z`。
- 代码：`9abdc26a`；8卡分担两模型，各100件、每件仅c00、共同seed。
- 后处理：与偏好数据一致的recover_v1；旧2026-10-17表使用legacy后处理，
  因此本次两侧都重新生成及评估，不直接拼接那张旧表。
- 6项严格DPO加载测试和7项汇总测试通过；真实GPU检查见[VALIDATION.json](VALIDATION.json)。
  原SVR首件latent及raw与先前c00逐位一致；DPO装回原生模型后的完整采样latent与
  PreferenceDiffusion.sample_latents逐位一致。314个reference tensor与344个DINO tensor
  均逐值等于原权重，原AE保留。
- 完整命令、checkpoint/source/input哈希和逐件记录在run的metadata、raw_predictions、
  evaluations中。失败保留在100件分母；合法但post采样失败不抹掉已完成的native检查。

## 实测结果

已完成，进程退出码0，采样和后处理总耗时268.91秒。两边各100次推理成功、
100件可读STEP、零超时、零unknown。以下固定比较原权重与最终5,000步DPO，
同图同seed、K=1，不挑候选。

| 指标 | 原SVR | DPO 5,000 | 配对件数 |
|---|---:|---:|---:|
| Native合法solid ↑ | 75/100 | 93/100 | 100 |
| Raw面CD² ×1000 ↓ | 5.725 | 5.206 | 100 |
| Raw边CD² ×1000 ↓ | 13.620 | 14.694 | 100 |
| Raw面F1 ↑ | 89.65% | 90.66% | 100 |
| Raw面数MAE ↓ | 1.09 | 1.18 | 100 |
| Post面CD² ×1000 ↓ | 4.089 | 6.064 | 70 |
| Post边CD² ×1000 ↓ | 7.108 | 8.914 | 70 |
| Post面F1 ↑ | 93.71% | 94.31% | 70 |

原来合法的75件全部保持合法，18件从invalid变成valid，7件双方invalid。
合法率提升18个百分点；按CAD配对bootstrap的95%区间为[11,26]个百分点。
这是固定seed下这100件的开发验证证据，不包含其他seed或重训波动。

几何没有全面改善：raw面CD²均值下降9.1%，raw边CD²上升7.9%；
共同70件的post面/边CD²分别上升48.3%/25.4%。这些几何差值的配对95%区间均跨零，
需要扩大评估来确定方向，不能把DPO loss下降或合法率提高当作几何精度提高。

两边分别有71/89件满足合法且三轮post几何完整，共同集合70件。
各有4件已判合法但STEP几何无法采样，仍保留其合法性；没有拿原点替代几何。
原SVR这100件的原始latent及raw与偏好数据生成时的c00全部逐位一致，
核验见[BASELINE_REPLAY.json](BASELINE_REPLAY.json)。

![固定100件对照](report/comparison.png)

完整数据：[逐件CSV](report/per_object.csv)、[汇总及配对区间](report/summary.json)、
[矢量图](report/comparison.svg)。汇总另存各自可测子集的均值及全部合法性转移ID，
避免把不同成功子集直接当配对比较。
本次只计算表中几何/有效性指标，没有新增FE/EV、顶点或LFD评估。
