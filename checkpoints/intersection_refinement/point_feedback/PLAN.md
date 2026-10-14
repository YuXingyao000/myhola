# 最终曲线逐点几何反馈

2026-10-14 用户确认实现并运行一晚。状态：实现、真实单卡/8卡入口和150项自动检查通过；正式运行状态见 NOTE.md，尚无下游效果结论。

## 问题与假设

上一轮 conditioned VAE 在固定 2417 件中得到 2111 个原生合法 solid，原 refinement 为 2113。306 个失败件中，独立拟合后的边面偏差出现 299 次，互反半边不一致 279 次，选边后端点不一致 279 次，自由边 239 次，某次全部候选环闭合接纳后仍裁面失败 167 次。观察可重叠；边面偏差在救回成功的 43 件中同样普遍，不能把频率当作因果排名。

现有 GeometryRead 已逐点读取几何，但随后将 16 个点 mean/max 汇成一个向量，更新 E/F 再解码。研究假设：在保留主干的前提下，将最终预测面的局部几何偏差直接映射到每个曲线点的坐标增量，可以改善局部几何一致性。它仍属于 refinement，不引入新的拓扑表示，也不补回 latent 已丢失的信息。

## 固定范围

- 原 Encoder/posterior 冻结，原 Intersection 保持 8 层、512 维、8 heads、FFN 2048、dropout 0.1。
- 原 coarse/final 几何头、refiner、classifier、已有 curve conditioning 和全部 loss 权重保留。
- 从上一轮已训练的 conditioned 验证最佳 checkpoint 初始化全部旧参数；只新增最终逐点反馈，不同时改变 attention、H/P、顶点、环或后处理。
- 新模块仅使用预测的 C1、S1、E1、F1，枚举候选不由 GT 筛选；GT 仅用于现有监督。
- 每点输入 17 维几何观察加 E64、Fi32、Fj32，MLP 145→128→64→3，末层零初始化。无 pooling、点序号或 dropout。
- 几何读取、反馈 MLP、坐标增量与 local/global 换算为 FP32，已有主干继续 BF16 mixed。新增 27,139 参数、6 个 tensor。
- 同时更新最终 global XYZ 和 local XYZ，保持原 center/scale 与后 3 维方向分量数值；几何监督与导出读取同一最终结果。BF16 原 bbox 可能产生精确零 scale（含 GT 正候选）；该退化情形无法在固定 bbox 下表示非零 global 增量，因此仅将新增增量置零，保留原 geometry/bbox 监督更新。非零 scale 不裁剪或加 epsilon；非有限 scale 明确报错。
- 当前投影是三角网格近似，保留 incidence 中 GT 离散误差基线，不执行硬吸附。

## 数据、初始化与预算

源权重：`/mnt/d/python/checkpoints/intersection_refinement/curve_conditioning/20261013T083729.740439Z/weights/epoch=0157-step=00057354-val_loss=0.038204.ckpt`，epoch 157、step 57354。使用 `model.init_curve_feedback_checkpoint` 创建新 run / optimizer；不是恢复旧 optimizer。

数据仍是 DeepCAD v7 与已有 accepted split：train 46480、val 3495、test 2417，cube24、seed42。8卡×每卡16，全局128；Adam lr2e-5、无scheduler、原梯度裁剪norm1；每epoch全量验证。新增统一 trainer.max_time 透传，此次上限8小时（初始化与退出保存另计），保留原10000epoch/1000000step安全上限。

## 去留证据与边界

实现验证包括全部旧 tensor 精确保留、零初始化预测/loss 等价、真实数据形状/loss/梯度/冻结状态、batch16显存，以及统一多卡入口。检查通过不代表重建性能提高。

训练后使用同一2417件、后处理与原生solid判据；验证集选择权重，不用test选择。分别报告原始输出与修复后质量，关注归属正确的边几何尾部、端点/边面残差和最终合法solid，而非仅平均loss。额外训练预算必须单列；若无等预算无新增分支对照，不能把差异全部归因新结构。

同平面共同漂移、错误owner曲面、共享端点与环嵌套并不由此保证解决。新增final投影会增加计算开销，不宣称达到1e-4。后处理已有每无向face-pair选择代表半边并共用初始曲线，不能把raw twin共享直接等同于修复所有free-edge。

依据：[上轮失败报告](../curve_conditioning/failure_census_20261014/REPORT.md)、[已有反馈实现](../../../src/brepnet/models/blocks.py)、[HiFi-BRep局限分析](https://arxiv.org/html/2608.16485#S5)。
