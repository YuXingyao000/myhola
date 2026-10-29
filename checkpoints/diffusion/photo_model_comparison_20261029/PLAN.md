# Task

停止空间条件训练，先可视化并人工检查训练曲线，再在完整 validation 上评价空间模型与 photo310k DPO5000，和既有 photo310k baseline 比较。

# Assumptions

沿用此前完整 validation 的 3505 件合成真实感照片；这里的“测试”指冻结方案后的 validation 重建评价，不打开独立 test。新架构从头训练，photo310k/DPO继承SVR预训练，质量比较不能直接归因为单个架构组件。

# Inputs

空间 run `diffusion_spatial_photo/20261028T104221Z`，W&B `1e37b32cb626`；照片 DPO run `diffusion_preference_photo/20261028T073228Z`，W&B `acc61a68da66`。空间最低验证loss checkpoint经元数据确认 step20930；最后完整 checkpoint step63700。DPO最后 checkpoint元数据 step5000、photo source step310000、无几何辅助loss。

baseline来源 `checkpoints/diffusion/photo_baseline/evaluation_full_20261026` 的 `photo_latest` 行，checkpoint SHA256 `d8ba2f83fa6989ad235c021d2a41ca23d79505076a522954086597d575caa745`。原VAE/DINO来源SHA256 `9dcafb192529c1ddf65942392c25b409fbbf2bce407805f34a460d0b44bb4839`。

# Outputs

训练曲线 PNG/SVG、曲线原始数据与摘要；固定协议和源文件摘要；每件生成/后处理/评价状态及原始latent；四行定量表（baseline310k、DPO5000、空间best20930、空间last63700），95%配对bootstrap区间。

# Processing Protocol

先核验并保留已有checkpoint，停止当前训练。绘制完整训练/验证DDPM loss与后期放大图；DPO偏好目标单列，不把训练总loss当DDPM loss。曲线读取完整历史，不依赖W&B下采样截图。

模型选择在下游结果出现前冻结：空间按已记录的最低validation MSE和最后完整保存状态两份；DPO固定用户指定5000步，不按本轮重建结果挑权重。每个方法同一3505件、同一输入摘要、K=1、同seed、原CPU初始噪声/GPU逐步噪声语义、1000步DDPM、FP32且TF32关闭。空间模型仅通过接口适配器接入相同sampler，不改架构或缩短采样。

原VAE与两级去重不变；后处理recover_v1、drop_num0、200次优化、每件整体180秒，合法STEP几何测量重复3次。baseline只在逐件记录、输入、计算协议及源码核验通过后复用；不把不同协议的历史表直接拼进新表。

# Metrics

完整分母的生成覆盖、有效solid比例、STEP产出、超时/失败/未知；原始面/边CD²（两方向平方距离均值之和，展示×1000）、face F1（阈值0.1）、面数绝对误差；成功solid几何均值独列并注明n，另列共同有效子集；采样和评价耗时。此固定历史协议不含真正图像重投影/可见区域指标，不将完整GT几何分数称为可见区域一致性。

# Decision Rule

训练下降、验证持续回升仅视作过拟合迹象；是否损害B-Rep重建由固定下游结果判断。比较validity与几何的取舍，不只按最小diffusion loss选“最佳模型”。仅成功子集几何不代表全量成功率；偏好训练loss下降不等于合法率提高。

# Edge Cases

缺失输入、非法形状、NaN/Inf、超时、生成失败保留固定分母；空测量集合返回null，绝不补零。中断候选不换seed重抽；只有明确完成并经摘要校验的记录可复用。DPO必须核对reference等于照片310k而非原SVR；错误来源立即报错。恢复评价不得覆盖旧baseline、改清单或改变已冻结checkpoint。

# Validation / Sanity Checks

确认8卡训练已停止；核对checkpoint实际步数和源权重；复用baseline前检查源代码变化是否涉及实际调用函数、runtime、每件输出摘要及既有表。测试来源错配拒绝、固定种子、四方法等长汇总、失败分母、部分结果不冒充完成；真实一个样本跑完整采样/后处理链。正式评价中每件核对输入/输出摘要，结束再检查源码未变。

# Implementation Constraints

复用现有sampler、VAE decode、OCC和metric代码；新增照片模型比较入口，避免改动工作区已有DPO/geometry实验。可使用8GPU及32CPU worker；日志/权重/大量几何不进Git。中文报告，科学曲线使用matplotlib并实际查看。

# Deliverables

停训记录、完整曲线及解释、可复现评估程序和协议、完整validation对照结果。若长任务仍运行，明确状态/进度，不能把启动或少量结果写成已完成。后续训练检查固定先画曲线并查看，再做选模与下游评价。
