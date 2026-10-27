# UV曲线监督第二阶段

用户已授权：保留当前架构，在UV模型30k checkpoint上继续全量训练。研究问题是：直接UV定位、三维走向与相对精度监督能否改善当前曲线退化，而不牺牲已有面精度。本阶段不是新decoder消融，不能单独归因于某一个新增loss。

本次保留固定曲面、两个UV输出及incidence；不加refinement，不改面分支，不冻结encoder，不处理OCC。详细数据处理协议见 [SPEC](SPEC.md)，网络/目标见 [ARCHITECTURE](ARCHITECTURE.md)。

实施顺序：离线标签合成与真实validation前32件验收 → 冻结标签算法/阈值 → 生成完整train/validation缓存 → 严格checkpoint、Adam及真实批次梯度验收 → 142七卡续训30k更新。生成失败不能静默丢样本；UV不可信点只mask UV项。

正式训练后，固定完整validation清单、解码协议和原始几何指标，对比30k源模型与新阶段checkpoint。至少报告face/edge CD²、双侧间距、拓扑precision/recall/F1、按GT边尺度分层的曲线误差和耗时。保持全部预定对象在分母；本阶段无合法STEP结论。历史原始预测可复用，但需核对checkpoint与完整ID列表。

实现检查只证明训练链路成立；改善是否成立取决于训练后的验证。新增目标改变了总loss单位，旧总loss与新总loss不能直接比较。旧curve_xyz MSE诊断继续保留，新的绝对目标单列curve_xyz_l1。
