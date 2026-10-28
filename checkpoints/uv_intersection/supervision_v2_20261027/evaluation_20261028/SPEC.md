# Task

按用户要求停止142上的UV监督v2训练，保留可恢复checkpoint，在固定validation上测量效果，与UV30k和旧64D独立XYZ曲线模型比较，给出是否继续当前方向的研究判断。不会自动重启或更改架构。

# Assumptions

本次是VAE重建评估，不是照片重建；使用历史冻结3495 validation，不用test。选择停训前最新完整last checkpoint作为唯一主结果，不据当前结果挑best。训练历史/预算不等，三方比较不是等预算单变量消融。

# Inputs

- v2固定快照：`paused_step_00051460_20261028T075416Z.ckpt`，原stage `.../joint64/20261025T144428Z/stages/supervision_v2_20261027T081450Z/weights/`；step51460、epoch123，SHA256 `da666c7db3f66e5257de8d17238d7996154f223c046ac2c0006cb807443f677a`。最新完整保存不包含停训时当前epoch内存中的少量额外更新。
- UV30k：SHA256 `c4f409b1a904d431cf82cf4754001ca590b59dc44abbaf987181819977a32318`；复用 `checkpoints/uv_intersection/evaluation_20261027/full_v1/`逐件记录。
- 旧64D：SHA256 `4262f4998fe047e19c3ca21a835d74c183b7430f7a656c709ec8ed045262ada4`；复用既有no_edge_feature_last64逐件记录及export来源收据。
- 3495 accepted validation SHA256 `ae607bfe5499e919d1e68a5110c0c211daf1158aedbe57a4756eda99f6f3f0ed`，原DeepCAD v7；v2 loss读已验证的当前曲面UV cache，仅用于监督统计。

# Outputs

STOP记录、固定权重/源码/命令协议、完整逐CAD指标与原始预测、三方汇总及配对比较、可导出的分布图、中文判断。大文件留远程，紧凑结果及说明进Git。

# Processing Protocol

复用上一轮evaluate/metrics，允许把硬编码checkpoint SHA/step改为显式参数，不改变任何指标计算。FP32模型、float64距离、posterior mean、单CAD、无增强/autocast/TF32、概率>0.5、不去重。7个独立GPU步进分片，无DDP补齐；保持推理输出不被GT筛选。先2件实现检查再完整3495件。核对各分片退出、完整ID、输入SHA与历史一致，冻结源码在运行前后未改。

# Metrics

面/边合并采样点raw CD²（双向最近点平方距离均值之和）、面XYZ L1、双侧分别的mean/worst CD²、GT-pair曲线XYZ MSE与每边CD²、双侧对应点gap mean/RMS/分位数及最近点CD²、拓扑micro P/R/F1/完全正确CAD。CAD等权均值/中位/P90、配对改善/退步数量并列。新UV/切向/相对XYZ loss仅作解释；不把不同版本total loss直接比较。距离为数据坐标，不换算毫米；CD²不是欧氏平均距离。gap为参数对应点残差，不等同OCC裁剪距离。

# Decision Rule

先看新监督是否改善UV30k的实际曲线误差，同时是否损害面或拓扑；再看与旧64D的剩余差距。训练loss持续下降不单独支持继续投入，旧64D明显更好则不能称曲线已恢复。架构/优化失败机制需要由几何、双侧不一致及误差分布共同解释，区分结果和假设。没有原生dualUV OCC构造入口，合法率明确未评估，不以0或两侧均值冒充。

# Edge Cases

推理失败、缺失、非有限逐件保留，全部3495仍为计划分母；空边CD记null但面/拓扑照常计。错SHA、漏/重复ID、混合协议阻止完整比较。已有输出不覆盖，停止仅针对本stage进程，不触碰GPU0其他任务。

# Validation / Sanity Checks

严格load checkpoint与Adam步数、确认所有训练进程退出；两件真实数据独立UV回代/曲面网格检查；保留已有loss MSE和独立几何计算一致性检查；全量输入hash与三方ID匹配，独立聚合复核。指标实现沿用已验收版本，不为本轮调阈值。

# Implementation Constraints

不改网络/loss/后处理，不重启训练。记录在当前主题目录；只做通用评估脚本必要参数化和本次三方汇总。保留未提交图件和历史checkpoint。用户时间用北京时间。

# Deliverables

停训确认、实际固定step/权重位置、3495完整结果、与两版baseline的数字对照和可复查分布图、明确继续或转换方向的理由及未验证边界。
