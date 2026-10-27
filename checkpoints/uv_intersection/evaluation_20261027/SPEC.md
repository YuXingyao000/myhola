# UV 全量训练 30k 原始几何验收

## Task
测试已完成的64维UV全量训练，测量面、双侧曲线、拓扑及两侧一致性，并和原64维no-edge-feature结果对照。先完成已有表示的原始几何验收。

## Assumptions
本次“测试”按既有开发分工使用固定validation 3495 CAD；原3505件中已有10件合同排除，不新增筛选。固定last 30000步为主，不用本次结果挑checkpoint。VAE输入包含CAD几何/拓扑，结论限重建，不是照片生成。新旧训练历史不同，不是等预算单变量消融。

## Inputs
- 142：`/mnt/d/python/checkpoints/uv_intersection/joint64/20261025T144428Z/weights/last.ckpt`，核对实际step、配置、SHA及有限值。
- `checkpoints/joint_boundary/validation/val_contract_audit/accepted.txt`，3495件，SHA256 `ae607bfe5499e919d1e68a5110c0c211daf1158aedbe57a4756eda99f6f3f0ed`；DeepCAD v7。
- 旧64D：SHA256 `4262f4998fe047e19c3ca21a835d74c183b7430f7a656c709ec8ed045262ada4`，上一轮原始预测及逐CAD指标；复用前核对名单、数据hash和指标定义。

## Outputs
独立evaluation目录保存SPEC、代码、权重/数据/代码溯源、逐CAD JSONL、双UV原始NPZ、summary、分布图及中文报告。原始大文件不入Git，紧凑结果纳入版本管理。

## Processing Protocol
严格完整加载最后权重；eval、posterior mean、FP32、无autocast/TF32、seed42、单CAD前向、无增广、threshold>0.5、不去重。允许7个独立单GPU分片，固定ID步进切片，无DDP补齐。预测保留仅由模型概率决定。GT pair选择仅用于标明诊断的逐边误差，不影响raw预测。先两件真实CAD检查，再同代码完整运行；失败保留。

## Metrics
- face/edge raw CD²：float64双向最近采样点平方距离均值之和，复用`tools/vae_compare_metrics.py:squared_chamfer`。face按全部16×16点；edge按全部激活双侧16点，旧版也使用全部互反半边。它衡量双侧轨迹点集，不代表已构造出一条合法共享边。另列两侧各自CD及其平均，避免仅合并点集掩盖坏侧。
- face对应XYZ L1；GT无向pair拓扑TP/FP/FN及micro P/R/F1。
- GT pair曲线诊断：无论分类是否命中，双侧与GT的逐点MSE（同一个正/反方向最小）、逐边最近点CD²；与训练loss口径分别记录。
- 双侧对应点欧氏gap：mean、RMS、P50/P90/P95/P99/max；另列双侧最近采样点CD²，区分参数不对齐与几何不一致。激活边与GT pair诊断分开。
- CAD宏平均/中位/P90及逐件改善数量。gap的CAD等权汇总与全边/点分布分别标明。单位为数据坐标，非毫米。任何owner自贴合为表示恒等式，不能作为准确性证据。

## Decision Rule
报告实际变化与局限，不以训练loss代替几何结论。后处理尚不支持`dual_uv_surface_v1`，OCC项记为接口未实现/未评估，不能填0或复用旧合法率；禁止把两侧均值伪装成旧pred_edge并称正式新架构合法率。完整同域UV/OCC构造属于后续明确实现任务。

## Edge Cases
缺失/损坏输入、NaN/Inf、推理失败逐CAD保留状态；空边时face指标仍计算、edge CD不可计算并标null，拓扑仍统计。无GT边则诊断null。重复/漏ID、错split、错权重阻止完整汇总。已有输出不覆盖。聚合显式记录每项有效分母，不静默忽略失败。

## Validation / Sanity Checks
已知平移/空曲线/反向曲线检查指标；真实样本严格load、loss有限、导出NPZ可无pickle回读，UV回代XYZ与导出一致、曲面网格重建误差有记录。全量检查3495唯一ID、分片退出、失败统计、baseline固定ID/hash匹配与独立聚合复算。

## Implementation Constraints
不改网络、loss、训练或post；脚本仅写本主题，复用已有metric与模型。保留所有历史产物及其他GPU任务。新脚本在隔离工作树开发并显式部署，记录代码commit。

## Deliverables
测试完成审计、可复现脚本、逐CAD指标、原始预测、汇总与分布图、中文REPORT；明确已测量原始几何和未完成OCC接口的界限。
