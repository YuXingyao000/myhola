# UV 曲面插值与预测 UV 的只读诊断

## Task
检查面节点误差是否在当前固定插值后放大，以及保留预测 UV、改用 GT 节点插值曲面后曲线误差变化。仅诊断，不修改网络、loss、节点或后处理。

## Assumptions
数据：既有 dual_uv_surface_v1 NPZ 的 face order、UV 域和 owner pair 已由正式评估验证。边界：只解释此32件，不声称全3495件或真实CAD曲面的结论。实现：当前三次16控制点open-uniform节点；NumPy/SciPy float64 CPU。没有待确认项。

## Inputs
固定 validation accepted.txt 原顺序前32件；142既有 full_v1/raw_predictions/<id>/data.npz，读取预测/GT面网格、预测UV/双侧XYZ/owner pairs和GT边。记录清单、文件SHA及脚本SHA。

## Outputs
逐CAD JSON、汇总JSON、原始固定清单和来源；大型预测不复制。输出独立目录，不覆盖原始预测。

## Processing Protocol
保持原预测active边；只在与GT owner pair可对应的子集上比较曲线。GT仅作诊断。重建pred和GT节点，用同一固定样条规则在65×65网格求值；另在原预测UV求值GT样条。单进程CPU最多2线程。

## Metrics
节点、整个65×65网格及0<min(u,v,1-u,1-v)<.05边缘带的坐标MAE、三维欧氏RMS和最大三维距离；它们是同参数插值差，不是真实CAD曲面精度。共同正/反GT点序最小坐标MSE，双侧对应点gap mean/RMS/max，原预测与GT曲面替换两种；报告active/GT/交集分母。UV标量距边界min(x,1-x)固定bins=[0,.001,.01,.05,.1,.25,.5]，另报UV点距最近边界。矩阵cond与4001点H增益。汇总给CAD宏平均和充分统计量池化值。

## Decision Rule
仅报告差异，不据此自动选架构或调参数；节点更准不推定连续面更准；替换GT曲面仍很差支持UV定位本身有问题，但不分离训练耦合因果。

## Edge Cases
缺文件、损坏、形状异常、非有限值、重复pair均逐CAD失败并保留32件分母；空交集曲线指标null；零分母null。不按结果换样本。

## Validation / Sanity Checks
检查插值节点回代、常数保持、重新计算原预测UV曲线与保存XYZ的最大差值；记录误差而不静默修正。统计JSON禁止NaN。

## Implementation Constraints
单一脚本、numpy/scipy、无模型前向/OCC/网络训练；用scp部署临时脚本，记录实际SHA；不更改训练代码或原始产物。

## Deliverables
SPEC.md、diagnose_surface.py、32件逐CAD结果及汇总；报告限制和全部失败。
