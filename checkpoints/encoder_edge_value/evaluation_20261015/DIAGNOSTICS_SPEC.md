# Task

在固定评估完成后独立汇总三方逐ID分布、配对solid变化，并补充按owner面对对应的曲线几何，解释全边合并CD可能掩盖的边归属误差。

# Assumptions

主协议见PLAN。conditioned、encoder_gat_control、encoder_edge_value读取相同CAD/face顺序，各自posterior不同。selected checkpoint在读取test前冻结。描述性子集不替代主分母、不证明因果。

# Inputs

candidate/control子目录raw_metrics/posterior_mean、post/{fit_only,optimized}的CSV/summary；其raw_predictions；GT data.npz；固定2417 split与training_audit。原始几何/生产后处理不改。

# Outputs

diagnostics.json、owner_per_object.csv（运行产物）、summary_figure PNG/SVG、REPORT.md与独立校验。保存脚本、CSV、GT/raw源文件摘要集合及协议SHA。

# Processing Protocol

每组恰好2417唯一ID，baseline在两表的既有指标/状态一致；核验same_geometry_source和GT来源摘要。独立复算raw均值/中位数/P90与summary。所有post必须终态，按native_valid_solid解析真/假/未知，不以success.txt代替。

读取每CAD GT有向facepair→edge采样，及三方法预测同映射。只对三方法都存在且GT存在的相同有向facepair计算owner对应CD²；另报告各方法漏边/多边数、共同pair数及覆盖对象数。直接用16个预测点和16个GT点的两向平方最近距离均值之和，不拟合/对齐/找最佳其他边，也不称CAD物理多重边指标。空共同pair返回缺测。

# Metrics

raw face_mae、face_cd_sq、edge_cd_sq及两向分量、reciprocal平均/最大；每项mean/median/P90/P95/P99/max。同IDdelta：改善/持平/退步阈值1e-12，保留缺测。candidate top1%按误差降序/ID升序选ceil(N×.01)，报告总误差份额，不用于剔除主结果。三个预定配对为candidate/conditioned、control/conditioned、candidate/control。另对双方FP=FN=0的对象复算几何。owner曲线每CAD取共同pair的mean/max后在对象上汇总。solid报告各自全集、四种配对转换+未知；几何报告共同可计算集合。

# Decision Rule

先用全集solid、owner覆盖及原始几何判断，再解释尾部/邻接子集。不能由同平面漂移或告警高频直接确定根因。验证曲线中KL和moving teacher与GT几何分开。

# Edge Cases

缺测保留；非法非有限/负误差、重复ID、错误源哈希、错latent摘要/face数/owner索引或重复有向pair明确失败。空集合与零分母null。所有源文件读取前后校验不变，不覆盖既有诊断。

# Validation / Sanity Checks

NumPy/标准库独立统计与上游summary核对；三方表baseline一致、配对集合守恒、owner pair相同；已知移位曲线CD²与相同曲线零值小检查。报告/图从同一JSON生成，最终独立agent复算。

# Implementation Constraints

只读已冻结预测与GT；不重新采样/后处理、不修改已运行评估工具；新增一个任务内汇总脚本。owner采样CD与线性tolerance不同。

# Deliverables

三方结果、图、owner对应诊断及来源审计，结合完整fail census形成结论。
