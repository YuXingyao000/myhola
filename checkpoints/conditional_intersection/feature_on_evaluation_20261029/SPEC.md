# Task
评估 Conditional Intersection 首轮 edge feature=1 训练的 validation best 与停训时最后完整 checkpoint，并和来源32维91%模型比较原始几何与原生OCC合法率。

# Assumptions
本次是有CAD输入的VAE重建，非照片重建。best按该run已保存的validation total loss选择；last为停训前最后完整保存，不代表尚未保存的内存权重。两者同时报告，不根据OCC挑权重。baseline冻结为best42471。训练历史不同，不称等预算消融。

# Inputs
固定3495 validation accepted清单，SHA256 ae607bfe5499e919d1e68a5110c0c211daf1158aedbe57a4756eda99f6f3f0ed，原3505有10件历史合同排除。原始DeepCAD v7。Conditional旧run 20261028T164331.682169Z；best step415，last身份以STOP记录为准。baseline SHA256 ce749416c1635dda1563161b971bd4260d064cfdb0c31f1fa9d06ad0e70961ff，历史结果 /mnt/d/python_curve_eval/checkpoints/shared_curve_uv/evaluation_20261020/best。

# Outputs
每个checkpoint固定SHA/step、逐CAD raw预测及指标、OCC完整终态与STEP、原生合法率/配对救回退步、共同合法可测STEP几何和汇总报告。大文件留运行目录，紧凑记录进Git。

# Processing Protocol
142单独物理GPU7导出，两模型各自encoder重编码；posterior mean、identity、不增强、FP32、无autocast/TF32、seed42、每CAD独立稳定seed，概率>0.5、不去重。严格载入原state，不在decoder读取GT拓扑/UV。187原PythonOCC7.8.1.1、PyTorch2.4.1/cu124环境；与baseline相同11个post源码SHA、recover_v1、optimized、200次优化/既有自适应600、drop_num0、CPU每worker1线程、32 workers、每CAD整个子进程180秒。依次处理best/last。无GT修复或候选选择。

# Metrics
复用tools.vae_compare_metrics.evaluate_prediction：raw面/边点CD平方（两方向均值之和）、面MAE、无向face-pair micro P/R/F1及完全正确对象数。复用tools.vae_compare_post：STEP read.precision.mode=0重读、原生TopAbs_SOLID且BRepCheck通过为主合法；成功标记不替代。STEP采样及CD沿用原协议，每面1024面积加权点、每边128原生参数点，无对齐缩放。完整3495及原3505分母并列；几何只报告可测覆盖与同对象配对集合。

# Decision Rule
同时报告best/last相对来源91%模型的几何和合法率，区别训练loss下降和下游进步。仅当前feature开启版不能唯一归因feature loss或注意力架构。下一轮关闭feature实验由root另行执行。

# Edge Cases
空预测、非有限、缺文件、模型失败、超时保留在预定分母；几何缺测独立于已经确认的合法性。错误SHA、重复/缺ID、源代码变化阻止完整比较。合法后几何阶段超时保留合法、几何缺测。不调整阈值、不按结果重跑挑版本。

# Validation / Sanity Checks
导出小样本检查32维latent、face16x16、edge16、legacy_half_edges、连接范围、浮点有限与NPZ无损；全量逐件SHA/ID和checkpoint一致，CAD输入SHA与baseline逐件相同；回传SHA一致。post源码11文件逐SHA一致。独立重算有效数和配对四格，完整终态3495。

# Implementation Constraints
仅新增本目录代码/记录；不改模型或生产后处理。先核验GPU释放再导出，导出完立即释放GPU给新训练。保留旧产物；不用test；参数化wrapper仅扩展方法身份。

# Deliverables
best/last完整validation raw与OCC结果、baseline对照和可复查路径、限定范围的中文结论。
