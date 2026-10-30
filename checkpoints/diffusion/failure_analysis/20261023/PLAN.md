# 单视图 diffusion 完整验证与失败溯源

## 1. Task
完成原SVR与DPO完整3505件validation的评估，产出失败阶段统计、内部几何异常、
可见区域一致性和重复性图表。将评估写成workspace内可复用的公开CLI/API，而非一次性脚本。

## 2. Assumptions
延续当前已冻结的双模型、原HoLa VAE、view0、c00与recover_v1后处理；不改训练或修复。
单图存在多解性：完整GT误差不单独判定重建错误；可见GT误差也不等于图像可直接测得的深度。
终态阶段和异常共现是观测事实，不能自动归因为diffusion或VAE的根因。

## 3. Inputs
- 主run：`checkpoints/diffusion/preference_eval/20261023T090834Z`，每方法3505件、K=1。
- 实际路径/权重/清单/seed取该run冻结的`metadata/protocol.json`；已有100件只读引用。
- 原始latent/NPZ、逐件inference/metrics、post_diagnostics及最终STEP，禁止重跑后处理。
- 原SVR的K=8附录：`checkpoints/diffusion/preference_data/20261020T134930Z/candidates/validation`，
  同一固定100件，每件c00…c07。DPO没有对应K=8，本轮不宣称DPO的同图多样性变化。
- 灰模图像`imgs.npz[svr_imgs][0]`与归一化GT STEP；相机来自旧`extract_imgs.py`及迁移元数据。

## 4. Outputs
在独立分析目录保存协议/源码hash、逐件JSON与CSV、可追溯源文件路径和哈希、汇总JSON、
中文PNG/SVG图表及分析报告。新工具放`src/brepnet/eval/`，包含帮助、文档和行为测试。

## 5. Processing Protocol
完整结果必须等主run达到complete；临时检查明确pending，不把尚未评估算失败。
只读保存的结果与STEP；不healing、ICP、旋转选优、单独重归一化或GT指导候选选择。
可见表面采用同一显式旧SVR相机：224²、FOVy45°、eye=(2,2,2)、center=(0,0,0)、
up=(2,2,3)，约定x右/y上/z前。重新以正确透视逆深度插值渲染双方STEP并反投影；
不用旧visibility缓存的线性深度，也不用未经裁剪的raw曲面冒充实体轮廓。
输入白底mask固定RGB阈值245；同时报告GT重渲染对输入mask的误差，保留OCC轮廓线造成的差异。
完整GT指标保持现有定义；新增工具不修改正在运行的采样/后处理/指标源码。

## 6. Metrics
- 已有：覆盖率、native有效solid、invalid/unknown、raw/post面边平方Chamfer、面匹配F1、面数误差。
- 失败溯源：互斥最终阶段（含超时/运行错误/未决）与每条恢复路径；多标签wire、trim、solid、STEP拒绝证据。
- 无GT内部检查：自连接/单向或重复半边、孤立面/图连通分量、互反曲线差异、曲线到owner面三角网格距离。
  距离是离散几何代理；固定归一化长度0.01作为异常筛查阈值，同时保留连续值。
- 单视图：预测轮廓对输入图的IoU、对称轮廓边界距离（图像对角线归一化）；GT对输入同指标；
  双方实际可见点的双向平方Chamfer（两方向均值之和），以及相交前景上的深度误差与覆盖率。
  仅用GT可见表面，不惩罚隐藏补全；读得出的invalid STEP也可计算，validity另列。
- 重复性：raw面/边点在共同坐标系按h=1e-4/1e-3/1e-2量化去重排序指纹；
  报告不同指纹数、重复组成员及重复对比例。不称其为严格B-Rep等价或同图多样性。
- 同图K>1工具：全部raw点集的平均成对双向平方Chamfer；先每图平均再对图片等权平均，
  全候选与双方valid的候选对分别报告数量。K<2为不可计算，距离大不等于合理多样性。

## 7. Decision Rule
先比较有效性及失败阶段，再检查可见区域与raw内部一致性是否同时改善。
完整GT差但观测一致的结果标为“需检查隐藏补全”，不能自动称为正确多解或错误模型。
raw异常与失败的相关性只能提出假设；未跑GT latent→同一VAE对照，不把失败分配为VAE或diffusion根因。
阶段图的类别互斥；异常图的标签可重叠，分母清楚；展示配对共同可测及各自覆盖率。

## 8. Edge Cases
缺失、损坏、空几何、异常、超时、相机不适配、不可计算指标保留在完整分母，不填零。
合法性已判定但几何采样失败不改写其合法标签。相机/轮廓基准异常作为数据审查信号，不自动剔除。
未记录拟合残差、优化位移及缝合内部过程标记not_recorded，不虚构溯源。

## 9. Validation / Sanity Checks
验证失败/valid/pending分类、多个恢复路径、输入hash；半边互反与owner距离的已知反例；
同相机自比较、遮挡背面变化、前景改变、透视深度及投影/反投影；重复集合/排列/平移与K1/K2行为。
真实小批量检查后运行完整3505件；旧100和完整集不混分母，绘图使用原始数量级。

## 10. Implementation Constraints
明确参数、类型注解、可导入函数与CLI；保持职责分层，复用数值/STEP读取工具。
不依赖用户机器硬编码路径；不修改已有run、权重、后处理或用户未提交改动。

## 11. Deliverables
可复用失败分析、单视图观测和多候选指标工具；完整集统计图/表/报告及源码/输入溯源。
主集K=1不评估同图多样性；已有原模型K=8仅作有清楚标识的附录，不冒充DPO对照。
