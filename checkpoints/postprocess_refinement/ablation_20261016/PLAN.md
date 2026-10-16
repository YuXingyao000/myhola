# Task

固定原始VAE与两次已采纳改动的VAE，完成模型×后处理的2×2消融比较，给出改动前后内部block结构图。附源码解释get_solid的内存检查与保存重读检查。

# Assumptions

- 两项采纳改动作为整体：原intersection上的一轮几何反馈refinement，以及最终curve几何头的face-pair conditioning。历史用户已保留conditioning；local attention、point feedback、encoder GAT新分支不纳入。
- 这是已训练权重的系统组件对照。新模型经历额外微调和损失变化，没有等预算loss-only训练对照，不把结果完全归因于某一新增block。
- validation与test均使用既有accepted清单。方法已冻结，test只做统计，不据其个例设计新规则或选checkpoint。历史DeepCAD test参与开发的事实保留，ABC不使用。

# Inputs

- 原模型：`/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt`，AutoEncoder_light，epoch8499/step1045500。
- 改后模型：`/mnt/d/python/checkpoints/intersection_refinement/curve_conditioning/20261013T083729.740439Z/weights/epoch=0157-step=00057354-val_loss=0.038204.ckpt`，SHA256 `4c7910b8169206c74fcf38c358cf171359149e0429c8debbc227d391aa56613f`，历史validation callback选定，不在本轮换权重。
- 数据：`/mnt/d/data/deepcad_v7`；validation accepted3495/原3505，test accepted2417/原2424，清单位于`checkpoints/joint_boundary/validation/{val,test}_contract_audit/accepted.txt`。
- 两模型encoder/posterior严格核对243张量完全相同。共同输入采用本主题validation_20261016、test_20261016原refinement导出的canonical FP32 posterior_mean；其encoder也严格等于原AE。核对每个源CAD、latent、raw/protocol hash，然后逐值复用；不混入旧归档的其他数值运行。
- 后处理保持commit c0d69f8a对应生产代码；legacy与recover_v1，固定原容差/圆柱特殊loop，不修改任何生产模型或修复算法。

# Outputs

本目录按validation/test保存raw_predictions、两个post_variants、metadata、report；只将紧凑协议、审计、结果表与图版本化，不覆盖历史产物。既有仅refinement两条后处理结果作为明确标记的旁列参照。

# Processing Protocol

1. 严格加载两个checkpoint，核对encoder/posterior相同、改后配置仅refinement+conditioning；核对split不相交、源码和权重哈希。
2. 从相同已核验均值解码两模型；FP32、TF32 off、seed42、8个不重叠GPU分片，真实小批检查后全量导出。无GT decoder拓扑、无额外去重、阈值0.5；原模型N²有向候选与新模型N(N−1)/双向logit平均均按各自已有推理合同保留并披露。
3. 每个split、每种后处理独立root，读同一份各模型raw。既有tools.vae_compare_post使用legacy/conditioned两方法，optimized、drop_num=0、200步名义优化/600步自适应上限、CPU单线程、每对象含测量180秒总预算。四个split×policy运行并发，每个32 workers，共128。
4. recover_v1四条路径顺序不变，共享总预算，首个通过即接受，不用GT选候选。记录额外尝试与成本。
5. 全部预定记录终态后生成配对结果、共同可计算几何统计与图，固定split分别汇总。仅refinement旁列复用既有同协议结果，标注历史测时不作本轮速度比较。

# Metrics

主指标：原生STEP重读TopAbs_SOLID且BRepCheck通过，accepted及原始分母同时给出。报告模型改动在两种post下、post改动在两个模型下、最终组合对原版的配对救回/退步。辅助：raw面XYZ MAE、面/边CD²与连接P/R/F1（若复用现有工具符合合同）；最终STEP面/边CD²共同可计算与共同合法子集的均值/中位数/P90、覆盖、失败/超时/缺失、存储容差、耗时。CD²沿用每面1024点/每边128点双向平方距离无对齐缩放，不当作1e-3最大误差。

# Decision Rule

如实报告四格，无论方向和幅度。只判断当前已采纳整体及固定post带来的实际变化；零旧成功退步可能来自recover_v1保留旧路径的设计。均值改善不等于每件改善，合法率不保证拓扑和精度完全正确，不预设结果能证明每项都有效。

# Edge Cases

空/坏/NaN/Inf预测、缺失、崩溃、超时留在预定分母，无法算几何标未知；已有确认的合法性不被后续几何采样错误抹去。输入/hash冲突停止受影响阶段，不静默覆盖/更换权重/重试筛好结果。无共同有效样本则指标记null。accepted之前拒绝的10/7件另列原始分母。

# Validation / Sanity Checks

真实小批严格加载与有限shape、两模型每件输入latent逐值一致；均值前三件重编码atol/rtol1e-5只作数值检查，不替换实际共同输入。全量唯一覆盖及split/hash/源码冻结核验。报告从逐件记录独立重算，配对计数守恒，PNG/SVG视觉检查。工具新代码只改实验执行和报告，不改指标数学定义；对关键拒绝行为进行必要检查。

# Implementation Constraints

在`/mnt/d/python_curve_eval`的codex/postprocess-validation分支操作，保留原工作区；环境`/mnt/d/anaconda3/envs/torch/bin/python`。先冻结本协议，再写实验脚本。复用现有加载、解码、后处理和测量工具，仅在本实验目录新增少量装配/报告代码。不给现有trainer和模型增加分支。所有运行保存命令、git与源码/输入/权重哈希。

# Deliverables

中文消融表与对比图、精确checkpoint与协议、逐件审计、架构前后PNG/SVG（覆盖face transformer、pair intersection、几何头、反馈和条件调制数据流/维度），get_solid源码解释。完成后提交紧凑研究记录。
