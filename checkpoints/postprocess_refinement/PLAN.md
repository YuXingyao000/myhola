# Task

在固定的原 Intersection refinement VAE 上改进后处理，使用 DeepCAD validation 比较合法实体、几何保真、边界保留和耗时。配合中文图文说明全部已讨论问题，明确哪些已实现、哪些仍是候选。用户已授权实现与评估；不训练或修改 VAE。

# Assumptions

- 数据：沿用已有 validation 合同审计，原清单 3505 件、accepted 3495 件，10 件拒绝保留在总体分母。不存在已经确认可复用的完整 validation raw，本轮固定权重重新编码 posterior mean。
- 任务边界：使用最初原 refinement epoch10/step3993，不使用后续 conditioning、point-feedback 或 GAT 分支。DeepCAD 为开发基准，优先 validation；本轮不使用 ABC。
- 实现：默认 legacy 路径保持原行为；新增显式 post_policy。圆柱完整自然 loop 的特殊处理保持，不按解析圆柱直觉删改。旧 decoder 去重不在本次路径中，不顺带修改。
- 尚待核验的只有资产读取、实际 checkpoint SHA 和推理工具入口；核验不通过时记录具体阻断，不猜测替代权重。

# Inputs

- checkpoint: `/mnt/d/python/checkpoints/intersection_refinement/frozen/20261011T123836.485257Z/weights/epoch=0010-step=00003993-val_loss=0.039476.ckpt`。
- legacy encoder 来源核验: `/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt`。
- CAD: `/mnt/d/data/deepcad_v7`。
- accepted validation: `checkpoints/joint_boundary/validation/val_contract_audit/accepted.txt`，SHA256 `ae607bfe5499e919d1e68a5110c0c211daf1158aedbe57a4756eda99f6f3f0ed`。
- 原清单与拒绝理由见同目录 summary.json/rejected.csv。所有后处理方案读取同一组原始预测，禁止根据 GT 选边、调几何或挑候选。

# Outputs

本主题保存 raw_predictions、各策略独立输出、配置与命令／代码／权重／输入哈希、逐对象终态、汇总、中文说明图、成败转移图及代表样本。运行产物遵循已有 gitignore；紧凑报告、计划和验证记录保留版本管理。

# Processing Protocol

1. 严格加载固定权重，FP32、TF32 off、seed42、canonical CAD、posterior mean；encoder 可读取输入 CAD 拓扑，decoder 不读取 GT 拓扑，不做额外 face 去重。并行推理按固定 ID 分片，记录错误而非换样本。
2. B0 `legacy`：当前 optimized 后处理，drop_num=0，名义优化200步／既有自适应上限600；原容差序列和圆柱分支不变。
3. B1 `roundtrip_retry`：独立候选状态；候选必须完成保存和原生重读检查，失败继续既有候选序列；普通 wire 不允许无解释跳过，圆柱完整自然 wire 替代单独记录。记录覆盖、失败阶段和耗时。该策略单列，不能把更严格验收称为提升。
4. F1 `fit_score_v2`：独立修正拟合误差公式与边界采样覆盖。固定以 RMS 和最大三维距离评分，遍历原容差候选，只有全检查点最大误差不超过 1e-3 才提前停止；不改变周期处理与后续 legacy 算法。比较其收益与退步，不预设有益。
5. B2 `recover_v1`：优先保留原流程得到的原生合法 solid；失败时按固定顺序尝试 B1 optimized、B1 fit-only、修正拟合的 B1 optimized。最多4条完整路径，共享每对象总墙钟上限180秒，各次从不可变 raw 重新开始；每条路径结果及最终选择写入诊断，不用 GT。新增救回须通过完整面／wire 验收，不能以丢孔换合法率。
6. 首先用预先固定的小批 validation 样本核验接口、原策略回放及圆柱保留，再完整运行 B0/B1/F1/B2。已确认的策略缺陷可修正，但改动与重跑必须记录，最终完整结果来自冻结版本。
7. 每对象独立子进程，初始32 CPU workers、单进程线程1；每策略每对象含测量总上限180秒。已经写出并确认的合法结果不会因后续几何测量超时被抹掉。复用缓存必须核验完整协议和哈希。

# Metrics

主指标：STEP 按原生读取精度重读后为 TopAbs_SOLID 且通过 BRepCheck；不统一抬高 tolerance。分别报告 /3495 与 /3505、STEP覆盖、超时、崩溃、缺失、救回／退步和成对ID。兼容0.1容差口径单列。正式后处理运行前补充固定记录：原生 STEP 的面／边／顶点最大存储容差、逐面wire总数及多wire面数；原生检查使用存储容差，不能当作统一1e-3精度认证。

几何：沿用 native STEP 每面1024点、每边128点、双向平方CD协议；共同可计算集合与各自集合、均值与分位数分开报告。GT仅供此阶段测量。

完整性：记录输入面数、实际使用面数、普通wire是否跳过、圆柱自然wire替代次数及新增救回对象的边界检查。合法solid不等同于1e-3几何保证；不宣称未测得的连续误差上界。记录原始有效候选与增加修复后的实际代价。

# Decision Rule

报告所有预定变体，不只报告最好者。候选去留优先看固定全集原生合法率与成对变化，并核验新增救回没有无解释的面／孔丢失；结合几何和耗时披露取舍。B2保留B0合法结果的机制本身说明退步可能被避免，不把额外尝试预算隐藏成单次算法收益。若没有可靠改善如实报告，不为了满足“必然上涨”改变分母或口径。

# Edge Cases

缺失、损坏、非有限值、空预测、非法索引、崩溃、超时都保留预定分母。无GT几何时合法性仍独立记录，几何标未知；零分母显示不可用。候选之间不复用被修改的OCC对象，不覆盖历史数据；未知边界完整性不当成通过。圆柱自然loop是已知转换，不能按edge数量相等粗暴拒绝。

# Validation / Sanity Checks

独立复核权重／split、严格加载和真实小批输出；测试旧默认行为不变、候选重读失败继续、状态隔离、完整边界检查、拟合端点与误差单位、圆柱分支。汇总检查ID唯一、固定分母、转移守恒、选择只依赖预测及OCC结果。新获救与退步样本作完整几何检查和图示，不能只凭success.txt。

# Implementation Constraints

实现放在既有 post 模块，参数显式传递，默认legacy；评估复用已有隔离runner。保留当前工作区未提交的指南修改。PythonOCC 7.8.1 API用实际环境验证；禁止启动训练、读取ABC样本或改数据split。

# Deliverables

可复现的后处理实现与命令；完整validation对比；包含全部后处理问题、处理办法和本轮范围的中文图文；合法率、配对救回／退步、保真和成本报告；真实失败与局限说明。
