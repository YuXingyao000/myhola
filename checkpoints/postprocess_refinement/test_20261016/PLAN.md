# Task

将validation选定的`recover_v1`固定后用于DeepCAD test，与相同原Intersection refinement VAE的`legacy`后处理比较。用户要求分析与方案修改继续依据validation；本轮test仅测量冻结方案的数据，不据结果修改规则、容差、候选顺序或模型。

# Assumptions

- 数据：正式test原清单2424件，沿用此前数据合同审核accepted的2417件；7件已拒绝仍计入原清单分母，不重新筛样本。
- 任务：只比较B0 `legacy`与validation选定的B2 `recover_v1`。不在test重新筛选B1/F1，不做逐件失败机制挖掘或挑图，不运行ABC。
- 实现：生产推理、后处理与评估runner冻结为commit `1fe0ebb8`。允许给已有报告工具增加显式策略列表与split标签，不能改变指标数学定义或修复行为。
- 同一checkpoint、同一raw、同一180秒总预算；B2最多4条路径共享该预算。B2额外尝试数和耗时公开报告。

# Inputs

- checkpoint：`/mnt/d/python/checkpoints/intersection_refinement/frozen/20261011T123836.485257Z/weights/epoch=0010-step=00003993-val_loss=0.039476.ckpt`；SHA256 `9b18509de2a21f50d8fad41b8c37695842fe2e92745dc779713b0ecac06e2107`。
- 原AE encoder核验：`/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt`。
- CAD根：`/mnt/d/data/deepcad_v7`；已知CAD输入VAE encoder，decoder不读取GT拓扑，后处理不读取GT或按GT挑候选。
- accepted清单：`checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt`，SHA256 `2864849e8240134c6033cc6cd64efa19337f67fea94b89a493690ae84355b7b3`。
- 原test清单：`src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt`，SHA256 `16352426a1878ee6836c3ddb306080a65313e92a5204a519e1386bdc59fc643a`。原审核、拒绝原因保留，不依据模型结果过滤。

# Outputs

本目录独立保存raw_predictions、两策略post_variants、metadata、report和validation核验记录。紧凑结果与图表入Git，原始几何、权重、临时日志及逐对象运行文件沿用gitignore。以前validation/test历史产物不覆盖。

# Processing Protocol

1. 核验权重、split、源码哈希及split交集。按validation协议重新生成canonical posterior_mean：seed42、FP32、TF32关闭、无额外decoder去重；8GPU不重叠分片，收集完整raw清单与缺失/错误。
2. 两策略读同一raw根，分别独立输出；均为optimized模式，drop_num=0，名义优化200步、既有自适应上限600步、CPU、每进程单线程、32 workers/策略、总上限180秒/件。
3. `legacy`保留原处理；`recover_v1`固定为legacy → B1 optimized → B1 fit-only → F1+B1 optimized，成功即停止。原有圆柱完整自然loop行为及所有阈值不改。
4. 保持validation采样协议：原生STEP每面1024点、每边128点，双向平方CD，无对齐/缩放；GT只用于事后测量。原生及兼容0.1合法性独立记录。
5. 收齐两侧全部2417个终态后汇总；不得中途根据结果挑版本、追加候选、改变预算或删除失败。报告validation与test的同口径提升，但不把两者混为一组样本。

# Metrics

主指标：原生读取后TopAbs_SOLID且BRepCheck通过，分别报告/2417与/2424、配对救回/退步。辅助：STEP覆盖、推理成功、超时/崩溃/缺失、几何可计算数、共同可计算集合及共同合法集合face/edge CD²的均值/中位数/P90、存储容差、时间均值/中位数/P90。可以汇总既定诊断中的成功路径数，但不据test具体失败样本形成规则。

# Decision Rule

只回答validation上观察到的提升是否也出现在test，以及成本和几何指标是否伴随退步。无论结果好坏都报告两侧固定全集与配对转移；不声称本轮原生合法率证明1e-3精度或拓扑完全正确。历史DeepCAD test曾参与研究，当前固定评估不改变这一事实。

# Edge Cases

失败、缺失、损坏、NaN/Inf、空输出、超时都留在分母；无GT几何或采样失败标未知，不移除合法性结果。不把缺失几何当改善、不把旧Compound当solid，不把通用runner未调度的legacy模型占位记录混入实际legacy后处理策略结果。

# Validation / Sanity Checks

复核checkpoint严格加载、encoder243张量一致、推理ID唯一与源/原始预测哈希；两策略的协议除策略和独立输出根外一致。核验生产源码和validation冻结版完全相同。报告工具若扩展，运行相关报告测试并验证原四策略汇总数值不变。全量结束后独立核对2417×2终态、分母、守恒与输出来源。

# Implementation Constraints

不改VAE、训练配置、OCC算法、容差、几何指标定义或split；不启动训练。只需既有工具执行与最小报告接口扩展。全程使用`/mnt/d/anaconda3/envs/torch/bin/python`与隔离工作区`/mnt/d/python_curve_eval`，保留用户其他工作区。

# Deliverables

中文test结果、与validation并列的合法率图、几何/耗时/覆盖表、固定协议及溯源核验。说明本轮仅评估，后续方案改动仍回到validation。
