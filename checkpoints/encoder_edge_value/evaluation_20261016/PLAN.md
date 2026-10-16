# Task

测试 Encoder edge-content GAT 续训后的实际权重，回答继续训练是否改善原始几何和合法 solid，并完成沿用既有格式的失败统计。用户本轮授权测试；不自动恢复训练或更改模型。遵循 spec-research-data-code，在任何新 test 推理前冻结本协议。

# Assumptions

- 观察事实：续训进程 exit 0，日志为 max_epochs=80 reached；最后完整 checkpoint 为 epoch78/28,677 steps，未达到计划29,040。续训loop计数原因独立审计，不把正常退出等同完成原预算。
- 在 test 前选择两份权重：主结果使用最后完整保存28,677；补充使用本次续训窗口 epoch40起 validation 总loss最小的27,588。全程 validation 最佳仍为旧1,089，沿用已完成结果，不称27,588为全程最佳，不按test重新选checkpoint。
- 同一规范CAD由各checkpoint自己的Encoder编码为posterior mean；这不是照片条件或旧diffusion/cache兼容性评估。
- 固定accepted 2,417件；原2,424件中的7件不支持合同继续单列，不按结果过滤。

# Inputs

- 运行目录 `/mnt/d/python_curve_eval/checkpoints/encoder_edge_value/candidate/20261015T073112.923985Z`。
- last：`weights/last.ckpt`，epoch78/28,677，SHA256 `040c05a45dd7edae825f2e6dd8e2cb3489a418146f9a7b2122cc54d38070618c`。
- resume_best：`weights/epoch=0075-step=00027588-val_loss=0.041263.ckpt`，epoch75/27,588，validation loss `0.04126333072781563`，SHA256 `e1e579e7bc6bdf64b3c078d6b5414e33630222de989b180ca8f3eb22b59c6ccf`。
- 推理使用同run `weights/evaluated_snapshots/<SHA>.ckpt` 的独立复制，复制后重核SHA；记录原路径映射，不改历史元数据。
- 旧参照：conditioning基线 `checkpoints/intersection_refinement/curve_conditioning/evaluation_20261014`；新GAT1,089与14,520分别为 `../evaluation_20261015/candidate`、`../evaluation_20261015/endpoint_14520/candidate`，只读复用并验证来源。
- GT `/mnt/d/data/deepcad_v7`；split `checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt`，SHA256 `2864849e8240134c6033cc6cd64efa19337f67fea94b89a493690ae84355b7b3`。
- 评估源码初始commit `32fdfa54`；模型、指标公式和生产后处理须与昨日核验版本一致。

# Outputs

本目录 `last/`、`resume_best/` 独立保存原始预测、原始指标、fit_only/optimized与调用元数据；训练审计、五快照逐件总表、分布/尾部/owner分析及图表在本目录。失败诊断放相邻 `../failure_census_20261016`，按两快照分组，分别生成四张中文图与自包含离线页面。历史产物不覆盖。

# Processing Protocol

1. 核验checkpoint、运行结束原因、完整validation history及配置。归档后固定两份SHA；同一新权重不因test结果替换。
2. 复用昨日已验证推理入口，严格加载当前checkpoint，`checkpoint_posterior`、FP32、TF32 off、seed42、posterior mean、p>0.5，不额外去重。Encoder可使用GT connectivity；Decoder仅获latent和面数，不获GT筛边。
3. 每份先两个固定ID真实smoke，再8 GPU独立分片全集；保存并核验逐件CAD来源SHA、当前float32 latent及其SHA、checkpoint/protocol。比较合同为 `same_geometry_source`，不是旧latent相同。
4. 原始指标每组16 CPU；post每模式32个单线程worker，每对象包含测量180秒、drop_num=0、原算法名义200/自适应600。同时最多96个post worker、所有任务合计不超过128核；与诊断协调。超时不擅自重跑筛选成功。
5. 固定全集后处理两模式完成后，各新快照对conditioning冻结全部失败+救回，同ID双侧诊断；合格旧baseline轨迹只读核验复用，缺失补跑；所有fit合法但optimized失败做完整路径对照。
6. 失败沿用.002/.006/.01/.08四阈值、正/负/未知、四组样本与四中文图。轨迹需复现原终态和STEP DATA；高频观察与互斥末端阻断分列。
7. 与14,520/1,089的阶段变化从全2417终态重新配对；仅在两侧轨迹可用的共同ID比较观察，报告覆盖。缺旧诊断不当阴性，不以不同失败集合百分比宣称改善。

# Metrics

固定分母的导出覆盖、STEP、原生TopAbs_SOLID且BRepCheck通过数、缺失/失败/超时；raw对应face XYZ MAE、face/pooled-edge双向平方CD、整件adjacency全对率、均值/中位数/P90/P95/P99、逐件改善率和top1%贡献。CD平方距离不称线性CAD tolerance。owner曲线比较采用五快照共同存在的GT有向pair，同时单列漏/多pair和覆盖率。post分列各自可计算与共同可计算集合。每个新快照对三个旧参照及last对resume_best报告逐ID成败与几何变化。

# Decision Rule

以固定全集solid及原始几何、尾部共同判断续训收益，不能只读总validation或edge均值。本次没有同28k步原GAT对照，不能把继续训练变化全部因果归于edge-content结构；两个新快照均完整报告，不按test选择新的正式模型。已有全程best保持其身份。后续训练/结构调整仅提出基于结果的判断，本轮不自行启动。

# Edge Cases

所有缺失、坏文件、空几何、NaN/Inf、测量超时保留预定分母并标状态；已验证solid后的测量超时不抹除有效性。错method/源SHA/latent摘要/重复ID/错误checkpoint拒绝有效比较。未知不作阴性；空cohort明确0。归档若已存在必须SHA相同；禁止覆盖历史结果。

# Validation / Sanity Checks

真实smoke检查shape/有限值/实际posterior和严格权重加载；全量逐件来源、split和终态核验；生产post源码11文件与原协议一致。独立重算五快照的数量、百分比、配对守恒与尾部贡献；抽取owner点数组独立计算。每组失败独立review，普通/离线browser检查4cohort×4阈值、详情、四图、390px布局、无JS错误、离线无外部请求。不因只改任务装配而重跑无关训练测试。

# Implementation Constraints

只新增任务级薄适配与必要汇总，复用已验证公共工具，不改模型、loss、trainer、生产后处理或公共指标定义。代码放当前worktree `/mnt/d/python_curve_eval`；历史原目录只读。训练文件不加入Git；紧凑报告、审计、重现入口和PNG科研图保留版本管理。

# Deliverables

两权重完整2417件raw/fit/optimized评估，五快照对比与训练结束审计，两组完整失败统计/中文四图/离线页面、独立复核记录、简洁结果说明与复现入口。实际测试全部终态后再判断本次效果。
