# Task

评估本轮 Encoder 边内容消息与原 GAT 解冻对照，沿用固定2,417件、原始/拟合/优化后指标及完整失败图表，解释与保留的 conditioning 模型相比的变化。用户已停止训练并授权分析；随后更正曲线判断并授权：实际结果有提升则继续训练。先完成固定评估，再依据结果执行这一条件；评估期间不修改模型或生产后处理。

# Assumptions

- 新 GAT candidate 已完成14,520步；原 GAT control被中断，最后完整保存1,815步。两者的验证最优均为epoch2/step1,089，且均是共同1,815步验证窗口中的最优，因此主比较有相同checkpoint训练曝光；完整运行预算不同另列。
- 两组都解冻Encoder/posterior/decoder并恢复KL，历史conditioning冻结Encoder，直接比较总validation loss会混入目标差异。单列total、weighted KL和total−weighted KL。
- Encoder已改变，必须对同一CAD分别编码取得各自posterior mean。旧latent replay不能标成新Encoder重建，也不用于本轮照片/diffusion结论。
- 全集2424中的7件合同不支持保持单列；不按结果删样本。

# Inputs

- candidate方法`encoder_edge_value`：`../candidate/20261015T073112.923985Z/weights/epoch=0002-step=00001089-val_loss=0.040878.ckpt`，SHA256 `d9e911e23a5dfc0e45fb90735a806706c9c89329898f0873df06927c8d2fb62d`。
- control方法`encoder_gat_control`：`../control/20261015T073112.923985Z/weights/epoch=0002-step=00001089-val_loss=0.040526.ckpt`，SHA256 `f1de7f11fee749e51921e0fb45d32ae7b6b7d7899316b248560bab4501d6c279`。
- 历史baseline方法`conditioned`：`checkpoints/intersection_refinement/curve_conditioning/evaluation_20261014`和相邻`failure_census_20261014`，仅只读核验复用。模型源SHA `4c7910b8169206c74fcf38c358cf171359149e0429c8debbc227d391aa56613f`。
- GT `/mnt/d/data/deepcad_v7`；accepted split `checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt`，SHA256 `2864849e8240134c6033cc6cd64efa19337f67fea94b89a493690ae84355b7b3`。
- 运行与W&B离线数值history、checkpoint callback/state/config，源commit `0b145d5d`。

# Outputs

本目录存训练审计、三方总表/原始分布/图表/独立复核。`candidate/`与`control/`分别是每组对历史conditioned的完整固定评估根，保存各自raw/post/metadata；三方或candidate-control配对由这些逐ID结果直接合并，不重复后处理。主失败统计放相邻`../failure_census_20261015`，原GAT对照诊断可放其`control_comparison/`，各自保留身份。离线HTML直接可打开。

# Processing Protocol

1. 在任何test推理前按validation确认上述两个固定checkpoint，不根据test换模型。审计实际保存步数、共同窗口、loss定义、权重变化/新分支和配置。
2. 用各自checkpoint配置严格加载为frozen_inference，FP32、TF32 off、seed42。每个CAD使用既有`load_identity_geometry`原始全局六通道采样/GT connectivity编码，posterior取mean，decoder只有latent和face数；阈值p>0.5，不额外去重或GT筛边。
3. 显式推理输入模式`checkpoint_posterior`，只允许posterior_mean。逐ID保存并核验同一data.npz SHA、规范face顺序、实际float32 latent及其SHA；与baseline比较要求`same_geometry_source`，不要求不同Encoder的latent相同。历史`same_latents`合同继续严格校验，不能静默放宽。
4. 8GPU独立推理分片；先真实smoke再全集。baseline raw/post严格核验来源、GT输入、split及原生产算法，只读复用，不改历史method标签。
5. 每个新方法分别fit_only/optimized：CPU单线程workers32，每件包括测量180秒，drop_num=0，名义优化200/原算法自适应600。分批调度控制CPU总并发。两组各2417条终态后固定分母汇总。
6. 冻结候选所有失败+救回与baseline同ID，全部做双侧诊断。历史合格baseline记录复用，缺失补跑。候选fit-only合法但optimized失败全部追加路径对照。对照同样保留完整fail统计和与candidate共同ID的阶段比较。
7. 沿用.002/.006/.01/.08阈值、正/负/未知、四组中文图及4cohort×4阈值离线筛选。观测重放须核对原终态与STEP DATA；高频标签、互斥末端阻断和因果假说分开。

# Metrics

主分母2417，另报2424覆盖；导出/STEP/原生合法solid/缺失/超时。raw面/边双向平方CD、对应面XYZ MAE、邻接整件全对率、均值/中位数/P90/P95/P99、逐ID改善率、尾部贡献；按owner pair检查边误差，不能仅看全边合并CD。post分别报可计算全集与共同可计算集合。fail报告面连接、孪生、端点、贴面、拟合、wire/裁剪/缝合/保存与未知。训练曲线仅作学习过程诊断，不是solid判据。

# Decision Rule

主结构比较是同epoch2/1089步的candidate对control；与conditioning对比回答全量微调后是否值得保留。缺少等参数量无edge输入MLP消融，不把差异全部因果归于边内容。改善平均loss但损害尾部/solid不能称成功。validation总loss还包含移动的特征对齐目标，不能把总loss上升直接解释为几何退化；本轮checkpoint选择保持冻结，不按test择优。若全集solid与原始几何支持相对保留基线有实际提升，按用户最新授权继续训练；明确记录恢复的checkpoint、已完成步数与新增预算，保持结构、学习率和loss配置。只在一个指标上有改善而主要有效性/误差尾部明显恶化，不视为满足继续条件。

# Edge Cases

错method、错输入合同、来源冲突、重复ID或错误latent摘要拒绝汇总。坏文件/NaN/空几何/缺失/超时保留分母；未知不当阴性；测量超时不抹去先前已验证的solid。baseline历史文件不覆盖；未经记录不重试筛成功。空cohort明确0。

# Validation / Sanity Checks

新推理/metrics输入模式有针对性回归：旧默认不变、同CAD不同latent合法、错CAD/坏latent摘要拒绝、新method与checkpoint结构匹配、mean由当前Encoder产生。真实smoke后核验所有source/checkpoint/split及模型/生产post源码未改。最后独立复算ID/配对守恒/指标/图表；普通与离线浏览器核验四图、cohort阈值/详情、390px无横向溢出、无JS或离线HTTP依赖。

# Implementation Constraints

只最小扩展既有评估方法与显式输入合同，不改训练、模型、loss、post或既有指标公式。代码/文档在当前worktree，其他历史目录只读。新产物放checkpoints，raw/weights/logs/大型轨迹不提交；紧凑报告受Git管理。依据spec-research-data-code流程在实施前固定此合同。

# Deliverables

训练审计与曲线、两模型完整2417原始/后处理评估、三方配对结论、全失败统计与中文4图/离线页、原因分析、复现命令与独立核验。实际结果全部终态后才作效果判断。

## 最新用户授权后的终点补充（在终点推理前固定）

用户更正曲线判断并要求“有一点提升就接着训练”。由于 validation 总loss含移动的 edge-feature 对齐目标，早期总loss最优不能代表最后保存状态；另评估候选 epoch39/step14,520 的 `last.ckpt`，SHA256 `e9120c78254d9b94aaaef88333c0f0656fd6a01346225d092fb8341297bff2bf`。只按“最后完整保存”选此权重，在读取其test结果前固定，不扫描其他checkpoint或按test挑选。

补充目录 `endpoint_14520/candidate/`，入口 `run_endpoint.py`；同2417 ID、同自有posterior mean、同raw指标与两种post协议。该组用于判断继续训练所接上的状态；不替代上述1,089步的公平结构比较，不与1,815步的control称作同预算对照。先等主推理释放GPU再推理；其post最多64进程，仅在主post全部完成后运行，可与64进程的主失败诊断并行。报告终点与conditioned/早期候选的固定逐ID变化及失败分母，候选与对照完整失败统计范围保持不变。
