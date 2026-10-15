# Task

对 Encoder 边内容消息与原 GAT 全量解冻两组分别做固定全集失败溯源，各自对照只读历史 `conditioned`；给出相同 ID 的阶段证据及中文四图/离线页面。整体 checkpoint 与选择协议见 [评估计划](../evaluation_20261015/PLAN.md)。本文件先冻结协议，不预填结果。

# Assumptions

固定 accepted 2,417 件，原始 2,424 件中的 7 件合同不支持单列。两组均按 validation 选择 epoch2/step1089；主架构比较为相同曝光的 `encoder_edge_value` 对 `encoder_gat_control`。各组相对 `conditioned` 都包含 Encoder/posterior 解冻和 KL 恢复等变化，不能将差异全部归因于边内容 MLP。GT 面顺序保持，但各自 posterior mean 不要求相同。

# Inputs

- 新值消息评估：`../evaluation_20261015/candidate`；原 GAT 对照评估：`../evaluation_20261015/control`。
- 历史评估：`checkpoints/intersection_refinement/curve_conditioning/evaluation_20261014`。
- 历史诊断：`checkpoints/intersection_refinement/curve_conditioning/failure_census_20261014`，只读核验复用。
- GT：`/mnt/d/data/deepcad_v7`；split：`checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt`，SHA256 `2864849e8240134c6033cc6cd64efa19337f67fea94b89a493690ae84355b7b3`。
- 新合同 `same_geometry_source`：每行 `input_source_sha256`、`actual_gt_source_sha256`、`input_contract`，以及各方法各自真实 `input_latent_sha256`；不把不同 Encoder 产生的 latent 改成相同。

# Outputs

本目录保存新值消息诊断；`control_comparison/` 保存原 GAT 对照诊断。各自冻结 manifest/cohort/来源哈希，候选全部失败和全部救回的 provenance/upstream/saved STEP、同 ID baseline 复用或补跑、所有 fit-only 合法但 optimized 失败的完整路径对照、紧凑统计/CSV/中文报告、四组 PNG/SVG、交互及自包含离线 HTML、浏览器/独立复核。

# Processing Protocol

1. 各组 raw、fit-only、optimized 的 2,417 ID 全部终态后才冻结；显式核对方法、同源 CAD、GT 面顺序及各自 latent 哈希。历史 `identical_latents` 默认合同保持严格，新增合同不能静默绕过。
2. 诊断范围为各组全部 optimized 失败和全部救回；同一 baseline ID 核验历史资产/协议/方法/诊断指纹，合格只读复用，缺失在本次 `baseline_controls` 补跑。重复选择的 baseline 不改写历史 method。
3. 生产 post 与测量公式不变；CPU、每进程1线程、drop_num=0、名义200/自适应600优化步、每件180秒、每 runner8并发、每 census最多4个 runner，总诊断最多64并发；与主后处理合计不超过128。
4. 每侧 upstream 等 provenance 完成，saved STEP 独立；所有 fit-only 成功但 optimized 失败 ID 在 `fit_only_counterfactual` 比较同 raw 与选择后 face/edge/connectivity 数组。
5. 科学源码在测试后/真实 smoke 前冻结。每个非空cohort选一个候选，加最多两个缺失 baseline 控制做 smoke，必须复现原终态、STEP DATA且无观测错误；随后同源码 resume 全集。重放未知不改变原成败分母。
6. 完成后复核来源未改；多标签/paired转换/独立拟合与周期替换贴合/最终失败面候选检查四图，另列互斥末端阻断。离线打包只处理展示，不重算测量。

# Metrics

阈值为归一化 XYZ 长度 .002/.006/.01/.08，每项正/负/未知和对应 ID；连接多漏、孪生差异、选边端点、后处理优化漂移、拟合、边面贴合、wire/裁剪/缝合/保存。对象和最终失败面分母分开，保留连续量及同 ID 变化。不能把平方 CD、六通道 local L1 或超阈值标签直接解释成 CAD tolerance/唯一原因。

# Decision Rule

固定全集 solid 与原始几何先决定实际效果；完整失败/退步/救回阶段证据解释机制。告警也可能出现在成功件，不能按最高频标签推出因果；主架构判断需要同时参考新值消息与原 GAT 同曝光结果。无照片或旧 diffusion 兼容结论。

# Edge Cases

错方法/输入合同、重复 ID、未知 bool、来源哈希冲突拒绝冻结；坏文件、NaN/Inf、失败/超时和不一致保留未知，不删样本、不扩大重试择优。空 cohort 显式0；没有候选可检查不是检查通过；闭环不等于可裁剪，贴面也不保证嵌套/同心关系。

# Validation / Sanity Checks

回归覆盖旧默认同 latent 合同、新同 CAD 不同 latent 合法、错 CAD/缺来源/错合同拒绝、方法身份与 cohort 守恒、当前实际 GT SHA核对。正式 smoke 前冻结源码；最终逐 ID/双侧来源/分母复核，普通和离线页面遍历4组×4阈值、观察转换、ID详情、四图解码、390px无横向溢出/无 JS错误；离线屏蔽HTTP/HTTPS。

# Implementation Constraints

只扩展既有 method 白名单、显式输入合同、中文标签和新目录调度/打包；不改模型、loss、生产后处理、测量采样或阈值。旧记录只读，旧默认不变。源码改变不能接旧 frozen run 静默 resume；展示变更单独记录，不新开3D图库。

# Deliverables

两组完整成对失败诊断与反事实路径、报告/解释边界、四张中文统计图/离线页面、来源审计/未知清单/复现命令、针对性测试与浏览器核验。未完成前不以 training/validation loss 或 smoke 宣称有效。
