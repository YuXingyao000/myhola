# Task

对 Encoder 边内容消息续训后的两份预先固定权重，分别完成与 conditioned 基线的完整失败溯源，并按同 ID 检查相对旧 step1089/14520 的变化。执行依据为上级[已冻结评估计划](../evaluation_20261016/PLAN.md)。本计划在任何诊断或新结果观察前固定。

# Assumptions

主结果是最后完整保存的 epoch78/step28677；补充结果是续训窗口 validation 最优 epoch75/step27588，后者不是全程 validation 最优。比较各 checkpoint 自己的 Encoder posterior mean，规范 CAD 来源和面顺序一致。固定 accepted 2417 件；原2424中的7件合同不支持另列。没有28k步原GAT结构对照，不能将训练时长变化归为结构因果。

# Inputs

- `last`：`../evaluation_20261016/last`；SHA256 `040c05a45dd7edae825f2e6dd8e2cb3489a418146f9a7b2122cc54d38070618c`，epoch78/28677。
- `resume_best`：`../evaluation_20261016/resume_best`；SHA256 `e1e579e7bc6bdf64b3c078d6b5414e33630222de989b180ca8f3eb22b59c6ccf`，epoch75/27588。
- 两权重均使用原run `weights/evaluated_snapshots/<SHA>.ckpt` 独立归档；候选方法 `encoder_edge_value`。
- 历史conditioned评估/诊断：`checkpoints/intersection_refinement/curve_conditioning/{evaluation_20261014,failure_census_20261014}`，只读核验复用。
- 历史step1089/14520诊断：`../failure_census_20261015` 与其 `endpoint_14520_comparison`，用于同ID时间比较，不重标旧方法/组别。
- GT `/mnt/d/data/deepcad_v7`；split `checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt`，SHA256 `2864849e8240134c6033cc6cd64efa19337f67fea94b89a493690ae84355b7b3`。

# Outputs

`last/` 与 `resume_best/` 各自保存 manifest/cohorts/来源哈希、全部失败+救回双侧轨迹、缺失baseline控制、全部fit成功而optimized失败路径、四中文PNG/SVG、紧凑统计/报告/NOTE/自包含离线页/独立复核与浏览器核验。根目录时间比较记录全2417配对成败、共同轨迹覆盖及未知，不改旧目录。

# Processing Protocol

1. 各组 raw/fit/optimized 全2417件终态且身份、数据/latent SHA、预算通过核验后冻结manifest。各自posterior mean允许不同，合同必须为 `same_geometry_source`。
2. 范围为候选全部optimized失败和全部救回；每个ID同侧baseline核验原输入/协议/方法/资产后只读复用，不合格或缺失则在本组baseline_controls补跑。
3. 公共科学源码与昨日SHA一致后冻结；不改生产post、阈值、采样或测量。每runner8单线程进程、每组最多4runner=32核；与主post总量不超过128核，两个census合计最多64核。
4. 每非空cohort第一个ID和最多两个缺失baseline控制做真实smoke；需复现原终态、STEP DATA且无观测错误，才同源码resume全集。upstream等待provenance，saved STEP独立。
5. 所有fit-only合法而optimized失败ID，比较同raw完整两条路径及选择后face/edge/connectivity数组；不单跑末端、不把后续路径差异归因于单项loss。
6. 时间比较从全2417终态表重新划分旧→新成败，只在两时点可用的共同ID轨迹比较观察；缺旧轨迹明确未知，不新增全量旧轨迹，不比较不同失败集百分比来宣称机制改善。
7. 完成后核验来源资产未变，独立重算观察/配对转换与最终失败面统计，再生成四图和离线展示。

# Metrics

归一化长度阈值 .002/.006/.01/.08，每观察正/负/未知及ID；连接多漏、孪生、端点、优化漂移、拟合、边面贴合、wire、裁剪、缝合和保存。多标签观察与互斥末端分列，最终失败面与对象分母分列。保留独立拟合/周期替换连续残差及同ID变化；平方CD或采样阈值不能等同CAD tolerance。

# Decision Rule

固定全集solid和原始几何由主评估决定实际收益；诊断解释发生在哪一阶段。高频贴面告警同样可能见于成功对象，不能直接称唯一原因。两新权重均完整报告，不按test挑选赢家，本轮不启动训练或更改模型。

# Edge Cases

方法/合同/SHA冲突、重复ID、未知成败拒绝冻结。坏文件、失败、NaN/Inf、超时和轨迹不一致保留原分母并记未知。空组显式0；无可检查候选不是检查通过；闭环不等于可裁剪；旧诊断未覆盖不等于无告警。

# Validation / Sanity Checks

公共科学工具复用昨日已通过35项合同回归版本，逐文件SHA确认，不将历史测试写成新执行。每组真实smoke后全集；独立脚本重算28观察×4阈值、paired转换、原STEP DATA、fit选择数组及来源资产；普通/离线浏览器各遍历4组×4阈值、ID详情、四图、390px布局与无JS错误，离线屏蔽HTTP/HTTPS。

# Implementation Constraints

仅新增本目录薄调度、展示身份、时间汇总；公共科学工具和生产post不改。旧产物只读，新增source_freeze记录本计划与上级计划SHA；源改变拒绝静默resume。权重/几何/重放/完整页面payload不提交；紧凑报告、审计和科研图由root统一提交。

# Deliverables

两组完整失败统计/四中文图/离线页、全部fit退步路径、独立和浏览器核验、来源/未知清单及有覆盖说明的同ID时间变化；向root提供可用于研究判断的结果与边界。
