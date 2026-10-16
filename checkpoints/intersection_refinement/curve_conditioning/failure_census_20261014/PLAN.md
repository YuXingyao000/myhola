# Task

对训练完成的最终曲线条件调制（`conditioned`）与原 Intersection refinement（`intersection`）做全集失败溯源。沿用上一轮四组中文科研图、四组×四阈值交互筛选、逐 ID 双侧详情和自包含离线页面；不以抽样代替全集。总体评估协议见 [评估计划](../evaluation_20261014/PLAN.md)。当前为执行计划，不提前填入结果。

# Assumptions

固定 accepted 2417 件，原 test 2424 件中另列 7 件合同不支持。主失败分母为本轮 `conditioned` optimized 全部非合法 solid；所有救回单列对照。baseline 为已评估的原 refinement，cohort 只由本轮两侧终态决定。阶段多标签表示证据与共现，最终阻断与最早观察分开报告，不把观察强行改成唯一根因。

# Inputs

- 新评估：`../evaluation_20261014`；checkpoint、SHA256、验证选择与训练曝光以该目录冻结计划和实际元数据为准。
- 原 refinement 评估：`/mnt/d/python/checkpoints/intersection_refinement/evaluation_20261011`。
- 历史 baseline 诊断：`/mnt/d/python/checkpoints/intersection_refinement/failure_census_20261011`，只读复用经身份、算法和资产哈希审核通过的同 ID 结果。
- GT：`/mnt/d/data/deepcad_v7`；split：`/mnt/d/python/checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt`，SHA256 `2864849e8240134c6033cc6cd64efa19337f67fea94b89a493690ae84355b7b3`。
- 所有新代码和产物写入 `/mnt/d/python_curve_eval`，不改正在用于其他任务的 `/mnt/d/python` 工作区。

# Outputs

冻结 manifest、cohort、source 审计；本轮全部失败和救回的 trace/upstream/saved STEP 诊断；baseline 全部同 ID 对照；全部 fit-only 成功但 optimized 失败的完整路径对照；cases/summary、中文 REPORT/NOTE、四组 PNG/SVG、交互页面、自包含 HTML 和独立复核记录。

# Processing Protocol

1. 等同一 2417 ID 的 raw、fit-only、optimized 全部终态后冻结 manifest，严格核对 method、输入 latent、bool、生产算法、预算与 baseline 来源。不按现有目录缩小分母。
2. 本轮诊断集合为全部失败加全部救回；baseline 相同 ID 审计历史诊断复用，缺失或不可复用的在 `baseline_controls` 重放。保持历史 method、manifest、协议、来源哈希，不重命名冒充新结果。
3. 保持 CPU、每进程 1 线程、drop_num=0、名义200/自适应600优化步、每件180秒；每 runner8并发。provenance 全部结束后运行同侧 upstream，saved STEP 可独立并行。同 root 同 runner 不重复并发。
4. 全部本轮 fit-only 成功且 optimized 失败的 ID 追加 fit-only 重放；核对同 raw 及选择后面、边、连接数组。零件则显式0，跳过空 runner。
5. 原终态与 STEP DATA 一致、没有观测工具异常时才采用阶段轨迹。缺测、超时和不一致保留未知及原失败分母；最后复核所有来源资产未变化。
6. 四图沿用多标签、成对变化、独立拟合/周期替换贴合、最终失败面候选检查。页面覆盖四组×四阈值、双侧 ID 详情；静态图固定全集分母，筛选只改变交互统计。自包含 HTML 内嵌全部核心数据与四图。

# Metrics

保持归一化长度阈值 .002/.006/.01/.08；各标签正/负/未知及 ID；连接多漏、半边选择、端点、优化漂移、独立拟合、边面贴合、wire/裁剪/缝合/保存；最早异常证据及最终阻断；同 ID 成对连续值与标签变化。最终失败面同错误跨容差去重，对象和面分母分开。

# Decision Rule

先比较固定全集合法 solid 与原始几何质量，再用全部失败、退步、救回及路径对照解释变化。超阈值和 OCC 错误码不是唯一根因；成功对照用于验证观察不等于充分失败条件。额外微调曝光与结构同时变化，不把全部收益因果归于条件调制；结果仅属于此 VAE 评估。

# Edge Cases

错 method、重复 ID、未知 bool、协议或哈希冲突拒绝冻结/汇总；坏文件、缺文件、NaN、超时和重放不一致显式记未知，不按缺诊断改成功/失败。零 cohort 明确无数据；近边不证明应焊合，丢 wire 不证明丢孔，没有候选可检查不算通过。

# Validation / Sanity Checks

方法扩展先做 CPU 回归；首次少量真实重放核对 STEP DATA 与工具状态后再以相同源码 resume 全集。独立检查 cohort 守恒、双侧全部判据与来源资产；普通/离线浏览器覆盖四组四阈值、ID详情、资源、390px窄屏与 JS 错误。离线版禁止 HTTP/HTTPS 请求并确认四张内嵌图可解码。

# Implementation Constraints

仅最小扩展既有 method 显式入口与展示标签，不改模型、loss、生产 post、失败判据或测量参数。历史 local 默认行为保持。正式 smoke 前冻结 runner 及其依赖源码；展示打包可独立调整。新结果全部在本目录，不覆盖历史文件，不新建3D图库。

# Deliverables

全集失败/退步/救回及优化退步路径对照，中文报告、四组 PNG/SVG、交互页、自包含 HTML、来源审计、未知清单、复现命令和独立验证。结果未完成前不以训练 loss 或 smoke 宣称模型改善。
