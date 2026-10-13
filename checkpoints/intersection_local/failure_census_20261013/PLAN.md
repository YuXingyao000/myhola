# Task

对本轮完整容量local与上一版refinement做全集失败溯源，沿用四组中文图、四分组/四阈值筛选和同ID双侧详情。总体协议见 [评估计划](../evaluation_20261013/PLAN.md)。

# Assumptions

主分母是本轮local optimized失败全集；救回单列对照。baseline=intersection，candidate=local，cohort从本轮原始终态派生。历史2层local不作为本轮输出；原资产只读。多标签是阶段证据和共现，不强行分唯一根因。

# Inputs

新评估 `../evaluation_20261013`；baseline `../../intersection_refinement/evaluation_20261011`；历史baseline诊断 `../../intersection_refinement/failure_census_20261011`。GT `/mnt/d/data/deepcad_v7`；accepted2417/原2424/合同不支持7，split SHA256 `2864849e8240134c6033cc6cd64efa19337f67fea94b89a493690ae84355b7b3`。本轮权重固定epoch42/step15609，SHA256 `c16d9ebe829659f3f4b8eb631e115fb10d03a28e9319bdb6bb5968b8d4bdb8f3`。

# Outputs

冻结manifest/cohort/source审计，当前全失败与救回的trace/upstream/saved_step、baseline同ID对照、全部fit-only成功但optimized失败的路径对照、cases/summary、REPORT/NOTE、四组PNG/SVG、可筛选页面与自包含HTML、独立复核。

# Processing Protocol

1. 等相同2417ID的raw、fit-only、optimized全部终态后冻结manifest，严格校验身份/bool/输入/算法/预算，不按已完成目录缩分母。
2. 当前诊断集合为全部失败及全部救回；baseline相同ID审计原诊断复用，缺少的在baseline_controls完整重放。10/12新增baseline controls若复用需保持原method及原manifest/协议/asset哈希，不把不同模型结果混入。
3. 固定CPU、单线程、seed、drop_num0、名义200/自适应600、每件180秒，每runner8并发；原post决策不变。provenance先结束再跑同侧upstream，saved_step可独立并行，同root同runner不得重复并发。
4. 全部本版fit-only成功而optimized失败的ID追加fit-only重放；比较raw及选择后面/边/连接。零个则显式0，不造空运行。
5. 原终态和STEP DATA一致且无instrumentation错误才归因；不可测/超时/偏离保留未知及原失败分母，最终重查所有源资产。
6. 四图为多标签、成对变化、独立拟合/周期替换贴合、最终失败面候选检查；页面四组×四阈值及ID双侧详情与机器统计精确一致。额外自包含HTML不依赖本地HTTP才能显示核心结果。

# Metrics

.002/.006/.01/.08归一化长度阈值，每标签正/负/未知及ID；连接多漏、半边选择、端点、几何优化漂移、拟合、边面贴合、wire/裁剪/缝合/保存。最终失败面同种错误跨容差去重；对象与面分母分开。成对连续值与标签变化；最早异常证据与最终阻断分开。

# Decision Rule

结合反复阻断及同ID两版变化解释几何/拓扑问题，成功对照用于判断超阈值并非充分失败条件。不把额外训练曝光收益全归因于结构，不把OCC码当唯一根因。

# Edge Cases

坏/缺文件、NaN、重复ID、未知bool、错method、协议/哈希冲突、超时、重放不一致显式处理。协议冲突拒绝汇总；终态失败留分母、阶段未知不当改善。零cohort明确无数据，正常互反边不当重复，近边不证明应焊合，丢wire不证明丢孔。

# Validation / Sanity Checks

首次少量真实重放核验STEP DATA与观测工具状态，然后全集运行。独立核对cohort守恒、全部判据/paired计数及来源未改；浏览器遍历四组四阈值、逐ID详情、所有资源、窄屏、JS错误，并验证自包含HTML不依赖data.js或HTTP加载核心数据。

# Implementation Constraints

复用当前测量工具，不改模型、loss、生产post或判据。runner及依赖源码在正式smoke前冻结；展示适配与诊断算法分开。全部新产物在本目录；不覆盖历史结果或别人的工作。标准绘图工具，不新建3D图库。

# Deliverables

本轮全集失败、退步、救回及路径对照；中文报告、四组PNG/SVG、自包含HTML及交互页、来源审计、未知清单、复现命令、独立核验。当前未确认计数不提前填入最终报告。
