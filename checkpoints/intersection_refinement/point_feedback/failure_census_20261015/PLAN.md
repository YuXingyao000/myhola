# Task

对逐点曲线反馈（`point_feedback`）与直接初始化基线曲线条件调制（`conditioned`）做全集失败溯源。沿用四组中文科研图、四组×四阈值筛选、逐 ID 双侧详情及自包含离线页面。整体评估、checkpoint 选择和预算见 [本轮评估计划](../evaluation_20261015/PLAN.md)。此文是运行协议，不预填结果。

# Assumptions

固定 accepted 2417 件；原 test 2424 件中的 7 件合同不支持另列。主分母是本轮 `point_feedback` optimized 全部非合法 solid，所有救回另列对照。本轮 baseline 严格是 `conditioned`，不将原 `intersection` 诊断重命名混入；原 refinement 的三级比较由评估报告单独呈现。高频观察、多标签共现、最早异常和最终阻断分开，不把标签当作唯一根因。

# Inputs

- 本轮评估：`/mnt/d/python_curve_eval/checkpoints/intersection_refinement/point_feedback/evaluation_20261015`。
- baseline 评估：`/mnt/d/python_curve_eval/checkpoints/intersection_refinement/curve_conditioning/evaluation_20261014`。
- baseline 历史诊断：`/mnt/d/python_curve_eval/checkpoints/intersection_refinement/curve_conditioning/failure_census_20261014`，只读核验复用。
- GT：`/mnt/d/data/deepcad_v7`。
- split：`/mnt/d/python_curve_eval/checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt`，SHA256 `2864849e8240134c6033cc6cd64efa19337f67fea94b89a493690ae84355b7b3`。
- 新产物只写 `/mnt/d/python_curve_eval`，不动 `/mnt/d/python` 的其他任务。

# Outputs

冻结 manifest、来源哈希、cohort；候选全部失败加全部救回的 provenance/upstream/saved STEP；相同 ID 的 baseline 复用或补跑；候选全部 fit-only 合法但 optimized 失败的完整路径对照；紧凑统计、逐 ID CSV、中文报告与研究笔记、四组 PNG/SVG、交互页及离线 HTML、浏览器和独立复核记录。

# Processing Protocol

1. 等同一 2417 ID 的 raw、fit-only、optimized 全部终态，核对 `conditioned`/`point_feedback` 身份、输入 latent、bool、算法、预算和历史来源。缺文件不缩分母。
2. 冻结所有候选失败和救回。对相同 baseline ID 逐一核验历史 manifest、method、资产、协议和诊断指纹；不符合复用条件的在本轮 `baseline_controls` 重放。历史路径和 method 原样保留。
3. CPU、每进程 1 线程、drop_num=0、名义200/自适应600优化步、每件180秒、每 runner8并发、最多4个 runner并行。同侧 upstream 等 provenance 完成，saved STEP 可独立执行。
4. 候选全部 fit-only 成功而 optimized 失败的 ID 在 `fit_only_counterfactual` 跑完整路径，核对同 raw 和选择后 face/edge/connectivity 数组；空集合显式记录 0。
5. 少量真实 smoke 必须确认原终态、STEP DATA 完全一致且没有观测工具错误，再用同一冻结源码 resume 全集。缺失、超时和不一致记未知，仍保留原失败分母。
6. 最终复核来源资产未变化；报告沿用多标签、paired 转换、独立拟合/周期替换贴合、最终失败面候选检查四图。离线打包只改展示，不重算测量。

# Metrics

阈值保持归一化 XYZ 长度 .002/.006/.01/.08；每项正/负/未知及 ID。连接多漏、孪生差异、选边和端点、后处理优化漂移、独立拟合、边面贴合、wire/裁剪/缝合/保存；同时保留 paired 连续量。面级错误按最终失败 decoder 面去重，对象和面分母分开。

# Decision Rule

先看固定全集合法 solid 及原始几何，再以完整失败/退步/救回和路径对照解释变化。合法对象也可能具备几何告警；成功对照用于辨别告警是否充分。训练曝光与架构同时变化，不能把全部差异因果归于新增逐点分支；不外推为端到端照片重建结论。

# Edge Cases

错 method、重复 ID、未知 bool、协议/hash 冲突拒绝冻结。坏文件、NaN/Inf、诊断失败/超时或重放不一致保留未知，不改原成功失败。空 cohort 显式无数据；无候选可检查不算通过；近边不等于应焊合，闭环不等于可裁剪。

# Validation / Sanity Checks

CPU 回归覆盖旧默认、conditioned baseline、新 candidate、混方法拒绝和 cohort 守恒。冻结 runner 依赖后执行真实 STEP DATA smoke，再跑全集。最终独立核对全部 ID、双侧来源、主/对照分母；普通和离线浏览器遍历四组四阈值、ID 详情、资源、390px布局和 JS 错误。离线禁止 HTTP/HTTPS，并确认四图可解码。

# Implementation Constraints

仅最小泛化 method/baseline 选择及中文展示；不改模型、loss、生产 post、采样参数和测量判据。保留旧默认，历史记录只读。正式 smoke 前冻结 `failure_census_contract`、科学 runner 及其依赖；不能修改后静默 resume。展示源码可单独修订并记录。不开新3D图库。

# Deliverables

完整双侧失败诊断、优化退步路径对照、报告和研究判断、旧格式四组 PNG/SVG、四组×四阈值交互及离线 HTML、来源审计、未知清单、可复现命令和浏览器验证。未完成前不以训练 loss 或 smoke 宣称提升。
