# Task

对训练完成的局部注意力 intersection 做全集失败溯源，与上一版 intersection refinement 按同一ID比较；沿用中文四组静态图、分组/阈值切换和ID详情。总体协议见 [评估计划](../evaluation_20261012/PLAN.md)。

# Assumptions

主分母由新评估的local optimized失败全集决定，包含持续失败与退步；救回单列。baseline机器标识intersection、candidate为local，不重标历史诊断。全部原产物只读；主统计描述可核验的阶段阻断和共现，不宣称唯一根因。

# Inputs

新评估 `../evaluation_20261012`；上一版 `../../intersection_refinement/evaluation_20261011`；已有411件诊断 `../../intersection_refinement/failure_census_20261011`。固定GT `/mnt/d/data/deepcad_v7`，accepted test 2417/原2424/不支持7；split SHA256 `2864849e8240134c6033cc6cd64efa19337f67fea94b89a493690ae84355b7b3`。

# Outputs

冻结manifest、cohort IDs、协议/资产与复用审计；当前所有失败及救回的trace/upstream/saved_step，未覆盖baseline同ID对照；全部local fit-only成功optimized失败的路径对照；cases/summary、中文REPORT/NOTE、PNG/SVG和可筛选页面。

# Processing Protocol

1. 严格读取相同2417 ID的raw、fit-only和optimized终态；校验显式bool、输入latent一致和生产算法/预算/来源。
2. 主候选集合为local失败与intersection失败local成功的并集；cohort由原评估结果推导，不硬编码上轮数量。
3. 已有intersection诊断按ID审核原manifest、协议、raw/STEP/GT和诊断哈希；只读引用保留原指纹。未覆盖或不能复用的同ID对照在独立baseline_controls目录重放。
4. 相同CPU固定seed、drop_num=0、名义200/自适应600优化、单件180秒；测量不改变生产post。工具名legacy保留历史命名，内容由manifest显式method决定。
5. 全local fit-only成功optimized失败追加fit-only重放，比较原raw与选择数组；零个反事实案例显式输出零，不伪造空诊断。
6. 只有STEP DATA一致且无instrumentation错误的轨迹用于阶段判断；缺测/超时/偏离为未知且保留分母。源资产最终重复hash确认未改。
7. 四组图保留多标签、成对变化、边面贴合替换前后、最终失败面的检查错误；页面按当前失败/持续失败/退步/救回及四阈值精确重计。

# Metrics

.002/.006/.01/.08归一化长度阈值；每观察正/负/未知和具体ID。raw连接多漏、选择丢边、端点偏差、优化漂移、拟合误差、边面贴合、闭环/裁剪/缝合/保存阻断；最早已观察阶段与最终阻断分开。所有面级计数以最终失败decoder面去重；相同ID两侧连续值和标签转移。

# Decision Rule

根据反复阻断与成对变化提出研究判断；成功对照用于说明阈值异常不等于失败原因。不得把每个失败强归到互斥根因，或把额外微调收益全归于注意力结构。

# Edge Cases

缺失/坏文件、NaN、未知bool、错method、重复ID、协议/哈希冲突、超时和重放不一致显式处理。协议冲突拒绝汇总；终态失败留分母、阶段未知不算改善。零cohort显示0和无数据；历史资产缺失不冒充阴性。正常互反半边不当重复；近边不证明应焊合，丢wire不证明丢孔。

# Validation / Sanity Checks

测试method扩展的旧入口兼容、动态cohort守恒、未知传播、复用拒绝错来源、零对照/反事实、报告与CSV/JSON一致。首次真实少量重放核对STEP DATA与无instrumentation错误，再全集执行。浏览器遍历4组×4阈值、全部观察计数/ID和成对详情，检查JS错误、资源与窄屏。

# Implementation Constraints

仅新paired工具与case_method允许local；历史METHODS二元组保留以免破坏旧manifest。复用既有测量函数，不改模型、loss或生产post。旧结果不覆盖；所有新诊断写本目录。图表使用标准绘图库，无新增3D图库。

# Deliverables

全集失败、救回与退步的可复核统计；通俗中文报告与交互页面；四组PNG/SVG、路径对照、来源保护、验证记录及可复现命令。未确认项明确列出。
