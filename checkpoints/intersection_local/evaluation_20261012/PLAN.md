# Task

完整测试训练后的局部注意力 intersection，与上一版 intersection refinement 做相同输入和后处理的逐对象比较，并对失败、救回、退步案例做全集溯源和中文图表页面。用户授权本轮评估；不再训练、不改模型或生产后处理、不按测试结果挑选权重。

# Assumptions

- 数据：沿用已审计 accepted test 2417件，原始2424件中7件合同不支持样本单列；固定split SHA256 `2864849e8240134c6033cc6cd64efa19337f67fea94b89a493690ae84355b7b3`。
- 边界：主要对照是上一版 refinement，而不是原HoLa。方法机器标识固定 `intersection`（上一版）与 `local`（本版），图中明确中文名称。原HoLa只用于既有latent和encoder审计的来源核验。
- 实施：所有原运行、checkpoint、预测、STEP与诊断只读。复用必须通过协议、算法和资产哈希核对；源产物不冒称本次重算。
- 用户观察末期可能过拟合；训练曲线审计与下游结果分别报告，不预设该判断成立。新模型有额外训练曝光，本次不是等预算架构消融。

# Inputs

- Local run：`checkpoints/intersection_local/frozen/20261012T073904.575690Z`。按val/loss最小与checkpoint callback固定 `weights/epoch=0019-step=00007260-val_loss=0.507484.ckpt`；读取真实epoch、step、回调、配置、哈希与训练历史后才能开始推理。last仅用于训练状态核对，不做test选优。
- 上一版：`checkpoints/intersection_refinement/evaluation_20261011`，其已评估权重epoch10/step3993（val_loss=0.039476）；raw和两种post保持来源。
- 相同输入latent：`checkpoints/joint_boundary/comparison_20261011/raw_predictions/{posterior_mean,archived_latents}/legacy/{id}/data.npz`。原checkpoint `/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt` 只用于encoder/cache核验。
- GT `/mnt/d/data/deepcad_v7`；split `checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt`。
- 已有refinement阶段诊断：`checkpoints/intersection_refinement/failure_census_20261011`，其中411个主侧案例可按逐ID审计后复用，缺少的旧成功对照另行重放。

# Outputs

- 本目录：training_audit、固定协议、来源哈希、两种输入模式逐对象raw预测／指标、posterior mean的fit_only和optimized后处理／原生STEP几何、配对成功转换、中文对比报告与PNG/SVG统计图。
- `checkpoints/intersection_local/failure_census_20261012`：冻结诊断清单、来源审计、阶段重放／上游测量／原STEP检查、多标签原因、分组与成对统计、中文NOTE/REPORT、四组静态图和可筛选页面。
- 大文件、权重和日志留既有ignored目录；工具、文档与紧凑验证结果受版本管理。不混入历史结果或validation smoke权重。

# Processing Protocol

1. 核对训练已停止、val与train曲线、checkpoint元数据和encoder/posterior逐张量一致性，固定上述权重和SHA。val使用posterior sample；本次后验重建使用已保存mean，不能直接比较两个total loss。
2. 严格加载local，FP32，关闭TF32，seed42，阈值`p>0.5`。按ID分片并行，无sampler补齐；保留有向半边，不增加face去重、不输入GT拓扑。
3. 两种模式均使用原归档float32 latent，逐ID校验latent与来源SHA；均值重编码审计沿上一轮atol=1e-5、rtol=1e-5，不改变实际输入。上一版raw通过只读引用保留，方法仍标`intersection`。
4. raw几何／拓扑指标保持原公式。posterior mean计算对应面MAE和连接F1；archived latent没有GT面对应，不计算这两项，也不重新采样diffusion。
5. 对local的posterior mean执行fit_only和optimized。CPU、drop_num=0、名义200次／原算法自适应600次、逐ID固定seed、逐对象总超时180秒、32并发。上一版post需完整算法／采样／预算／split／raw哈希一致才只读复用，记录原时间与协议；STEP资产本次记录哈希，不能伪称有历史未记录的哈希证明。
6. 主有效性为原读取精度下TopAbs_SOLID+BRepCheck，容差0.1兼容口径单列。STEP存在、success marker、合法solid、几何可计算独立记录。汇总固定分母及共同可计算集合。
7. 由全体终态派生cohort：本版失败全集、持续失败、上一版成功本版失败、上一版失败本版成功；不硬编码上一轮304/262/42/107。新侧诊断所有失败与救回，旧侧审计复用已有案例并补齐其余同ID对照。
8. 诊断保持相同CPU seed／生产post决策／180秒上限，独立目录记录raw、半边选择、优化、独立拟合、周期替换、闭环、裁剪、缝合与保存重读；对全部本版fit-only成功但optimized失败件追加fit-only路径对照。
9. 只有原STEP DATA精确相等、无instrumentation错误且测量完整的重放用于原运行阶段归因。偏离／缺测保留未知，不改变原成功结论；统计使用.002/.006/.01/.08归一化距离阈值，非毫米。
10. 生成旧式中文四组图、cohort／阈值切换、具体ID展开和同ID成对详情。网页与统计JSON一致，页面不另造新3D案例图库；独立重计关键频次和来源保护。

# Metrics

原始面XYZ MAE、面／边平方CD、无向面连接P/R/F1、互反半边几何差异与单向pair；raw完整覆盖。fit_only／optimized的native solid分别/2417和/2424，缺失、失败、超时及阶段；原生STEP几何报告各自与共同可计算集合、均值和中位数。配对四类成功转换及ID。

失败统计为多标签的确认正例／阴性／未知：无可用闭环、环全部接纳仍裁不出面、返回面但丢边界、面齐仍未成solid、保存重读分歧；另记录连接多漏、半边选择丢正确pair、边面贴合、端点接近、拟合／优化漂移与自由边。最早异常、最终阻断和相关代理分开。救回／退步均使用相同对象成对对照。

# Decision Rule

依据固定输入的几何、连接、solid和成对机制变化判断当前模型价值。不能用val loss或单一距离超阈值替代质量／唯一根因，也不能把额外微调收益全归因于local attention。已有diffusion latent解码表现与完整照片条件重建能力分开。只按验证集选checkpoint，不做GT best-of-K或新增修复预算。

# Edge Cases

空预测、缺失／坏文件、非法shape／索引、NaN/Inf、输入哈希不一致、未知bool、不可读STEP、采样失败、超时、崩溃和诊断重放分歧均显式记录。缺失失败保留固定分母，不将不可算指标当零；已确认solid的后续采样超时不抹去solid成功。零分母显示不可用；正常互反边不算重复。薄壁近边不自动判应焊合，丢wire不自动判丢孔，未知不算改善。协议冲突停止该阶段并诊断，保留产物，不静默覆盖。

# Validation / Sanity Checks

先检查新方法参数化仍保持旧工具默认结果与测试通过；小批真实推理核对输入和输出合同。post复用前完整检查协议／raw哈希，少量阶段重放需验证STEP DATA一致再展开全集。全集结束独立核对2417身份、cohort守恒、分母、CSV/JSON指标、原资产哈希不变，抽核具体事件计数；浏览器检查cohort×阈值、ID展开、配对详情、资源、移动布局与JS错误。

# Implementation Constraints

Python3.10+；复用已验证模型、metric与OCC测量，不改生产post或损失。显式baseline/candidate身份与来源，旧legacy/joint/intersection默认入口兼容。多进程仅调度独立ID，限制CPU线程为1，GPU按分片分配。只启动授权的推理／后处理／诊断，不启动训练；不碰其他任务的未提交工作。图表使用标准绘图库，页面沿用此前中文统计交互。

# Deliverables

完整对比结果、失败全集与成对诊断、可复现命令、训练状态审计、中文结论、PNG/SVG图和可打开统计页面、精确判据与未知项、独立核验记录。任何未完成阶段必须明确标注，不用小样本冒充全集；最终结论可直接链接逐ID证据。
