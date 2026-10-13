# Task

完整测试恢复容量后训练完成的局部注意力 intersection，与上一版 intersection refinement 在相同输入和后处理预算下比较；完成全集失败、救回及退步的逐件溯源和中文图表。用户已授权执行；不启动训练、不改模型或生产后处理、不按测试表现挑权重。

# Assumptions

- 数据：沿用 accepted test 2417件，原2424件中7件数据合同不支持样本单列；split SHA256 `2864849e8240134c6033cc6cd64efa19337f67fea94b89a493690ae84355b7b3`。
- 比较边界：主要baseline仍为上一版refinement（机器名 `intersection`），当前完整容量版为 `local`。2026-10-12的2层small local仅作额外历史对照，不能混为当前模型或主baseline。
- 用户观察部分loss先升后降、部分曲线趋稳；这是训练现象，不预设下游改善或已充分收敛。本次增加了训练曝光，不能视为等预算结构消融。
- 原checkpoint、latent、预测、STEP、诊断只读；复用需核验方法身份、协议、算法及资产哈希。全部新结果写新目录。

# Inputs

- 当前run：`checkpoints/intersection_local/full_capacity/20261012T141134.581695Z`。按验证callback固定 `weights/epoch=0042-step=00015609-val_loss=0.098798.ckpt`，val/loss=0.09879826009273529，SHA256 `c16d9ebe829659f3f4b8eb631e115fb10d03a28e9319bdb6bb5968b8d4bdb8f3`。
- last为epoch43/step15972、val/loss=0.1005650982260704，SHA256 `ee55d005ad79259e43d53fcb77177d106f2f9485c7e1fca39564fd9e5e79dd05`；只用于状态审计，不额外做test选优。当前inter实际为512维8层、8heads、FFN2048、dropout0.1、ReLU、完整TransformerDecoderLayer。
- 上一版refinement：`checkpoints/intersection_refinement/evaluation_20261011`，权重epoch10/step3993、SHA256 `9b18509de2a21f50d8fad41b8c37695842fe2e92745dc779713b0ecac06e2107`。
- 相同输入：`checkpoints/joint_boundary/comparison_20261011/raw_predictions/{posterior_mean,archived_latents}/legacy/{id}/data.npz`；原HoLa checkpoint `/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt` 仅作encoder/latent来源审计。
- GT `/mnt/d/data/deepcad_v7`，split `checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt`。
- 历史refinement诊断 `checkpoints/intersection_refinement/failure_census_20261011`。10/12 census中的baseline controls若另行复用，必须保持intersection身份并完整核验原来源；不复用旧small local作为当前预测。

# Outputs

本目录保存训练审计和曲线、两种输入raw预测与指标、posterior mean的fit_only/optimized输出与原生STEP指标、配对成功转换、中文报告和静态对比图、独立审计。`../failure_census_20261013` 保存全集诊断、成对机制变化、四组中文PNG/SVG、交互页面、自包含HTML及来源审计。大产物留ignored目录，紧凑报告保留版本管理。

# Processing Protocol

1. 核验实际训练配置、callback、checkpoint SHA、训练曲线和encoder/posterior逐张量一致性。验证使用posterior采样，测试使用已有mean，两者loss不可直接等同。
2. 严格按checkpoint配置加载，FP32、TF32关闭、seed42、`p>0.5`；8GPU按ID分片，不通过DDP sampler补齐。全部有向非self候选保留，不增加面去重、不向decoder输入GT拓扑。
3. 两种mode使用原归档float32 latent并逐ID校验输入；mean重编码仅作atol/rtol1e-5审计。上一版raw以只读引用保留真实方法名和来源。
4. raw指标沿用已验证公式。posterior mean计算对应面MAE和面连接P/R/F1；archived latent无GT面对应，不算对应MAE/F1，也不重采样diffusion。
5. 当前posterior mean分别运行fit_only和optimized：CPU单线程、每mode32并发、drop_num=0、优化名义200/原算法自适应600、固定逐IDseed、整对象180秒（包含测量）。上一版post仅在算法/采样/预算/split/raw完整一致时只读复用，历史耗时不作为受控速度比较。
6. 主有效性为原读取精度下TopAbs_SOLID+BRepCheck；0.1兼容口径单列。STEP存在、success marker、合法solid、几何可计算各自记录。固定2417分母，并另报/2424；共同可计算不称为共同成功。
7. 全部后处理终态完成后生成失败全集、持续失败、退步、救回cohort。当前所有失败和救回逐件诊断，baseline补齐同ID；全部fit-only成功但optimized失败追加fit-only路径对照。
8. 诊断与原生产决策/seed/180秒预算一致，每runner8并发、每进程1线程。只有终态与STEP DATA精确相等且无观测工具错误的重放可用于阶段归因，缺测与偏离保留未知。
9. 按.002/.006/.01/.08归一化长度阈值统计多标签和paired变化，制作四组中文图及四组×四阈值页面、具体ID双侧详情。页面增加自包含HTML，静态图在最终回答直接展示，不仅依赖localhost。
10. 独立复算来源、分母、cohort守恒、关键判据及CSV/JSON；浏览器检查页面计数、交互、资源、窄屏和JS错误。

# Metrics

raw面XYZ MAE、面/边双向平方CD、逐对象宏平均无向面连接P/R/F1、整件面邻接全对数、互反半边差异。全部覆盖率、均值/中位数/尾部和逐件改善；去尾仅作描述，不改变主分母。

fit_only/optimized原生合法solid比例、四类成对转换及ID、缺失/超时/不可读/几何可计算状态，原生STEP几何各自及共同可计算集合。失败观察分别正/负/未知，涵盖面连接、选边、端点、边面贴合、拟合、闭环、裁剪、缝合、保存；最早证据、最终阻断与共现分开。

# Decision Rule

用固定输入的几何、连接、最终solid和逐件机制判断当前权重是否优于上一版；不能用截图loss趋势或单个超阈值标签替代测试，不能把全部变化归因于局部注意力。固定diffusion latent解码不等于重新测试完整实拍重建。只按验证集选择权重。

# Edge Cases

空预测、重复/缺失ID、坏文件、非法shape/索引、NaN/Inf、未知bool、不可读STEP、采样失败、超时、崩溃及重放分歧显式记录并留分母。不可算指标不置零，零分母显示不可用，已确认solid后的采样超时不抹去成功。协议冲突停止对应阶段诊断，保留产物，不静默覆盖或重试筛选成功。未知不算改善，正常互反边不算重复，近边/丢wire不直接解释成应焊合/丢孔。

# Validation / Sanity Checks

实际完整容量与严格加载核验；真实小批合同和冻结encoder一致。baseline复用先查协议和raw哈希；少量诊断验证STEP DATA一致后继续全集。最终独立核对2417身份、四类转换、所有诊断终态、源资产未改、图表/页面与机器统计一致。已验证工具未变不机械重复整套测试，修改的报告/自包含页面只作相称验证。

# Implementation Constraints

不改训练代码、模型权重、损失、生产post或几何判据。复用当前工具，必要适配仅限新目录/报告标签/链接及自包含展示，所有依赖哈希在正式诊断前冻结。CPU线程1，独立ID并行，旧结果只读。用户展示时间用Asia/Shanghai，运行元数据保留UTC并显式标注。保留其他任务未提交工作。

# Deliverables

全集对比、全集fail census及配对证据、训练状态与来源审计、中文研究结论、四组PNG/SVG、可直接打开的自包含HTML及交互页、可复现命令、独立验证。所有阶段完成才给最终结论；中间数量明确是进度，不冒充全集。
