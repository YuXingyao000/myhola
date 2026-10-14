# Task

完整评估训练完成的最终曲线条件调制，与已评估的原 Intersection＋refinement 比较，并复现此前的全集失败统计、中文图表和离线交互报告。本轮不训练、不改模型和生产后处理。

# Assumptions

用户已授权沿用前次评估；固定 accepted test 2417 件，原 2424 件中 7 件合同不支持样本单列。当前 `/mnt/d/python` 已切到 data-generation 且有未提交工作，因此在 `/mnt/d/python_curve_eval` 独立 worktree、`codex/curve-conditioning-evaluation` 分支运行。旧运行资产从 `/mnt/d/python/checkpoints` 只读引用。

候选有额外训练曝光，比较回答当前权重是否改善，不是等预算架构消融；归档 diffusion latent 重放不等于重新采样的完整照片重建。

# Inputs

- 按 last.ckpt 的验证集 ModelCheckpoint callback 选择候选 epoch157/step57354：`/mnt/d/python/checkpoints/intersection_refinement/curve_conditioning/20261013T083729.740439Z/weights/epoch=0157-step=00057354-val_loss=0.038204.ckpt`。实际 SHA、分数、冻结状态和训练曲线见 `training_audit.json`。last epoch184/step67155 不用于 test 选优。
- baseline：原 refinement epoch10/step3993，已有完整评估 `/mnt/d/python/checkpoints/intersection_refinement/evaluation_20261011`；历史诊断在同主题 `failure_census_20261011`。
- 同一输入来源：`/mnt/d/python/checkpoints/joint_boundary/comparison_20261011` 中 posterior_mean 与 archived_latents 的 legacy 输出；原 encoder 权重 `/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt`。
- GT：`/mnt/d/data/deepcad_v7`；split：本 worktree 的 `checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt`，SHA256 `2864849e8240134c6033cc6cd64efa19337f67fea94b89a493690ae84355b7b3`。

# Outputs

本目录保存配置/权重/代码来源、训练审计、两种 raw 指标、posterior mean 的 fit_only/optimized 结果、配对变化与中文图表。相邻 `failure_census_20261014` 保存全体失败及救回的双侧诊断、四组中文图、交互页面、自包含 HTML 和独立核验。原始预测、STEP、日志和全量轨迹按现有规则留在本地。

# Processing Protocol

1. 核对训练 callback、实际配置、旧 encoder/posterior 逐张量一致、条件分支已更新。严格加载实际 checkpoint。
2. 小批验证后 8 GPU 独立分片推理，先 posterior_mean、再 archived_latents；FP32、TF32 off、seed42、p>0.5，复用同一归档 float32 latent，禁止 GT 拓扑输入和新增 face 去重。每片前三件 mean 重编码仅作 atol/rtol1e-5 审计，实际输入逐值不变。
3. baseline raw/post 只读复用，校验方法名 intersection、split、原始输入哈希与完整终态；核对 11 个生产后处理文件 SHA。
4. posterior_mean 跑 fit_only 与 optimized，每种 32 CPU workers、进程内单线程、每对象含测量 180 秒、drop_num=0、名义优化200/原算法自适应上限600。原生 TopAbs_SOLID＋BRepCheck 为主，兼容容差0.1单列。
5. 所有终态完成后冻结失败、持续失败、退步、救回 cohort；对候选全部失败和救回逐件诊断，baseline复用合格历史诊断并补齐同ID对照；所有 fit-only成功而optimized失败追加路径对照。
6. 沿用此前 .002/.006/.01/.08 的失败观察阈值、正/负/未知口径和四组图表。工具缺测或重放不一致不得改写为确定根因。
7. 生成结果与离线页面后独立复算分母、ID集合、配对守恒、阶段证据及资源/交互。全部完成才报告最终结论。

# Metrics

两种输入的面/边双向平方CD、分位数及同ID改善比例；posterior_mean额外对应面XYZ MAE、无向面连接P/R/F1与整件邻接全对率。raw与最终几何分开。固定2417及原2424分母报告覆盖、STEP产出、原生合法solid与失败/超时；几何各自与共同可计算集合分别报告。失败报告覆盖连接、选边、端点、边面贴合、拟合、wire、裁剪、缝合、保存，多标签不相加为失败总数。

# Decision Rule

先看合法solid的配对增减，再结合原始几何和失败机制评判这次条件调制的实际收益。不按测试结果换 checkpoint，不以验证loss改善推导最终精度改善；若结果相反如实说明。不能把额外训练曝光带来的全部变化归因于新增模块。

# Edge Cases

缺失、坏文件、NaN/Inf、非法shape/索引、空预测、崩溃、超时均留在预定分母，指标不可算记缺测。已确认原生solid之后测量超时不能抹去有效性。协议或来源冲突停止受影响阶段，不静默覆盖历史结果或重试筛选成功。零分母显示不可用，未知不算改善。

# Validation / Sanity Checks

严格权重加载和冻结encoder核验、真实小批输出/输入哈希、后处理算法与baseline来源一致性、相关方法名/报告工具回归。独立复算全2417件及全部fail census，报告数字与图表/交互页面逐项核对。运行时保存命令、依赖SHA、退出码、分片与终态。

# Implementation Constraints

模型、loss、生产post及所有历史资产保持不变；工具改动仅为接入 conditioned 方法身份、输出标签和路径，不改变指标或判据。CPU线程1；不修改 data-generation 分支。研究记录在本 worktree 的 checkpoints 下，元数据使用UTC，用户展示时间为Asia/Shanghai。

# Deliverables

完整评估、完整fail census、中文结论和静态图、可直接打开的自包含HTML、可复现命令及来源/独立验证报告。正式运行完成后同步实际状态；初始化等价性或smoke不作为研究效果结论。
