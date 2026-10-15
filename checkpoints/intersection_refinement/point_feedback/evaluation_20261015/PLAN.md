# Task

完整评估最终曲线逐点反馈的验证最佳权重，直接对比上一轮 curve conditioning，并保留原 Intersection＋refinement 的历史参考。沿用固定测试、两条后处理、全集失败诊断与四组中文图表／离线页面。用户已授权执行；不训练、不修改模型或生产后处理。

# Assumptions

- 固定 accepted test 2417 件；原 2424 件中 7 件合同不支持，保持单列。
- 新 run 在 conditioning 的最佳权重上额外训练。比较回答当前 checkpoint 是否改善，不是等预算架构消融；不能把全部收益归因于新增模块。
- posterior_mean 是 VAE 重建；archived_latents 是固定历史 diffusion 输入重放，不是新图片采样。
- `/mnt/d/python` 为其他任务工作区，仅只读引用资产。本次在 `/mnt/d/python_curve_eval`、`codex/curve-point-feedback` 执行。

# Inputs

- 候选方法 `point_feedback`：`../20261014T153723.959456Z/weights/epoch=0034-step=00012705-val_loss=0.037230.ckpt`。选择依据是 last.ckpt 中验证集 ModelCheckpoint 回调，不根据测试换权重；实际 SHA、step、encoder 状态由 training_audit.json 固定。
- 直接 baseline 方法 `conditioned`：`../../curve_conditioning/evaluation_20261014`；诊断为相邻 `failure_census_20261014`；checkpoint 为 epoch157/step57354、SHA256 `4c7910b8169206c74fcf38c358cf171359149e0429c8debbc227d391aa56613f`。
- 历史三级参考 `intersection`：`/mnt/d/python/checkpoints/intersection_refinement/evaluation_20261011`，仅在可验证同协议的指标上并列。
- latent 来源 `/mnt/d/python/checkpoints/joint_boundary/comparison_20261011`；legacy encoder `/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt`。
- GT `/mnt/d/data/deepcad_v7`；split `checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt`，SHA256 `2864849e8240134c6033cc6cd64efa19337f67fea94b89a493690ae84355b7b3`。

# Outputs

本目录保存训练审计、源码／权重／输入来源、两个输入模式的 raw 指标、posterior_mean 的 fit_only／optimized、逐件配对结果、中文报告及分布图。相邻 `../failure_census_20261015` 保存完整失败／救回双侧轨迹、四组图、交互及自包含离线页面、未知与重放核验。

# Processing Protocol

1. 严格加载实际 checkpoint；核验 243 个 encoder/posterior 张量仍与源模型完全一致、配置和新增反馈参数状态。先真实小批 smoke，再完整运行。
2. 复用同一归档 float32 latent，8 GPU 独立分片、FP32、TF32 off、seed42、p>0.5；GT 不筛候选，不新增 face 去重。posterior 重编码只作原协议容差审计，不替换输入。
3. baseline raw/post 只读复用，严格检查 method=conditioned、split、输入 latent 哈希、终态、生产后处理文件哈希及协议。不可将历史 method 改名伪装成新运行。
4. posterior_mean 的 fit_only／optimized 各 32 CPU 单线程 workers，每件含测量 180 秒，drop_num=0，名义优化200／原算法自适应600。原生 TopAbs_SOLID＋BRepCheck 为主，兼容容差0.1单列。
5. 2417 件全部终态后冻结持续失败、新增失败、救回、持续成功集合。诊断候选全部失败和全部救回；baseline 同 ID 合格历史记录复用、缺少的补齐。候选 fit-only成功而optimized失败全部追加完整路径对照。
6. 沿用 .002/.006/.01/.08 阈值、正／负／未知及四组图；多标签共现与互斥末端阻断分别统计。所有观测重放核对终态和 STEP DATA，不把工具缺测解释成已确定根因。
7. 独立复算固定分母、配对守恒、原始误差分布、报告图表和离线页面。阶段未完成不按成功子集提前宣布整体提升。

# Metrics

固定2417及原2424分母的导出覆盖、STEP产出、原生有效solid、失败／超时。raw 两模式面／边双向平方CD及均值、中位数、分位数、逐ID改善比例；posterior_mean 另报对应面XYZ MAE、面邻接P/R/F1与整件全对率。post 几何各自可计算与共同可计算集合分别报告。完整失败统计覆盖面连接、选边、端点、贴合、拟合、wire、裁剪、缝合和保存；未知保留。

# Decision Rule

先判断合法solid的配对增减，再结合原始几何、修复后几何与失败机制评估收益。三版历史参考不冒充等训练预算比较。不按test选checkpoint、不把training／validation loss当solid结果、不把平方CD当线性最大容差。

# Edge Cases

重复ID、方法身份或来源冲突、协议漂移拒绝汇总；缺失、空预测、坏文件、NaN／Inf、崩溃、超时保留预定分母。测量超时不抹去此前已确认的solid有效性。未知不是阴性；零分母显示不可用；空cohort显式0。只允许有明确记录的恢复，不能通过重复运行筛成功样本。

# Validation / Sanity Checks

工具方法扩展需相关回归与真实 smoke；不修改任何生产几何算法。baseline审计、逐IDlatent一致、完整终态及重放一致性必须通过。最终独立复算数值；浏览器检查四组四阈值、双侧详情、窄屏、无JS错误；离线页不依赖HTTP且图像可解码。

# Implementation Constraints

优先复用已有 tools 与报告模板，仅增加明确的新method和baseline选择。模型／loss／post 不变。新记录统一位于 checkpoints；weights/raw/logs按现有ignore不提交。元数据UTC，用户时间Asia/Shanghai。保存实际命令、退出码、源码SHA、checkpoint及split SHA；保留所有历史资产。

# Deliverables

完整原始／后处理评估与配对结论；完整fail census、四组PNG/SVG、可直接打开的自包含HTML；训练和来源审计、独立复算、可复现命令、最新状态。若有无法测量或重放不一致，明确数量及结论限制。
