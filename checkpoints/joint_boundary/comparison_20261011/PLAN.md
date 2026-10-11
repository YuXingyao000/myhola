# Task

冻结协议 v1：先比较训练完成的 joint-boundary VAE 与原 HoLa light VAE，再讨论结构优化。分别回答 posterior-mean 重建、原始边界集合、OCC 后处理有效性，以及同一批历史 diffusion latent 换 Decoder 后的变化。此次不训练、不调阈值、不改模型或后处理算法。

# Assumptions

- 数据：主比较使用事先审计接受的 2,417 个 test CAD；原 test 共 2,424，7 个契约不支持对象单列并保留总体覆盖率分母。identity view，全局坐标不额外对齐／缩放。
- 任务：这是已训练模型的替换对比，不是相同训练预算的架构消融。旧 checkpoint 的训练集／预算与新模型不同；不能把差异全部归因于双向交互。
- 实现：预期两个 encoder/posterior 权重完全相同，必须检查；主比较以同一个 encoder 生成同一 posterior mean，两个 Decoder 自主预测边拓扑，不输入 GT connectivity、不按 GT 边数筛选。
- 下游：历史 baseline raw NPZ 已包含 face_latents；第二组重放同一个历史生成 latent，无 GT 面数／拓扑、不重新生成或挑选候选。只对归档覆盖作声明，不把重放当作重新执行并完整验证的照片生成流程。
- 需核实的资产事项：旧权重严格加载合同、归档 latent 覆盖及元数据、encoder 权重相等；若不成立，记录并停止依赖该假设的比较。

# Inputs

- Joint：`checkpoints/joint_boundary/frozen/20261010T084310.423740Z/weights/epoch=0538-step=00097559-val_loss=0.028098.ckpt`，按验证集最佳选择，测试前固定。
- Legacy：`/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt`，正是初始化 joint encoder 与历史 diffusion 的权重。
- 原始数据：`/mnt/d/data/deepcad_v7/{id}/data.npz` 与 `normalized_shape.step`。
- 清单：`checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt`，SHA256 `2864849e8240134c6033cc6cd64efa19337f67fea94b89a493690ae84355b7b3`；原始清单及拒绝原因沿用已有审计。
- 归档：`experiments/baseline/diffusion_baseline_real_v7_cube24/test_outputs/{id}/data.npz` 的 float32 `face_latents[F,32]`；仅只读。

# Outputs

本目录下 `raw_predictions/{posterior_mean,archived_latents}/{legacy,joint}/{id}/data.npz`，逐件状态、raw 指标 CSV、汇总 JSON、后处理逐件状态、报告及确定性选例图。大体积原始输出／运行日志保留本地；协议、脚本、紧凑汇总及结论受 Git 管理。`metadata/` 记录配置、命令、Git、权重／清单哈希与输入来源。

# Processing Protocol

1. 权重严格加载，核对 encoder/posterior 每个 tensor 与两个解码器的输入；FP32 eval、关闭 cuDNN 搜索、固定 seed 42，不执行训练 loss 的 GT 条件解码支路。
2. posterior_mean：逐对象 encode 一次，取 mean，两个 Decoder 使用同一 latent 和面顺序。保持原面集合，不调用旧 `decode_latents` 的隐式删面。Joint existence `>=0.5`，旧 classifier `>0.5`，均保留原始预测。
3. archived_latents：对同一 intended 清单按归档可用性逐件记录，两个 Decoder 重放同一 finite [F,32]；不额外去重或用 GT 筛选。不计算依赖 GT face order 的 MAE／拓扑精度。
4. raw 指标遍历全部 intended IDs；错误、空预测、缺失等显式记录。不使用旧 forward 的 `edge_coords_loss`，因为它来自 GT 连接分支。
5. posterior_mean 的全部对象做两种独立后处理：fit-only（关闭几何优化）和 optimized（名义 200 次迭代，原实现可自适应延长到 600）；均 CPU、drop_num=0，保留当前表示适配、拟合、修补、缝合。原代码仍可丢弃少量 trim 失败面，不宣称无修复。每方法／每对象／每模式硬超时 180 秒，固定逐 ID seed。
6. 后处理独立读取 STEP：报告原读取容差下的 solid + BRepCheck，及历史强制 tolerance=0.1 的兼容检查，两者不混用。记录输出面数／STEP／success marker／失败／超时，分母固定。缺失预测不替换成原点。
7. 后处理几何：对可读 STEP 与 GT native STEP 使用固定逐 ID seed 的表面／边采样，双向平方 CD；两个方法在各自可读集合和共同可读集合分别汇总。报告采样方式和容差，不把 merged raw 边指标冒充 native CAD 拓扑。
8. 选例预先按 accepted 清单稳定 hash 并按 GT 面数区间分层，展示 GT／legacy／joint 原始曲面与边。补充失败例必须明确其选取规则，不能仅展示获胜样例。

# Metrics

- 面几何：posterior_mean 的逐对象对应 UV 点 XYZ MAE；全部预测／GT face 采样点的双向平方 CD。面点为原 16×16 grid，不做面积加权或参数重排。
- 边几何：全部阈值激活的预测曲线点云与 GT shared 代表曲线点云 CD，保留冗余预测。CD 为两方向 mean nearest squared distance 之和，同时保留方向项；空预测且 GT 非空则值缺失并计失败，不赋零。
- 主拓扑：posterior_mean 对有效不同面 pair 取 unordered set，与 GT 集合计算逐对象 precision/recall/F1 与 micro 汇总；另报活跃 query／有向半边数、唯一 pair 数、same-face 和越界数。
- 重复：joint 同 pair 多于一条 query 是重复；legacy 正常两条 reciprocal half-edges 不是重复，单列有向重复／单向 pair 及互反曲线差异。不能把两种原始 edge_count 直接相减作同一指标。
- 汇总：mean、median、p90，配对差异、共同成功分母、每个阶段全 intended 成功率；原始 2,424 总体覆盖率另列。归档组报告几何与集合内部诊断，不借用 GT face order 计算拓扑 F1。
- 有效性：原容差 solid validity 与 legacy-compatible validity 分别列，不能仅凭 STEP 存在或 success.txt 判为成功。geometry 指标成功子集与失败覆盖率并列。

# Decision Rule

只在相同输入、阶段与指标定义下判断改进／接近／倒退。几何、拓扑、实体有效性若方向不一致，结论写明取舍，不压缩为一个总分。VAE posterior 重建提升不自动等于 diffusion 提升；归档生成 latent 重放提供该固定上游分布上的证据，不证明所有新 diffusion 架构或真实照片任务。不得按测试结果改 checkpoint、阈值或修复预算。

# Edge Cases

空清单或重复 ID 在启动时拒绝。逐件缺失／损坏／NaN／Inf／无效形状／非法索引分别记录；不静默跳过。零 GT 与零预测 pair 的 P/R/F1 约定均为 1；只有一侧为空则 F1=0，precision/recall 的空分母约定在实现中固定。OCC 崩溃与硬超时由隔离子进程记录，无法恢复的输出不算成功。所有失败保留在 intended 分母。原始不支持的 7 件不偷偷移出原 test 覆盖率。

# Validation / Sanity Checks

检查精确 encoder 权重相等与同输入 latent；同一预测对自身几何指标为零；已知平移的 CD/MAE；正确／错误 pair、双向半边非重复、重复 query、空预测与非法索引；两种表示的小规模真实导出和 OCC 往返。通过后跑全清单，不用 smoke 代替最终比较。

# Implementation Constraints

Python、既有 PyTorch/NumPy/SciPy/PythonOCC 环境；尽量复用模型与 post 函数，不改训练、阈值或公开后处理签名。推理可按 ID 明确分片到多张 GPU，无 distributed sampler 补齐；指标以 CPU cKDTree 为主。后处理隔离进程且单进程线程=1，显式并发与时间上限。所有新资产写入 checkpoints，不修改 experiments。权重以安全加载为先，必要时仅明确允许已知数据容器，不允许静默任意 pickle 执行。

# Deliverables

可复用的推理／raw metrics／post runner、必要的行为测试、完整实际对比结果、直观选例、对 diffusion 的有边界结论，及 Git 提交。记录任何执行失败、协议修订和未完成项；不启动新模型训练。
