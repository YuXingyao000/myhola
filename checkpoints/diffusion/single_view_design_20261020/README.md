# 单视图 diffusion：从目标到公共主干

状态：2026-10-20 讨论记录；未修改生产网络、未启动训练。用户进一步明确：**当前专注提升已有单图生成模型的各项指标**。下文“公共主干／先无条件再条件”的提议保留为讨论历史，已不作为下一步计划。此前 768 维实验结果保留，不能由其 0/1000 合法 solid 单独推断宽度导致失败。

## 本轮方向修正

- `/mnt/d/ckpt/HoLa/Diffusion_svr_sq30_1500k.ckpt` 已经是训练完成的单图 concat 条件模型；`/mnt/d/ckpt/HoLa/Diffusion_uncond_1100k.ckpt` 已经是训练完成的无条件模型。恢复原结构属于资产复用，不能当成新的研究方案。
- 单图改进以已有 SVR 权重作为候选初始化；其已有图像投影同样保留，不采用下文针对“无条件转条件”提出的图片末层零初始化。无条件模型保留作诊断参照，不是必须先重训的前置任务。
- 论文 Table 6 的 single-view validity 为 86.37%。用户报告早期原代码复现的 Image 指标达不到论文值。当前独立、固定 100 件 validation / K=1 的旧 SVR 对照为 74/100，不能直接与不同样本和未完整恢复协议的论文数值作同口径比较；也不能据此认定论文如何处理过结果。
- 当前固定对照中，旧 SVR → layerwise last 的原始面 CD² ×1000 为 5.523 → 4.172，边为 14.213 → 9.274，有效 solid 为 74 → 54。几何距离改善和实体有效性没有同步改善；训练量、宽度、图像编码和视角增强均不同，不能据此断定 cross-attention 或某个损失是原因。[完整报告](../svr_comparison_20261017/README.md)
- 原始网格 CD 与论文最终 B-Rep primitive 指标分列。现有该批主表尚未覆盖 vertex 和 FE/EV，不能把它描述为论文全指标复现。原生有效性和历史 0.1 容差兼容检查在该批旧 SVR 上均为 74/100，容差不能单独解释该批差距。

## 待讨论候选：用生成结果继续训练已有单图模型

保留旧 SVR 的 1024 维、24 层主干、已有图像分支及匹配的冻结 VAE/cache，研究如何让 diffusion 学习最终解码结果的质量，而不只优化噪声预测误差。这是研究选择，现有比较尚未证明缺少这类反馈是主因。

一种候选实现是在训练集的同一图片下生成多个完整 latent，保留去重前 `[30,32]` 的生成结果；经固定 VAE 和固定后处理评价，构成质量较好／较差的样本对。使用 diffusion preference loss 微调已有模型，同时保留原 DDPM 监督和冻结旧模型作为参照。OCC 只提供评价，不需要也不能被描述为已有可微反传链路。训练期多候选收集与部署时多次采样是两种预算，不混为单次质量提升。

样本优劣需要共同考虑几何匹配、可见图像一致性和实体有效性；仅按合法性排序会偏好简单但错误的形状。训练 GT 可用于监督和匹配，不进入部署时选样。单个 GT 不代表唯一合理的隐藏补全，不能把“更接近这个 GT”自动等同于“条件分布更正确”。冲突样本的取舍、标签和多样性约束尚需设计，未将该候选确定为正式训练计划。

另一种辅助方式是把低噪声下的干净 latent 估计送入冻结 decoder，增加具体的几何／连接监督。它可能改善接近正确 latent 时的细节，却未必改善从高噪声开始的结构选择；随机复制对应、去重差异及 decoder 自身误差也需处理。因此不将泛化的“几何 loss”当作已确立的首选。

相关先例：[CADCrafter §3.5](https://arxiv.org/html/2504.04753v1#S3.SS5) 使用 CAD 编译成功／失败的反馈进行 diffusion DPO 微调。它生成 sketch/extrude 命令序列，不能直接证明对 HoLa 的 B-Rep latent 有效；本项目仍需以固定协议下的单图几何、拓扑、有效性及多样性判断。

## 候选生成、指标与偏好训练协议审核稿

状态：2026-10-20，应用户要求列出供检查；**未冻结、未实现新流程、未生成偏好数据、未启动训练**。本节细化并收窄前面的候选提议：第一轮只使用“几何没有明显退步的合法候选优于几何无效候选”，暂不启用两个合法结果之间的 GT 距离排序。下面标为“建议”的参数不是已运行配置。

### 1. Task

从已有单图 SVR 权重出发，研究偏好微调能否改善固定预算下的实体有效性，并检查完整 GT 重建、拓扑和同图多样性是否退步。完整 GT 重建准确性和单图合理补全是不同目标；本稿使用 GT 的质量门槛，不声称已经解决对合理隐藏部分的评价。

### 2. Assumptions

- 已核对：旧 SVR 为 1024 维、24 层、图像全局 concat；现有主分支 layerwise 模型不能直接作为它的严格 resume 入口。
- 建议默认：冻结旧 SVR 作为 reference，从相同权重初始化 trainable 副本；固定最初HoLa-BRep的配套VAE/cache，保留已有图像投影，不从无条件模型重新开始。用户明确指出当前VAE有多次迭代，不能仅凭模型名或32维shape假定latent兼容。
- 建议默认：每图固定 `K=8`，同图不同 seed，先做一轮离线候选收集。没有合格 pair 的图片不追加采样。
- 待定：训练 CAD 清单与规模、GT 几何质量门槛、允许退步的容差、LFD 近重复阈值、DPO 时间权重与损失权重、训练预算。
- 待定：已有灰模相机与 renderer 是否能够逐像素对应。未经核实不能使用重投影指标作为标签。

### 3. Inputs

- 图片：train split、灰模 view 0 的 `imgs.npz["svr_imgs"]`，不混入生成照片。
- 基础权重：`/mnt/d/ckpt/HoLa/Diffusion_svr_sq30_1500k.ckpt`，记录完整 hash。
- 固定 decoder：最初HoLa-BRep的原版AE，优先从旧SVR完整checkpoint中加载。已有[逐参数审计](../svr_comparison_20261017/report/provenance.md)确认其704个AE tensor与 `/mnt/d/ckpt/HoLa/AE_deepcad_1100k.ckpt` 完全一致；这是此前审计记录，不是对当前所有磁盘缓存的重新验收。训练cache仍需核对编码器、mean/std约定、rotation及对应关系。
- 用户允许考虑最新UV改进VAE，但本轮旧SVR继续训练建议先采用上述原版组合。若改用改变encoder/posterior的UV版本，应重新编码数据并适配或重训diffusion，除非有独立证据证明latent保持兼容；不能把旧cache或旧diffusion输出直接送入新decoder就称为已完成迁移。
- 训练 GT：对应 CAD、原始 face/half-edge 几何、latent cache；指标计算可使用 GT，不作为推理条件。
- validation：用于规则、阈值与 checkpoint 选择；test 仅在方案冻结后作最终评估。不能从 validation/test 候选构造梯度训练对。
- CAD identity、view、rotation、归一化坐标与 split 一致，不通过最优旋转、ICP 或预测独立缩放提高指标。

### 4. Outputs

- `candidate_manifest`：CAD/image/candidate ID、seed、图片与权重 hash、采样配置、原始 latent 路径、后处理版本和预算。
- 每个候选的**原始最终 `[30,32]` latent**、去重映射、raw 解码结果、STEP 与后处理记录。不能把去重后重新补齐或重新编码的 latent 当作原始生成样本。
- `candidate_metrics`：每项数值、可计算状态、失败原因、指标协议版本；缺失值保留为缺失，不能填零。
- `preference_pairs`：同图的 `winner_id / loser_id`、触发规则、实际指标、阈值版本；无 pair 的图片记录原因。
- `per_image_diversity` 与汇总：全部 K 次的状态、合法数、可测 LFD 数、距离分布和近重复比例。

### 5. Processing Protocol

1. 固定清单、split、模型/VAE、坐标、采样与后处理协议。
2. 为 train 每图生成 K 个候选：建议沿用 1000 步原生 DDPM、FP32、无 CFG、无 GT 选样与无额外重试。保存完整采样 latent 和随机状态溯源。
3. 使用原版去重与 VAE 解码，保存 raw 面、半边、连接；模型原始输出不被后处理覆盖。
4. 按用户修正，使用当前优化后的 `post_policy="recover_v1"`，不再使用旧对照的纯legacy策略。固定 optimized、drop=0、名义200步（原实现可扩至600）。建议沿用现有隔离runner每候选180秒的整子进程预算，包含后处理、有效性和几何测量；recover_v1最多四条内部路径共享预算，首次合格即停止且不读GT。策略见[后处理总览](../../postprocess_refinement/README.md)。生产函数默认仍是legacy，接入时必须显式传策略；不能仅沿用旧CLI就认定启用了recover_v1。
5. 先记录状态与原生有效性，再计算 raw/post 几何和拓扑；不成功的候选也保留在清单。
6. 按下述规则建立同图正负对。第一轮每图最多取一对；多个合格 pair 用固定 seed 均匀选取，不按最高总分挑选，避免候选多的图片获得更多权重。
7. 用 pair 训练单图模型；保留原 GT DDPM 监督，旧 reference、DINO 与 VAE 冻结。可训练图像投影的冻结范围需在实现配置中明确。
8. validation 同时检查预先指定的 K=1 输出和固定 K 候选分布；不能把用 GT 挑出的最好候选当作单次质量。冻结后再评估 test。

### 6. Metrics

以下均为指标定义；它们不自动决定正负标签。

| ID | 指标与方向 | 计算口径 | 第一轮用途 | 实现状态 |
|---|---|---|---|---|
| S1 | requested / inferred / decoded / STEP / measurable | 各阶段成功数除以预定 `M×K`；每图另列 K 次状态 | 分母和失败分析 | 已有固定分母包装，可扩展 |
| V1 | native valid solid ↑ | 原容差读取 STEP，最外层类型为 SOLID 且 BRepCheck 通过；不放宽容差 | 正负配对核心 | 已有 `tools/vae_compare_post.py` 实现 |
| V2 | compat valid at 0.1 ↑ | 独立重读、precision=0.1、SetTolerance(0.1) 后检查；重设容差不保证总比原值宽松 | 历史对照，不选 pair | 已有；不是 V1 |
| G1 | raw face CD² ↓ | 原始所有 face grid XYZ 合并后双向最近平方距离均值之和 | 配对质量保护 | 已计算过 |
| G2 | raw edge CD² ↓ | 原始所有半边点合并；与 face 分开计算 | 配对质量保护 | 已计算过 |
| G3 | raw face P/R/F1 ↑ | primitive 两两 CD² 代价、Hungarian；匹配代价<0.1为TP | 配对质量保护 | 已计算过 |
| G4 | post face / edge / vertex CD² ↓ | 最终 STEP 几何；三类分别合并点集计算，不是匹配后CD平均 | 最终质量；face/edge用于正样本质量门槛 | face/edge已有；vertex需接入该协议 |
| G5 | post face / edge / vertex P/R/F1 ↑ | 三类分别 Hungarian 一对一匹配；代价<0.1为TP | 最终质量报告 | 通用模块已有，固定采样协议需接入 |
| T1 | post FE / EV P/R/F1 ↑ | Hungarian映射后比较face-edge与edge-vertex incidence | 第一轮只报告，不直接选 pair | 通用模块已有；未在100件主表计算 |
| C1 | face / edge / vertex 数量及误差分布 | raw/post分别记录；与GT数量误差分列 | 检查是否趋向简单形状；不是多样性的替代 | 大部分已有 |
| O1 | 同输入视角 mask IoU ↑ | 同相机/投影渲染最终裁剪后实体；与输入前景mask比较，保留背景孔洞 | 待实现后的观测一致性 | 尚未实现/核验 |
| O2 | 同视角 mask boundary distance ↓ | 前景mask边界（含孔洞边界）的两个方向最近欧氏像素距离均值相加，再除以2和图像对角线 | 轮廓位置诊断 | 尚未实现/核验 |
| O3 | 可见内部线条误差 ↓ | 比较输入中确实有证据的可见孔槽/折线，不能把所有CAD边投影当图像边 | 待研究；O1/O2看不到的槽不能假称已检查 | 边界定义与参考提取未冻结 |
| D1 | 同图 pairwise LFD | 每张图内部、原生合法且LFD可计算的候选间计算所有无序对；逐图汇总mean/median | 多样性监测，不作奖励 | 特征提取已有，分组统计待新增 |
| D2 | 同图近重复 pair 比例 ↓ | 同一子集中 `LFD<τ_dup` 的无序对数/全部无序对数 | 检查候选是否重复 | 待新增与校准阈值 |
| D3 | 每图合法数/K、LFD可测数/K | 区分有效性不足与描述子提取失败；少于两可测合法候选时D1/D2为N/A | 与D1/D2一起报告 | 待新增分组汇总 |
| N1 | 对训练集的最近 LFD | 查询候选到固定训练参考集的最小LFD，明确参考集 | 可选新颖性分析，不是同图多样性 | 已有任意输入矩阵的行最小值汇总；参考集需另行核验，不是D1 |
| R1 | 推理、后处理、评价耗时 | 每候选及总预算，超时独列 | 成本报告 | 基础记录已有 |

**CD² 数值定义**：对于预测点集合 P、GT 点集合 Q，

\[
CD^2(P,Q)=\frac{1}{|P|}\sum_{p\in P}\min_{q\in Q}\|p-q\|^2
+\frac{1}{|Q|}\sum_{q\in Q}\min_{p\in P}\|q-p\|^2.
\]

不取根号、不除以2；两方向分量同时保存。所有数值先保存原单位，展示时才乘1000。重合重复面不一定明显改变CD，因此同时保留primitive F1和计数。

**F1 定义**：P=TP/预测primitive数，R=TP/GT primitive数，F1=2PR/(P+R)。0.1是**双向CD²代价阈值**，不是点到点欧氏距离0.1。是否适合筛选仍待审查，不能把它当作工程精度要求。

**FE/EV 的现有语义**：FE是face-edge所属关系，EV是edge-vertex所属关系。每个CAD内先累计全部正确对应的关系，分别除以预测关系总数和GT关系总数得到P/R，再计算F1；不是逐面或逐边的F1平均。现有实现没有按0.1阈值删除距离很大的Hungarian对应，因此不能描述为“几何匹配合格后才比较拓扑”。第一轮只报告；若以后新增几何阈值限定的版本，另起指标名和协议，不覆盖历史结果。

**采样口径建议**：raw使用原网格；post沿用已完成100件对照的每面1024点（面内按三角面积随机采样）、每边128点（原生curve参数均匀，非弧长均匀），vertex取真实顶点。逐候选固定采样seed。统一 `condition.py` 默认的面积/长度自适应点数是另一套协议，不能混为同列。

**0.1容差检查的含义**：容差是几何内核判断点、边、面是否吻合时允许的误差尺度。旧兼容检查先按0.1精度读取，再强制把形状子元素容差设为0.1；不是0.1%的错误率，不等于把几何修好了，也不能不核对坐标单位就称为0.1毫米。它可能增大或减小原容差。V1不在评价阶段重设容差，但仍使用STEP本身存储的容差，因此也不等于统一工程精度认证。V2只作历史对照，不进入训练标签。

**观测与多样性限制**：现有 `visible_condition` 把全部预测面与GT中可见比例≥0.05的整张面比较，会处罚合理隐藏面，不充当O1–O3。D1变大不证明隐藏补全更合理；只有O1–O3可信后，才能另报观测合格子集的多样性及该子集占K的比例。第一轮D1明确叫“合法候选间差异”，不是“合理补全覆盖率”。LFD提取/归一化/对齐与网格质量固定，姿态/尺度误差另外报告。

现有100件报告仅覆盖raw face/edge CD、raw face P/R/F1与面数、post face/edge CD、两种validity和状态/耗时；没有vertex、post primitive F1、FE/EV、重投影或同图LFD的既成结果。

### 7. Decision Rule

第一轮建议只使用 `validity_pair`，不将上述指标任意加权为总分：

1. A/B来自同一图像，同一模型/VAE/采样/后处理协议；都具有完整原始latent和可计算的G1/G2/G3。
2. `V1(A)=true`；`V1(B)=false`且已确认是生成几何/构造问题。文件丢失、OOM、未知超时和指标异常不能冒充B。
3. A需通过绝对质量门槛：post face/edge CD²分别不超过τ_post_face/τ_post_edge，raw face F1不低于τ_face_f1。这些门槛使用训练/validation GT，含完整隐藏部分，因此属于GT重建质量约束。
4. 相对B，A的raw face/edge CD²分别最多增加ε_face_cd/ε_edge_cd；raw face F1最多下降ε_face_f1。A的后处理几何不可测时不建pair，即使其V1通过。
5. 所有τ/ε待validation校准并冻结。本轮没有虚构数值；未冻结前不构建正式训练标签。
6. 两个都合法、两个都无效、质量互有优劣或任何必需指标缺失：第一轮不建pair，保留原因和全部候选记录。

**K=8不保证能配对，也不要求每图都有正负样本。**八个都合法时不构造有效性pair；八个都无效时也不临时指定一个“较不坏”的正样本。这些图片仍参加原始GT DDPM训练。每图有效候选数、实际pair图片数/全部图片数，以及配对图片的面数分布都应报告，避免把只覆盖部分难度的结果泛化。若pair过少，应判断是失败稀少还是集中在少数图片；不无限加K、不人为破坏latent、不从另一张图片借负样本。新后处理可能救回旧失败，必须重算旧模型在recover_v1下的基线和候选标签，不能用旧74/100估计本轮pair产出。

这比前面“两个合法结果也按完整GT距离排序”的提议更保守，避免直接把合法的不同背面当负样本。GT质量门槛仍可能偏好接近唯一GT的候选，不能据此保证隐藏多样性不受影响。

若另行研究提高GT几何重建精度，两个合法候选也可形成相对优劣：例如post face/edge CD²均有明确改善且face F1不退步时才建立 `geometry_pair`。这里“负”表示相对较差，不要求无效。但这是独立的新训练目标，需规定改善幅度与保护阈值；会偏向完整GT的隐藏部分，不默认为第一轮有效性方案的一部分，也不因K=8全合法就临时启用。

后续若启用 `observation_pair`，需先完成O1–O3的相机与参考提取核验：两个都合法时，仅在至少一个观测指标有超过噪声的改善、其余受保护观测指标没有明显退步时建立pair；仅隐藏部分不同的不强排。它不属于第一轮已定实现。

### 8. Edge Cases

- 所有 M×K 个请求都进入状态表；没有合格pair不追加候选，也不从尝试分母移除。
- 缺文件、OOM、基础设施错误、未知超时：保留失败，不作为质量负标签。可明确重复定位的几何构造失败才可进入V1=false负集。
- 读得出STEP但非solid、内核检查失败：记录几何无效；有可测raw仍可进入配对候选。
- 无edge/face、NaN/Inf、部分primitive无法采样：对应指标N/A，保留其他可计算项；不造原点几何，不静默跳过坏面。
- 没有真实vertex的形状：不填原点；记录vertex指标不可用。零分母不当作完美F1。
- 原始几何与后处理几何分列；成功标志、存在STEP、可读STEP、合法solid是不同状态。
- 连续指标均值使用可计算集合，同时给出n/全部请求数；跨模型给出共同可计算集合，不能拿幸存均值代替整体结果。
- LFD可计算的合法候选不足2：D1/D2=N/A；仍报告m/K及描述子失败数。只对可计算图片求均值时必须披露图片覆盖率，不做静默3σ删除。
- GT用于训练标签不等于推理oracle。测试若按GT选best-of-K，必须另列为oracle诊断，不能用于主结果。

### 9. Validation / Sanity Checks

- 固定输入和seed重放旧SVR，严格加载，核对VAE与cache，保留原始latent与解码输出的可追溯映射。
- 用已知同形、平移、重复面、缺面、缺边与错误连接样例核验各指标方向；不启动大型实验来代替接口验证。
- 对原生/0.1有效性分别读取，避免容差修改污染后续几何或拓扑指标。
- 同一个STEP重复三角化/采样的误差用于估计评价噪声；阈值设计和个例分析只用validation。
- 检查pair选择无跨图片混配，所有阈值版本明确，任何异常状态不会变成普通低分；人工复核validation中的典型接受/拒绝对。
- 新模型接口可靠后才做正式训练；正式效果以固定K=1和K候选联合报告判断。

### 10. Implementation Constraints

- 本稿只定协议，不修改生产网络/指标、不启动候选收集。
- 复用已有特征和几何计算内核，增加清晰的固定协议包装；不得直接使用原点fallback、现存目录交集或`only_valid`当请求分母。
- 不覆盖历史指标。新增采样/阈值/FE-EV定义/可见渲染规则独立版本化。
- DPO需按旧SVR实际epsilon参数化和noise schedule实现，不能直接照搬CADCrafter的x0参数化公式。保存完整原始 `[30,32]` latent；正负候选之间不要求逐面对应，不按GT重排候选。
- 第一轮离线候选来自固定旧checkpoint，reference固定；GT DDPM使用原训练数据。每个训练step不需要重新跑1000步生成或OCC。
- 普通讨论中的K=8、后处理预算为建议，需以冻结后的配置与数据清单为准。

### 11. Deliverables

本轮交付：供用户审核的流程、指标定义、配对规则及待定参数。下一阶段如实施，需交付冻结配置/清单、可复核的候选指标与pair原因表、接口验证报告、训练命令及固定评估报告。

源码依据：[condition](../../../src/brepnet/eval/metrics/condition.py)、[native/post](../../../tools/vae_compare_post.py)、[raw评估](../svr_comparison_20261017/evaluate.py)、[visible](../../../src/brepnet/eval/metrics/visible_condition.py)、[LFD wrapper](../../../src/brepnet/eval/metrics/lfd.py)。方法先例：[CADCrafter §3.5](https://arxiv.org/html/2504.04753v1#S3.SS5)；其公开文字没有给出本稿的综合质量门槛、每图K和具体配对算法。

以下保留本轮较早的形式化推导与结构建议；其中结构核对可复用，训练顺序已由上面的用户目标修正。

## 已核对的事实

- 旧无条件 checkpoint 的结构是 `32→768` 输入 MLP，拼接 256 维零条件，进入 1024 维、16 heads、24 层 TransformerEncoder，FFN 为 2048。时间嵌入与输出头均按 1024 维工作。
- 旧单图分支使用冻结 DINOv2 ViT-L/14 with registers 的 CLS，再经 MLP 得到 256 维条件。256 维在输入端有条件语义，进入 Transformer 后与其余维度一起运算。
- 前一轮新无条件配置继承当前 768 维、12 heads 的主干；同时训练数据、增强行为、冻结 AE 的运行模式和实际训练量也与旧 checkpoint 有差异。没有理论要求移除条件拼接时同步缩窄主干。
- 旧无条件 checkpoint 已有固定 1000 次生成的评估资产；它是真正训练过的无条件模型。模型名称包含 `Diffusion_condition` 不改变这一事实。

依据：[旧基线](../uncond_baseline_20261018/README.md)、[新训练协议](../uncond_training_20261018/PLAN.md)、[完整生成对比](../uncond_comparison_20261019/README.md)。

## 要学习的随机变量

设 CAD 为 \(G\)，冻结编码器输出的 mean latent 为 \(E_\mu(G)\in\mathbb R^{N\times32}\)。训练变量为：

\[
Z_0=\Pi\,\mathrm{Pad}_{30}(E_\mu(G)),\qquad
Z_t=\sqrt{\bar\alpha_t}Z_0+\sqrt{1-\bar\alpha_t}\epsilon.
\]

Pad 随机复制已有面，\(\Pi\) 表示随机排列；\(\epsilon\) 是各元素独立的标准高斯噪声。网络联合预测全部 `[B,30,32]` 元素的噪声。slot 对应的是同一 CAD 的面集合，不具有固定面编号语义。

同一真实面的多个副本在加噪后不同，模型需要学会恢复其共同干净目标。最终去重恢复变长面集合；这使重复关系也成为生成任务的一部分。30 个 slot 全部保留不直接证明去重实现错误。

在固定相机、CAD 坐标和归一化约定下，省略相机条件，有：

\[
p(Z_0\mid I)\propto p(Z_0)p(I\mid Z_0).
\]

这是整组 latent 的联合分布。表示承载几何和拓扑，不意味着任意 latent 集合都能被 decoder 解成合法 solid。单图不能确定的隐藏结构应保留分布，而不是强制唯一的面数或背面形状。

在噪声时刻 \(t>0\)，定义 \(s_t=\nabla_{Z_t}\log p_t\)，则：

\[
s_t(Z_t\mid I)=s_t(Z_t)+\nabla_{Z_t}\log p_t(I\mid Z_t),\quad
p_t(I\mid Z_t)=\int p(I\mid Z_0)p(Z_0\mid Z_t)\,dZ_0.
\]

这个恒等式解释图片如何改变去噪方向，不指定网络必须分为两个模块，也不指定 concat、cross-attention、层数或宽度。条件残差层不能自动等同于右侧的 likelihood 梯度。高噪声单步均值的渲染也不等于这个积分。

未知相机时需要讨论 \(p(Z_0,c\mid I)\) 或边缘化相机；这里的固定相机推导不能直接当作未知实拍相机协议。

## 较早提议：公共主干结构

保留旧 1024 主干的完整计算结构，条件采用已有的全局拼接接口：

\[
g_\phi(I)=\mathrm{MLP}_\phi(\mathrm{DINO}_{CLS}(I))\in\mathbb R^{256},
\quad
h_0=\mathrm{Concat}(P_{32\to768}(Z_t),\mathrm{Repeat}_{30}(g))+T_{1024}(t).
\]

- 输入投影：Linear 32→768，LayerNorm，SiLU，Linear 768→768。
- 图片投影：冻结原 DINO CLS 1024 维；Linear 1024→1024，LayerNorm，SiLU，Linear 1024→256。
- 无条件输入：投影后的完整 256 维向量显式置零；不是将图片或 DINO 特征置零后再过带 bias 的 MLP。
- 新图片投影末层的权重和 bias 建议零初始化；完整旧主干加载后，训练开始时条件与无条件输出均能复现旧 prior。上游图片投影通常从末层开始更新后才获得梯度。
- 每层：pre-LN self-attention（16 heads×64）与残差；pre-LN FFN（1024→2048→1024，ReLU）与残差；共 24 层，参数不共享。
- 最终 LayerNorm；输出 MLP 为 Linear 1024→1024，LayerNorm，SiLU，Linear 1024→32。
- DDPM、latent 去重、冻结原版 VAE 和后处理使用同一接口。无固定 face-slot 位置编码；应满足 \(f(\Pi Z_t,t,g)=\Pi f(Z_t,t,g)\)（推理模式）。

完整保留旧投影和时间注入；仅把当前 `hidden_dim` 改为 1024 不等同于上述结构。

这条路径实际以 \(g(I)\) 为条件。要达到 \(p(Z_0\mid I)\)，图像特征必须保留任务需要的信息；维度 256 本身既不能证明充分，也不能证明不足。HoLa §5.6 已讨论全局一维条件向量和 padding/de-dup 的局限。

保留 patch 并通过 cross-attention 读取是针对图像信息压缩的备选。它可以保留 face 排列等变性，也不需要预先将 patch 与 face 一一对应；但 attention 权重不自动成为几何对应。后续比较保持 1024 主干与相同数据/训练预算，仅改变条件读取接口。本轮不将其设为必需模块。

## 较早提议：训练与之后的无条件验证

令 \(m\sim\mathrm{Bernoulli}(1-p_{\mathrm{drop}})\)，采用：

\[
\mathcal L=\mathbb E\|\epsilon-\epsilon_{\theta,\phi}(Z_t,t,m\,g_\phi(I))\|_F^2.
\]

先验证 \(g=0\) 的公共主干，再从该 prior 初始化条件阶段；条件阶段共同训练主干和图片投影，随机丢弃图片条件。保留原版 VAE。条件丢弃提供持续的无条件监督，不保证微调后无条件质量绝不退步，也不等同于必须启用额外的 CFG 强度放大。

已经有无条件 checkpoint，首先应验证公共主干严格承接其权重和推理行为，不需要为证明已知旧能力重新从零训练。若研究问题是当前数据协议下的重新训练能力，再单独以这份完整 1024 结构运行，固定数据、预算和评估。旧/new 数据分布不同时，权重复用属于初始化或适应，不是等训练条件的宽度消融。

评估沿用已有合法 solid、MMD-CD/COV/JSD、LFD 与复杂度等指标及逐件失败分母，同时记录去重后面数。之后条件阶段需分别检查关条件的生成能力、开条件的可见几何一致性及固定采样预算的多样性；单独的噪声 loss 不能代替这些证据。

## 原始文献

- [HoLa](https://arxiv.org/html/2504.14257v1)：§3–4.3 表示与生成，§5.4 单图多候选，§5.6 条件压缩和 padding 局限。网络精确宽度以 checkpoint 和源码审计为准。
- [DDPM](https://arxiv.org/abs/2006.11239)：高斯加噪、噪声预测与反向采样。这里使用已审计代码的 epsilon 参数化；不把论文简写的 latent 重建损失当作 checkpoint 配置。
- [Classifier-Free Diffusion Guidance](https://arxiv.org/html/2207.12598v1)：Algorithm 1 的有条件/无条件共同训练；条件丢弃与推理 guidance 强度是两个不同选择。
- [Latent Diffusion Models](https://arxiv.org/abs/2112.10752)：cross-attention 是已有条件注入方式，不构成其必然适合当前 HoLa 训练的证据。
