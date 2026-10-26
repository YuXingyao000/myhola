# 照片偏好数据：完整 train 与 validation

## Task
用户授权从照片310000步权重重新生成偏好对，一次运行覆盖完整训练集与完整validation。产出候选、OCC评价、同图偏好对及可恢复协议；本次执行范围为生成数据，DPO训练另行启动。

## Assumptions
照片为已有FLUX合成真实感图，固定rotation0。保留此前K=8和选对门槛，避免同时改变训练方法及偏好定义。初始policy/reference拟使用照片310000步；本次候选仅由该冻结模型产生，不能混入灰模权重候选。训练与validation按CAD ID独立。

## Inputs
照片权重为 `checkpoints/diffusion/photo_baseline/evaluation_20261026/weights/photo_latest.ckpt`（step310000，SHA256 d8ba2f83fa6989ad235c021d2a41ca23d79505076a522954086597d575caa745）；原HoLa checkpoint（SHA256 9dcafb192529c1ddf65942392c25b409fbbf2bce407805f34a460d0b44bb4839）提供同源DINO/AE/历史配置。完整train46627、validation3505；输入 `real_photo.npz[flux]` uint8[512,512,3]，GT `data.npz` 与 `normalized_shape.step`。固定视角0，不使用test。

## Outputs
独立时间戳run：401056个预定候选的原始latent[30,32]、raw几何、去重映射、native STEP与指标/失败；每图最多一对，train发布pairs.jsonl，validation发布validation_pairs.jsonl；无pair对象也记录原因。单独GT mean32/std32缓存、输入/源码/权重摘要、进度、日志及启动/恢复命令。

## Processing Protocol
先完整validation3505×8，再train46627×8，无须人工在阶段间续接。若validation无偏好对也继续生成train，不按validation结果改阈值。8张4090，每卡2个独立FP32/batch1采样worker，CPU32、队列64；原生DDPM1000、TF32关闭、identity/candidate派生seed42，候选不重抽。冻结同源AE重新编码GT，只供后续DDPM监督；GT不进入候选生成或修复。原AE解码后使用recover_v1、drop0、max_optimize_iter200、整候选180秒预算、post指标3轮采样。

## Metrics
raw面/边平方CD（两向最近邻均值平方距离之和）与Hungarian面F1阈值0.1；post同类指标3轮采样；native SOLID类型及BRepCheck，独立保留超时、未知、计量失败及容差诊断。照片路径保留已完成的native有效性结论，即使后续三角化失败；post指标缺失不补零。

## Decision Rule
同图优先合法胜确认非法，且raw面/边CD退步不超过2%、面F1退步不超过0.01；否则选两个合法候选间的几何pair，三轮post均满足退步限制且每轮至少一项明确改善（CD降低5%或F1增加0.02）。固定seed从合格对中选至多一对。unknown不当负例，无合适pair不降门槛。GT仅用于离线标签；单张照片隐藏面可能有多种合理补全，这里的全GT偏好不等于观测唯一性判断。

## Edge Cases
缺失或损坏输入、GT编码失败记各预定候选unknown，不删分母；已记录输入/输出变化拒绝恢复。checkpoint/条件类型/源码/清单/预算变化拒绝复用。中断候选不reroll；完整候选恢复检查hash。train/validation重叠、重复或空ID拒绝启动。输出目录不能覆盖历史run。全图原始输入及缓存失败也保留独立记录以支持恢复。

## Validation / Sanity Checks
单测验证照片键/形状/颜色预处理、严格checkpoint来源、协议改变拒绝恢复、数据集隔离及unknown行为，兼容现有灰模入口。真实GPU用完整1000步重现既有validation c00，比较latent/raw逐位等价；两张validation与两张train的K8完整流水线验证选对与独立清单，恢复不重复生成/评价、不改标签。小规模仅验证实现，正式生成不复用预检结果。正式启动核对8卡16worker及CPU队列真实推进。

## Implementation Constraints
沿用preference_generate/io/selection/eval职责；独立来源模块处理照片条件与新checkpoint，显式配置而非按文件名猜模型。保持模型结构、采样与选择函数，保留用户未提交的其他DPO工作。运行时源码快照覆盖脏依赖；使用原环境，训练程序不自动启动。正式run生成期间不修改其计算源码。

## Deliverables
实现与针对性验证、已启动的完整validation→train运行、固定协议、日志/进度/可恢复命令、预计成本与Git提交。开始稳定生成后后台持续执行；完成状态以两个完整split汇总和最终manifest为准。
