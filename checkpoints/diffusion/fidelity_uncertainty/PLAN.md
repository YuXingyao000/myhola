# 新版DPO数据 → 重训对照 → 几何辅助训练

## 1. Task

2026-10-23，按用户纠正收窄：重评已有候选并重选win/lose，使用新数据重训DPO比较，然后进入冻结decoder几何训练。2026-10-25 更新：第一阶段全量重选[验收通过](DATASET_RESULTS.md)，第二阶段 5,000 步训练[checkpoint 已核验](TRAINING_RESULTS.md)，完整 3,505 件 validation 对照亦已完成，[结果见 README](README.md)。本轮新标签未呈现明确整体收益；几何项和继续DPO两组训练均已完成额外5000步，[完整validation已启动](geometry/README.md)，尚无下游结论。执行契约见[DATASET_SPEC.md](DATASET_SPEC.md)，实现验证见[CPU_VALIDATION.md](CPU_VALIDATION.md)。[历史宽泛草案](history/PLAN_v1_broad.md)不再作为执行队列。

## 2. Assumptions

固定原1024维concat SVR、原HoLa VAE、灰模view0与recover_v1。首轮复用每图8候选，仍最多一对，不同时换decoder/生成policy。新增观测阈值是建议起点，非验证最优值；正式制对前以validation和已知几何反例核验测量稳定性并冻结。训练仅使用train，test不调规则。

## 3. Inputs

- 既有候选：`checkpoints/diffusion/preference_data/20261020T134930Z`；train 46,627件，旧pair 42,037对。保留原始`[30,32]`生成latent，不重编码STEP。
- 原模型/冻结reference：`/mnt/d/ckpt/HoLa/Diffusion_svr_sq30_1500k.ckpt`。
- 旧DPO：`checkpoints/diffusion_preference_svr/20261023T030527Z/weights/steps/step=00005000.ckpt`。
- validation及已有对照：`checkpoints/diffusion/preference_eval/20261023T090834Z`。
- 数据/相机/post沿用[原训练契约](../failure_analysis/20261023/TRAINING_CONTEXT.md)与[评测工具](../../../src/brepnet/eval/SINGLE_VIEW_EVALUATION.md)，运行前核对protocol/hash。

## 4. Outputs

新版候选评分、pair manifest、每图选择/弃权理由和旧新pair变化；新DPO与旧DPO报告；后续有/无几何损失的训练对照。旧数据、权重、run和失败分母全部保留。

## 5. Processing Protocol

1. **新版数据**：复用8候选，补算第6节观测指标，按第7节选对。每图优先合法/非法对，否则双合法对；合格集合中按seed42+image_id抽一对。无pair对象仍用于GT去噪。报告pair保留/替换/翻转/弃权及覆盖。
2. **新DPO重训**：从原SVR初始化，reference仍原SVR；`L=L_ddpm+L_dpo`、beta20、lr1e-5、全局batch64、seed42、5,000步，沿用旧精度/优化器/clip。与现有旧DPO step5000比较，隔离标签变化与额外训练量。固定3,505件validation、K=1、同seed/VAE/post，报告合法率、观测指标、完整raw/post几何。已有100件K=8仅作同图变化附录，不作全量多样性结论。
3. **几何训练**：第二步评估后开展；新数据未显出价值时先报告，不直接叠加几何项。选定起点后分叉：继续相同DPO的控制组，以及加几何项组；均建议再5,000步、相同优化器初始化、reference与数据曝光。2026-10-25 根据已完成 validation，建议保留旧标签 DPO step5000 + 旧 pair 为下一阶段共同起点，reference 仍为原 SVR，避免把标签变化和几何项同时引入。此为研究建议，尚未启动训练；基于开发 validation 作出的选择不能视为独立最终评估结论。

几何项在GT去噪分支由epsilon预测还原clean latent，再按padding/排列来源还原每面输入，经冻结decoder输出全局XYZ。首版直接监督真实GT raw采样点：利用已知GT面对应/owner pair，分别计算每面/每边双向平方Chamfer，再各自平均相加。

`L = L_ddpm + L_dpo + lambda_geo * L_geo`

decoder参数冻结、eval，但保留latent梯度；不经过OCC/NumPy。GT owner pair只用于训练，推理仍预测连接。建议仅在SNR(t)>=1时加几何项；权重在固定train校准批检查梯度量级后写入配置并固定，不根据validation反复搜索。`D(z_GT)`只作decoder误差参照，不另设训练分支。点损失不直接监督漏边存在性，最终合法性仍由下游检验。

## 6. Metrics

| 指标 | 配对用途 |
|---|---|
| V：native valid solid | 合法/明确非法/unknown |
| S：输入轮廓IoU ↑ | 前景与背景孔洞 |
| C：可见表面双向平方Chamfer ↓ | GT辅助的可见几何监督，不是无GT推理评分 |
| B：轮廓边界距离 ↓ | 局部边界偏移，不覆盖所有内部孔槽 |
| raw面CD²、边CD²、面F1 | 仅合法/非法pair保留原粗几何退步保护 |

S/C/B沿用现有224²相机与公式，要求保存STEP的面能完整三角化。完整post GT几何继续评估，不独自决定两个合法候选的胜负。新内部槽线指标、多pair与新policy重采样不是本轮前置项目。

## 7. Decision Rule

拟选winner为W、loser为L。

**观测不明显退步，全部满足：** S(W)>=S(L)-0.01；C(W)<=1.02*C(L)；B(W)<=1.02*B(L)。

**观测明确改善，至少一项：** S(W)>=S(L)+0.02；或C(W)<=0.95*C(L)且严格小于；或B(W)<=0.95*B(L)且严格小于。IoU的0.01/0.02是1/2个百分点。

| 情况 | 判断 |
|---|---|
| W合法、L明确非法 | 观测不明显退步，且raw面/边CD最多恶化2%、raw面F1最多下降0.01；满足则W胜，不要求观测额外改善 |
| 两者合法 | 观测不明显退步且至少一项明确改善，则W胜 |
| 两者非法、存在unknown或必要指标缺测 | 不配对 |
| 双方合法但近似持平，或改善/退步冲突 | 不配对，不靠隐藏部分GT差异打破平局 |

弃权只代表没有可靠偏好，不证明两补全同样合理。完整GT仍通过raw保护参与合法/非法pair；它不参与双合法候选的隐藏差异排序。无合格pair不重抽、不放宽阈值。

## 8. Edge Cases

非法STEP无法完整三角化或缺必要指标，不自动判输。缺失/超时/NaN/空几何留档；零对零不算改善。距离用乘法比较避免除零，近零变化需核验稳定性。合法标签不因后续测量失败改写。几何训练无低噪声样本时辅助项为零；空边避免NaN；复制slot不能当独立真实面或重复加权。

## 9. Validation / Sanity Checks

核验配对阈值/冲突/零基准/缺测排除/确定性选择与同相机重渲染。几何接口用真实batch验证shape/loss/latent梯度、decoder参数不更新、GT对应及不同面数。小批量只验收实现；效果由固定validation最终采样评价。

## 10. Implementation Constraints

沿用现有模块，不覆盖旧run。两次对照分别只改pair数据、只加几何项。复用旧DPO结果前核对代码/配置/seed/hash；不一致时按同协议重跑旧标签控制。记录GPU小时，等步数不宣称等计算成本。

## 11. Deliverables

- [x] 第一阶段重选代码、loader 协议兼容、CPU 测试与手动运行说明。
- [x] 新候选评分、pair与变化/覆盖报告。
- [x] 新数据DPO 5,000步训练与完成核验。
- [x] 新旧DPO的完整validation同协议重建比较（本轮未见明确整体收益）。
- [x] 几何项代码、严格初始化/恢复、真实 CPU 梯度检查及手动命令。
- [x] 有/无几何项的同起点训练运行（各额外5000步，严格权重与配置核验通过）。
- [ ] 有/无几何项完整validation下游对照（已启动）。
