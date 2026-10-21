# 单图 diffusion 偏好数据与训练实现协议

状态：2026-10-20，用户已授权实现、测试并启动数据生成，启动后不持续监看；训练代码实现并验证，不启动正式训练。本文取代 `single_view_design_20261020/PAIR_SELECTION.md` 的早期v2规则。

## 1. Task

保留原HoLa单图SVR的1024维、24层concat网络和原VAE，从同图8个生成结果中选择相对优劣对。冻结生成模型、后处理和指标协议，建立有效性与几何两类偏好数据，并实现原DDPM监督加Diffusion-DPO的训练接口。

## 2. Assumptions

- `/mnt/d/ckpt/HoLa/Diffusion_svr_sq30_1500k.ckpt` 完整加载，使用其中原始AE与DINO；不换UV VAE，不混用新版latent空间。
- 每图固定8候选，不为凑pair重抽；同图最多一对，有效性优先，合格对内固定seed均匀选取。
- 已确认的阈值是工程设计默认，不是实测最优值：CD²改善5%、容忍退步2%；F1改善0.02、容忍退步0.01，后两者是绝对百分点。
- 不再使用旧稿Q90/Q10正样本门槛，不要求每项都改善，不使用“2票胜1票”的多数投票。
- 几何偏好针对完整配对GT，包括隐藏面，不能保证保留所有合理补全。

## 3. Inputs

- train 46,627 CAD：`src/brepnet/data/list/deduplicated_deepcad_training_7_30.txt`。
- validation先用已冻结100件：`checkpoints/diffusion/svr_comparison_20261017/validation100.txt`；仅作规则/运行验证，不训练。
- CAD：`/mnt/d/data/deepcad_v7/{id}/data.npz`、`normalized_shape.step`。
- 灰模：`/mnt/d/data/deepcad_v7_cond/{id}/imgs.npz["svr_imgs"][0]`。
- 原GT cache：`/mnt/d/data/ae_cache/1119_deepcad_aug1_11k_24/{id}_0/features.npy`。三件核验发现与旧checkpoint内AE的FP32重编码mean有0.0049–0.0078的最大绝对差异，来源不能完全确认；本轮不复用该cache。
- 生成时使用同一个冻结旧AE重新编码rotation0 GT，保存run内`gt_cache/{id}_0/features.npy`，FP32 mean32/std32，DDPM仅用mean。
- 不读取test个例，不使用FLUX/实拍/ABC。图片与latent身份、坐标、rotation0必须一致。

## 4. Outputs

独立时间戳run中保存协议/源码hash/命令、固定split、每候选原始 `[30,32]` latent、去重映射、raw预测、recover_v1 STEP、指标重复测量、状态和错误；每图选择记录及两侧指标。train `pairs.jsonl` 与validation候选/选择严格分开；所有尝试保留分母。

## 5. Processing Protocol

1. CPU规则/指标测试、真实旧checkpoint采样等价检查、真实小规模生成→后处理→选对接口验证。
2. 启动独立后台生成进程，先validation100×8，再train46627×8，阶段之间不改变规则。不持续监看长期运行。
3. validation若没有任何合格pair，程序停止并记录原因，不继续大量生成；其他阈值不自动调节。
4. 初始仅GPU4/5/6/7；2026-10-21用户授权扩到GPU0–7，每卡2个独立进程。仍逐seed原生DDPM1000/FP32/batch1采样；候选完成后交给CPU后处理队列；有界队列避免无界产物积压。并行进程/跨GPU与原候选的逐位一致性及调度迁移见`performance_20261021/`。
5. recover_v1，optimized/drop0、名义200步/自适应600，每候选180秒整隔离进程预算，内部修复共享预算且不读GT。
6. 独立OCC子进程隔离崩溃/超时；指标原始值与失败原因保存。启动中断后只显式resume，不覆盖旧run。
7. 数据生成可靠并启动后，完成偏好训练实现/验证，输出命令但不启动训练。

## 6. Metrics

- raw：原面/半边固定网格的CD²，以及面primitive F1。
- post：最终STEP每面1024点（面内按三角面积）、每边128点（参数均匀）的CD²与面primitive F1。
- CD²=双向最近平方距离均值之和，不取根号、不除2。F1用Hungarian、primitive CD²<0.1计TP。
- 同图同轮共享GT采样；post做3个固定seed重复测量，仅当所有重复都支持同一标签时接受，不用均值隐藏标签反转。
- native validity：按STEP原存储容差检查SOLID与BRepCheck；不强制0.1。保存容差诊断，不宣称工程精度认证。
- FE/EV、LFD、数量等不用于本版标签；后续完整模型评估可另行计算，不阻塞标签数据生成。

## 7. Decision Rule

定义A相对B“没有超出退步上限”：

```text
face_cd(A) <= 1.02 * face_cd(B)
edge_cd(A) <= 1.02 * edge_cd(B)
face_f1(A) >= face_f1(B) - 0.01
```

定义“至少一项明确改善”：

```text
(face_cd(A) < face_cd(B) and face_cd(A) <= 0.95 * face_cd(B))
or (edge_cd(A) < edge_cd(B) and edge_cd(A) <= 0.95 * edge_cd(B))
or face_f1(A) >= face_f1(B) + 0.02
```

- A合法、B确认几何无效：比较raw三项，均不超退步上限即可。
- A、B都合法：比较post三项，每次重复都满足不超退步上限，并至少一项明确改善；不再额外叠加raw条件。
- CD基准为0时不能靠0→0获得“改善”；退步仍按乘法边界检查。
- 优先枚举有效性pair；没有则枚举几何pair；固定seed选择一对。全无效、指标冲突、差异不足、重复测量翻转则没有pair。
- unknown运行故障不是negative；必需指标缺失不配对。该方案可能得到较少pair，但不为凑数量临时放宽。

## 8. Edge Cases

- 缺文件、OOM、未知超时、进程崩溃、程序异常：unknown，保留失败分母，不当普通低分。
- 正常完整后处理后的非solid/BRepCheck失败可为invalid；无STEP仅在诊断确认几何失败时为invalid。
- 无边、NaN、不可采样：依赖该项的指标缺失；不填原点或无穷大。
- validation/test不进入梯度；无pair训练图片仍参加GT DDPM。
- 不将去重/后处理后几何重新编码或补齐成训练latent。
- protocol/模型/指标/输入改变时拒绝混用run；显式resume复用可核验已完成结果。经验证且明确指定`--accept-scheduling-update`，仅允许生成编排与运行记录源码迁移；保留原protocol及fingerprint，另存每次runtime源码、diff、资源、Git和命令，不静默替换来源。

## 9. Validation / Sanity Checks

规则边界、相反方向、零基准、F1百分点、同图约束、unknown负例排除、重采样不稳定、每图最多一对。真实GPU旧采样等价、原始latent和导出、真实OCC/post face F1与共享GT点。训练验证DPO梯度方向、reference冻结、DDPM保留、pair latent不变、严格加载/resume和真实小批量loss/梯度；不把这些作为研究效果结论。

## 10. Implementation Constraints

数据编排与选择写入 `src/brepnet/data/`；几何评价复用 `eval/` 与既有OCC内核；旧模型接口与偏好loss在 `diffusion/`；训练沿用统一Hydra/Lightning入口。明确参数、简短注释，不静默fallback。训练checkpoint/reference/数据协议匹配，VAE/DINO冻结，不复制两份无用AE，不在训练step调用OCC。

## 11. Deliverables

生成与训练源代码、配置、相关测试报告、已启动生成run位置/进程/日志；训练命令和状态“未启动”。保留用户未提交文件，提交本任务变更并检查后台同步状态。
