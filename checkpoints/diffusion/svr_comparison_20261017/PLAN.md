# SVR 当前模型与历史 HoLa 的配对比较

## Task

比较用户停止后的逐层条件 diffusion 与指定 HoLa SVR checkpoint，输出有固定分母的定量表及可视化。
用户记忆中的 CLIP 最佳版本单独核验；不凭名称或记忆把指定 checkpoint 自动认定为该版本。

## Assumptions

本次是开发阶段 validation 对照，不使用 test 调整代码或规则。模型的训练步数、历史数据/增强与架构可能不同，
结论属于 checkpoint/系统比较，不是等预算架构消融。默认保留每个 checkpoint 配套的冻结 AE；若 AE 不同需明确记录。
不启动训练，不修改网络或后处理策略以改善本次结果。

## Inputs

- 历史：`/mnt/d/ckpt/HoLa/Diffusion_svr_sq30_1500k.ckpt`，读取元数据确认1,500,000步；源码按state keys/shape恢复并严格加载。
- 当前最佳：`checkpoints/diffusion_layerwise_svr/20261016T143450.891057Z/weights/epoch=0002-step=00054639-val_loss=0.042497.ckpt`。
- 当前最后保存：同run的`weights/last.ckpt`，已读取确认为182,130步；日志至少198,650步不能当作已保存权重。
- 历史alignment候选：`/mnt/d/data/new_cond_ckpt/0423_deepcad_natural_blender_align.ckpt`，150,000步；继续追溯同一run的callback后，在查看本轮结果前定位其保存的best `outputs_brepnet_diffusion/Diffusion_0413_deepcad_natural_blender_align/img2brep/ya2o4xi1/checkpoints/epoch=856-step=250000.ckpt`，作为额外候选。两者不是已确认的用户所指历史最佳；历史run训练/验证使用同一750件小集，其val loss不可当独立验证指标。
- 图片：`/mnt/d/data/deepcad_v7_cond/{ID}/imgs.npz`的`svr_imgs[0]`；相同像素，经各模型核验过的原预处理。
- GT：`/mnt/d/data/deepcad_v7/{ID}/data.npz`及`normalized_shape.step`，仅用于评估，不进入采样或修复。
- validation源清单：`src/brepnet/data/list/deduplicated_deepcad_validation_7_30.txt`，3505件，SHA256 `b6403d9b235db7ebeaac76ba3a9d076440f149863ac692ae938129e20404ee32`。

## Outputs

受Git管理的固定100件清单、源码/权重/协议审计与紧凑汇总、可复现比较脚本。
原始NPZ、STEP、逐对象运行状态与诊断图放本主题的运行产物目录；可视化数值对照表另存PNG/SVG。

## Processing Protocol

按`sha256('svr-compare-20261017-seed42:' + ID).digest()`排序源validation IDs，取前100，推理前冻结；不按成功或复杂度换样本。
每模型每图K=1，batch=1；样本seed由固定字符串与ID的SHA256派生，各模型相同；FP32、eval/no_grad、1000步原DDPM，无CFG、无GT面数/拓扑/ordering。
各模型以对应原始代码严格加载；相同GPU型号/线程设置，CUDA同步计时，逐样本日志记录输入hash、seed与完成状态。
保留原生初始噪声的生成设备：历史模型在CPU生成后搬到GPU，当前模型直接在GPU生成；相同seed不代表初始噪声逐元素相同。
原坐标评估，不做最优旋转、缩放或ICP，不使用GT选候选。
后处理复用`tools.vae_compare_post`隔离runner：legacy policy、drop_num=0、CPU、名义200次优化（原实现可自适应扩展）、180秒单件硬预算含几何评估、16并发、无额外重试。
CLIP版本如确认权重/源码可得，则在同一清单/协议下加一行；无法确认则单列缺口，绝不以别的权重冒充。

## Metrics

- raw推理覆盖率/100；原始面/边CD²（双向平方最近邻均值之和），面数量绝对误差。
- raw face primitive F1：按面CD²构建cost、Hungarian匹配、cost<0.1计TP；不是点级阈值F1。
- STEP产出/100、原生精度下可读STEP及BRepCheck合法TopAbs_SOLID/100；0.1容差兼容结果仅作旁列。
- 后处理面/边CD²：每个STEP面1024点、每边128点，原生坐标；分别报告各自可计算集合与共同可计算集合。
- CUDA同步推理秒/件、后处理时间和超时数；比较表注明N、K、checkpoint步数与实际指标分母。

## Decision Rule

先看固定全集的覆盖/solid比例，再比较共同可计算集合的几何质量；训练loss不替代重建指标。
只报告实测方向与幅度，100件/单seed属于本次validation开发结果；不宣称真实照片泛化或全领域SOTA。

## Edge Cases

缺失、非有限、加载不匹配、无STEP、非法solid、超时与指标错误各自记状态，保留原分母；不使用原点fallback。
历史CLIP含义、旧数据/相机约定或旧split无法确认时如实标记，不进行未经验证的权重重命名或宽松加载。
已有历史产物不覆盖；算法失败不重采样；基础设施错误修复后的恢复必须记录原因与原记录。

## Validation / Sanity Checks

核对100件源数据可用性、旧训练/验证split关系、checkpoint实际step/参数及AE来源。
在名单首件检查两端条件张量、输出字段/尺度、严格加载，再完整跑名单；首件仍计入固定分母，不按其结果调算法。
验证原始导出兼容后处理合同、CD/F1已知几何样例和失败分母汇总；报告代码与配置hash。

## Implementation Constraints

仅本主题加入比较/兼容运行代码，不改当前训练架构、不把legacy模型宽松装进新模型。
复用现有数值及OCC runner，避免统一eval的缺失原点fallback与目录交集汇总；不修改OCC公开签名。
源码/当前运行已有未提交工作保留；数据和权重不提交。

## Deliverables

定量可视化表、机器可读summary、全部100件状态与协议、可复现脚本、源码/权重识别说明和限制。
