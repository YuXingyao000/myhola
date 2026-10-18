# 历史 HoLa 无条件生成基线

## Task

按用户指定的旧架构恢复 `Diffusion_uncond_1100k.ckpt`，生成固定1000个样本，建立供后续新版无条件模型比较的基线。
输入是随机噪声；每面32维latent经MLP变为768维，与256维零条件拼接后进入1024维、24层Transformer。
本轮不训练新版模型，不改训练主干、VAE或后处理算法。

## Assumptions

- 已核对checkpoint实际为1,000,000步、epoch324；名称中的1100k不作为步数证据。
- 旧模型预测epsilon，L2损失，1000步linear DDPM；保持其训练对应设置，不套用当前cosine配置。
- 此次先建立生成覆盖、实体有效性及复杂程度的基线；面边数量分布与随机图库不代替正式形状覆盖率或新颖性指标。
- 历史源码使用已恢复的固定Git版本，严格匹配全部state keys和shape；这不证明该版本恰好是训练时的commit。
- 不使用图片、GT面数、GT拓扑或test样本；本轮生成不需要逐对象GT。

## Inputs

- 权重：`/mnt/d/ckpt/HoLa/Diffusion_uncond_1100k.ckpt`，启动前保存完整SHA256、文件大小、配置与实际step。
- 原始源码候选：`c2660c00c6a9c8f9f82bd5b9f4d3c774fa38d361`中的`Diffusion_condition`与`AutoEncoder_1119_light`，以严格加载验证兼容。
- AE使用完整diffusion checkpoint内权重；另核对与当前1119 AE的704个tensor一致性，不覆盖来源。
- 固定样本编号`uncond_000000`至`uncond_000999`；seed为`sha256('hola-uncond-20261018-v1:' + sample_id)`前4字节按little endian解释的无符号整数。

## Outputs

`raw_predictions/`保存原始预测NPZ；`post_inputs/`仅适配整数索引dtype；`post/`保存原始后处理产物；`metadata/`保存逐样本状态、日志、来源、冻结协议和源码快照。
受Git管理的`report/`保存紧凑summary、逐样本CSV、随机结果图库与旧模型架构图；`README.md`记录事实、结果及限制。

## Processing Protocol

1. 固定1000个编号与seed，完整权重严格加载。先在首样本检查张量、一次去噪与完整采样；首样本属于正式分母，不按结果替换。
2. 保留原生`inference(bs=1, device, v_data=None)`；FP32、eval/inference_mode、TF32关闭、1000步DDPM、无CFG，每件仅一次提议。
3. 起始噪声沿用旧实现的CPU生成后搬至GPU；Python、NumPy、torch CPU/CUDA按逐样本seed重设，记录初始噪声hash与采样耗时。
4. 每GPU一个独立进程，使用8张空闲RTX4090分片。线程预算每worker为1；禁止改动其他任务进程。
5. 保留原始导出；历史float connectivity仅在有限、精确整数且往返相等时转换int64，另存并记录hash，几何和索引值不改变。
6. 后处理采用与SVR比较相同的legacy策略：CPU、drop_num=0、名义200次优化（原实现可扩展到600），每件180秒硬预算、16并发，无重采样/算法重试。
7. OCC先按STEP保存精度读取并检查TopAbs_SOLID与BRepCheck；0.1容差兼容结果若计算，独立列出，不能替代原生有效率。不依赖GT计算或选择输出。
8. 随机图库在看到结果前取seed清单前24件；失败显示失败，不换成成功样本。

## Metrics

- 原始推理成功数/1000、可用于后处理的导出数/1000、STEP产出/1000、原生合法solid/1000；有效率附Wilson95%区间。
- 原始面数、有向半边数；STEP面/边/顶点数量，均报告各自可计算数、分布及分位数。
- 分开报告生成少于7面、7至30面、超过30面的比例；不按面数过滤主有效率分母。
- 推理与后处理耗时、超时及错误分类；全部1000件在汇总中有一行。
- 这些指标不包括逐件重建CD、GT面数误差，也不声称已经测得COV/MMD/LFD。

## Decision Rule

本轮只建立旧checkpoint在固定协议下的实测基线，不据此判断新模型或cross-attention优劣。
后续新版需真正接受无条件训练，并对齐VAE、数据、训练量和评估协议；原论文数字因过滤与有效性容差不同，不能直接当作本轮验收阈值。

## Edge Cases

缺失、非有限、加载失败、空几何、后处理失败、无STEP、非solid、超时都单独记录，保留1000分母。无原点替代、失败剔除或自动补样。
已有结果仅在权重、代码、协议hash一致时复用；未完成记录标记interrupted并保留。基础设施错误若修复，必须保留原记录并写明恢复原因，不能重采样几何失败。
没有可计算值时输出null和n=0，不输出NaN或误填0。

## Validation / Sanity Checks

检查32→768→拼零256→1024及时间嵌入位置；strict加载missing/unexpected为0；全参数已离开meta。
检查原生`None`条件与显式零条件一次去噪数值相同，输出为[B,30,32]；真实首样本完整DDPM与AE解码有限。
验证无GT的后处理runner、整数dtype适配、失败分母、Wilson边界与确定性清单；图库包含预定24件而非人工筛选。

## Implementation Constraints

仅在本主题增加恢复和评估代码，复用既有旧源码隔离工具及OCC检查函数；不修改当前网络和历史产物，不改变PythonOCC调用签名。
保留`checkpoints/uv_face_encoder/`的未提交工作。数据、权重和生成几何不提交Git。
研究说明使用中文，Python标识符英文，函数职责清晰且带简短注释。

## Deliverables

可复现的旧模型加载/采样/后处理命令，1000件完整状态及汇总、随机图库、准确标注768＋256拼接的架构图，以及可供新版沿用的冻结协议。
