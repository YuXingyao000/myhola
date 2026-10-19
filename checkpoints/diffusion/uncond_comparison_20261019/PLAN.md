# 无条件生成：测试与数据对比

## Task

比较旧HoLa、新版last和新版最低val-loss checkpoint，仅收集数据与图表，不改模型、训练或后处理规则。

## Assumptions

旧模型1,000,000步；新版last实际303,500步，最低val-loss为276,185步。不是等训练量消融。
三者均使用相同1119 AE，sampling为FP32、TF32关闭、1000步linear DDPM、一次生成/seed。
不输入图片、GT面数或拓扑；无条件生成不计算逐件重建CD/F1。

## Inputs

- 旧结果：`../uncond_baseline_20261018/`，固定1000个ID和seed，原始输出与后处理只读复用。
- 新版：`checkpoints/diffusion_uncond/20261018T085204.415712Z/weights/last.ckpt`及`epoch=0090-step=00276185-val_loss=0.027705.ckpt`。
- 参考集：当前validation清单按`sha256('uncond-ref-20261019:'+CAD_ID)`排序取前1000，使用v7 GT mesh；test不参与本轮开发评估。
- 新颖性参考：当前训练清单全部46,627件，使用原始几何与连接。

## Outputs

每模型raw/post/逐件状态；checkpoint身份、参考清单、指标协议；受Git管理的紧凑CSV/JSON/PNG/PDF。
报告仅给实测值、有效分母、失败与缺失原因，不写架构优劣结论。

## Processing Protocol

1. 新版完整strict加载保存的配置与权重，核对AE704个tensor与1119相同。
2. 使用旧1000个seed；起始噪声沿用旧CPU RNG生成后搬GPU，保存hash；其余使用现有采样与VAE解码函数。
3. 沿用既有evaluate.py：legacy、drop0、名义200/自适应600次优化、CPU16并发、180秒/件、无算法重试。
4. 原生精度solid与0.1兼容validity分开，失败保持1000分母；旧结果复用前核对后处理源码hash。
5. 点云从post的裁剪表面mesh按面积采样2000点，GT同法；使用既有normalize_pc/align_pc与point_cloud_set定义。
6. 分别报告可三角化输出全集、原生合法且可三角化子集的分布指标及各自n；另用固定seed将三组合法子集降采样到相同数量作补充，不据指标挑样本。
7. Unique/Novel沿用4bit面几何+邻接图同构判据；Novel对完整训练清单精确匹配，用必要几何签名剪枝，不把缺nearest文件当作新颖。
8. Complexity调用既有实现（其有效性gate为0.1兼容口径），逐件保留不可计算原因。LFD使用现有工具链；依赖缺失明确N/A，不编造近似LFD。

## Metrics

生成/STEP/原生合法solid/兼容solid率及Wilson95%；面/边/顶点数量、环复杂度、平均曲率；MMD-CD、COV-CD、JSD；Unique/Novel；可用时LFD；单件采样与后处理耗时。
CD为双向最近点平方距离均值之和，MMD对每个参考取最近生成；COV为生成最近邻覆盖参考比例，JSD用既有28³网格。
数量分布单列，不能替代形状分布指标。所有成功子集指标显示n/1000；图表标明实际训练步数。

## Decision Rule

本轮不判断胜负、不依据结果修改模型或规则。last为主要训练产物，最低val-loss作为事先确定的第二产物，两者并列报告。

## Edge Cases

缺失/非有限/后处理失败/超时/三角化失败保持原记录；几何指标仅在可计算子集求值并列完整覆盖率。
空集合/依赖缺失标为null及原因，不输出伪0或把缺失视为新颖。保留原始几何，预处理不覆盖旧结果。

## Validation / Sanity Checks

核对实际step、完整strict加载、AE一致、CPU起始noise hash；首件属于正式1000。检验CD自距离、signature剪枝与直接同构小例一致，复核固定分母与CSV行数。

## Implementation Constraints

仅新增本主题脚本，复用现有模型、后处理和metrics函数；不建立新的训练/模型加载框架，不增加生产网络防御逻辑。保留`checkpoints/uv_face_encoder/`。

## Deliverables

具体数据表、对比图、固定样例图及机器可读逐件/集合结果；标清未完成指标与原因，不做方向性总结。
