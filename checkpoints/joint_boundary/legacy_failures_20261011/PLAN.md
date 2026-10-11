# 原版 VAE 后处理失败案例集：冻结执行协议

## Task

从已完成的同 latent VAE 对比中收集 50 个原版 Decoder 在 optimized 后处理后未形成有效 solid 的案例，提供可追溯数值、逐件观察和可浏览的共同讨论页面。用证据约束后续架构设计；本任务不训练或修改模型。

## Assumptions

- 数据：使用 comparison_20261011 的 posterior_mean / legacy 输出。它是 GT B-Rep 编码后的重建诊断，不是照片条件生成失败集。
- 边界：筛选集合为 2,417 accepted test IDs 中 native_valid_solid=false 的全部 369 件；不是 geometry_error 集合，也不以 success.txt 代替有效性。
- 实现：CPU PythonOCC 环境沿用已有评估；原始输出只读。新增重放诊断有单独目录，不替换原评估或提高其成功率。
- 未确认：每个失败的根因。数值、可见现象、日志所指阶段与架构假设分别标记。

## Inputs

- `checkpoints/joint_boundary/comparison_20261011/` 的 raw_metrics、post/{fit_only,optimized}、metadata、raw_predictions。
- `/mnt/d/data/deepcad_v7/{id}/{data.npz,normalized_shape.step}`。
- 原版 checkpoint `/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt`；本任务不重新推理。
- 当前 Git 与原比较协议中的文件指纹。

## Outputs

- 冻结的 50 件清单及来源／分层／指标，失败总体与成功总体的紧凑对照。
- 每件原版 raw、GT、原版 optimized STEP 的几何诊断；保留 fit-only 与 joint 的既有指标作为上下文。
- 多视图或同步旋转的 GT / raw / post 页面，显示无法三角化的面数，不能把不可渲染部分静默画成完整结果。
- PLAN、AUDIT、REPORT、README、可重建页面源码与验证记录。大几何、截图、日志留本地且忽略提交。

## Processing Protocol

1. 校验 2,417 intended、2,048 native valid、369 failed 和逐件唯一性。
2. 在深入逐件诊断之前冻结抽样：按原生输出类型（compound / shell / invalid solid）、native BRepCheck 是否通过、raw pair topology 是否精确分为九个非空观测层。各层先保留 1 件，其余名额按剩余层容量 `n−1` 比例和最大余数分配；余数相同按层名称字典序。层内按 GT 面数、SHA256(`legacy-failure-v1:{id}`)、ID 排序，配额为 q 时取零起始秩 `floor((j+0.5)*n/q)`，j=0…q−1，以覆盖复杂度，无人工替换。该选择依据总体统计、尚未依据深入 case 解释或渲染结果；有意保留稀有观测类型，50 件比例不用于估计总体发生率。成功／失败总体比较另按精确 GT 面数控制，并展示 1–9 / 10–19 / 20–30 的描述性分层。
3. 冻结 50 件 manifest 并记录输入表 SHA256。唯一 invalid-solid 案例必须因所在层纳入。
4. 分析原始导出中的几何误差、reciprocal half-edge 不一致及拓扑；对已保存 STEP 独立提取拓扑、开放边、无效子形状和可渲染性。
5. 必要时同预算重放这 50 件后处理并仅增加诊断日志，记录与原输出是否一致。差异不能改写原结论。
6. 页面统一使用同坐标系和相机，按观察过滤并能跳转任意 ID。每件显示来源、已知事实和未证实的解释。

## Metrics

- 原指标：native/compat validity、shape_type、BRepCheck、face/edge CD²、face MAE、pair precision/recall、reciprocal CD²、面／边数量及耗时。
- 附加指标：原生 face/edge/wire/shell/solid 数；基于拓扑身份的边-面 incidence 与 free/nonmanifold edge 数；逐面可三角化状态；可能的半边端点分离及曲线到面距离须明确单位与离散近似。
- 拟合／裁剪／缝合 trace 只表明失败出现的位置，不单独证明学习架构的根因。
- 聚合区分失败全集 369、案例集 50、成功对照 2,048。分层案例集的比例不冒充总体发生率。

## Decision Rule

- 纳入条件仅为原评估 `native_valid_solid=false`；缺失测量保留案例并显示 null / 错误，不能换成更容易画的案例。
- 一个观察可支持研究方向，但未经控制干预不得写成确定因果。native solid、compound/shell、几何精度和可视化完整性分别呈现。

## Edge Cases

- STEP 可读但无完整三角网格：保留可读曲线／可渲染面，明确缺失；不拿 raw mesh 冒充 post mesh。
- seam／退化边对 incidence 的影响单列；free edge 数不能无条件解读为曲面间实际距离。
- 空数组、NaN/Inf、重复 ID、缺文件、超时分别报告。GT 同样检查，不预设所有资产完美。
- trace 重放若改变输出类型或 validity，保留原始分类并提示重放差异。

## Validation / Sanity Checks

- 清单唯一且恰好 50 件，全部属于冻结失败全集，各层配额一致，稀有 invalid-solid 案例存在。
- 数值函数使用有区分力的合成输入或已知 OCC 闭合／开放形状校验。
- 页面逐件资源存在，50 件可访问；检查桌面与窄屏、筛选／切换／相机操作及浏览器错误。
- 抽查多视图与原始数据一致；原评估目录无修改。

## Implementation Constraints

- 不改模型、训练配置或后处理算法；新增诊断工具放 tools/，记录与页面放本目录。
- 新产物不进入 experiments/。不提交数据、STEP、模型、截图大文件或凭证。
- 固定 CPU 线程与有界并发，OCC 危险批量步骤有子进程隔离／超时；不静默修复测量对象。

## Deliverables

可浏览页面、50 件 manifest、数值／trace 诊断、证据分级报告、复现命令与 Git 提交。架构建议围绕观察到的失败机制形成，不根据平均 loss 推断根因。
