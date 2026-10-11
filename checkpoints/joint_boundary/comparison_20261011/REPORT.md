# 2026-10-11 — 已训练 Joint VAE 与原 HoLa VAE 对比

**完整测试结论：当前新 Decoder 明显退步，尚不适合替换旧 Decoder。** VAE posterior 重建的曲面、曲线和自主拓扑均落后；在固定历史 diffusion latent 上，也未获得整体几何收益。重复边界是明确问题，但它不能解释全部差距。OCC 全量同样退步：几何优化后，原读取容差有效 solid 从旧版 84.73% 降至新版 32.19%。

本次只测试、审计和增加可复用评估工具，没有训练、调阈值或修改模型／后处理算法。代码与协议提交为 `d72ecb08`；[复现命令](README.md)、[冻结协议](PLAN.md)、[资产审计](ASSET_AUDIT.md)、[软件环境](environment.json)分别记录执行与边界。

## 比较成立的条件

- 固定 2,417 个 accepted test CAD，两个 Decoder 使用逐对象完全相同的输入 latent；两组均 2,417／2,417 成功导出，输入摘要全部一致。原始 test 为 2,424，另 7 件契约不支持，保留在总体覆盖率分母；本次输出覆盖原始清单的 99.7112%。
- 主组只 encode 一次，使用 identity 坐标、posterior mean、GT 面集合与顺序。Encoder 读取 GT B-Rep 是 VAE 重建任务本身；**Decoder 不读取 GT adjacency**，旧模型不隐式删面。它不是纯图像输入的端到端测试。
- Joint 在测试前固定为验证集最佳 step 97,559；原模型固定为 `1119_deepcad_aug1_11k.ckpt`。243 个 encoder／posterior 张量完全一致，原／新模型严格加载。精确路径、SHA256、训练划分与历史日志见资产审计。
- 这是两个已训练 checkpoint 的替换比较。旧模型约 181.6M 参数，新模型约 79.5M，冻结 encoder 约 73.7M；Decoder 的容量、global XYZ 与 local+bbox heads、目标函数、训练数据版本和训练预算都不同。不能将差异单独归因于双向交互，也不能直接否定共享边界表示。

同一个 latent 能被旧 Decoder 更好地解码，说明本轮差距直接存在于新 Decoder 的架构／训练组合；当前证据不能把它归咎于 encoder 丢失信息或 diffusion 生成失败。具体是容量、几何参数化、交互更新还是监督造成，仍需区分。

## 1. Posterior mean 重建：几何和拓扑都退步

以下均为全 2,417 件的逐对象宏平均；坐标是数据的归一化全局 XYZ，无配准或额外缩放。CD² 是两方向最近点平方距离的均值之和，保留全部激活边曲线；原始点为 UV grid／曲线采样，不是面积／弧长均匀采样。

| 指标 | Legacy | Joint | 比较 |
|---|---:|---:|---|
| 对应面 XYZ MAE ↓ | 0.00207324 | 0.00456291 | 2.20× 误差 |
| 面点云 CD² ↓ | 0.000050153 | 0.000237510 | 4.74× 误差 |
| 边点云 CD² ↓ | 0.000089235 | 0.001525565 | 17.10× 误差 |
| 唯一面 pair precision ↑ | 99.7366% | 95.1263% | −4.61 个百分点 |
| 唯一面 pair recall ↑ | 99.9136% | 98.8269% | −1.09 个百分点 |
| 唯一面 pair F1 ↑ | 99.8171% | 96.8011% | −3.02 个百分点 |

Joint 的 face MAE 仅在 63／2,417 件上更低，edge CD² 仅在 10／2,417 件上更低。中位数也退步，因此不只是少数异常对象拉高均值。训练中 face loss 达到 `1e-3` 数量级，不能取代与既有模型在统一全局坐标上的直接比较。

![原始重建比较](visualizations/raw_comparison.png)

图右栏揭示数量问题，但不同表示的曲线条数不能直接等同：

| 每对象平均数量 | 数值 |
|---|---:|
| GT 共享边界／不同面 pair | 29.2346 |
| Legacy 唯一面 pair | 29.3240 |
| Joint 全部激活 query | 69.4042 |
| Joint 唯一有效面 pair | 30.9636 |
| Joint 同 pair 的额外 query | 38.1589 |
| Joint 同面 assignment | 0.2818 |

旧模型平均 58.5912 条**有向半边**，正常互反两条不是重复；它没有同方向重复，平均单向 pair 为 0.0567。新模型的 38.1589 指按当前“每对面一条合并边界”表示应多出来的 query，不是 native physical-edge 计数。

此前很高的 `matched_ownership_accuracy` 只描述匹配到 GT 的那些 query 的面归属，甚至不要求这些 matched query 越过 existence 阈值。本次对全部激活预测中的合法不同面 pair 去重后，precision 为 95.13%；同面 assignment 等另列，并非逐 query precision。因此“匹配归属准”“完整预测集合准”“几何落在两侧曲面上”“可构成 solid”是不同问题。

还不能将 17.10× 边误差全归因于多余 query。边 CD² 的两个方向分别为：

| 方向 | Legacy | Joint |
|---|---:|---:|
| 预测 → GT | 0.000060531 | 0.001153333 |
| GT → 预测 | 0.000028704 | 0.000372232 |

GT → 预测也退步约 13 倍；单纯复制完全相同的曲线不会改变这一方向的最近点距离。当前输出同时存在真实边界附近覆盖不足或位置不准。至于匹配内曲线误差、误激活和漏边各占多少，本次没有用 oracle 筛选拆分，不能指定唯一原因。

## 2. 固定 diffusion latent 重放：接近旧版的面误差，较差的边误差

同一批历史生成 latent，两个 Decoder 都不再去重、不读取 GT 面数／连接、不重采样或选候选。同期测试日志及旧 loader 支持 identity／view 0，归档缺少逐件相机与源码摘要的限制见资产审计。全部 2,417 件同时可计算，未知面对应关系，因此**不报告对应 MAE 和 GT topology F1**。

| 指标 | Legacy | Joint | Joint 变化 |
|---|---:|---:|---:|
| 面 CD² ↓ | 0.005113912 | 0.005156570 | +0.83% |
| 边 CD² ↓ | 0.012894253 | 0.014124439 | +9.54% |
| 边：预测 → GT ↓ | 0.005480984 | 0.006875906 | +25.45% |
| 边：GT → 预测 ↓ | 0.007413270 | 0.007248534 | −2.22% |

Joint 的面／边 CD² 分别在 42.66%／27.97% 对象上更低；平均重复 query 仍有 36.0430 条。更多曲线可能带来覆盖方向的轻微收益，但预测到 GT 的误差更大，整体边 CD² 仍退步；这些数字不能支持“新 Decoder 已有 diffusion 收益”。

![输入 latent 来源对替换结果的影响](visualizations/diffusion_replay.png)

研究解释：生成 latent 下两模型的共同形状误差较大，与 Decoder 差异在总误差中被稀释的解释一致；这只是推断，不是已完成的误差归因实验。冻结 encoder 且历史 diffusion 内 AE 权重相等，使这次重放具有接口意义；未来若解冻 encoder，旧 cache／diffusion 的语义兼容性需重新检查。

这是固定历史上游输出的替换测试，不是重新训练 diffusion，也没有重跑完整图像生成流程或该组 latent 的 OCC 后处理。历史条件为 FLUX 合成真实感图像，不是实拍泛化结论。

## 3. OCC 后处理

已完成 **2,417 × 2 方法 × 2 模式 = 9,668** 个独立任务，双方每个模式均有 2,417 个终态记录。固定 CPU、32 个隔离 worker／模式、单 worker 单线程、180 秒全子进程预算、drop_num=0。optimized 名义 200 步，既有算法可扩展至 600。fit-only 仍包含拟合／修补／裁剪／缝合，drop_num=0 也不排除既有代码丢弃少量裁剪失败面。

| 模式／指标 | Legacy | Joint |
|---|---:|---:|
| fit_only：原读取容差有效 solid | 1,058／2,417（43.77%） | 478／2,417（19.78%） |
| fit_only：强制 0.1 容差兼容有效 | 1,058／2,417（43.77%） | 480／2,417（19.86%） |
| fit_only：几何可计算 | 1,989／2,417（82.29%） | 1,786／2,417（73.89%） |
| optimized：原读取容差有效 solid | 2,048／2,417（84.73%） | 778／2,417（32.19%） |
| optimized：强制 0.1 容差兼容有效 | 2,048／2,417（84.73%） | 782／2,417（32.35%） |
| optimized：几何可计算 | 2,262／2,417（93.59%） | 1,881／2,417（77.82%） |

所有失败／缺失／超时均保留在上表分母。原始 2,424 test 分母下，native solid 比例为 fit-only **43.65%／19.72%**、optimized **84.49%／32.10%**（Legacy／Joint）。原读取容差有效性采用 STEP 自带／读取容差，并非统一严格工程公差保证；强制 tolerance=0.1 结果单列。

![后处理完整覆盖率](visualizations/post_comparison.png)

`STEP exists` 与 `success.txt` 均不能替代有效性检查。optimized 的 success marker 为 2,049／785，而 native valid solid 为 2,048／778。两模式全部输出了 STEP；fit-only 的 Legacy 有 10 件 STEP 读取失败，optimized 全部可读。

后处理几何使用 native STEP：每面 1,024 个三角网格内面积均匀点（各面等数量，非整个物体按面积加权），每条 native edge 128 个参数均匀点；mesh deflection 为 0.01／0.1，固定逐 ID seed，GT 与预测都不对齐／缩放。它与 raw grid／merged boundary 指标不是同一种采样，数值不能直接跨阶段相减。

| 模式 | 共同可计算对象 | 面 CD² Legacy | 面 CD² Joint | 边 CD² Legacy | 边 CD² Joint |
|---|---:|---:|---:|---:|---:|
| fit_only | 1,528／2,417 | 0.004626021 | 0.005820433 | 0.002978008 | 0.004843924 |
| optimized | 1,807／2,417 | 0.000736142 | 0.003386459 | 0.000398878 | 0.003648483 |

这两个交集是**共同可计算几何子集**，包含可读取但无效的 solid／compound，不是共同有效 solid 子集。optimized 子集的面／边误差仍分别退步约 **4.60×／9.15×**；中位数也退步。两个模式共同子集不同，不能直接以其均值之差估计优化操作的因果收益。各自可计算子集的均值／中位数／p90、共同子集的配对差异见 [fit-only 汇总](post/fit_only/summary.json)和 [optimized 汇总](post/optimized/summary.json)。

失败与成本：

- fit-only：Legacy 417 件 geometry_error、10 件 STEP 读取失败、1 件超时；Joint 624 件 geometry_error、7 件超时。共同几何覆盖 63.22%。
- optimized：Legacy 155 件 geometry_error；Joint 524 件 geometry_error、12 件超时。共同几何覆盖 74.76%。
- 全部 geometry_error 都发生在预测 STEP 的面采样，没有 GT 采样或 API 签名错误：固定网格协议下某些面没有三角化或有限正面积。所有超时也都在 geometry 阶段；其中 fit-only Joint 3 件、optimized Joint 2 件已通过 native solid 检查，有效性仍被保留。不能把几何失败等同于实体无效。
- 既有 post 算法耗时均值（不含后续有效性／GT 缓存／采样／CD）：fit-only Legacy **1.23 s**、Joint **2.29 s**；optimized **3.93 s／7.02 s**。这是本机并发执行记录，不是通用性能承诺；全子进程耗时另列于 summary。

后处理并没有把所有重复 query 原样交给 OCC。已完整核验的 fit-only 组里，joint 的 167,750 个原始 query 中，移除同面分配 681 个、几何门限不通过 6,986 个、随后重复候选 87,547 个，最终保留 72,536 个，即平均 **69.40 → 30.01**。两模式的 2,417 份筛选记录逐字段相同，全部对象最终面 pair 唯一且无同面连接；700 件在筛选后完全失去至少一个原始 pair，共减少 2,303 个 pair。

[适配统计](adapter_summary.json)保留完整计数。适配器先做几何门限，再按预测 confidence／残差／索引选唯一候选，不读取 GT。此处实际去重阶段删除的数量，和 raw 门限前 duplicate 指标 92,230 不相等。因此后处理质量差距仍可能来自错误／缺失的面关联、候选选择、曲线位置、曲面或裁剪一致性，不能仅用“重复被送入 OCC”解释。

## 4. 预先选例与验证记录

在推理前按 GT 面数四档，每档选择稳定 ID hash 最小的两件；[固定清单](selected_examples.json)共 8 件。原始图显示未裁剪 UV 曲面及全部预测曲线，曲线采用透视显示以便看到内部结构，不能将它们误读为最终 solid。没有按新模型获胜条件挑选。

- [Posterior 全部选例](visualizations/posterior_mean/raw/contact_sheet.png)
- [历史 diffusion latent 全部选例](visualizations/archived_latents/raw/contact_sheet.png)
- [fit-only STEP 选例](visualizations/posterior_mean/fit_only/contact_sheet.png)、[optimized STEP 选例](visualizations/posterior_mean/optimized/contact_sheet.png)：无法完整采样的输出标明状态，不以部分网格冒充完整实体。
- 原始逐对象表：`raw_metrics/{posterior_mean,archived_latents}/per_object.csv`；[主组汇总](raw_metrics/posterior_mean/summary.json)、[归档组汇总](raw_metrics/archived_latents/summary.json)受 Git 管理。

28 项针对性测试通过，覆盖严格加载／同输入／导出、半边与重复 query 区分、空预测／非法索引、确定性 CD、固定分母、native／compat 分离、隔离超时与断点指纹。另有真实 GPU 导出、两模式两表示 OCC 接口 smoke。它们验证实现合同，科研结论来自全量测试。

首次 raw 指标汇总遇到 NumPy bool 累计产生 int64、JSON 无法序列化的问题；已修复类型合同，并保留失败 CSV 与日志。两组重跑的逐对象 CSV 与首次文件 SHA256 相同，指标计算未改变。这不是训练或测试样本的失败。

当前有依据的优先级：保留原 Decoder 作为可用基线；新版本同时面对曲面／曲线重建精度与预测集合数量问题，不能仅以 matched ownership 高或 loss 收敛作为完成信号。本次没有启动下一轮结构优化。
