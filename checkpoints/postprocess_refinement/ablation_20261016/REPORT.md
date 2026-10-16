# 原始 VAE、两项已采纳改动与后处理的 2×2 对照

**已完成。** 相对“原始 VAE + 旧后处理”，整套改动使 validation 原生合法率从 **85.67% 升至 90.16%**，test 从 **84.90% 升至 89.41%**，分别净增 **157 / 109 个**合法 solid。它不是每件都改善：整套改动在 validation 救回 195 件、退步 38 件，在 test 救回 130 件、退步 21 件。

同一模型单独更换后处理，两份 split 均无原成功样本退步；更换模型仍有个别退步。原始解码平均几何误差与最终 STEP 的配对平均误差有所下降，但 test 的双向预测曲线一致性均值变差，详见第 5 节。合法率提升不能解读为所有指标、所有样本都变好。

本轮回答两个问题：同一份 latent 交给原始 decoder 和“refinement + 最终曲线条件调制”decoder，输出有什么变化；把各自输出交给旧后处理和固定的新后处理，又分别改变多少合法率。协议见 [PLAN.md](/mnt/d/python_curve_eval/checkpoints/postprocess_refinement/ablation_20261016/PLAN.md)。

## 1. 四格分别比较什么

| 模型 | 旧后处理 `legacy` | 新后处理 `recover_v1` |
|---|---|---|
| 原始 `AutoEncoder_light`，没有 refinement | 原始模型与原流程 | 只换后处理 |
| 已采纳的 refinement + face-pair curve conditioning | 只换模型整体 | 模型整体与后处理都换 |

横向比较同一个模型的两份后处理结果，纵向比较同一种后处理下的两个模型。右下角对左上角给出整套改动的结果。配对统计会同时列出“原来失败、现在成功”和“原来成功、现在失败”，不能只看总数的差。

这里的“改后模型”包含用户先后保留的两项改动：一轮 intersection refinement，以及最终 curve 几何头中的 face-pair conditioning。**此前只有 refinement 的模型作为历史旁列，不能拿它代替这次改后模型的一行。** 局部 attention、逐点 feedback 和新 GAT 分支均不在本次比较中。

### 固定权重与共同输入

| 项目 | 原始模型 | 改后模型整体 |
|---|---|---|
| checkpoint | [1119_deepcad_aug1_11k.ckpt](/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt) | [epoch=0157-step=00057354-val_loss=0.038204.ckpt](/mnt/d/python/checkpoints/intersection_refinement/curve_conditioning/20261013T083729.740439Z/weights/epoch=0157-step=00057354-val_loss=0.038204.ckpt) |
| checkpoint 内记录 | epoch 8499 / step 1,045,500 | epoch 157 / step 57,354 |
| SHA256 | `8b714943d24c3f3f18f3a1e64222b1f3577323ddbf42e4b90d603186ab7e917a` | `4c7910b8169206c74fcf38c358cf171359149e0429c8debbc227d391aa56613f` |

两个 checkpoint 的 encoder/posterior 共 **243 个张量逐值相同**。本次没有各自重新随机采样，而是读取已核验的 10 月 16 日参考导出，为每个 CAD 给两个 decoder **完全相同的 FP32 posterior mean**。原 CAD 文件、参考协议、latent 与输出文件均记录并核验哈希；不是沿用不同日期、不同数值设置的两份 latent。

推理固定 seed 42、FP32、关闭 TF32。validation 有 3,495 个 accepted CAD，原清单 3,505 个；test 为 2,417 / 2,424。完整审计确认两模型的 11,824 份输出覆盖全部 5,912 个 CAD，无重复、遗漏或非有限数值，实际输入 latent 逐值相同。当前 accepted train / validation / test 清单两两无交集；这一检查不独立证明历史训练清单的全部内容。证据见 [inference_integrity.json](/mnt/d/python_curve_eval/checkpoints/postprocess_refinement/ablation_20261016/integrity/inference_integrity.json)。

这是**已知 CAD 的 VAE 重建**：encoder 使用 CAD 几何与连接信息，latent 数量也来自该 CAD 的面数；不等于单张图片重建。decoder 没有额外输入 GT 拓扑。两边都不额外去重，连接阈值为 0.5；保留各模型既有接口区别：原模型枚举含 self-pair 的 N² 个有向候选，改后模型枚举 N(N−1) 个候选并平均双向 logits，曲线几何仍分别输出。原模型导出明确调用 `v_deduplicated=False`，见 [decode_legacy](/mnt/d/python_curve_eval/tools/vae_compare_inference.py:217)。这些区别属于本次“模型整体”的一部分。

## 2. 模型内部具体增加在哪里

![原始与改后 decoder 的 block、维度及数据流](/mnt/d/python_curve_eval/checkpoints/postprocess_refinement/ablation_20261016/architecture.png)

[查看可缩放 SVG](/mnt/d/python_curve_eval/checkpoints/postprocess_refinement/ablation_20261016/architecture.svg)。图中保留原 face transformer、intersection、几何头与分类器，并展开两个新增模块内部的数据流。

**第一项是让网络看到自己预测出来的几何，再修正一次特征。** 原 intersection 仍先从 face pair 得到边特征，原几何头先产生面与边。refiner 将预测面的 16×16 网格拆成三角片，把预测边的采样点投到两个所属面上，读取最近点、偏移和距离。它把这些量编码为消息，再给面特征、边特征各加一个增量，最后复用原几何头重新解码。这里的投影提供反馈，没有直接用投影点替换最终边。源码见 [IntersectionRefiner](/mnt/d/python_curve_eval/src/brepnet/models/blocks.py:969)。

**第二项是让最终曲线头在输出前再次看到两张所属面的特征。** 边特征与两个面特征拼接后，经一个小 MLP 产生缩放和偏移，作用于 curve CNN 与 center/scale MLP 的最终输出层之前：`h_new = (1 + gamma) * h + beta`。新增输出层从零初始化开始，初始行为与原头一致。它只作用于最终 refined curve 头，没有替换 intersection，也没有增加显式 vertex、wire 或 coedge 拓扑预测。源码见 [FacePairConditioning](/mnt/d/python_curve_eval/src/brepnet/models/blocks.py:1116) 与 [IntersectionDecoder](/mnt/d/python_curve_eval/src/brepnet/models/blocks.py:1264)。

两项设计都旨在减少“面看起来对、边看起来也对，但拼不到一起”的情况；是否实际改善，需要看下游结果。新增模块分别有 143,328 与 239,424 个参数，合计 382,752 个。**本轮同时包含额外微调与训练目标变化，没有等预算、只改 loss 的对照，所以不能把全部差异都归因于某一个 block。**

## 3. 旧流程究竟检查了什么，新流程改变了什么

旧代码**本来就有两层检查**。第一层在 [get_solid](/mnt/d/python_curve_eval/src/brepnet/post/occ_topology.py:410) 内部：缝合、修复 shell、构造和修复 solid，最后要求形状类型为 `TopAbs_SOLID` 且 `BRepCheck_Analyzer` 通过，才返回结果。第二层在 [旧外层保存函数](/mnt/d/python_curve_eval/src/brepnet/post/construct_brep.py:287)：把结果写成 STEP，再读取检查。

`get_solid` 内部按以下五步执行：

1. 打乱输入面的顺序，用当前 `connected_tolerance` 做 Sewing，把面缝到一起。
2. 如果缝合结果是 `Compound`，立即返回 `None`。源码后面虽然还留着“挑选最大 shell”的片段，但它在 `return` 之后，实际不会执行。
3. 将缝合结果的容差设为当前值；若 shell 检查不通过，则尝试修复面与朝向，再设置容差。
4. 用 `MakeSolid` 构造 solid，再调用 `ShapeFix_Solid`。这里条件中有 `or True`，因此修复实际总会执行，不只是检查失败时才执行；修复配置禁止创建 open solid。
5. 再次设置当前容差，只返回类型为 `TopAbs_SOLID` 且 `BRepCheck_Analyzer` 通过的结果，否则返回 `None`。

**本轮没有修改这个函数本体，主要改变的是外层如何尝试、保存和验收候选。**

问题是旧外层的停止位置偏早。只要 `get_solid` 返回了内存中的合法 solid，就先跳出容差候选循环；如果随后保存、重读失败，后面的候选不会再试。

```python
# 旧外层的控制顺序，省略无关代码
for tolerance in CONNECT_TOLERANCE:
    solid = get_solid(trimmed_faces, tolerance)
    if solid is not None:
        break
save_step_file(step_path, solid)
if not check_step_valid_soild(step_path):
    return False
```

新路径把**保存与重读放回候选循环内部**，见 [新候选循环](/mnt/d/python_curve_eval/src/brepnet/post/construct_brep.py:451)。一个候选只有写出、原生重读、合法性与面/环数量检查都通过，才结束；否则继续原有的下一个容差。

```python
# 新路径的控制顺序，省略诊断与预算检查
for tolerance in CONNECT_TOLERANCE:
    candidate_faces = copy_occ_shapes(trimmed_faces)
    solid = get_solid(candidate_faces, tolerance)
    if solid is None:
        continue
    save_step_checked(temporary_step, solid)
    summary = read_native_step(temporary_step)
    if summary.native_valid_solid and boundary_counts_preserved(summary):
        return True
```

上面是便于阅读的顺序示意，函数名与返回结构作了简化。实际旧检查还有一个区别：其 helper 默认会把重读形状的容差统一重设为 `0.1` 后再验，见 [旧 validity helper](/mnt/d/python_curve_eval/src/brepnet/eval/metrics/validity.py:40)。这一操作既可能提高原容差，也可能降低它。本轮所有模型、所有后处理的主指标统一按原生重读检查，不把这个旧 helper 的通过直接当作主指标成功。新路径使用 [原生重读函数](/mnt/d/python_curve_eval/src/brepnet/post/construct_brep.py:356)，不在验收前统一重设形状容差。

**每个新候选还会复制一份 OCC 对象。** Python 中复制列表不等于复制其底层几何；修复一个候选可能影响之后继续使用同一对象的候选。这里将一组形状整体复制，同时保留这份副本内部已有的共享边/顶点关系，见 [copy_occ_shapes](/mnt/d/python_curve_eval/src/brepnet/post/occ_topology.py:184)。它没有自动建立原先不存在的共享边关系。本轮没有单独消融“复制对象”这一项，不能单独报它贡献了多少合法率。

### recover_v1 的固定尝试顺序

| 顺序 | 路径 | 具体行为 |
|---|---|---|
| 1 | 原流程 | 先完整运行旧优化、拟合与构造，原生验收通过即保留 |
| 2 | 优化 + 新构造 | 重新从相同 raw 出发，保留旧拟合，采用上述候选循环 |
| 3 | 不优化 + 新构造 | 直接拟合原预测，避免一次几何优化反而破坏可构造的输入 |
| 4 | 优化 + 新拟合评分 + 新构造 | 最后一条固定恢复路径，使用 `fit_score_v2` |

顺序在 [策略装配](/mnt/d/python_curve_eval/src/brepnet/post/construct_brep.py:574) 中固定。首个通过就接受，不读取 GT 来选择候选；每条路径从相同 raw 与 Python/NumPy 随机状态开始。保留原容差列表、原周期处理以及本项目圆柱面的完整 natural-loop 特殊表示，不把它替换成标准圆柱的“两圆环加 seam”假设。

`fit_score_v2` 修改检查点与拟合评分，并不直接约束曲线贴住所属面。它也不构成连续几何的最大误差保证。详细公式与提前停止边界见 [此前拟合说明](/mnt/d/python_curve_eval/checkpoints/postprocess_refinement/weekly_report_20261016/WEEKLY_REPORT.md)。

## 4. 最终结果表

表中依次列出 **合法数量（accepted 合法率 / 原始清单合法率）**。validation 分母分别为 3,495 / 3,505，test 为 2,417 / 2,424。为便于阅读，下文用 A=原模型+旧后处理、B=改后模型+旧后处理、C=原模型+新后处理、D=改后模型+新后处理。

| split | 模型 | legacy | recover_v1 |
|---|---|---|---|
| validation | 原始模型 | A：2,994（85.67% / 85.42%） | C：3,091（88.44% / 88.19%） |
| validation | refinement + conditioning | B：3,066（87.73% / 87.48%） | D：3,151（90.16% / 89.90%） |
| test | 原始模型 | A：2,052（84.90% / 84.65%） | C：2,111（87.34% / 87.09%） |
| test | refinement + conditioning | B：2,114（87.46% / 87.21%） | D：2,161（89.41% / 89.15%） |

accepted 之前排除的 10 / 7 件保留在原始分母；失败、超时也保留在预定分母。全轮 23,648 份预定记录均有唯一终态，没有缺失。已记录原生合法的 STEP，不因随后几何测量出错或超时而改记非法。

![两个 split 的模型 × 后处理合法率](/mnt/d/python_curve_eval/checkpoints/postprocess_refinement/ablation_20261016/report/native_validity.png)

[可缩放 SVG](/mnt/d/python_curve_eval/checkpoints/postprocess_refinement/ablation_20261016/report/native_validity.svg) · [四格完整统计 CSV](/mnt/d/python_curve_eval/checkpoints/postprocess_refinement/ablation_20261016/report/cells.csv)。

| 配对比较 | validation 救回 / 退步（净增） | test 救回 / 退步（净增） |
|---|---|---|
| 原始模型换后处理，A→C | 97 / 0（+97） | 59 / 0（+59） |
| 改后模型换后处理，B→D | 85 / 0（+85） | 47 / 0（+47） |
| 旧后处理下换模型，A→B | 147 / 75（+72） | 104 / 42（+62） |
| 新后处理下换模型，C→D | 107 / 47（+60） | 76 / 26（+50） |
| 整套改动，A→D | **195 / 38（+157）** | **130 / 21（+109）** |

整套改动的 accepted 合法率增量分别为 **4.49 / 4.51 个百分点**。**零退步只适用于同一模型换后处理，不能推广到整套改动。** `recover_v1` 优先保留旧路径的原生合法输出，这也是零退步的设计原因；模型输出变了以后，这种保留关系不再成立。

仅 refinement 的历史模型单独旁列如下，同样使用 accepted / 原始分母的顺序：

| 历史模型：仅 refinement | legacy | recover_v1 |
|---|---|---|
| validation | 3,076（88.01% / 87.76%） | 3,147（90.04% / 89.79%） |
| test | 2,111（87.34% / 87.09%） | 2,155（89.16% / 88.90%） |

数据来自 [validation 历史报告](/mnt/d/python_curve_eval/checkpoints/postprocess_refinement/validation_20261016/REPORT.md) 与 [test 历史报告](/mnt/d/python_curve_eval/checkpoints/postprocess_refinement/test_20261016/REPORT.md)，本轮汇总核对了共同输入均值与后处理合同。新模型整体相对原始模型有提升，但 conditioning 相对仅 refinement 并非各格单调改善：validation 的旧后处理下少 10 件，新后处理下多 4 件；test 分别多 3 / 6 件。它不是等预算训练的第三个消融模型，不能据这张旁列单独确定 conditioning 的因果收益；历史测时也不用于本轮速度比较。

## 5. 几何指标与合法率一起看

### 原始解码：已完成

以下统计直接来自解码后的 16×16 面网格和 16 点曲线，尚未做几何优化或 OCC 修复。validation 的 3,495 件和 test 的 2,417 件在两模型下均完整、可计算，没有因缺失或非有限值缩小这些指标的比较集合。数据见 [validation raw summary](/mnt/d/python_curve_eval/checkpoints/postprocess_refinement/ablation_20261016/validation/raw_metrics/posterior_mean/summary.json) 与 [test raw summary](/mnt/d/python_curve_eval/checkpoints/postprocess_refinement/ablation_20261016/test/raw_metrics/posterior_mean/summary.json)。

| split | 模型 | face XYZ MAE 均值 ↓ | face CD² 均值 ↓ | edge CD² 均值 ↓ | macro topology F1 ↑ |
|---|---|---:|---:|---:|---:|
| validation | 原始模型 | 1.99455e-3 | 4.67046e-5 | 7.39322e-5 | 0.998627 |
| validation | refinement + conditioning | 1.80889e-3 | 3.84247e-5 | 6.29827e-5 | 0.999000 |
| test | 原始模型 | 2.07324e-3 | 5.01529e-5 | 8.92354e-5 | 0.998171 |
| test | refinement + conditioning | 1.86796e-3 | 4.06114e-5 | 7.76858e-5 | 0.998531 |

在本组权重上，validation 的面/边 CD² 均值分别降低 **17.73% / 14.81%**，test 分别降低 **19.02% / 12.94%**；face MAE 分别降低 9.31% / 9.90%。这说明在进入后处理之前，平均几何重建误差已经有所下降。macro F1 是先逐 CAD 计算面连接 F1，再对 CAD 求平均；它只评价不同面之间的连接对，不验证 wire、端点或 solid 是否正确。

再看同一 face pair 的两个方向各自产生的曲线是否重合。下面先在每个 CAD 内求 reciprocal curve CD² 的均值，再报告这些 CAD 均值的总体均值和中位数；**它比较的是两条预测曲线彼此的差异，不是曲线到 GT 的误差**。

| split | 原始模型：均值 ↓ | 改后模型：均值 ↓ | 原始模型：中位数 ↓ | 改后模型：中位数 ↓ |
|---|---:|---:|---:|---:|
| validation | 6.54833e-5 | 4.30381e-5 | 3.28864e-5 | 2.08126e-5 |
| test | 8.26855e-5 | **1.28781e-4** | 3.42845e-5 | **2.23139e-5** |

validation 的这两个统计都改善；但 **test 的 reciprocal CD² 均值升高 55.75%，中位数降低 34.92%**。因此不能概括成“所有几何指标都变好”或“双向曲线始终更一致”。均值与中位数反映了不同的分布变化，仅凭这些汇总不能定位具体失败机制；本轮也不依据 test 个例调整规则。

raw CD² 是两个方向最近点平方距离均值的和，使用原始采样点，不做面积/弧长加权或额外参数对齐；重合的重复几何不会被它充分惩罚。edge CD² 使用全部过阈值的预测曲线，不通过 GT 预先挑选边；GT 侧沿用已审计的合并非 seam 边界代表，不等同于原生 CAD 的全部物理边或裁剪环。以上平均误差改善不等于每件改善，也不等于最大误差已达到 `1e-3`。

### 最终 STEP：整套改动的配对几何

最终 STEP 的面/边 CD² 沿用每面 1,024 点、每边 128 点的双向平方距离，不额外对齐或缩放。它衡量采样后的平均几何差异，单位是当前数据坐标单位的平方，不能直接当成 `1e-3` 的最大距离。

两种成对集合分开报告：**共同可计算**是两边都获得有限几何指标的 CAD；**共同合法**还要求两边都原生合法。每项指标只在双方均为有限值的同一批 CAD 上比较，并列出覆盖数；未算出或非有限的指标不填零、不进入均值。不会拿不同成功子集的均值作公平改善，也不能因为合法率上涨就默认每件更接近 GT。

下表比较 A→D；箭头前后是**同一集合**内的旧、新均值。其余四种配对比较，以及中位数、P90，见 [paired.csv](/mnt/d/python_curve_eval/checkpoints/postprocess_refinement/ablation_20261016/report/paired.csv)。

| split | 配对集合 | 件数 / accepted 覆盖率 | face CD² 均值，A→D | edge CD² 均值，A→D |
|---|---|---:|---:|---:|
| validation | 共同可计算 | 3,196 / 91.44% | 6.86077e-4 → 6.43768e-4 | 3.34392e-4 → 2.68261e-4 |
| validation | 共同合法且可计算 | 2,781 / 79.57% | 4.81043e-4 → 4.74118e-4 | 1.10201e-4 → 9.71895e-5 |
| test | 共同可计算 | 2,176 / 90.03% | 6.76799e-4 → 6.08197e-4 | 3.52238e-4 → 2.82558e-4 |
| test | 共同合法且可计算 | 1,906 / 78.86% | 4.52776e-4 → 4.44378e-4 | 1.17262e-4 → 1.02483e-4 |

共同可计算集合的面/边平均误差，在 validation 降低 6.17% / 19.78%，test 降低 10.14% / 19.78%。共同合法且可计算集合中，面误差只降低 1.44% / 1.85%，边误差降低 11.81% / 12.60%。改善幅度随集合改变，不能用覆盖较小的合法子集替代全部输出。两边均合法的 CAD 实际有 2,956 / 2,031 件；其中 175 / 125 件至少一边缺少可计算几何，因此没有进入共同合法几何均值。

### 实际存储容差

**原生合法仍不保证几何误差低于 `1e-3`。** OCC 会结合形状中记录的容差判断合法。下面先取每个合法 CAD 内所有边或顶点的最大存储容差，再汇总这些最大值的中位数 / P90；不是直接测得的几何误差。各格取各自全部原生合法输出，集合不同，不把表间差异直接当作配对改善。

| split | 格 | 合法件数 | 边最大容差：中位数 / P90 | 顶点最大容差：中位数 / P90 |
|---|---|---:|---:|---:|
| validation | A | 2,994 | 0.005856 / 0.013374 | 0.005870 / 0.013408 |
| validation | B | 3,066 | 0.005709 / 0.011666 | 0.005736 / 0.011666 |
| validation | C | 3,091 | 0.005878 / 0.013850 | 0.005896 / 0.013978 |
| validation | D | 3,151 | 0.005741 / 0.011963 | 0.005761 / 0.012109 |
| test | A | 2,052 | 0.005789 / 0.014042 | 0.005820 / 0.014086 |
| test | B | 2,114 | 0.005663 / 0.012116 | 0.005683 / 0.012212 |
| test | C | 2,111 | 0.005834 / 0.014834 | 0.005848 / 0.014865 |
| test | D | 2,161 | 0.005710 / 0.012913 | 0.005723 / 0.013154 |

面最大存储容差的中位数与 P90 在八格中均为 `1e-7`；边、顶点的上述统计明显大于 `1e-3`。这正说明“能原生通过 OCC 检查”和“所有几何已精确到 `1e-3`”是两回事。

完整面/环数量检查只能防止该阶段明显少掉已有边界，不能证明每个孔都正确，也不能发现组环前已经合并的边界。周期面的不同表示还可能改变 wire 数量，所以单看与 GT 的环数差异，不能直接断言物理孔被填平。

## 6. 预算、执行状态与结论边界

所有正式后处理固定生产代码版本 `c0d69f8a`、`optimized`、`drop_num=0`，单对象单 CPU 线程。每次优化名义 200 步，旧优化器可以自适应延长至 600 步，见 [步数上限](/mnt/d/python_curve_eval/src/brepnet/post/geometry_optimization.py:64)。`recover_v1` 最多执行 **3 次优化总计**，比旧策略最多多 2 次；另有一条不优化的路径。全部路径及测量共享单对象 180 秒总预算，不是每条路径各有 180 秒。

四个 split × policy 运行各 32 workers，同时运行 raw 指标统计。因此记录的耗时反映本轮并发负载，**不是互不干扰的独立速度 benchmark**。同样的单件时间上限也不意味着尝试次数相同：新策略本来就多了失败后的恢复机会。

记录的单对象耗时如下，单位为秒，各列给出**均值 / P90**。`post_elapsed_seconds` 计后处理调用本身；`wall_seconds` 包含该对象的检查与几何测量等全流程时间。

| split | 格 | post_elapsed：均值 / P90 | wall：均值 / P90 |
|---|---|---:|---:|
| validation | A | 6.60 / 9.60 | 7.78 / 11.77 |
| validation | B | 6.46 / 9.28 | 7.37 / 11.14 |
| validation | C | 7.89 / 13.78 | 8.96 / 15.73 |
| validation | D | 7.52 / 13.08 | 8.35 / 14.81 |
| test | A | 7.42 / 10.63 | 8.62 / 12.52 |
| test | B | 7.32 / 10.58 | 8.26 / 12.16 |
| test | C | 9.40 / 15.88 | 10.44 / 18.18 |
| test | D | 8.96 / 15.06 | 9.84 / 17.11 |

新增恢复路径增加了平均成本与较慢样本的耗时。以下终态是**测量完成情况**，不能拿 `geometry_error` 或 `timeout` 数量直接当非法 solid 数量：

| split | 格 | 几何可计算 | 几何测量错误 | 超时 | STEP 重读错误 |
|---|---|---:|---:|---:|---:|
| validation | A | 3,278 | 214 | 3 | 0 |
| validation | B | 3,343 | 151 | 0 | 1 |
| validation | C | 3,286 | 206 | 3 | 0 |
| validation | D | 3,350 | 144 | 0 | 1 |
| test | A | 2,252 | 163 | 1 | 1 |
| test | B | 2,289 | 128 | 0 | 0 |
| test | C | 2,255 | 161 | 1 | 0 |
| test | D | 2,292 | 125 | 0 | 0 |

本轮超时均发生在几何测量阶段，已取得的原生合法性记录继续计数；测量失败不代表此前构造一定失败。没有缺失终态。完整数字与阶段见 [summary.json](/mnt/d/python_curve_eval/checkpoints/postprocess_refinement/ablation_20261016/report/summary.json)。

执行中外层 post 监督进程返回了 143（SIGTERM，原因未确认），四个独立 post 子进程继续完成。过程中核对过 PID/启动标识，未重跑或追加单件预算；结束后子进程退出码无法恢复，因此没有记作 exit 0。完成依据是 **23,648 条终态记录完整、四个子进程汇总齐全、独立复算一致、生产源码与协议哈希未改变**，见 [post_integrity.json](/mnt/d/python_curve_eval/checkpoints/postprocess_refinement/ablation_20261016/integrity/post_integrity.json) 与 [completion_observed.json](/mnt/d/python_curve_eval/checkpoints/postprocess_refinement/ablation_20261016/metadata/post/completion_observed.json)。

正式报告的八格合法数量与十组配对结果也逐项通过独立审计复核，见 [report_crosscheck.json](/mnt/d/python_curve_eval/checkpoints/postprocess_refinement/ablation_20261016/integrity/report_crosscheck.json)。

本轮只评价这些固定 checkpoint 与固定后处理的系统表现。训练预算不等，且已采纳改动伴随额外损失，不能拆解出单个模块的独立因果贡献。平均几何改善、合法率提升与样本级退步同时存在，后续方向仍需在 validation 上判断具体失败机制。

失败原因与方案设计只使用 validation。test 仅按冻结方案做汇总，不查看个例来调整容差、规则或 checkpoint；DeepCAD test 历史上参与过开发，不追溯称其为完全未见测试集。ABC 本轮未使用。
