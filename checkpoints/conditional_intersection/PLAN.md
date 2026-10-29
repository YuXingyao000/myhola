# Conditional Intersection：研究计划

日期：2026-10-29。网络实现提交`74c7ecb4`；原feature-on启动提交`7609d276`，现已按用户要求停止。旧best415／last5395已完成固定validation评估，合法率90.73%／90.76%，来源90.99%；原始几何改善未转化为合法率提升，见[结果](feature_on_evaluation_20261029/REPORT.md)。关闭特征匹配的新运行以`d2fa70c3`从同一91%来源重新初始化，七卡全量训练已启动，见[当前运行](no_edge_feature_20261029/README.md)。新运行尚无完整validation结论。

## 问题与决定

已恢复的32维显式共享XYZ模型在固定3495件validation上得到3180件合法solid（90.99%）。它的原Intersection让一个面token查询另一个面token；cross-attention只有一个key，无法在面内选择局部位置。用户同意借鉴Conditional DETR，在原Intersection内部加入内容／位置分开的attention，继续输出原64维有向边特征。

本轮不追加末端correction或第二个refinement，不更换曲线表示。面CNN已经生成的16×16中间特征成为空间memory；每个有向面对仍只有一个query。ReferenceUVHead预测被读取面上的一个参考位置，没有GT位置标签，不代表曲线中心或曲线上的某一个点。

## 固定范围

- 来源为 [91% XYZ32基线](../vae_baseline/xyz32_20261028/README.md) 的shared_curve_uv best42471，SHA256 `ce749416c1635dda1563161b971bd4260d064cfdb0c31f1fa9d06ad0e70961ff`。
- 保留原8层512维Intersection的self-attention、FFN、LayerNorm、输入／输出投影与侧别位置参数；仅替换每层cross-attention的计算。
- 保留refinement、curve conditioning、shared XYZ、辅助UV。原轮次feature匹配权重为1.0；2026-10-29用户要求关闭这一项，当前入口`train_vae_conditional_intersection_no_edge_feature`将其设为0.0，其他loss权重不变。旧入口保留历史值以复现旧运行。
- 本次Encoder、Posterior及Decoder全部参与训练；历史来源训练冻结Encoder，因此本轮不是严格的仅attention单变量消融。
- 新run、新Adam，不恢复来源run的optimizer、步数或Timer。首阶段上限30000个optimizer更新或24小时，先到者结束；设备／microbatch以142实测为准并保存在resolved config。
- 142采用PyTorch2.8/cu128，正式计划使用物理GPU1–7、每卡microbatch8、累积2，即112 CAD/update；GPU0保留给现有ComfyUI。来源历史为128 CAD/update，批量差异明确记录。Intersection和原refinement均按64候选分块并启用激活重算，refinement几何分块128；这是计算／显存调整，不改参数或loss。
- 保留原数据、UV sidecar、增广、半边索引和OCC接口；不将本次参考UV写成曲线几何或新增GT输入。

## 实现验收

1. 新旧模块在eval、FP32、零局部投影时，对相同输入产生近等价边特征；完整模型迁移严格保留旧权重。训练dropout的随机分布改变，不能宣称逐步训练等价。
2. 真实validation小批量检查训练／推理张量形状、有限loss与梯度、反向共享XYZ、全模型可训练及checkpoint完整加载。
3. 确认零初始化的局部投影首步得到梯度，并在更新后打开真实空间内容路径。关闭attention dropout时，初始value空间恒定，参考UV／位置分支首步梯度应为零；正式训练保留0.1 attention dropout，随机mask使加权和依赖attention，首步位置梯度可以非零，但这不是空间内容信号。CNN局部特征输入经零投影的首步梯度仍为零（不代表面CNN的其他训练路径无梯度）。验收同时覆盖这两种情况，不仅依据首步位置梯度判断路径是否学到空间内容。
4. 比较分块与未分块输出／梯度；检查激活重算和BF16训练。142上使用实际最大面数样本检查显存，再进行短DDP训练验收。
5. 完成验收后，在142启动独立正式run；记录Git、初始化摘要、全部超参数、GPU、命令及W&B链接。

已完成的真实检查见[PREFLIGHT.json](PREFLIGHT.json)与[PREFLIGHT_LARGEST_BATCH8.json](PREFLIGHT_LARGEST_BATCH8.json)：190,236,065个参数全部可训练，新增7,937,922个参数；778个来源state项严格保留。最大尺寸检查将两个30面CAD重复组成8个对象，仅用于显存与实现验证，单卡BF16训练峰值7.91GiB。初始化coarse误差很小，但经原refinement后全部候选输出并非逐位相同；分类选择无分歧，实际选中曲线最大XYZ差约2.03e-6。完整误差及边界见README，不将小批量一致性当作下游收益。

## 研究评价

用固定3495件validation与来源权重比较。posterior mean、原始预测及固定recover_v1后处理分别报告：面／边CD²、拓扑指标、原生合法solid数、救回／退步及耗时；失败保留在预定分母。训练loss与attention热图仅帮助解释，不替代完整几何和合法率结果。test不参与结构选择或容差调整。

注意力图按层和head展示在owner面16×16网格上，核查参考UV与实际读取区域；多head未被监督为边上不同区段，参考UV也没有两侧3D对应保证。判断应基于实际权重与下游几何，而非仅凭参考点看似合理。

## 参考

- [Conditional DETR论文](https://arxiv.org/abs/2108.06152)：内容与位置query分开匹配。
- [作者Transformer实现](https://github.com/Atten4Vis/ConditionalDETR/blob/main/models/transformer.py)：逐head拼接内容／位置，value传递内容。本项目的一面对单query、固定参考UV与旧权重迁移是任务适配，不是原论文完整复现。
