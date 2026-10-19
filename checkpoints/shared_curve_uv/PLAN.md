# 共享曲线与逐侧 UV（2026-10-19）

## 本轮问题

已保留的 UV boundary best 改善 raw 几何，但仍独立预测互反半边的 XYZ。用户批准尝试：保留早期有向特征，融合一对特征后只解码一条三维曲线，再将实际预测的逐点 XYZ、共享特征、该侧有向特征和所属面特征送入 UV 头。

这是结构实验，尚无训练或有效性结论。它只保证互反输出使用同一条三维曲线，不保证两个 UV 回代位置相等，也不保证 OCC 裁剪成功。

## 固定与改变

- 基础：`codex/vae-uv-boundary`，代码基准 `580bec89b378796c144476771cdce6c396cc1215`；初始化来自 UV boundary best，epoch176/step64251，SHA256 `f7b550689440afb36ffbc16df3ed0adbc2394ecbd59f00d47f337e24f43adfdf`。
- 保留 CNN encoder/posterior 冻结、8层512维 intersection、有向候选、coarse、refiner、classifier、原曲线 CNN/bbox MLP、face头；不更改数据、后处理或已有 loss 权重。
- 仅 final refined 分支按互反关系构造共享64维特征：`g = h_ij + MLP(cat(h_ij,h_ji))`，选择 `i < j` 为代表方向，128维残差MLP末层零初始化。每个无序候选只执行一次曲线几何头，保留代表方向的已训练face conditioning。
- UV头每个有向候选输出所属面的一条路径；读取实际曲线点，不 detach。保留旧UV头的2层128维Transformer、4heads、FFN256、位置MLP；替换context/output尺寸并增加逐点XYZ embedding。迁移保留旧backbone，明确记录部分权重映射。
- 输出继续为 legacy_half_edges：一条共享曲线展开成两行，互反行使用相反的点序；UV仍导出 `[P,2,16,2]`，第二侧从互反行的所属面UV反序获取。不会删去两侧面的UV或候选行。

## 监督对应

原始NPZ/UV sidecar不改。coarse与encoder teacher继续使用各自原标签。final共享曲线为同一面对选择 `face_i < face_j` 的GT连接行作为共同参数化，互反行使用该行点序的反序，两个owner的UV/mask同时交换和反序。完整保留代表行闭环首尾UV分支，不对UV做周期取模，不平均两条可能有不同起点的曲线。这个代表选择只在loss内使用，推理只读取预测特征及候选索引。

所有GT原始数据仍保留；不是把所有互反GT强行声明为逐点相等。新旧final点序目标不同，总loss曲线不直接作效果对比。共享几何消除互反XYZ差异，但原twin距离实现仍可有数值残差，不能将该项下降解释为额外学习收益。

代表选择依赖face索引：重新排列face可能选中另一条旧半边，因此不宣称对任意face重排严格等变。这是本轮保留旧曲线精度的取舍；同一次解码中两侧共享几何则是严格保证。

实现检查排除了直接平均有向特征的初始化：两个validation对象的初始edge CD²从2.08e-5恶化至7.36e-3，不能交给正式训练去挽救。现采用保留代表方向的零残差融合。该检查只定位初始化破坏，不是训练质量评估；原检查报告保留于validation/initialization_mean_fusion_rejected.json。

## 验证与运行

必须验证：旧开关关闭时输出/state不变；共享曲线互反严格反序；初始化保留代表方向的旧曲线；UV逐侧与曲线点序一致；XYZ输入及UV loss梯度连接；空边和混合precision；代表行的UV/mask/bbox/六维local目标；严格初始化与full/resume；真实validation小批量FP32/BF16和无GT推理。

代码与层级记录置于既有 `models/vae` 分类模块、独立入口config和本目录ASCII架构快照。正式训练由用户在142前台启动，W&B online。后续失败分析只用validation；不得用test选择实现或阈值。本次实现不自动开始正式训练。
