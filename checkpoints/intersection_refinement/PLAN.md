# Intersection refinement — execution contract

## Task
保留 HoLa light encoder、posterior、face Transformer、面/边几何头和线性 classifier，新增一轮基于实际几何与候选边消息的特征残差校对。用户已确认损失与实现范围；不加入独立 Δlogit 分支。

## Assumptions
沿用当前非 seam、每个有向 face pair 一条合并边界的数据合同；使用已审计的 shared-boundary accepted split，但模型输出仍为有向半边。冻结原 encoder/posterior，显式迁移完整 legacy checkpoint；后续可配置联合微调。旧 latent cache 仅在 encoder、数据与 posterior 合同一致时继续使用。

## Inputs
原 VAE batch 的面/边采样、归一化目标、face_counts、GT edge_face_connectivity；decoder 只接收 face latents 与分组计数。原模型 checkpoint 是初始化来源，恢复运行另用 trainer.resume_from_checkpoint。

## Outputs
模块化新 VAE、训练配置、逐对象 legacy_half_edges 预测，以及初稿/最终损失与诊断。所有研究记录和运行产物在 checkpoints/intersection_refinement/；不改历史 experiments/。

## Processing Protocol
1. 先枚举每个对象的全部有向非对角面对，训练和推理一致。GT 只进入 encoder 和 loss；不据 GT 删除 decoder 候选。
2. 原 face Transformer 运行一次；原 inter/classifier 得到 E0 与连接 logits。双向 logits 取均值，方向特征不平均。
3. 原几何头解码 S0/C0；曲线分别读取两张所属面的连续三角网格近点、偏移和距离。近点选择不求导，选中的几何计算向边与面反传。
4. 预测 p0 加权的消息按所属面聚合并归一化；融合两侧上下文、F0/E0 和几何读取得到 ΔF/ΔE。残差输出零初始化；保留线性 classifier。
5. 复用原几何头解码 F1/E1，重算并对称化 logits，最终才按概率筛边。单次校对不重复运行 face Transformer。
6. 最终局部几何/bbox 重建 + 0.25 初稿辅助；正负分别归一化 BCE 权重 0.1；E0 teacher matching 与 encoder-edge 直解辅助保持；KL 保持原定义（冻结时仅监测）。加入最终全局 XYZ、正边到两张预测面的贴合、互反曲线 XYZ 一致性。三者初始权重分别 1.0、0.1、0.1，可显式配置。
7. 贴合使用 GT 正边 mask，不乘可学习概率；GT 同算法离散化距离加容差作为误差下限。互反曲线使用 GT 导出的固定点到折线对应，不强行假设采样同序/相位；保留原有方向通道监督。

## Metrics
分别记录原始 loss 与 weighted loss，不把诊断加进 total。记录 coarse/final adjacency precision/recall、正边概率、ΔF/ΔE 幅度和全局几何误差。真正研究结果仍需固定 split 的原始预测、OCC 后处理与 solid 成功率比较；本次实现验证不宣称提升。

## Decision Rule
真实小批量前反向、迁移、冻结、排列/对象隔离、预测独立性与导出通过后，交付可运行训练命令，不自行启动正式长训练。所有未通过检查需修复或明确说明。

## Edge Cases
空候选（单面对象）、无正边/无负边、退化曲面三角片/曲线段需有限输出和梯度。拒绝非有限输入、跨对象/重复/非互反 GT、索引越界和不支持的 seam。推理不得使用 GT 数量/连接筛选；无静默丢对象。

## Validation / Sanity Checks
核对原 encoder mean/std 与原几何头迁移等价；验证 classifier 不变、残差初始化为零、双向对称概率、GT 改动不改变 decoder forward、连续面内点距离为零、梯度到边/面及校对模块、冻结 encoder、checkpoint 严格恢复、CLI 单卡与多卡小运行、逐对象导出。

## Implementation Constraints
以 diffusion/*.py 与 blocks.py 的显式模块风格组织；状态使用有形状说明的 dataclass；避免在 decode 中计算监督。保留原入口与历史模型。全候选解码支持分块/activation checkpoint，以控制显存；不靠静默截断减少候选。

## Deliverables
独立 codex/intersection-refinement 分支；实现与验证提交；源码定位、损失公式、完整训练/测试/续训命令与紧凑验证记录。
