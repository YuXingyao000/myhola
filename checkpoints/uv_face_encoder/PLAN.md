# UV token 面编码器：先保留空间采样，再汇聚

## 研究问题与范围

用户观察到：形状大体正确时，边与面仍可能有足以破坏裁剪／闭合的小误差。本轮检验一个编码端假设：旧面 CNN 在关系融合之前，通过 MaxPool 将 `16×16` 压到 `2×2`，是否过早丢掉了有用的几何差异。

这是研究假设，尚不能从既有 failure case 或 loss 曲线证明。新的编码器也最终压缩到 32 维，不承诺无损，也没有解决采样本身、曲面拟合或 decoder 的全部误差。

2026-10-17 已获同意的变量：**只替换局部面 CNN**。保持 256 个有序网格点，通过六层面内 attention 交换信息，再由一个 query 汇聚。边 CNN、后续面间 attention、五层 GAT、全局融合、posterior、原 intersection、refinement、最终 curve conditioning、几何头和 loss 定义不改。不开启 local-inter、point-feedback 或 edge-value 实验。

## 固定架构

输入为每面 `16×16` 网格上的 XYZ 和 normal，共六通道。UV 是归一化网格索引 `(i/15,j/15)`，不是真实长度、额外预测坐标或独立 CAD 标签。

1. 几何：`[M,6,16,16] → [M,256,6] → Linear(6,128) → GELU → Linear(128,512)`。
2. UV：`[1,256,2] → Linear(2,128) → GELU → Linear(128,512)`，广播后与几何相加。
3. 六个独立初始化的 PreNorm Transformer block；每层保持 `[M,256,512]`。8 heads、每头64维，FFN `512→2048→512`，GELU、dropout0。每层两条 residual。
4. memory LayerNorm；一个共享 learned query 扩展到 `[M,1,512]`，经 query LayerNorm 后 cross-attention 读取全部256个 K/V，再加 query residual。
5. readout 的 LayerNorm／FFN `512→2048→512`／residual；最后 LayerNorm／Linear `512→32`，输出 `[M,32]`。

`M` 是当前 packed batch 的总面数；不同面在这个 block 中独立计算。输出仍送入原来的面间 attention、GAT、全局融合和 posterior。这里的32维不是已经完成 posterior 的 holistic latent。

旧面 CNN 为22,048,040参数，新模块22,219,040参数，增加171,000（约0.78%）。包含 conditioned refinement 的整模型从181,965,981变为182,136,981。参数接近不等于计算量接近：面内一直保留256个token，粗略乘加量约为旧面 CNN 的12倍；实际显存和吞吐尚未测量。

## 初始化与训练协议

来源是已评估的 conditioned refinement checkpoint，epoch157／step57,354：

```text
/mnt/d/python/checkpoints/intersection_refinement/curve_conditioning/20261013T083729.740439Z/weights/epoch=0157-step=00057354-val_loss=0.038204.ckpt
SHA256: 4c7910b8169206c74fcf38c358cf171359149e0429c8debbc227d391aa56613f
```

完整校验来源后，只弃用 `encoder.face_coords.*` 的旧CNN47个state项；保留其余689项。新UV模块随机初始化。这是新run、新Adam，不是继续源checkpoint的优化器。不能用旧CNN权重去近似填充attention层。

因为新面编码器必须学习，本轮显式解冻encoder、posterior和decoder；KL仍采用既有sum reduction和1e-6权重。这一点与源运行的冻结encoder状态不同。若做CNN续训对照，两组必须同样解冻，保留相同KL、数据、batch、学习率、seed和预算。

配置预设8卡×每卡batch16、lr2e-5、seed42、bf16、最多100,000 optimizer steps（1,000 epochs上限），没有自动启动。实际启动前记录设备、batch和预算；若因显存调整batch，两组同步调整并核对样本曝光量。参数量接近不能替代等预算比较，应同时报告steps／样本量与耗时。

CNN续训对照继承已有CNN权重，UV候选则重学局部面表示；两者早期收敛差异不能单独作为最终容量优劣的结论。这一初始化差异必须保留在报告中。

## 数据与判断标准

沿用 DeepCAD v7 与 joint-boundary 合同校验后的 accepted split：train46,480、validation3,495、test2,417。输入仍是现有dataloader的面／半边采样及连接关系，不新增数据标签。训练沿用cube24 augmentation，validation关闭augmentation。

方向判断使用validation：固定对象列表，对比边与面几何误差、贴合误差的尾部、原始解码和同版后处理后的合法solid率、救回／退步对象数量，以及耗时。不能只看平均loss，也不能靠继续修改后处理使候选占优。缺失、失败、超时保留在分母。冻结方案后才使用test做数据评估；本轮实现验证没有读取test样本。

如采用该encoder，必须重新生成同源latent cache，并重新验证／训练后续diffusion。最终维度仍为32，并不表示旧latent分布、cache或diffusion checkpoint兼容。

## 实施状态

- 已实现：可选UV block、现有模型装配、严格组件迁移、统一trainer的resume清理、独立config。
- 验证结果见 [README](README.md) 与 [VALIDATION.json](VALIDATION.json)。这些只验证实现，不代表重建精度提升。
- 未启动正式训练；未开展新模型的OCC合法率评估或GPU吞吐测试。
