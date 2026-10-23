# 64维关闭edge_feature：训练目标变更快照

2026-10-23。网络代码基准 `97355cf1371b1a1b684c9409b9debeb8ee6f3e20`。
独立入口 `configs/train_vae_latent_expansion_no_edge_feature.yaml`。
完整层级、宽度、索引、初始化及纯latent推理合同沿用[64维架构](../ARCHITECTURE.md)；本轮不改变任何网络层或state key。该文档的历史“未训练”状态不代表后来运行的状态，实际运行记录见各自metadata。

```text
GT边 [R,16,6] -> EdgeCNN -> e_teacher [R,128]
                              |                |
                              |                +-> shared CurveGeometry
                              |                    -> local/bbox GT监督【保留】
                              v
                     stop_gradient(e_teacher)
                              |
z [F,64] -> FaceFeatureDecoder -> F0 [F,64]
              -> pair gather -> 8层512维AttnIntersection -> h0 [P,128]
                                                          |
                 GT正边对应的L1(h0, e_teacher)【仅日志；权重0】
                                                          |
                    coarse geometry / refiner / shared curve / conditioning
                                                          |
                    final面/边XYZ + UV；原几何/分类/KL等目标【保留】
```

Encoder、posterior、Decoder均可训练；face latent64、edge feature128。训练仍随机posterior，诊断mean推理仍从各自Encoder获得latent；GT只用于编码训练CAD及监督，不注入纯latent Decoder。Loss实现在 `src/brepnet/models/vae/losses/intersection.py`，已有 `feature_weight` 直接控制weighted项。本轮将它从1改为0，raw值仍记录以便观察不受匹配约束时的变化；`encoder_edge_local`、`encoder_edge_bbox`权重仍为1。所有其他模块/loss和原扩容初始化保持一致。

此变化无需checkpoint迁移新逻辑；新run从同一32维source按原64维规则扩容，optimizer重置。训练后的latent语义可能变化，不能继续使用旧cache或据此声称旧diffusion兼容。
