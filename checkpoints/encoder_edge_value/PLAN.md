# GAT 边内容消息实验：固定执行协议

日期：2026-10-15。状态：实现与验证完成；正式运行状态以 `metadata/active_pair.json` 为准，尚无训练后评估结论。

## 1. Task

检验在既有 GAT 的 attention 权重之外，将边内容直接写入 face 消息，是否改善 holistic VAE 的边几何与最终 B-Rep 有效性。用户已授权实施；不将几何 loss 大小相近视为 Encoder 已被证实有问题。

## 2. Assumptions

- 源码事实：当前五层单头 GATv2 的 edge attributes 进入 score，value 仅为变换后的邻面；边已间接影响 face，不能称没有融合。
- 研究假设：直接传递边内容能减少聚合时丢失的边界信息；仍有每面 32 维压缩，不能保证几何精度或合法 solid。
- 本轮从已保留的 curve-conditioning checkpoint 开始，不沿用已放弃的 point-feedback。原 GAT 全量解冻对照用于分离重新训练 Encoder 的影响。
- 初始拟定每组 8 GPU × batch16，40 epoch / 14,520 optimizer steps；正式启动前真实 batch 验证显存。如必须调整，先更新本协议，两组一致，不静默调整。

## 3. Inputs

- 同一源权重：`/mnt/d/python/checkpoints/intersection_refinement/curve_conditioning/20261013T083729.740439Z/weights/epoch=0157-step=00057354-val_loss=0.038204.ckpt`。
- SHA-256：`4c7910b8169206c74fcf38c358cf171359149e0429c8debbc227d391aa56613f`；源 epoch157 / step57354。
- DeepCAD v7：`/mnt/d/data/deepcad_v7`；已有 accepted train/val/test：46,480 / 3,495 / 2,417，路径为 `checkpoints/joint_boundary/validation/{train,val,test}_contract_audit/accepted.txt`。
- GT face/edge samples 与现有 connectivity；不增加新的拓扑标签，不给 decoder 额外 GT。

## 4. Outputs

候选与对照分别存放 `checkpoints/encoder_edge_value/{candidate,control}/<UTC>/`；沿用 weights/logs/wandb/raw_predictions/metadata 分目录。保存初始化审计、真实 batch 验证和启动记录。输出仍为 legacy_half_edges，latent 形状仍为 N×32。

## 5. Processing Protocol

1. 候选：每层 `value = W_source F_source + MLP([F_target,F_source,E])`；MLP 为 `128→64→SiLU→32`，末层零初始化。保留原 GAT attention、5层/32维/单头、self-loop 与 mean 合成 edge attribute、bias、外部 residual/activation。
2. 每个真实边与 self-loop 使用同一 alpha 加权；新增值来自原始 face/edge feature，目标节点聚合方向保持 PyG 约定。仅增加五个 value MLP，共51,680参数。
3. 对照保留原 GAT，无新增 MLP。两组保留相同 decoder/conditioning/refinement/分类头；point-feedback 关闭。
4. 两组均 `freeze_encoder=false`：CNN、face attention、GAT、global、posterior 和 decoder 一起微调。保留原 loss 权重；KL sum×1e-6 重新参与梯度，两组一致。teacher edge feature 随 Encoder 学习，不能视为固定信息上限。
5. 新 run、新 Adam；lr2e-5、clip1、bf16、seed42、cube24、每epoch完整validation、无scheduler、无时限。候选严格保留全部旧 tensor，仅初始化 value_net；完整加载/resume严格匹配。
6. 八卡顺序训练，两组固定相同 optimizer steps 和全局 batch；训练随机初始化不消耗额外全局 RNG，不声称 CUDA 逐位可复现。源权重、split、Git与配置写入run元数据。

## 6. Metrics

训练记录既有 raw/weighted geometry、classification、incidence、twin、KL。后续正式评估固定2,417件与已有后处理：分别报告原始/修复后边与面几何、owner对应边误差及尾部、adjacency、合法原生solid、失败分类与耗时。local含6通道、bbox为center/scale；不将L1/CD²解释为CAD最大tolerance。

## 7. Decision Rule

各组仅按同预算窗口中的最小 val/loss 选择checkpoint，保留同终点last作审计。主要架构比较为新值消息对等预算原GAT解冻对照，历史冻结模型只作参考。若只改善平均loss而边误差尾部/最终solid不改善，不认定方向有效。没有等参数量MLP消融，不把收益全部归因于边输入本身。

## 8. Edge Cases

空edge、孤立face保留原self-loop语义；packed对象之间不传递。dense homogeneous graph是当前支持合同，sparse/bipartite明确拒绝。迁移拒绝缺失/多余/尺寸错误/重复归一化key及已有value分支的来源；旧完整checkpoint默认分支仍严格可加载。非有限loss/gradient、OOM或不完整验证记录为失败，不能静默降低协议。

## 9. Validation / Sanity Checks

验证旧tensor保留、初始化输出与loss等价、MLP层数/维度/参数量、edge内容可在固定alpha时改变消息、梯度可达、排列等变、packed隔离、空边、自环、strict full/resume迁移。实际训练数据完整前向/反向与optimizer更新，核验有限梯度和显存，再验证统一多卡入口。

## 10. Implementation Constraints

网络block只写入已有 `blocks.py`；装配/初始化放 `vae.py`；trainer只增加新初始化字段的resume清理，不新增训练类。一个新实验配置继承原入口，原模型默认关闭新分支。禁止静默更改原层数、宽度、loss、数据契约和后处理。新Encoder训练后的latent语义改变，旧cache/diffusion不视为兼容，须重建/重训或独立验证。

## 11. Deliverables

代码、显式config、本协议/README、紧凑验证报告、可停止/恢复的运行记录；不提交权重、数据、W&B凭证或巨型日志。实现验证与训练完成/效果评估分开记录。

依据：[PyG GATv2](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATv2Conv.html)、[HoLa](https://arxiv.org/abs/2504.14257)、[前轮固定评估](../intersection_refinement/point_feedback/evaluation_20261015/NOTE.md)。
