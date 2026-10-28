# Conditional Intersection（32维共享XYZ）

日期：2026-10-29（北京时间）。实现提交`74c7ecb4`。状态：已实现，8项CPU检查通过；142真实preflight进行中，未正式训练。正式运行的信息与验证结果在启动验收后补充。

[研究计划](PLAN.md) · [逐层结构与张量形状](ARCHITECTURE.md) · [91%来源模型](../vae_baseline/xyz32_20261028/README.md)

## 改动及初始化

原Intersection的单query读取对方一个整体面token，改为读取该面的256个空间token。query/key各自包含内容与位置分量；ReferenceUVHead从有向面对特征预测被读取面上的一个参考UV，8层共用该坐标，各层query继续调制位置打分。最终仍输出64维有向边特征，后续refinement、conditioning、共享XYZ和辅助UV保持原接口。

入口：`configs/train_vae_conditional_intersection.yaml`，继承`train_vae_shared_curve_uv`。主要模块位于`src/brepnet/models/vae/decoders/conditional_intersection.py`；装配、严格初始化与原统一trainer分别接在现有decoder、checkpoint和训练入口中。完整结构及代码路径见ARCHITECTURE。

- 源权重（142）：`/mnt/d/python/checkpoints/shared_curve_uv/candidate/20261019T064906.049992Z/weights/epoch=0116-step=00042471-val_loss=0.036517.ckpt`。
- SHA256：`ce749416c1635dda1563161b971bd4260d064cfdb0c31f1fa9d06ad0e70961ff`。
- 用`model.init_conditional_intersection_checkpoint`显式迁移；`init_shared_curve_checkpoint`清空。旧整体投影、8层self/cross内容投影、FFN、LN和其他模型参数保留；新增局部投影、参考UV及位置分支初始化。
- 局部投影零初始化，初始各位置value相同；eval输出应近等价于来源模型。训练attention dropout的采样不同，不保证同seed下训练输出逐位相同。
- 首步位置梯度为零只适用于关闭attention dropout的检查。正式训练保留0.1 attention dropout，随机mask加权和仍依赖attention，因此ReferenceUVHead／位置分支首步可能有梯度，但不是空间内容信号。CNN局部输入经零投影的首步梯度仍为零；局部投影更新后才打开实际空间内容及对应位置学习路径，面CNN其他监督路径不受此结论限制。
- 本次**全模型训练**：`freeze_encoder=false`、`trainable_scope=all`。来源模型历史上冻结encoder；本轮不假装与它只有一个变量不同。更新encoder后旧latent cache／diffusion不能默认兼容。
- 这是**新run、新Adam**，不是resume。完整新模型加载及后续resume仍要求同结构严格加载。

## Loss：沿用来源权重，完整列出

| 原始项 | 对total_loss的权重 | 监督对象 |
|---|---:|---|
| `coarse_face_local / face_bbox / edge_local / edge_bbox` | 各0.25 | coarse局部六维采样和center/scale |
| `refined_face_local / face_bbox / edge_local / edge_bbox` | 各1.0 | final局部六维采样和center/scale |
| `coarse_classification` | 0.025 | 0.25 coarse系数 × 0.1分类系数 |
| `refined_classification` | 0.1 | 有向面对是否相邻 |
| `edge_feature` | **1.0** | coarse边特征对encoder edge teacher；teacher目标detach |
| `encoder_edge_local / encoder_edge_bbox` | 各1.0 | encoder边特征经原边几何头重建 |
| `face_xyz / edge_xyz` | 各1.0 | final全局XYZ的L1；边使用共享GT参数化 |
| `incidence` | 0.1 | 预测边到两侧预测三角网格距离，扣除GT离散网格基线 |
| `twin` | 0.1 | 按GT对应关系比较互反曲线，并扣除GT残差 |
| `uv` | 0.01 | 原辅助UV对sidecar标签，按有效点筛选 |
| `uv_anchor` | 0.1 | 固定GT-UV位置上的预测面／曲线几何，带GT插值残差补偿 |
| `kl_loss` | 原KL乘`gaussian_weights=1e-6`，然后计入total一次 | sum-reduction KL；本轮解冻Posterior，KL参与优化 |

六维local的后三个通道保留原数据方向信息；本轮没有新增UV-only版本的独立切向／相对尺度loss。ReferenceUVHead没有直接GT UV loss；它与原输出辅助UV头是两个不同模块。`geometry_tolerance=1e-5`与原几何loss定义保留。

## 首阶段训练配置

Adam，lr`2e-5`，seed42，BF16 mixed，clip norm1；全模型训练。142使用PyTorch2.8/cu128，正式计划使用物理GPU1–7、每卡microbatch2、累积8，即112 CAD/update；GPU0保留给正在运行的ComfyUI。来源历史为8卡×batch16、128 CAD/update，因此本轮批量并非完全相同。实际配置以run metadata为准。

新Intersection和原refinement均按64候选分块并开启激活重算，refinement几何分块保持128，减少全量训练时既有edge CNN等激活的显存。这只调整计算／重算方式，不改变网络参数或研究loss。首阶段30000个optimizer更新或24小时先到停止，不把终点之前的旧validation值称作最后一步实测。

下列为启动模板；工作树、数据、split和GPU必须先核验。训练只读train／validation，test路径仅满足配置字段，不用于本阶段挑选。

```bash
cd /mnt/d/python_uv_train
conda activate torch
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 WANDB_MODE=online python -u -m src.brepnet.train \
  --config-name train_vae_conditional_intersection \
  dataset.raw_data_root=/mnt/d/data/deepcad_v7 \
  dataset.edge_uv_root=/mnt/d/data/deepcad_v7_uv/v1 \
  dataset.train_dataset=checkpoints/joint_boundary/validation/train_contract_audit/accepted.txt \
  dataset.val_dataset=checkpoints/joint_boundary/validation/val_contract_audit/accepted.txt \
  dataset.test_dataset=checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt \
  'model.init_conditional_intersection_checkpoint="/mnt/d/python/checkpoints/shared_curve_uv/candidate/20261019T064906.049992Z/weights/epoch=0116-step=00042471-val_loss=0.036517.ckpt"' \
  trainer.devices=7 trainer.batch_size=2 trainer.accumulate_grad_batches=8
```

正式run输出`checkpoints/conditional_intersection/xyz32/<UTC时间>/`。启动前保存命令、数据／权重身份、resolved config及Git状态；本独立工作树不在主工作区自动同步范围，检查142实际HEAD及源权重摘要后才启动。

## 实现及研究状态

实现Git hash：`74c7ecb4`。8项CPU检查通过，142真实preflight进行中；完整新模块参数量、真实批量测试、正式运行目录／GPU／W&B／步数在验收后补充。尚未启动正式训练，没有新方案的完整validation几何或OCC结果；90.99%是来源checkpoint的历史结果。
