# Intersection refinement

后续局部注意力 inter 微调使用独立的 [intersection_local 配置与说明](../intersection_local/README.md)。本页保留原 refinement 协议。

这一版保留原 HoLa light 的 encoder、posterior、face Transformer、面/边几何头、`inter` 和线性 `classifier`，新增一轮几何读取、候选边消息聚合和面/边残差回写。新增模块零初始化输出 ΔF/ΔE，完整旧模型权重用于初始化；默认冻结 encoder/posterior，微调 decoder。实现协议见 [PLAN.md](PLAN.md)，验证结果见 [NOTE.md](NOTE.md)。已完成首次微调测试：2417 件上，优化后合法 solid 由 2048 增至 2113，但固定 diffusion latent 的几何误差未改善。完整结果和局限见 [测试报告](evaluation_20261011/REPORT.md)，复现命令见 [评估说明](evaluation_20261011/README.md)。全部304失败件的溯源、107救回与42退步的成对比较见[失败统计](failure_census_20261011/README.md)，研究判断见[通俗解释](failure_census_20261011/NOTE.md)。

## 从哪里读代码

| 文件／类 | 职责 |
|---|---|
| `src/brepnet/models/vae.py:AutoEncoder_intersection` | 顶层装配、权重迁移、冻结控制、训练/推理与逐对象导出 |
| `src/brepnet/models/vae.py:HolisticEncoder/FacePosterior` | 显式 encoder 与 posterior；保留原计算与权重 |
| `src/brepnet/models/blocks.py:IntersectionDecoder/PrimitiveDecoder` | 全候选枚举、原几何头封装、初稿→校对→复用几何头、logits 对称化 |
| `src/brepnet/models/blocks.py:GeometryRead/IntersectionRefiner` | 几何读取、两侧消息聚合、ΔF/ΔE；没有新增 classifier |
| `src/brepnet/models/blocks.py:closest_surface_points/closest_polyline_points` | 点到连续三角网格/折线的近点和梯度 |
| `src/brepnet/models/vae.py:IntersectionLoss` | GT 对应、各项损失、权重与诊断；不混进 decoder |
| `src/brepnet/models/blocks.py:CandidatePairs/PrimitiveGeometry/DecoderStage/RefinementOutput` | 初稿/最终状态和候选索引的显式 tensor 合同 |
| `src/brepnet/models/blocks.py:LocalPairIntersection/FacePairConditioning` | 可选 local inter 与可选最终边几何头调制；分别由配置选择 |

2026-10-13 按原文件组织归并代码，移除原六个 `intersection_*` 文件、三个 `joint_*` 文件及 `holistic_encoder.py`、`local_intersection.py`。网络 block、几何工具和状态结构集中到 `blocks.py`，encoder/posterior、模型装配、损失和导出集中到 `vae.py`。模型注册名 `AutoEncoder_light`、`AutoEncoder_joint`、`AutoEncoder_intersection` 及已有 checkpoint 的模块属性层级/state key 不变；统一 trainer 继续按 config 构造模型。历史 PLAN／NOTE 中的文件路径代表当时实现，本表是当前源码入口。

## 可选最终 curve 条件调制

新增配置与训练入口见 [curve_conditioning README](curve_conditioning/README.md)。基础配置为 `model.curve_conditioning.enabled=false`、`hidden_dim=128`，旧配置没有此字段时也按关闭处理。开启需要原 `intersection.kind=legacy` 与 refinement；`FacePairConditioning` 读取 `[E1,F1_i,F1_j]`，在 refined edge CNN 和 center/scale MLP 的最终输出层前分别施加缩放／偏移，产生调制量的末层零初始化。coarse 和 encoder-edge teacher 保持原路径，原 inter 容量、refinement 运算及损失权重不变。后文原 refinement 协议对应条件调制关闭的设置；新增分支的研究效果尚需独立训练和评估。

`model.init_refinement_checkpoint` 有两种明确区分的组件初始化：

- `intersection.kind=local_attention`：用新 local inter 替换 `decoder.inter.*`，其余旧参数严格保留；旧 local 配置与迁移语义不变。
- 原 legacy inter 加 `curve_conditioning.enabled=true`：保留原 refinement 的所有旧参数，只初始化新增 `decoder.edge_geometry.conditioning.*`。任何缺少的旧 key、额外 key 或形状不符均拒绝；不能用已 conditioned 或 local checkpoint 冒充原 refinement 来源。

两者均创建新 run／optimizer。完整 conditioned/local 模型分别使用匹配配置的 `model.checkpoint` 或 `trainer.resume_from_checkpoint` 严格加载，不通过组件迁移做续训。

## 一轮校对的具体运算

原 Transformer 运行一次，得到 `F0 [sum_faces,32]`。每个对象独立枚举 `i != j` 的全部有向候选，得 `E0 [P,64]`，P 为各对象 `N(N-1)` 的和。训练和推理均如此，GT 不决定候选集合。

1. `inter([F_i,F_j]) → E_ij` 保留两份方向特征；`classifier` 仍是原 Linear(64,1)。
2. 对称化只处理 logits：`l_ij = (raw_l_ij + raw_l_ji)/2`，`p_ij = sigmoid(l_ij)`。不平均特征、曲线或方向通道。
3. 原几何头输出 S0/C0，暂不按概率筛边。每个曲线采样点读取两侧曲面各自的最近点；17 维观测为 `[curve_xyz, closest_i, closest_j, offset_i, offset_j, distance_i, distance_j]`。逐点 MLP 后 mean/max pooling 得 `G [P,64]`。
4. `message_encoder([E0,G]) → M [P,128]`。每条有向候选的消息按 p0 权重写给两个所属面，累加后除以收到的权重和。AB/BA 是两条独立消息，分母同样计入两份；两方向特征不会被提前合并。
5. `relation_encoder([E0,G,F_i,F_j,context_i,context_j]) → H [P,128]`。H 经线性残差头得 ΔE；H 再按 p0 聚合回各面，结合 F0，经面残差头得 ΔF。p0 不 detach，消息路由可导；没有先按概率做硬删除。
6. `F1=F0+ΔF`，`E1=E0+ΔE`。同一套几何头输出 S1/C1，同一个 classifier 重算并对称化 logits。导出时才用 `p1 > 0.5` 保留候选。

两个残差输出头的权重和 bias 初始化为零，因此初始 final/coarse 几何完全相同。新增模块第一步只有残差输出头获得非零梯度；它们离开零之后，梯度进入几何读取与消息模块。原 decoder 同时正常微调。固定 classifier 权重时，单方向 logit 的修正为 `wᵀΔE_ij`，最终对称 logit 的修正为 `wᵀ(ΔE_ij+ΔE_ji)/2`；没有新增 Δlogit 网络。

## 损失

所有 L1 取有效元素均值。`rec` 包含面/边 local 六通道与 center/scale 四项，边只监督 GT 正例。0/1 表示初稿/最终。

\[
\begin{aligned}
L ={}& L_{rec}^{1}+0.25L_{rec}^{0}
+0.1(L_{adj}^{1}+0.25L_{adj}^{0})\\
&+L_{feature}^{0}+L_{encoder-edge}+\beta L_{KL}\\
&+\lambda_{xyz}L_{xyz}^{1}
+\lambda_{on}L_{on}^{1}+\lambda_{twin}L_{twin}^{1}.
\end{aligned}
\]

初始权重 `xyz=1.0, incidence=0.1, twin=0.1`，可在 `model.loss_weights` 显式修改；尚未做权重标定。KL 沿用原 sum 与 `gaussian_weights=1e-6`，posterior 返回时已经乘 beta，loss 不重复加权。冻结时 KL 仅监测，不进 total。

\[
L_{adj}=-\tfrac12\big(\operatorname{mean}_{+}\log p+
\operatorname{mean}_{-}\log(1-p)\big),\quad
L_{feature}^{0}=\|E_{+}^{0}-\operatorname{sg}(E_{enc,+})\|_1.
\]

某一类为空时只取存在的类，候选为空时返回可导零。GT teacher 只约束 E0，E1 不再强行贴住旧 teacher。

\[
L_{xyz}^{1}=\|S^1-S^*\|_1+\|C_+^1-C_+^*\|_1.
\]

XYZ 在原零件全局归一化坐标系下计算，仍非物理毫米。贴合距离为点到预测 UV 网格按固定对角线组成的三角片的连续距离，先无梯度选择最近三角片，再对所选近点运算反传到两边几何。

\[
L_{on}^{1}=\operatorname{mean}_{(i,j)\in P_+,k,s\in\{i,j\}}
\big[d(q^1_{ij,k},S_s^1)-d(q^*_{ij,k},S_s^*)-\epsilon\big]_+.
\]

`epsilon=1e-5`；GT 同算法的距离是离散误差下限。GT 正边 mask 只用于监督，不进入 decoder，也不把此损失乘可学习 p。

Twin 不假设两个方向具有相同采样起点或均匀弧长。对每个 GT 半边采样点在 reciprocal GT 折线上找固定最近线段 `(a,b)` 和插值系数 t；将同一对应作用于预测 reciprocal 曲线。每个方向均参与平均，XYZ 以外通道不参与。

\[
T_{ji}(q^*_{ij,k})=(1-t)q_{ji,a}+tq_{ji,b},\quad
L_{twin}^{1}=\operatorname{mean}_{ij,k,c}
\big[|q^1_{ij,k,c}-T_{ji}^{1}|-|q^*_{ij,k,c}-T_{ji}^{*}|-\epsilon\big]_+.
\]

这是对讨论中“允许反向/起点对齐”的具体实现，不会让两份半边的六通道直接相等。没有 vertex/loop 次序监督，不保证闭环或合法 solid。

W&B 同时记录原始值和 `weighted_*`；`total_loss` 仅汇总明确加权项。`*_precision/recall`、`*_positive_probability` 和 `delta_*_mean` 是诊断，不加进损失。诊断阈值固定 0.5，均为当前 batch 的候选级统计，不代替完整测试集报告。

## 训练

在仓库根目录执行，以下命令使用本机已验证的 `torch` 环境与资产。W&B 默认开启，8 卡由 `trainer.devices=8` 启动。默认每卡 batch=2，全局 batch=16；candidate 分块只控制内存，不改变样本数。

```bash
conda activate torch
cd /mnt/d/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

python -m src.brepnet.train --config-name train_vae_intersection \
  model.init_legacy_checkpoint=/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt \
  dataset.raw_data_root=/mnt/d/data/deepcad_v7 \
  dataset.train_dataset=checkpoints/joint_boundary/validation/train_contract_audit/accepted.txt \
  dataset.val_dataset=checkpoints/joint_boundary/validation/val_contract_audit/accepted.txt \
  dataset.test_dataset=checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt
```

默认 lr=2e-5、max_steps=100000、cube24 训练增强，encoder 与 posterior 冻结且保持 eval；decoder 训练。这是完整旧 decoder 的微调，加上新校对模块的训练。验证沿旧入口使用 posterior sample，正式 test 使用 mean；不能将两者数值混作同一协议。

在本机 RTX 4090 24 GB 上，后续性能检查支持更大的执行分块和 batch。建议在上述命令追加以下覆盖；默认 YAML 暂保留初始保守设置，便于复现。实测、显存压力检查和 8 卡结果见 [性能记录](validation/performance/README.md)。

```bash
trainer.batch_size=16 \
model.refinement.candidate_chunk_size=256 \
model.refinement.geometry_chunk_size=128 \
model.refinement.checkpoint_candidates=false \
model.loss_weights.geometry_chunk_size=128
```

此时全局 batch=128，训练集 46480 件、8 卡与 drop_last 设置下每轮 363 步。分块和关闭重算不改变网络/损失定义；batch 改变优化轨迹和每 step 的样本曝光量，勿按相同步数直接比较训练预算。这里的 `checkpoint_candidates` 仅控制反向重算，与保存 `.ckpt` 无关。

为区分新增 loss 与反馈结构，可在同样的旧 checkpoint 初始化命令后追加：

```bash
model.refinement.enabled=false trainer.exp_name=intersection_refinement/loss_only
```

此配置仍保留相同全候选、logits 对称化与损失，省略校对块，便于后续比较；它不等同于完全未经修改的原模型。尚未运行该正式对照训练。

## 保存、续训与测试

```text
checkpoints/intersection_refinement/frozen/<UTC timestamp>/
  weights/          # last.ckpt 与 val/loss 最低的三个 checkpoint
  logs/             # 本地日志
  wandb/            # W&B
  metadata/         # 配置、Git、命令、seed
  raw_predictions/  # 逐对象 data.npz
```

保存发生在验证结束；达到 max_steps 前至少完成一轮验证后才有常规可续训 checkpoint。正式训练和本次验证运行放在不同目录；不要拿 validation 下的几步权重作为研究结果。

本配置设置 `trainer.checkpoint_enable_version_counter=false`，固定更新 `weights/last.ckpt`，避免从 best 续训后最新状态另存为 `last-v1.ckpt`。带 epoch/step 的 top-3 权重仍保留。旧实验默认行为不变。

续训使用原 run 的完整 checkpoint，不再提供初始化路径。保留原训练配置、设备和 batch；`max_steps` 为累计步数上限。

```bash
python -m src.brepnet.train --config-name train_vae_intersection \
  trainer.resume_from_checkpoint=/absolute/run/weights/last.ckpt \
  dataset.raw_data_root=/mnt/d/data/deepcad_v7 \
  dataset.train_dataset=checkpoints/joint_boundary/validation/train_contract_audit/accepted.txt \
  dataset.val_dataset=checkpoints/joint_boundary/validation/val_contract_audit/accepted.txt \
  dataset.test_dataset=checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt
```

测试在上面的续训命令后追加：

```bash
trainer.evaluate=true trainer.devices=1 trainer.batch_size=2 dataset.is_aug=0
```

导出保留有向半边合同 `representation=legacy_half_edges`，可进入原后处理。每个对象包含最终曲面/曲线、整数连接索引、对称 adjacency probability、所有候选概率和 coarse 概率；没有额外 face 去重，阈值筛边只依赖预测。不自动运行 OCC。与 legacy baseline 比较需固定 face 去重策略、后处理版本、split 和预算。

若以后联合微调 encoder，使用 `train_vae_intersection_joint`，提供已训练新模型的 `model.checkpoint=/absolute/run/weights/last.ckpt`，开启新的 run；这不是 resume。encoder 更新后需要重建 latent cache，并重新核对 diffusion 的输入分布。

## 实现验证命令

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m pytest -q \
  src/brepnet/test/test_intersection_geometry.py \
  src/brepnet/test/test_intersection_decoder.py \
  src/brepnet/test/test_intersection_losses.py \
  src/brepnet/test/test_intersection_vae.py \
  src/brepnet/test/test_training_device_contract.py \
  src/brepnet/test/test_run_management.py \
  src/brepnet/test/test_joint_decoder.py
```
