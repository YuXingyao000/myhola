# 最终曲线几何头的面条件调制

实现日期：2026-10-13。研究问题：保留已验证的原 Intersection＋refinement，在最终曲线几何头再次读取所属面特征，是否能减少边面几何误差？下文记录当日实现与验证，不将实现检查当作质量提升。

2026-10-14已完成用户训练后评估，按验证集选择epoch157/step57354。固定2417件，原生合法solid由2113变为2111；原始面精度和修复后边几何有改善，原始边中位误差未改善。见 [完整评估与解释](evaluation_20261014/NOTE.md)、[全集失败统计](failure_census_20261014/REPORT.md) 和 [离线交互报告](failure_census_20261014/page/standalone.html)。

## 本次新增代码与配置

本次将此前独立的 11 个 `intersection_*`、`joint_*`、`holistic_encoder.py`、`local_intersection.py` 实现归回原文件体系。模型／encoder／posterior／loss 放在 [vae.py](../../../src/brepnet/models/vae.py)，网络 block、几何头、投影和状态结构放在 [blocks.py](../../../src/brepnet/models/blocks.py)。

| 入口 | 本次改动 |
|---|---|
| [blocks.py：FacePairConditioning](../../../src/brepnet/models/blocks.py) | **新增**共享条件 MLP，以及分别输出曲线／bbox 缩放与偏移的两组零初始化线性头 |
| 同文件：`PrimitiveDecoder.forward` | **新增**两处条件调制，保留原 CNN／MLP 模块名称与层次 |
| 同文件：`IntersectionDecoder._decode_stage` | **新增**仅 final refined stage 传入 `[E1, F1_i, F1_j]`；coarse 和 encoder-edge teacher 不传入 |
| [vae.py：AutoEncoder_intersection._init_curve_conditioning](../../../src/brepnet/models/vae.py) | **新增**从完整原 refinement checkpoint 严格迁移，唯一允许的新参数前缀为 `decoder.edge_geometry.conditioning.` |
| [vae_intersection.yaml](../../../configs/model/vae_intersection.yaml) | **新增** `model.curve_conditioning.enabled: false`、`hidden_dim: 128` |

没有新增模型 Python 文件或模型注册名称。原 joint、原 refinement、两种历史 local intersection 都保留原配置入口；模块定义所在文件改变，模型内部参数名称和注册顺序保持兼容。测试和工具的 import 同步更新。`train.py` 沿用统一初始化／续训流程，保留续训时清除 `init_refinement_checkpoint` 的处理并更新错误提示，不增加训练分支。

此次提交也收录归并前已存在、尚未提交的两份 `train_vae_intersection_local*.yaml`、local 回归测试及其 GPU 检查脚本，以保留历史实验入口和完整检查依赖。这些是既有实验资产；本次新增调制仅通过上表中的 `curve_conditioning` 配置启用。

## 精确结构

每个有向候选仍然对应原来的 `E1 [64]`，两侧为更新后的 `F1_i [32]`、`F1_j [32]`：

1. 拼接为 128 维条件，经过 `Linear(128,128) → SiLU`。
2. `point_affine: Linear(128,192)` 输出曲线隐层的两组 96 维向量；在原最终 `Conv1d(96,6,1)` 前执行 `h = (1 + gamma) * h + beta`，沿 16 个采样点广播。
3. `bbox_affine: Linear(128,1536)` 输出 bbox 隐层的两组 768 维向量；在原最终 `Linear(768,4)` 前执行相同操作。
4. 输出仍为 16×6 局部采样（XYZ＋normal）及 center 3＋scale 1；全局 XYZ 的恢复方式保持原样。

只有两个 affine 末层的权重和 bias 初始化为零；共享条件投影正常初始化。没有新增 LayerNorm、attention、拓扑 head 或 loss。首步共享条件投影的梯度为零是预期行为，affine 末层更新后该路径开始学习。

原参数 181,726,557；新增 239,424；开启后合计 181,965,981。原 Intersection 保持 8 层、512 维、8 heads、FFN 2048、dropout 0.1。encoder／posterior 继续冻结，原 decoder 与新增分支一起微调。

默认关闭时不注册新参数。开启只支持 `intersection.kind=legacy` 且 `refinement.enabled=true`，避免同时混入另一种 inter 或关闭 refinement。旧配置缺少新字段时仍按关闭处理。

## 初始化与训练命令

初始化来源固定为此前已评估的原 refinement 验证最佳权重：epoch 10、step 3,993。不是 local attention 权重，也不是只迁移原始 VAE。

下列命令沿用该基线实际运行的 8 卡×每卡 batch 16、bf16、lr 2e-5、seed 42、cube24、accepted split 和分块设置。原实际运行设定上限为 1,000,000 steps／10,000 epochs；这只是上限，不能把它称为与 step 3,993 checkpoint 等预算。新模型使用新的 run 和 optimizer，不作为原 run 的 resume。

```bash
cd /mnt/d/python
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /mnt/d/anaconda3/envs/torch/bin/python -m src.brepnet.train \
  --config-name train_vae_intersection \
  trainer.exp_name=intersection_refinement/curve_conditioning \
  trainer.devices=8 trainer.batch_size=16 trainer.num_workers=2 \
  trainer.precision=bf16-mixed trainer.learning_rate=2.0e-5 trainer.seed=42 \
  trainer.max_epochs=10000 trainer.max_steps=1000000 \
  model.intersection.kind=legacy \
  model.curve_conditioning.enabled=true model.curve_conditioning.hidden_dim=128 \
  model.init_legacy_checkpoint=null \
  "model.init_refinement_checkpoint='/mnt/d/python/checkpoints/intersection_refinement/frozen/20261011T123836.485257Z/weights/epoch=0010-step=00003993-val_loss=0.039476.ckpt'" \
  model.refinement.candidate_chunk_size=256 \
  model.refinement.geometry_chunk_size=128 \
  model.refinement.checkpoint_candidates=false \
  model.loss_weights.geometry_chunk_size=128 \
  dataset.raw_data_root=/mnt/d/data/deepcad_v7 \
  dataset.train_dataset=checkpoints/joint_boundary/validation/train_contract_audit/accepted.txt \
  dataset.val_dataset=checkpoints/joint_boundary/validation/val_contract_audit/accepted.txt \
  dataset.test_dataset=checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt
```

旧 checkpoint 的全部 730 个 state keys 必须完整存在且形状一致；新增参数以外任何缺失／额外参数均报错。已有调制的 checkpoint 使用 `model.checkpoint` 完整加载，或使用 `trainer.resume_from_checkpoint` 恢复原 run；续训仍需开启同一调制配置，不再次初始化。

验证记录见 [validation/REPORT.md](validation/REPORT.md)。后续研究比较仍需固定样本列表、后处理和预算，比较原始几何与最终 solid；本次实现检查不替代这些下游结果。
