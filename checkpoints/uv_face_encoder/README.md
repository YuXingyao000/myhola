# UV face encoder

本轮用全分辨率的UV token attention替换局部面CNN，最终输出仍为32维。研究动机、固定结构和评估边界见 [PLAN.md](PLAN.md)。实现保持在原有文件中，共用原trainer。

## 新增代码与配置

| 位置 | 新增内容 |
|---|---|
| `src/brepnet/models/blocks.py:UVFaceEncoder` | 几何／UV embedding、6层面内attention、单query汇聚 |
| `src/brepnet/models/vae.py:AutoEncoder_intersection` | `face_encoder.kind`装配与`_init_face_encoder`严格迁移 |
| `src/brepnet/train.py:UnifiedTrainer` | resume时清除`init_face_encoder_checkpoint`，不重复迁移 |
| `configs/model/vae_intersection.yaml` | 默认`face_encoder.kind: cnn`，历史配置行为不变 |
| `configs/train_vae_uv_encoder.yaml` | 明确启用UV，解冻全模型，保留conditioned refinement |

新模块数据尺寸如下。这里`M`是batch内总面数，不是固定30。

| 层 | 输入 | 输出 |
|---|---|---|
| 展平网格，保留点的排列 | `[M,6,16,16]` | `[M,256,6]` |
| 几何MLP：6→128→512，GELU | `[M,256,6]` | `[M,256,512]` |
| UV MLP：2→128→512，GELU | `[1,256,2]` | `[1,256,512]`，广播相加 |
| 6× PreNorm self-attention＋FFN，各自有residual | `[M,256,512]` | `[M,256,512]` |
| memory LayerNorm | `[M,256,512]` | K/V为`[M,256,512]` |
| learned query＋query LayerNorm＋cross-attention＋residual | Q为`[M,1,512]`，K/V为上述memory | `[M,1,512]` |
| readout LayerNorm＋FFN＋residual | `[M,1,512]` | `[M,1,512]` |
| output LayerNorm＋Linear(512,32) | `[M,1,512]` | `[M,32]` |

所有attention都是8 heads；FFN为512→2048→512、GELU、dropout0。六层分别初始化，不共享参数。不在各层间降低网格分辨率，也不引入逐面独立learned query。固定UV buffer不写入checkpoint，UV embedding本身参与训练。

接下来仍走原来的面间attention→GAT→global fusion→posterior。**仅输出维度相同，不保证latent分布相同。**

## 新训练

源环境为`/mnt/d/anaconda3/envs/torch/bin/python`。从分支工作树运行，权重使用主工作区已有文件的绝对路径，不复制checkpoint。下面命令为8卡／每卡batch16；当前已有diffusion训练占用GPU，本次没有启动此命令。实际运行时先确认设备可用。

```bash
cd /mnt/d/python_curve_eval
/mnt/d/anaconda3/envs/torch/bin/python -m src.brepnet.train \
  --config-name train_vae_uv_encoder \
  "model.init_face_encoder_checkpoint='/mnt/d/python/checkpoints/intersection_refinement/curve_conditioning/20261013T083729.740439Z/weights/epoch=0157-step=00057354-val_loss=0.038204.ckpt'" \
  dataset.raw_data_root=/mnt/d/data/deepcad_v7 \
  dataset.train_dataset=/mnt/d/python_curve_eval/checkpoints/joint_boundary/validation/train_contract_audit/accepted.txt \
  dataset.val_dataset=/mnt/d/python_curve_eval/checkpoints/joint_boundary/validation/val_contract_audit/accepted.txt \
  dataset.test_dataset=/mnt/d/python_curve_eval/checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt \
  trainer.devices=8 trainer.batch_size=16 trainer.seed=42
```

训练配置预设100,000 steps、lr2e-5、bf16、全量解冻。原有KL和其他loss权重不改。源checkpoint中的旧面CNN被替换，其余参数逐张量保留；输出记录在`checkpoints/uv_face_encoder/candidate/<UTC>/`，由统一入口保存配置、Git、命令、日志和权重。

这次更换的面编码器是随机初始化，初始几何预测不会与旧VAE相等；不要将此初始化当作可直接评估的训练完成模型。

如需做同预算CNN续训对照，沿用以上命令的数据、预算、seed和其他设置，只将相关override改为：

```bash
trainer.exp_name=uv_face_encoder/control \
model.face_encoder.kind=cnn \
model.init_face_encoder_checkpoint=null \
"model.checkpoint='/mnt/d/python/checkpoints/intersection_refinement/curve_conditioning/20261013T083729.740439Z/weights/epoch=0157-step=00057354-val_loss=0.038204.ckpt'"
```

此对照也解冻encoder/posterior，避免将解冻收益误认为编码器收益。CNN对照保留旧CNN权重，UV候选重学面表示；这项初始化差异需要明确报告。

## 完整加载与续训

`init_face_encoder_checkpoint`只接受完整旧CNN＋conditioned refinement来源，连将被丢弃的CNN keys／shapes也会校验。拒绝缺少权重、错误形状、额外local／feedback／edge-value参数、重复前缀key或已UV来源。只初始化UV，**不恢复旧optimizer**。

完整UV权重加载使用同架构配置的`model.checkpoint`，同时将`model.init_face_encoder_checkpoint=null`，不要再次使用组件迁移。`strict=True`校验keys和shapes；nhead／dropout等配置可能不改变shape，必须使用原run保存的配置，不能只依靠strict加载判断语义完全一致。

续训使用原run的resolved config和完整checkpoint，恢复Adam／步数；源CNN checkpoint无需仍然存在。将以下路径替换成实际run记录：

```bash
/mnt/d/anaconda3/envs/torch/bin/python -m src.brepnet.train \
  --config-path /absolute/run/metadata/invocations/<UTC> --config-name config \
  'trainer.resume_from_checkpoint="/absolute/run/weights/last.ckpt"' \
  'trainer.run_dir="/absolute/run"' \
  'hydra.run.dir="/absolute/run"' \
  model.init_face_encoder_checkpoint=null \
  hydra.output_subdir=metadata/hydra hydra.job.chdir=false
```

## 实现验证

专用测试覆盖所有层保留256 tokens、UV对应关系、面batch独立性、梯度、默认CNN参数／RNG／输出兼容、严格迁移、完整模型与Adam恢复、resume不重复初始化。真实validation样本`00000178`和`00007585`合计14faces／48half-edges，用于完整模型前后向及推理导出检查。

2026-10-17结果：相关回归测试 **185 passed**（其中UV专用66项）；真实小批量FP32前后向、CPU bf16训练模式前后向、mean/std与逐对象推理导出均通过。旧模型的689个保留state项逐张量完全相同。概率矩阵在CPU sigmoid中出现小于1e-9的浮点舍入差异，使用数值容差检查对称性；没有为此修改旧模型逻辑。

完整记录见 [VALIDATION.json](VALIDATION.json)。GPU峰值显存、吞吐、正式训练和OCC合法率均尚未验证；CPU小批量检查不代表研究结论。关联工作树的提交不会触发主工作区到5090服务器的自动同步。
