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

## 2026-10-17：首次训练中断后的数值修复

上节是实现时的验收记录。随后5090服务器的正式run
`candidate/20261017T085436.965155Z`在约1160步报`face_latents must be finite`退出，
最后完整checkpoint为1089步；三次validation loss为1.330961、0.627364、0.406857。
权重、Adam状态在1089步均有限，原run连同日志已回传并校验；不要修改其封存文件。
这次只完成了早期训练，未完成100000步，也没有下游合法率结论。

**已确认的问题是几何投影的反向数值实现，不需要更改UV编码器结构。**
薄三角片的外部点最终投到边界，但旧实现先计算平面投影，再用`torch.where`选择结果。
即使平面分支没有被选中，其中的大权重／小分母除法仍可能在反向产生NaN。
极短线段投到端点时存在相同问题。前向位置、距离都可以正常，因此只查loss不足以发现。

- `blocks.py`的两处投影函数现在先确定分支，在除法**之前**屏蔽未采用分支的分子和分母。
  选中的平面／线段投影继续传梯度，几何容差和损失定义不变；不删除投影，不替换NaN为零。
- `train.py`在无GradScaler的norm裁剪中使用`error_if_nonfinite=True`，阻止坏梯度经裁剪扩散并写入Adam。
  出错时报告step、rank、样本ID和异常参数；有GradScaler时保留原溢出处理。
  正常梯度使用原来的裁剪规则，不增加逐参数的常规有限性扫描。
- 网络参数、checkpoint keys、batch、学习率及训练预算均未改变。

CPU旧／新代码对照已复现：薄三角片和长度`1e-13`线段的前向逐位相同，旧版反向NaN，
新版反向有限；选中投影另通过FP64梯度对照和gradcheck。相关回归累计为**189 passed、1 skipped**
（源机运行时主动隐藏CUDA）。实际Lightning两进程CPU/DDP坏梯度检查能在更新前终止。

同样的旧／新投影对照已在RTX 5090、PyTorch 2.8.0/cu128上复现，见
[CUDA对照](validation/nan_diagnostic_20261017/cuda_projection_ab.json)。两个案例均为旧版前向有限、
几何梯度非有限；修复后梯度有限，投影前向逐位相同。

旧版和修复版分别从1089步完整恢复模型与Adam，以8卡、每卡16、bf16、学习率2e-5
运行到1300步，各完成211次更新；两条诊断轨迹的loss与逐参数梯度均有限，修复版更新前
检查正常通过。诊断关闭validation和常规指标记录，不是正式续训，见
[短诊断报告](validation/nan_diagnostic_20261017/summary.json)。GPU已释放，未启动正式续训。

原run没有保存首次坏梯度对应的batch和随机状态，因此上述缺陷不能直接称为原约1160步中断的
唯一已确诊原因。短诊断和修复验证与正式训练分开记录，不能用未再次报错证明整个训练已稳定。

修复后续训仍恢复原1089步的完整模型／Adam，不重新初始化UV。原run已封存，显式指定新的
续训输出目录，保留旧报告和校验清单。先将计算节点代码更新到本次数值修复提交，再运行：

```bash
cd /mnt/d/python
UV_PARENT=/mnt/d/python/checkpoints/uv_face_encoder/candidate/20261017T085436.965155Z
UV_CONTINUATION="/mnt/d/python/checkpoints/uv_face_encoder/candidate/$(date -u +%Y%m%dT%H%M%SZ)_numeric_fix"
/mnt/d/anaconda3/envs/torch/bin/python -m src.brepnet.train \
  --config-path "$UV_PARENT/metadata/invocations/20261017T085739.243727Z" --config-name config \
  trainer.resume_from_checkpoint="$UV_PARENT/weights/last.ckpt" \
  model.init_face_encoder_checkpoint=null \
  trainer.run_dir="$UV_CONTINUATION" hydra.run.dir="$UV_CONTINUATION" \
  hydra.output_subdir=metadata/hydra hydra.job.chdir=false
```

继续沿用原run的8卡、每卡16、bf16、学习率2e-5和总步数100000，不使用短诊断的预算。
新目录仍记录来源checkpoint、调用配置与Git版本；启动后需为新目录重新设置训练结束后的回传任务。
