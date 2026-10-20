# VAE latent 32→64

本轮按现有shared-curve＋UV结构扩大面/边信息容量，Encoder、posterior和
Decoder全部参与训练。不是从随机模型重建，也不是resume原来的32维run。
实现及本机/142验证完成，提交与真实检查结果见下方；尚未启动正式训练。

## 改了什么

| 位置 | 本轮变化 |
|---|---|
| `src/brepnet/models/vae/checkpoints.py` | 按语义段扩展旧权重，完整记录81个变化tensor；697个保持 |
| `src/brepnet/models/vae/intersection.py` | 接入显式初始化来源；原forward计算不变，注释采用D维 |
| `src/brepnet/train.py` | resume清空扩容初始化来源，仍用统一trainer |
| `configs/model/vae_intersection.yaml` | 新增默认null的迁移字段，不影响旧配置 |
| `configs/train_vae_latent_expansion.yaml` | 64维候选，继承shared-curve/UV，解冻全部参数 |
| `configs/train_vae_latent_expansion_control.yaml` | 相同预算的32维解冻对照 |

面特征/latent32→64、边特征64→128。CNN仍在2×2/长度2处展平，
attention层数和内部宽度不变。参数182,298,143→182,982,439；
完整ASCII层级图、输入输出及迁移解释见[ARCHITECTURE](ARCHITECTURE.md)。

迁移复制通道并分摊后续输入权重，使初始posterior-mean结果尽量保留。
采用0.49/0.51分摊使复制通道能够收到不同梯度。独立posterior重采样的
训练分布不与32维严格相同；不能把这称为随机VAE完全等价扩容。

重建与UV等loss权重不变。恢复KL训练后，32维sum KL系数为1e-6，
64维为5e-7，抵消维度翻倍造成的总项标度变化。比较依据和边界见[PLAN](PLAN.md)。

## 在142前台运行

工作区 `/mnt/d/python`，分支 `codex/vae-uv-boundary`。
来源是shared-curve validation best，epoch116 / step42471；
SHA256 `ce749416c1635dda1563161b971bd4260d064cfdb0c31f1fa9d06ad0e70961ff`。

```bash
cd /mnt/d/python
conda activate torch
WANDB_MODE=online python -u -m src.brepnet.train \
  --config-name train_vae_latent_expansion \
  dataset.raw_data_root=/mnt/d/data/deepcad_v7 \
  dataset.edge_uv_root=/mnt/d/data/deepcad_v7_uv/v1 \
  dataset.train_dataset=checkpoints/joint_boundary/validation/train_contract_audit/accepted.txt \
  dataset.val_dataset=checkpoints/joint_boundary/validation/val_contract_audit/accepted.txt \
  dataset.test_dataset=checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt \
  'model.init_latent_expansion_checkpoint="/mnt/d/python/checkpoints/shared_curve_uv/candidate/20261019T064906.049992Z/weights/epoch=0116-step=00042471-val_loss=0.036517.ckpt"' \
  trainer.devices=8 trainer.batch_size=16 trainer.seed=42 \
  trainer.learning_rate=2e-5 trainer.precision=bf16-mixed \
  trainer.max_steps=100000 trainer.wandb.enabled=true trainer.wandb.offline=false
```

W&B名称 `vae-shared-curve-latent64`，输出
`checkpoints/latent_expansion/d64/<UTC时间>/`。Ctrl+C中断前台训练；
继续时沿用上述64维命令，改为
`model.init_latent_expansion_checkpoint=null`，
再加 `trainer.resume_from_checkpoint=<本次64维run的weights/last.ckpt>`。
该resume恢复optimizer和step，不能填32维来源checkpoint。

要跑32维解冻对照，保持同样数据和训练参数，把配置名改为
`train_vae_latent_expansion_control`，把初始化参数名改为
`model.checkpoint`，值仍是上述同一个shared-curve来源。
W&B名称 `vae-shared-curve-latent32-control`。本次不自动启动候选或对照。

新Encoder改变latent分布，64维不能使用旧32维diffusion/cache；后续重建评估
必须由各自Encoder编码同一批validation CAD，不能套用前几轮缓存的latent。

## 验证入口

```bash
python -m pytest -q \
  src/brepnet/test/test_latent_expansion.py \
  src/brepnet/test/test_shared_curve_uv.py \
  src/brepnet/test/test_uv_boundary_model.py \
  src/brepnet/test/test_intersection_vae.py \
  src/brepnet/test/test_intersection_losses.py

python checkpoints/latent_expansion/validate_implementation.py \
  --checkpoint 'checkpoints/shared_curve_uv/candidate/20261019T064906.049992Z/weights/epoch=0116-step=00042471-val_loss=0.036517.ckpt' \
  --data-root /mnt/d/data/deepcad_v7 \
  --uv-root /mnt/d/data/deepcad_v7_uv/v1 \
  --output checkpoints/latent_expansion/validation/remote_real_smoke.json \
  --device cuda:0 --batch-size 2
```

大batch的BF16检查添加 `--batch-size 16 --precisions bf16-mixed`，
输出另取文件名，不覆盖前次报告。小批量来自固定validation清单前段，
不执行optimizer step，不构成研究效果评估。

## 实现与部署记录

实现提交：`4f7b2df78f5ad22313dca0ea9e732db53ba4344e`，
基于 `85f6fb33da269fd75d0f7352164ce1af3663be27`。
同分支快进到142 `/mnt/d/python`，未改主工作区的diffusion分支。
本工作树不触发主工作区的自动同步hook，部署通过直接Git快进与远端HEAD核验。
同步前142的32份旧UV encoder未跟踪文件先stash保存，SHA与stash引用位于
`/mnt/d/migration/latent_expansion_20261020/untracked_backup.json`；
部署收尾按摘要恢复，未纳入本轮提交，未覆盖历史数据或权重。

- 本机72项相关测试通过（18.71秒）；142相同72项通过（19.07秒）。
- 两机真实checkpoint、7+20面的小批量FP32/BF16前后向通过；
  Encoder、posterior、Decoder均有有限非零梯度，CNN复制通道梯度不同。
- 142额外检查每卡16件，211面，共享/UV纯latent推理通过。
  BF16前后向约1.60秒、peak allocated 5.76GB（5.37GiB）。
  这是该validation batch的单卡实现检查，不含Adam状态或DDP额外开销，
  不代表所有面数分布的显存上限或正式多卡吞吐。
- 142的16件mean初始化：final face XYZ最大差8.34e-7、edge4.66e-5，
  全部候选的存在性判定未变化；全部参数可训练，新增684,296。
- 完整8卡训练命令Hydra解析、路径和在线W&B认证通过，未创建正式训练run。

检查记录：[测试](validation/tests.json)、[本机真实数据](validation/local_real_smoke.json)、
[142小批量](validation/remote_real_smoke.json)、[142每卡16件](validation/remote_batch16.json)、
[训练配置解析](validation/remote_preflight.json)。
实际模型/配置文件SHA记录在各GPU报告内，区分实现检查与训练结果。

最初把所有输出都限制在1e-5绝对误差的检查，受到扩维后FP32求和顺序影响，
在少数UV和local方向通道未通过。诊断后明确了分量验收界限：
XYZ/bbox绝对1e-4、UV/local2e-4、logits5e-4，统一相对1e-4，
并要求本批候选存在性判定不变；报告保留原始max/mean差。
[初次诊断](validation/local_diagnostic.json)明确标为非验收，不作为通过结果。
这不是OCC精度保证，也不是用容差替代扩容后的下游研究评估。
