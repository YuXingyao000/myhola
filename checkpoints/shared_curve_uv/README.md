# 共享曲线 + 逐侧 UV

状态：已实现，65项相关测试通过；真实validation小批量FP32/BF16前后向通过；尚未正式训练或评估合法率。代码基准`580bec89b378796c144476771cdce6c396cc1215`，实现hash和142同步状态在提交后补录。

[研究问题/范围](PLAN.md) · [逐层ASCII架构与张量形状](ARCHITECTURE.md) · [真实小批量报告](validation/real_smoke.json)

## 本次新增与调整

| 文件 | 改动 |
|---|---|
| `src/brepnet/models/vae/decoders/primitives.py` | 新增SharedCurveFusion、CurveConditionedUVPathHead |
| `src/brepnet/models/vae/decoders/intersection.py` | final从一对有向特征生成一条共享XYZ，再展开为两侧反向视图和UV |
| `src/brepnet/models/vae/losses/intersection.py` | final标签采用同一个GT参数化，coarse/teacher不变 |
| `src/brepnet/models/vae/checkpoints.py` | 已训练UV checkpoint的显式迁移 |
| `src/brepnet/models/vae/intersection.py`、`src/brepnet/train.py` | 原VAE装配与resume接入，无新增trainer |
| `configs/model/vae_intersection.yaml` | 默认关闭的shared_curve开关和初始化字段 |
| `configs/train_vae_shared_curve_uv.yaml` | 独立实验入口，继承原loss权重，W&B在线 |

原8层512维inter、coarse/refiner/classifier、CNN encoder、面头及曲线头宽度均保留。新增参数45,630。encoder/posterior继续冻结。数据、UV sidecar、后处理不改。

初始化没有直接平均互反特征：真实小批量表明这样会破坏旧曲线输出，现采用代表方向特征加零初始化的双侧融合残差。初始代表曲线保留，反向侧使用同一曲线反序。UV头的部分输入/输出权重映射和face索引代表选择的限制在ARCHITECTURE中明确记录。

## 在142直接运行

目标为`172.31.224.142:/mnt/d/python`，分支`codex/vae-uv-boundary`。不要resume旧UV checkpoint：这是新结构、新optimizer的run。初始化来源为UV boundary best（epoch176/step64251），SHA256：`f7b550689440afb36ffbc16df3ed0adbc2394ecbd59f00d47f337e24f43adfdf`。

```bash
cd /mnt/d/python
conda activate torch
WANDB_MODE=online python -u -m src.brepnet.train \
  --config-name train_vae_shared_curve_uv \
  dataset.raw_data_root=/mnt/d/data/deepcad_v7 \
  dataset.edge_uv_root=/mnt/d/data/deepcad_v7_uv/v1 \
  dataset.train_dataset=checkpoints/joint_boundary/validation/train_contract_audit/accepted.txt \
  dataset.val_dataset=checkpoints/joint_boundary/validation/val_contract_audit/accepted.txt \
  dataset.test_dataset=checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt \
  'model.init_shared_curve_checkpoint="/mnt/d/python/checkpoints/uv_boundary/candidate/20261018T080851.446914Z/weights/epoch=0176-step=00064251-val_loss=0.037775.ckpt"' \
  trainer.devices=8 trainer.batch_size=16 trainer.seed=42 \
  trainer.learning_rate=2e-5 trainer.precision=bf16-mixed \
  trainer.max_steps=100000 trainer.wandb.enabled=true trainer.wandb.offline=false
```

沿用8卡×16、lr2e-5、seed42、bf16-mixed、100000步上限。`uv_weight=.01`、`uv_anchor_weight=.1`，本轮未改。训练/选模用validation，训练命令声明的test路径不会在fit时据此做失败分析。W&B名称`vae-shared-curve-uv-v1`。新run输出`checkpoints/shared_curve_uv/candidate/<UTC时间>/`；Ctrl+C中断前台训练。

继续同一次新实验时，使用`trainer.resume_from_checkpoint=<本实验weights/last.ckpt>`；不要把旧UV来源填入resume。保留新实验结构配置，trainer会清空组件初始化来源并恢复optimizer/step。

## 实现验证

```bash
python -m pytest -q src/brepnet/test/test_shared_curve_uv.py \
  src/brepnet/test/test_uv_boundary_model.py \
  src/brepnet/test/test_intersection_vae.py \
  src/brepnet/test/test_intersection_losses.py

python checkpoints/shared_curve_uv/validate_implementation.py \
  --checkpoint 'checkpoints/uv_boundary/candidate/20261018T080851.446914Z/weights/epoch=0176-step=00064251-val_loss=0.037775.ckpt' \
  --data-root /mnt/d/data/deepcad_v7 \
  --uv-root /mnt/d/data/deepcad_v7_uv/v1 \
  --output checkpoints/shared_curve_uv/validation/remote_real_smoke.json
```

小批量从固定validation清单前8个读取生产sidecar，前2个对象（7+20面、124条GT连接行）执行GPU前后向，不更新optimizer。验证严格共享XYZ/UV反序、767项旧权重保留、初始代表曲线保留、encoder冻结、UV到XYZ梯度、纯latent推理、strict full/resume、旧开关关闭兼容、空候选和标签参数化。

本机检查：初始代表曲线最大坐标差8.35e-7（CUDA计算次序差异），FP32总loss约0.0389，BF16约0.0429，均有限；UV loss回传C的梯度非零。两对象初始edge CD²旧2.08e-5、新1.93e-5，仅说明没有初始化精度崩塌，不是有效性收益证据。全量合法率和失败分析等待正式训练完成。
