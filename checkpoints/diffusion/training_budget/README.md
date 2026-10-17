# SVR 训练预算与续训

2026-10-17：用户要求将总步数上限直接设为 **10,000,000**，自行手动停止。
`configs/train_diffusion_layerwise_svr.yaml` 与新训练命令已同步修改；模型、学习率、batch、正常验证与 checkpoint 规则保持原样。

## 已完成运行

- Run：`checkpoints/diffusion_layerwise_svr/20261016T143450.891057Z`。
- 配置：8 卡 × B64，bf16，seed42，lr=1e-4，SVR view0，缓存 mean latent，训练/验证 photo ratio 均为0。
- 结束日志明确为 `max_steps=100000` reached；达到预算正常停止，没有启用 loss early stopping。
- 已直接读取 `weights/last.ckpt`：`global_step=100000`、`epoch=5`，Adam 参数状态的 step 均为100000，lr=1e-4。
- 末尾三个 10k 窗口的 rank-0 local training loss 均值为 0.03572、0.03423、0.03301，仍在下降。
- 最近三次全局 validation loss：54,639步0.042497、72,852步0.042730、91,065步0.044174；100k未另做validation。
  验证仍随机采样 timestep/noise，不能仅凭这几次波动判断显著过拟合；已保留按验证 loss 排名的权重。
- 本轮只修改预算和命令，尚未启动续训，也未评估生成/B-Rep 重建质量。

## 从100k继续

在 `/mnt/d/python` 执行；总目标为10M，包括已完成的100k。Lightning恢复权重、Adam状态与步数，继续写入原run。
内存盘数据仍需存在；重启后按 [暂存说明](../throughput/README.md) 重建。

```bash
/mnt/d/anaconda3/envs/torch/bin/python -m src.brepnet.train \
  --config-name train_diffusion_layerwise_svr \
  trainer.devices=8 trainer.batch_size=64 trainer.seed=42 trainer.max_steps=10000000 \
  trainer.resume_from_checkpoint=/mnt/d/python/checkpoints/diffusion_layerwise_svr/20261016T143450.891057Z/weights/last.ckpt \
  dataset.raw_data_root=/mnt/d/data/deepcad_v7 \
  dataset.condition_data_root=/dev/shm/hola-svr-20261016/conditions \
  dataset.cached_latent_root=/dev/shm/hola-svr-20261016/latents \
  dataset.train_dataset=/dev/shm/hola-svr-20261016/splits/train.txt \
  dataset.val_dataset=/dev/shm/hola-svr-20261016/splits/validation.txt \
  model.autoencoder.checkpoint=/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt
```

`trainer.max_steps` 是总 optimizer steps 上限，续训时不能继续传旧命令里的100000。
`trainer.resume_from_checkpoint` 用于恢复训练，不应替换为仅加载权重的 `init_from_checkpoint`。
