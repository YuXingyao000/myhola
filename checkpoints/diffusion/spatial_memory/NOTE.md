# 8 卡正式训练启动记录

2026-10-28，用户要求“如果空着就直接8卡训练起来”。启动前 GPU0–7 均为 0% 利用率、15 MiB 显存占用，没有 CUDA compute process；因此启用全部 8 张 RTX 4090，覆盖此前预留 GPU7 的安排。

## 运行合同

- run：`/mnt/d/python/checkpoints/diffusion_spatial_photo/20261028T104221Z`。
- tmux：`spatial-photo-20261028T104221Z`；退出状态由 `metadata/exit_code.txt` 记录，运行中尚无此文件。
- W&B：[photo_spatial_memory_v1_8gpu_20261028T104221Z](https://wandb.ai/whatseven777/hola-brep/runs/1e37b32cb626)。
- 启动代码 commit：`9c38bdc27608424fc73c5cb44e4d1e3b93f3dd74`。现有其他主题的工作区修改被保留；统一入口记录 dirty diff，本次另存实际 runtime 源码快照及 377 个文件 SHA256 至 `metadata/`，不将其混入本轮 Git 提交。
- 模型：`SpatialPhotoDiffusion / spatial_photo_v1`，新空间 encoder 和 24 层去噪器从头训练；原 HoLa DINO/VAE 冻结，无 DPO、几何辅助 loss 或 GT 推理条件。
- 数据：46,627 个训练 CAD、3,505 个完整 validation CAD，FLUX 合成真实感图像、identity rotation 与同源原 VAE mean cache。文件路径与 SHA256 由 resolved config、数据 provenance 和逐样本校验记录。
- 8 GPUs × batch64 = 全局 batch512；Adam `1e-4`，bf16 mixed，seed42，最大 10,000,000 optimizer steps。
- `epoch_scale_factor=10`，实际每 epoch 910 optimizer steps；每个 epoch 完整验证 epsilon MSE。保留正常 sanity validation；没有把 smoke 的 batch 限制带入正式训练。
- checkpoint：完整 validation best3/last；每 5000 steps 的定期快照保留最近 3 份。权重、优化器和日志写入当前 run，各自目录独立。

## 启动验收

8 个 rank 已完成初始化和 sanity validation，进入正式训练。10:44:44 UTC 检查已到 epoch0 / step109，8 张卡均为 100% 利用率，显存约 16.8–17.0 GiB/卡，启动以来平均约 1.16 step/s，没有退出或 OOM。W&B 已记录 global_step49 的 rank0 local epsilon MSE 为 0.6625608；这是有限的训练诊断值，不是全局 validation loss。启动状态快照见 [LAUNCH_VALIDATION.json](LAUNCH_VALIDATION.json)。这些是运行健康检查，不是生成质量结果；尚未完成第一个完整 validation，也未做下游几何评估。

控制台日志：`logs/console.log`。完整启动命令、实际源代码快照、Git/配置、W&B ID 均保存在该 run 的 `metadata/`。
