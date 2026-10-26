# 原版 HoLa 照片 RGB baseline

2026-10-25 用户授权长时间训练。这里使用已有 FLUX **合成真实感图片**，不称作实拍数据。当前目标是获得照片域的原版网络基线，之后再讨论条件架构。

2026-10-26 用户要求停训并评价：训练已停止，保留最近完整 step310000 和最低监测 validation loss 的 step4553。固定照片 validation100 的三组对比、停止记录与图表见 [evaluation_20261026](evaluation_20261026/README.md)。下面“已启动”保留最初启动时的历史记录。

同日按用户要求扩展至完整3,505件validation。冻结协议与可复现命令见 [全量计划](evaluation_full_20261026/PLAN.md) 和 [启动脚本](evaluation_full_20261026/RUN_EVALUATION.sh)。三组权重、输入、采样和后处理参数保持一致；原300条结果通过摘要校验后引用，补跑新增10,215次重建。完成状态以全量目录的 `status.json` 和评估报告为准，不能用100件的结果替代全量结果。

## 已启动

正式 run：`/mnt/d/python/checkpoints/diffusion_photo_baseline/20261025T125340Z`，后台 tmux `hola-photo-baseline`。在线曲线：[W&B](https://wandb.ai/whatseven777/hola-brep/runs/ca1061135728)。12:55 UTC 启动核验窗口从 step520 推进至 704，约 7.145 step/s，8 卡平均 GPU utilization 为 99.4%–100%，显存约 7.4–7.5 GiB。这个速度不含定期验证 / 存盘开销；不代表一夜已达到原来的 150 万步。

运行代码 commit `7134f28c`；另外保存 520 个源码 / 配置文件快照与依赖版本，覆盖未提交依赖。正式 run 此时还没有到首次保存点；短跑的 checkpoint / Adam 恢复已通过。之后约每 4,553 steps 的验证会保存 last / best，每 5,000 steps 保存定期快照。

```bash
# 实时查看（不另起训练）
tail -f /mnt/d/python/checkpoints/diffusion_photo_baseline/20261025T125340Z/logs/console.log
```

分支 `codex/photo-rgb-baseline-20261025`。源机提交后的自动同步被目标机“必须已处于同名分支”规则拒绝；保留目标分支，未强推 / 切换远端。当前训练在源机 4090 上运行，不依赖该同步。

## 固定协议

| 项目 | 本轮设置 |
|---|---|
| 网络 | 原版 HoLa，1024 维、24 层 Transformer、CLS concat |
| 初始化 | `Diffusion_svr_sq30_1500k.ckpt`，原训练到 1,500,000 steps；严格加载 |
| 新训练 | 新 Adam / 新步数，SVR → 合成照片迁移；不是从零训练或原 optimizer 续训 |
| 输入 | `real_photo.npz["flux"]` 完整 RGB，固定 rotation 0；224 resize + ImageNet normalize |
| 图像编码 | 原 checkpoint 的冻结 DINOv2 ViT-L；离线 FP32 CLS `[1024]`，训练时仍更新 `img_fc` |
| 目标 | 原始 uniform-t / epsilon DDPM MSE，1000 个 diffusion timesteps |
| latent | 原 checkpoint 内 AE 同源的 FP32 mean32；随机复制 / 排列为 30 个 face tokens |
| 训练集 | 去重 CAD train，46,627 个；固定照片，不做输入增强或质量筛选 |
| loss 监测 | 既有固定 validation100；不作为全量重建质量结论 |
| 设备 / batch | 8 × RTX4090，64 / GPU，全局 512；与原 SVR checkpoint 全局 batch 相同 |
| optimizer | Adam，lr=1e-4，BF16，gradient clip norm=1，seed=42 |
| 预算 | 新照片训练最多 10,000,000 optimizer steps，无早停 / 时间上限；手动停止 |
| 保存 | 每 5,000 steps，最近 3 个定期快照 + last；validation 最优 3 个 + last |

每个训练 epoch 将 train 重复 50 次，约 4,553 个 optimizer steps，随后监测 validation loss。DDP 会把 100 个验证条目补齐至 104 次读取；随机 timestep / padding 的这一数值只用于运行监测。正式下游比较应使用固定样本和预算、单卡无重复推理，分别报告几何与 STEP/solid 有效性。

没有 DPO、几何辅助 loss、depth/normal 条件或架构增层。冻结输出缓存的依据是本轮无随机图像增强；之后加入增强时必须改变供数协议，不能继续把固定特征当作已增强输入。

## 来源与实现验证

- [PLAN.md](PLAN.md)：研究假设、输入输出与错误处理。
- [CACHE_VALIDATION.json](CACHE_VALIDATION.json)：全部 46,727 条照片 / latent 均成功，未静默删样本；split、文件与原 checkpoint 指纹。
- [MODEL_PREFLIGHT.json](MODEL_PREFLIGHT.json)：原 FP32 diffuse 与新入口逐位一致；真实 batch64 的 12 次 BF16 更新，有限梯度、DINO 冻结。
- [DDP_VALIDATION.json](DDP_VALIDATION.json)：8 卡从 step32 恢复到 step36，所有 Adam state 的 step 同步恢复；原 SVR 路径设为不存在仍能恢复。
- [RUN_STATUS.json](RUN_STATUS.json)：正式长跑的启动核验快照；其中 step 不是实时值。

相关 dataset / checkpoint / device / preference 单元检查共 27 项通过。正式 W&B 已记录的启动 loss 均有限；这些实现检查不替代下游重建评估。

原 checkpoint SHA256：`9dcafb192529c1ddf65942392c25b409fbbf2bce407805f34a460d0b44bb4839`。缓存父协议绑定原 AE；没有复用名字相似的另一版 VAE cache。缓存分片合并修正为每个 NPZ member 只读取一次；中断后的合并核对原协议和每条 shard identity，不重复编码图片。

新模块为 `src/brepnet/diffusion/photo_baseline.py`、`src/brepnet/data/photo_dataset.py`，共享既有 `LegacySVRDenoiser`，统一训练入口注册为 `PhotoBaselineDiffusion`。旧模型严格迁移，完整恢复从新 checkpoint 读取架构与状态，不再依赖原 SVR 权重文件。推理接口是 `sample_latents()`；生成完整 B-Rep 必须显式接回同源原版 AE 和固定后处理，不能套用当前逐层 cross-attention 模型的默认推理器。

## 训练与恢复

首次训练（默认生成独立时间戳 run；不要用它重复启动已在运行的任务）：

```bash
cd /mnt/d/python
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=4 \
  PYTHONPATH=/mnt/d/python /mnt/d/anaconda3/envs/torch/bin/python -u \
  -m src.brepnet.train --config-name train_diffusion_photo_baseline \
  trainer.devices=8 trainer.batch_size=64 trainer.seed=42 \
  trainer.learning_rate=1e-4 trainer.precision=bf16-mixed \
  trainer.max_steps=10000000 trainer.max_time=null \
  dataset.cache_root=/mnt/d/python/checkpoints/diffusion/photo_baseline/cache_v1 \
  model.legacy_checkpoint=/mnt/d/ckpt/HoLa/Diffusion_svr_sq30_1500k.ckpt
```

实际后台 run 的路径、日志和恢复脚本见 `RUN_STATUS.json`。恢复必须传原 `trainer.run_dir` 与已有 checkpoint；不要新建 run 后只加载 policy，否则 optimizer / step 不会续接。普通 DataLoader 不保存 epoch 内精确取样位置和随机数轨迹；这里验证的是模型 / Adam / 步数恢复，不声称与完全不中断运行逐位一致。

运行元数据保留 Git commit / dirty diff、命令、resolved config，另存相关源码快照，覆盖工作区中尚未提交的依赖。其他研究分支的已有修改保留，不纳入本次提交。

## 解释边界

这是从 SVR 预训练权重开始的照片域适配基线。新照片步数单独计算；原训练使用旋转增强，本轮固定单图，因此相同全局 batch / 步数不代表完全相同的数据多样性。达到 150 万新增 steps 需要多日，不能理解为一夜已经等量训练。实现检查不证明合成照片的几何标签可靠，也不证明实拍重建或有效率提升。
