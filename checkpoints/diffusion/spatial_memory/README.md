# Spatial photo diffusion v1

已实现照片 → 512 个空间 query → 24 层 conditioned denoiser，VAE 保持原版冻结。当前只有实现验证，尚无正式训练或下游重建效果结果。研究依据与比较限制见 [PLAN.md](PLAN.md)。

## 模块与结构

| 文件 | 职责 |
|---|---|
| `src/brepnet/diffusion/spatial_conditioning.py` | 8³ 格心、XYZ Fourier 编码、4 层 Image-to-3D query encoder、RGB 旁路 |
| `src/brepnet/diffusion/spatial_blocks.py` | 规范化 attention、时间 adaLN、24 层 face denoiser、activation checkpointing |
| `src/brepnet/diffusion/spatial_model.py` | 冻结原 DINO/VAE、DDPM loss、采样、原 VAE decode、严格加载 |
| `src/brepnet/data/spatial_photo_dataset.py` | 同源照片/GT cache、完整 split、逐件摘要核验 |
| `src/brepnet/models/__init__.py` / `src/brepnet/train.py` | 独立模型注册与现有统一 Lightning 入口 |
| `configs/train_diffusion_spatial_photo.yaml` | 独立研究配置，不改变 baseline / DPO 配置 |

运行 `python checkpoints/diffusion/spatial_memory/draw_architecture.py` 生成带每层/形状标注的 [结构及模块关系图](visualizations/spatial_photo_v1.png)。可训练参数 449,166,368；冻结 DINO/VAE 485,955,965。DINO 仍为 FP32，后续训练采用 bf16 mixed。

```python
# 训练：只有 target 用 GT；条件编码不读取 GT 几何。
memory = model.extract_condition(photo)             # [B,768,512]
epsilon = model.denoiser(z_t, timestep, memory)      # [B,30,32]

# 推理：每张照片的空间 memory 只计算一次。
model.eval()
latent = model.sample_latents(photo, num_steps=1000) # [B,30,32]
decoded = model.decode_latents(latent)               # 原 VAE raw + 去重映射
```

空间 memory 是 512×512，附加 RGB 旁路后是 768×512；30 个 face slots 没有变成 798 个网络输出。K/V 投影属于每个去噪 block，参数各自独立；当前只缓存条件编码结果，没有缓存每层 K/V。

## 训练命令

本机 4090 环境为 `/mnt/d/anaconda3/envs/torch`，PyTorch 2.4.1/cu124。GPU7 留给用户。下述命令启动**新的从头训练**，不是从 DPO 接着训练；默认完整 validation 监测 epsilon MSE，不会在每次 validation 跑 1000 步几何生成。

```bash
cd /mnt/d/python
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 OMP_NUM_THREADS=4 \
/mnt/d/anaconda3/envs/torch/bin/python -m src.brepnet.train \
  --config-name train_diffusion_spatial_photo \
  trainer.devices=7 \
  trainer.batch_size=64 \
  trainer.learning_rate=1e-4 \
  trainer.precision=bf16-mixed \
  trainer.max_steps=10000000 \
  trainer.seed=42
```

数据/权重路径和 SHA256 显式写在该配置中；换机器需迁移对应资产并覆盖路径，不可换成另一 VAE 的旧 cache。7×64=448 全局 batch；`epoch_scale_factor=10` 使完整 validation 约每 1040 optimizer steps 执行一次。完整 checkpoint 包含冻结参数与 Adam 状态，约 6.9 GiB，写盘会暂停训练；定期 step checkpoint 只保留最近 3 份，validation 保留 best3/last。10M 是总步数上限，正式停止与选模须依据训练/验证状态及固定下游评估。

续训在同一条命令后增加 `trainer.resume_from_checkpoint=/绝对路径/原run/weights/last.ckpt`，恢复原 run、优化器和计步。恢复必须匹配架构与数据协议；不允许宽松加载旧模型。推理可用 `load_spatial_photo(checkpoint, device="cuda")`，严格恢复完整新模型，再调用上述采样/解码接口。

## 验证记录

- 新空间条件测试及既有 diffusion 合同：42 passed / 1 skipped（跳过的是既有可选项），不是质量评价。
- [真实 batch 检查](VALIDATION.json)：4 张训练图、完整尺寸网络，前向/反向/Adam 更新；全部可训练参数有有限梯度，DINO/VAE 无梯度；空间坐标 MLP 和 24 个 cross-attention 均有非零梯度。
- [batch64 检查](BATCH64_VALIDATION.json)：真实 64 图同样通过，单 GPU peak allocated 10.27 GiB，不含 DDP 额外通信 buffer，也不是稳定吞吐基准。
- 真实 validation 图片产生 `[1,768,512]` memory；4 步短采样输出有限 `[1,30,32]`，条件编码仅一次。真实 GT mean 经原 VAE 解码得到 `[7,16,16,3]` 曲面；没有用随机初始化模型的几何来宣称质量。
- 多卡短运行与严格恢复记录见 [DDP_VALIDATION.json](DDP_VALIDATION.json)；冻结参数逐 tensor 一致性见 [RESTORE_VALIDATION.json](RESTORE_VALIDATION.json)。

检查工具：`python -m tools.check_spatial_diffusion --batch-size 64 --output <report.json>`；单元检查：`python -m pytest -q src/brepnet/test/test_spatial_diffusion.py src/brepnet/test/test_diffusion_conditioning.py src/brepnet/test/test_diffusion_refactor.py`。训练/检查产物与权重未加入 Git。
