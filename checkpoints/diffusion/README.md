# 单视图 Diffusion 实现地图

2026-10-16：实现从 `src/brepnet/models/` 迁入 `src/brepnet/diffusion/`。
本次保留计算行为与已有权重接口；未进行正式训练或效果评估。目的、范围见 `PLAN.md`，验证见 `VALIDATION.json`。

## 从哪里阅读

```text
src/brepnet/diffusion/
├── __init__.py          # 按需导出 Diffusion
├── model.py             # 网络装配、训练 forward、推理入口与 VAE 接口
├── conditioning.py      # 条件工厂、单图/点云路由
├── image_encoder.py     # 冻结 DINOv2 / Depth Anything ViT + 可训练投影
├── fusion.py            # CAD query → 图像 memory；全局 alignment 支路
├── denoiser.py          # 输入投影、时间编码、self-attention 主干、输出投影
├── topology.py          # 可选 oracle / learned attention mask 或 bias
├── latents.py           # cached mean/std 或冻结 VAE 编码
├── padding.py           # 面集合打包、validity、latent 去重
├── losses.py            # 双向 InfoNCE 与训练损失汇总
├── sampling.py          # DDPM 反向迭代
├── point_encoder.py     # 保留的点云条件分支
└── legacy_encoders.py   # 相机 embedding、禁用的文本路线兼容组件
```

`models/vae.py` 和 `models/blocks.py` 仍是外部 VAE/共享算子依赖。
`models/__init__.py` 中的 registry 已指向新包，`train.py` 继续调用同一个 `build_model`。
独立 `inference.py` 也使用新包。旧 `models/diffusion*.py` 和 `models/condition_encoders.py`
只做兼容导出，历史脚本无需批量改写。新代码从新包导入：

```python
import torch

from src.brepnet.diffusion import Diffusion

# 配置的字段与之前一致；生产使用前明确指定冻结 VAE 的 checkpoint。
model = Diffusion(model_cfg, condition_cfg)
losses = model(batch)
losses["total_loss"].backward()

# 推理条件为 conditions["imgs"]；默认模型无需 GT 面数或 GT topology。
model.eval()
with torch.no_grad():
    outputs = model.inference(num_samples=2, device=device, v_data=batch)
```

## 默认单视图网络

以下数值来自 `configs/model/diffusion.yaml` 与 `configs/condition/single_img.yaml`，
不是对每个历史 checkpoint 配置的推断。`B` 为 batch，`F=30` 为固定 slots，`C=32` 为每面 latent。

| 部件 | 层与张量 |
|---|---|
| 图像输入 | dataset resize/normalize 后 `[B,3,224,224]` |
| 冻结 DINOv2 ViT-L/14 | 24 blocks，16 heads，hidden 1024，FFN 1024→4096→1024/GELU；输出 `[B,257,1024]`，含 1 CLS + 256 patches |
| 图像投影 | Linear1024→1024、LN、SiLU、Linear1024→1024；wrapper 返回 `[B,1,257,1024]` |
| 训练 latent | dataset 复制并重排面到 30 slots；cache `[B,30,64]=[mean,std]`，默认只取 mean 得 `x0:[B,30,32]` |
| noisy latent 投影 | Linear32→768、LN、SiLU、Linear768→768 |
| 条件融合输入 | CAD query Linear768→1024，图像 memory 为 `[B,257,1024]` |
| 条件融合器 | 4× TransformerDecoderLayer：pre-LN self-attention(16 heads) + residual；pre-LN cross-attention(Q=30 slots,K/V=257 tokens) + residual；pre-LN FFN1024→2048→1024/ReLU + residual |
| 融合输出 | LN1024、Linear1024→768，输出 `[B,30,768]` |
| 时间支路 | t `[B]` → sin/cos768 → Linear768→768、LN、SiLU、Linear768→768 → `[B,1,768]`，在 fuser 之后广播相加 |
| 主干 | 24× TransformerEncoderLayer：pre-LN self-attention(12 heads) + residual；pre-LN FFN768→2048→768/ReLU + residual；最后 LN768 |
| 输出投影 | Linear768→768、LN、SiLU、Linear768→32，预测 epsilon `[B,30,32]` |
| 采样后 | latent 距离去重 → 每对象 `[Fi,32]` → 冻结 VAE；几何拟合/裁剪/缝合/STEP 属于外部后处理 |

Transformer dropout 默认 0.1。fuser 的 heads=hidden/64、FFN2048 和 dropout0.1 是其构造器内既有设置，
不受 `denoiser.feedforward_dim/dropout` 控制。图像每次推理编码一次；每个 DDPM timestep 重跑 fuser 和主干。
主干各层没有独立 image cross-attention，fuser 自身不显式接收 timestep。

## 训练目标与可用信息

默认 cosine DDPM，1000 timesteps，epsilon prediction，fixed_small variance，clip_sample=False。
训练 target 为采样噪声，L2 对全部 slots/通道求均值；默认总损失为 diffusion L2 + alignment。
`sample` prediction 为已有可选分支，其 target 是 clean latent；不能据此宣称支持任意 scheduler prediction_type。

Alignment 使用 clean latent 的共享输入投影和 query 投影：

```text
clean CAD [B,30,1024] → Linear1024→256 → mean slots → Linear256→256 → L2 normalize
image     [B,257,1024] → Linear1024→256 → mean tokens → Linear256→256 → L2 normalize
全局 embedding [B,256] → batch 内双向 InfoNCE，temperature=0.07，weight=1.0
```

这是对象级匹配，不是面/像素对应监督；mean 包含重复 slots 和 CLS。
梯度进入共享输入/query 投影、图像投影及四个 alignment heads，
不经过融合 decoder、时间网络或去噪主干；DINO/VAE 冻结。
当前没有跨卡 negatives all-gather、classifier-free guidance、condition dropout、渲染重投影或 solid validity 训练项。

默认 random padding 复制真实面并随机排列，mask 全 true，未向网络提供面位置编码或 GT 面数。
推理从高斯噪声开始，经 DDPM 后以 latent 平均绝对差 `<1e-2` 顺序去重，间接决定面数。
VAE 输出面/半边采样几何与连接信息，合法 solid 仍依赖后处理。

## 权重与迁移

保留原 `nn.Module` 属性名和注册顺序，未在 alignment 外新增 module 包装层。
尤其保留 `autoencoder.*` / `latent_provider.autoencoder.*` 的重复注册路径、
`face_padder.*` / `latent_provider.face_padder.*`、未用的 `camera_embedding`，
以及名称仍为 `oracle_topology_self_attention` 的可选 learned 分支。
新旧 `state_dict` 可严格互载；Lightning 的 `model.` 前缀不变。配置、latent cache、冻结 VAE 接口未迁移。
验证覆盖范围见 `VALIDATION.json`；没有载入所有历史生产 checkpoint 做训练续跑。

## 本轮记录而未修复的既有问题

- alignment 在推理每步仍被计算并丢弃；将当前 noisy latent 传给 alignment 不等于推理获得 GT。
- zero padding validity head 在训练读 prediction、推理读最终 latent，epsilon 配置下输入分布不一致。
- learned topology 预测固定 slots adjacency，尚未解决随机面 slot 对应；无自动 topology BCE。
- GT topology 属 oracle，默认关闭。可选 bias 仅作用于 denoiser 主干 self-attention，不进入 fuser。
- `camera_embedding` 未进入单图 forward；图像 cached_features 没有接通，始终读取 `imgs`。
- `inference.py` 的旧 AE 权重过滤未覆盖 `latent_provider.autoencoder.*`，可能覆盖独立指定的 AE；属于独立推理工具的既有问题。
- `autoencoder.checkpoint=null` 不自动阻止冻结随机 AE；正式运行需要显式来源并与 latent cache 同源。

本次没有将这些行为变更混入源码迁移。合成真实感照片和真实照片的评估区分、
单视图隐藏几何的不确定性，以及公平比较预算仍遵循根目录研究指南。
