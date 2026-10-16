# 单视图 Diffusion：逐层图像条件

2026-10-16：当前实现将前端 4 层 fuser 替换为 24 组「原 EncoderLayer → image cross-attention」，移除 alignment。
这是架构变更，尚未正式训练或评估重建质量。研究假设与比较范围见 [PLAN.md](PLAN.md)，本轮接口验证见 [LAYERWISE_VALIDATION.json](LAYERWISE_VALIDATION.json)。

旧模块化 baseline 保存在 `codex/diffusion-modular-baseline`，commit `b67c5d8e`。
[VALIDATION.json](VALIDATION.json) 只记录该 baseline 的行为保持拆分验证，不能用作当前架构的兼容性或效果证据。

## 从哪里阅读

```text
src/brepnet/diffusion/
├── __init__.py          # 按需导出 Diffusion
├── model.py             # 装配、训练 forward、VAE 接口、严格权重加载
├── conditioning.py      # 条件工厂与编码路由
├── image_encoder.py     # 冻结 DINOv2 / Depth Anything ViT + 可训练投影
├── fusion.py            # 单层 pre-LN image cross-attention + residual
├── denoiser.py          # 输入/时间投影、24 组 encoder + cross、最终 LN/输出
├── topology.py          # 可选 oracle / learned self-attention mask 或 bias
├── latents.py           # cached mean/std 或冻结 VAE 编码
├── padding.py           # 面集合打包、validity、latent 去重
├── losses.py            # diffusion loss 与可选 validity loss 汇总
├── sampling.py          # DDPM 反向迭代
├── point_encoder.py     # 保留的点云条件组件
└── legacy_encoders.py   # 历史独立组件，不进入当前单图模型
```

`models/vae.py` 和 `models/blocks.py` 仍是外部 VAE/共享算子依赖。
`models/__init__.py` 的 registry 与 `inference.py` 使用新包。旧路径保留的类导出只用于定位当前实现，
不恢复旧前端 fuser、alignment 或旧配置语义。新代码从新包导入：

```python
import torch

from src.brepnet.diffusion import Diffusion

# model_cfg 使用当前配置，冻结 VAE 必须显式指定与 cache 同源的 checkpoint。
model = Diffusion(model_cfg, condition_cfg)
losses = model(batch)
losses["total_loss"].backward()

# conditions["imgs"]: [B,3,224,224]；默认推理无需 GT 面数或 GT topology。
model.eval()
with torch.no_grad():
    outputs = model.inference(num_samples=2, device=device, v_data=batch)
```

## 默认单图网络

数值对应 `configs/model/diffusion.yaml` 与 `configs/condition/single_img.yaml`。
`B` 为 batch，`F=30` 为固定 latent slots，`C=32` 为每面 latent；图像 token 数 `T=257`。
配置使用 `model.denoiser.conditioning=layerwise_cross_attention`，memory 维度只取 `condition.output_dim`，不再重复配置 `condition_fuser.condition_dim`。

| 部件 | 层与张量 |
|---|---|
| 图像输入 | resize/normalize 后 `[B,3,224,224]` |
| 冻结 DINOv2 ViT-L/14 | 24 blocks，16 heads，hidden1024；输出 `[B,257,1024]`，含 1 CLS + 256 patches |
| 可训练图像投影 | Linear1024→1024、LN、SiLU、Linear1024→1024；编码 wrapper 返回 `[B,1,257,1024]`，供各层共享的 memory 为 `[B,257,1024]` |
| 训练 latent | cache `[B,30,64]=[mean,std]`，默认取 mean 为 `x0:[B,30,32]`；训练加噪后得到 `x_t` |
| latent 输入投影 | Linear32→768、LN、SiLU、Linear768→768，输出 `[B,30,768]` |
| 时间支路 | `t:[B]` → sin/cos768 → Linear768→768、LN、SiLU、Linear768→768 → `[B,1,768]`，在主干入口广播相加 |
| 每组原 EncoderLayer | pre-LN self-attention：12 heads × 64 + residual；pre-LN FFN：768→2048→768、ReLU、dropout0.1 + residual |
| 每组新增 image cross-attention | pre-LN query；Q768→768，K/V1024→768；12 heads × 64，attention `[B,12,30,257]`；Wout768→768、dropout0.1 + residual |
| 重复与输出 | 以上 encoder + cross 共 24 组，各组参数独立；最后 LN768，再 Linear768→768、LN、SiLU、Linear768→32，预测 epsilon `[B,30,32]` |
| 采样后 | latent 距离去重 → 每对象 `[Fi,32]` → 冻结 VAE；拟合/裁剪/缝合/STEP 是外部后处理 |

主路的关键顺序为：

```text
x_t → input_projection → + time_embedding
    → [完整 EncoderLayer → ImageCrossAttention + residual] × 24
    → final LayerNorm → output_projection → epsilon
```

时间仍在原主干入口加入。移除前端 fuser 后，首个 image query 自然已经包含时间信息；没有逐层额外加 t，也没有 adaLN。
每次推理仅编码一次图像，各层共享 projected image memory，但每个 cross-attention 有自己的 Q/K/V/Wout 参数。
每个 DDPM timestep 都运行完整 24 组；默认主干与 cross-attention 的 dropout 都为 0.1。

DINO 与 VAE 冻结。图像 projection、输入/时间/输出 MLP、原 encoder 主干和新增 cross-attention 可训练。
不存在前端 4 层 fuser、alignment heads、InfoNCE loss 或单图模型中未使用的 camera embedding。

## 训练目标与可用信息

默认 cosine DDPM、1000 timesteps、epsilon prediction、fixed_small variance、`clip_sample=False`。
训练 target 为采样噪声，L2 对全部 slots/通道求均值；random padding 的默认总损失只有 diffusion L2。
已有 `sample` prediction 的 target 为 clean latent；zero padding 仍有可选 validity loss，但不是本轮研究主线。
当前没有 classifier-free guidance、condition dropout、渲染重投影或 solid validity 训练项。

训练使用 GT 面数打包 cache/latent；random padding 复制真实面并随机重排，mask 全 true。
模型不加入 face slot 位置编码，默认推理不接收 GT 面数或 GT topology。
采样从高斯噪声开始，结束后以 latent 平均绝对差 `<1e-2` 顺序去重，间接决定面数。
隐藏几何仍由条件分布与训练先验补全；逐层 cross-attention 本身不保证唯一重建、面/像素对应或合法 solid。

此次研究与实质验证聚焦单图条件。保留的 point cloud、sketch、无条件接口不等于已经验证同等效果。
Depth Anything 选项仍只读取 ViT 特征，不产生深度预测或深度监督。

## 训练入口

当前入口为 `configs/train_diffusion_layerwise_real.yaml`，继承 `train_diffusion_baseline_real` 的数据协议和预算，
只将 `trainer.exp_name` 设为 `diffusion_layerwise_real`，避免新 run 被命名成历史 baseline。
原配置文件名含 `baseline` 不代表它固定了旧架构；本分支的共享 `model/diffusion.yaml` 已是逐层条件模型。

正式运行需明确设备数、每卡 batch、seed、数据/split、AE/cache 来源与训练预算。以下命令是待填环境变量的模板，本次未执行：

```bash
python -m src.brepnet.train --config-name train_diffusion_layerwise_real \
  trainer.devices="${DIFFUSION_DEVICES:?}" \
  trainer.batch_size="${DIFFUSION_BATCH_SIZE:?}" \
  trainer.seed="${DIFFUSION_SEED:?}" \
  dataset.raw_data_root="${CAD_ROOT:?}" \
  dataset.cached_latent_root="${LATENT_CACHE_ROOT:?}" \
  dataset.condition_data_root="${CONDITION_ROOT:?}" \
  dataset.train_dataset="${TRAIN_SPLIT:?}" \
  dataset.val_dataset="${VAL_SPLIT:?}" \
  dataset.test_dataset="${TEST_SPLIT:?}" \
  model.autoencoder.checkpoint="${AE_CHECKPOINT:?}"
```

## 配置、权重与历史 baseline

- 旧 `condition_fuser` 配置显式拒绝。不要把旧 run 的 resolved config 直接套到当前分支。
- 旧 diffusion checkpoint 含前端 fuser / alignment / camera 权重，且缺少逐层 cross-attention 权重，不能直接 resume。需要复现旧模型时使用 baseline 分支。
- `Diffusion.load_state_dict(..., strict=False)` 不再作为迁移路径；完整当前模型必须严格匹配。若以后复用部分旧权重，需另行定义显式组件迁移协议，不能宣称训练续跑。
- `trainer.init_from_checkpoint` 严格加载整个当前模型，包含 AE；它是权重初始化，不恢复优化器/步数，也不使用下述独立推理的 AE 隔离规则。
- 独立推理通过 `model.load_diffusion_state_dict(weights)` 加载已去除 Lightning `model.` 前缀的权重。该方法忽略 checkpoint 中的 AE 权重、保留单独指定的 AE，并严格检查条件编码器、denoiser、padder、topology 等其他 key。
- `trainer.resume_from_checkpoint` 由 Lightning 恢复完整训练状态。当前实现仍在恢复之前先构造模型并加载 `model.autoencoder.checkpoint`；即使 run checkpoint 包含 AE，该配置路径也需可读。这是既有初始化顺序的限制。
- VAE 表示与 latent cache 格式没有改变；复用仍要求实际 encoder、posterior、数据处理与 cache 同源。`autoencoder.checkpoint=null` 仍不自动阻止冻结随机 AE，正式运行必须指定来源。
- 原 `VALIDATION.json` 保留为历史记录。本轮验证报告另存 `LAYERWISE_VALIDATION.json`；接口/梯度检查不能解释为重建能力提升。

`inference.py` 已使用当前手写模型配置，修正图片多余 view 轴，统一为 `[B,3,224,224]`。
它不会自动恢复任意训练 run 的配置；使用前仍需核对 checkpoint 对应的参数、VAE 与条件处理。

## 尚未解决的限制

- zero padding validity head 在训练读取 prediction、推理读取最终 latent；epsilon 配置下二者分布不同。
- learned topology 固定 slots adjacency 与随机面 slots 的对应问题未解决，也没有自动 topology BCE；GT topology 仍是 oracle，默认关闭。
- 可选 topology mask/bias 只作用于原 encoder self-attention，不作用于图像 cross-attention。针对本地 PyTorch 2.4.1 在偶数 heads、eval/no_grad 下 fused EncoderLayer 误处理浮点 bias（出现 NaN 或改变语义）的问题，浮点 mask 路径显式执行同一个 EncoderLayer 的原子层与两次残差，保留原参数和 additive-logit 语义；无 mask / bool hard mask 仍调用原 forward。默认 topology 关闭，grad/no_grad 一致性回归见本轮验证报告。
- 图像 `cached_features` 尚未接通，图像分支始终读取 `imgs`。
- 本轮没有改变 VAE 解码或 OCC 后处理，不能仅凭 attention 结构或 diffusion loss 推断 STEP/solid 有效率。
