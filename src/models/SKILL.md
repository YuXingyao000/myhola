---
name: img2brep-diffusion-model-maintenance
description: Maintain or extend the Img2Brep diffusion implementation in src/models/diffusion, including layerwise image conditioning, latent/padding contracts, and checkpoint loading. Use for repository diffusion changes, not as a fixed future research plan.
---

# Diffusion 模块维护

当前实现与形状见 [README](/mnt/d/hola_research/diffusion/README.md)，研究范围和历史 baseline 见 [PLAN](/mnt/d/hola_research/diffusion/PLAN.md)。
以当前源码、配置和两级 `AGENTS.md` 为准；本文件不要求延续旧架构。

## 实现入口

- `diffusion/model.py`：装配、训练/推理与 checkpoint 加载。
- `diffusion/conditioning.py`、`image_encoder.py`：条件路由、冻结 ViT 与可训练图像投影。
- `diffusion/denoiser.py`、`fusion.py`：原完整 EncoderLayer 后接独立 image cross-attention；默认共 24 组。
- `diffusion/latents.py`、`padding.py`：latent 来源、面 slots 打包、有效性筛选和去重。
- `diffusion/topology.py`：可选 oracle / learned self-attention mask 或 bias。
- `diffusion/losses.py`、`sampling.py`：训练目标与 DDPM 反向迭代。
- `vae/`、`blocks.py`：外部 VAE 与共享算子，不为 diffusion 整理擅自修改其表示。

## 当前合同

单图条件为 `conditions["imgs"]: [B,3,224,224]`，图像 memory 默认 `[B,257,1024]`。
去噪输入 `[B,30,32]` 先投影到 768 并加时间嵌入，再进入 24 组 encoder + image cross-attention，最后 LN/输出投影。
条件维度只读取 `condition.output_dim`；`denoiser.conditioning=layerwise_cross_attention` 显式指定当前路线。
旧前端 fuser、alignment 和未用的 camera 参数已移除，不再要求提供 clean latent alignment 输入或相应 loss。

Cache 的后半维是 **std**，不是 log-variance。random padding 复制并重排真实面；任何 face-level 条件或标签必须复用同一 padding index。
若使用 oracle adjacency，必须确保行列顺序等于 latent slots 顺序；GT topology 是 oracle，默认关闭。非 cache 路径若另行随机 padding，不能假定外部 adjacency 自动对齐。
推理不输入 GT 面数；训练打包使用 GT 面数，两者需分开描述。

## 权重与变更

旧结构在 `codex/diffusion-modular-baseline`（`b67c5d8e`）。当前显式拒绝 `condition_fuser` 配置，旧 diffusion checkpoint 不能直接 resume。
完整当前模型权重使用严格加载；独立推理用 `load_diffusion_state_dict` 保留单独指定的 AE，并严格检查所有非 AE 参数。
不要以 `strict=False` 掩盖旧层/新层不匹配。表示或结构改变时明确说明旧权重、cache 与 run 的迁移边界。

配置字段属于对应 Hydra 配置组，不通过路径、文件名或静默 fallback 推断架构。
结构调整应解释要解决的失败机制，画清层、张量、残差和训练/推理可用信息；不把这份实现地图当作禁止重设计的要求。

按改动检查实际形状、梯度、采样和加载合同。单图以外接口保留不等于已做能力验证。
[VALIDATION.json](/mnt/d/hola_research/diffusion/VALIDATION.json) 是旧模块拆分记录；当前接口验证为 [LAYERWISE_VALIDATION.json](/mnt/d/hola_research/diffusion/LAYERWISE_VALIDATION.json)，二者都不构成重建质量结论。
