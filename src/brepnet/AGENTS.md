# BRepNet Source Guidelines

研究目标和协作原则见根目录 [AGENTS.md](../../AGENTS.md)。本文件描述当前实现与接口约束；允许有理由的架构重设计，不把已有实验方向设为固定路线。

## 入口与配置

- `train.py`：`python -m src.brepnet.train`，`UnifiedTrainer` 根据 `model.stage` 分派 VAE / diffusion，数据类通过 `dataset.name` 查找。
- `models/__init__.py`：`build_model`；主模型在 `models/vae.py` 与 `models/diffusion.py`。
- `inference.py`：独立推理工具，仍手工构造配置并带旧兼容逻辑；使用前核对参数与 checkpoint，不能认为它完整复现训练配置。
- `eval/run.py`：`python -m src.brepnet.eval.run`；`post/` 完成 OCC B-Rep 构造。
- 行为配置归属 `configs/model/`、`dataset/`、`condition/`、`trainer/`、`datagen/`；不要复活已移除的 `model.py`、`diffusion_model.py`、`train_diffusion.py` 入口。

## VAE：表示与解码

`models/vae.py` 的 `AutoEncoder` / `AutoEncoder_light` 编码面 UV 采样点、半边采样点与 edge-face connectivity；图网络、注意力与全局聚合把几何／拓扑融合到每面 latent。常用 latent 维度是 `8 × 2 × 2 = 32`。这是包含全局信息的 holistic 表示，不能直接当作独立局部 surface code。

`sample()` 内部参数化 mean/logvar，但返回的缓存统计量是拼接的 **mean/std**。解码器恢复面几何，并通过有序 face pair 的 neural intersection 预测连接和有向半边几何。每个 pair 只产生一个候选半边；多重公共边、复杂裁剪和全局闭合不能由 adjacency 分类保证。最终合法 solid 仍依赖 `post/` 的拟合、优化、裁剪与缝合。

改变 VAE 或归一化时说明旧 latent cache、AE checkpoint 与 diffusion checkpoint 是否兼容；不能复用名称相同但定义不同的缓存。正式 diffusion run 确认冻结 AE 加载了正确 checkpoint，并与 cache 同源。

## Diffusion：实际数据流

1. `dataset.py:Diffusion_dataset` 读取 cache 并完成 padding；`diffusion_latents.py:LatentProvider` 取 mean，或用 std 采样。非 cache 分支调用冻结 VAE 再由 padder 打包。
2. `diffusion.py` 对 `[B, F, 32]` 加噪；默认 DDPM 1000 steps、epsilon prediction、L2 loss。
3. `diffusion_condition.py` 选择条件编码器。当前支持 single image、sketch、point cloud 或无条件，显式拒绝 `multi_img` / `text`；有 YAML 或论文支持不代表当前路径可用。
4. `condition_encoders.py` 默认冻结 DINOv2 ViT-L，224 输入得到 257 个 token，再训练 projection。Depth Anything V2 选项只使用其 ViT 特征，不等于显式深度预测／监督。
5. `diffusion_denoiser.py` 的前端 fuser 用 noisy face tokens 查询 image tokens，默认 4 层 cross-attention。训练 alignment 用 clean latent 的汇聚表示；它不是 face-to-pixel 对应监督。
6. fuser 之后加入 timestep embedding，再经过默认 24 层 self-attention backbone（hidden 768），预测噪声。每个 diffusion timestep 都重新调用 fuser，但 backbone 各层没有独立 image cross-attention。
7. 采样后筛选／去重 face latent，交给冻结 VAE 解码。当前 diffusion loss 不直接优化 OCC solid validity，也没有直接渲染重投影损失。

## Face padding 与 topology 的共同契约

- 默认 random padding 复制真实面并随机排列到 30 slots；mask 全为 true。`RandomFacePadder.postprocess()` 用 latent 距离阈值去重，面数由此间接产生。
- zero padding 有 validity head；当前训练输入 denoiser prediction，推理输入最终 latent。epsilon prediction 下两种输入分布不同，切换 padding 不能假设等价。
- Oracle adjacency 在 dataset 中通过同一 padding index 重排。新增 face-level 标签也必须使用相同映射。
- 当前 `LearnedTopologyPredictor` 从图像 CLS 预测固定 slots 的 adjacency；默认 learned 配置仍用随机 latent 排列，没有显式 slot matching。先解决对应关系，再讨论预测精度或 attention 强度。
- `experiments/2026-06-19/cvae_diffusion_two_step.py` 读取 GT adjacency 计算 canonical ordering，并使用 GT 面数恢复预测；只能作为带 GT 信息的诊断路径，不能当作部署时纯照片推理。
- `source: gt_adjacency` 是 oracle；`source: learned` 读 raw `conditions["imgs"]`。存在预测缓存或 `freeze=false` 不表示已经实现 topology BCE 联合目标；当前 `Diffusion.forward()` 没有自动加入这些辅助损失。
- additive bias `+2` 将相同 base logit 下的未归一化 attention mass 乘 `exp(2)`；`log_multiplier` 使用 `log(1 + (max_multiplier-1)*P)`。这不保证 feature 输出被相同比率约束。旧数值解释错误不等于已经证明 two-step 退化的原因。
- `dataset.topology_visibility_weighted` 当前用 `min(v_i,v_j)` 修改 oracle adjacency；它不是完整的 visible/hidden 分解模型，也不同于实验脚本中的 visibility-weighted BCE。

## 已知配置漂移（2026-10-09 源码审计）

| 区域 | 使用前要处理的问题 |
|---|---|
| `configs/train_diffusion_real.yaml` | 继承 `is_overfit=true`、`validation_real_photo_ratio=0.0`；额外写的 `scale_factor` 不被 diffusion loader 读取 |
| `configs/train_diffusion_baseline_real.yaml` | 显式修正 overfit 与 real-photo validation，但有历史机器路径和 8-GPU 设置，需要 overrides |
| `AutoEncoder_dataset3` | 读取 `scale_factor`、`data_root`、`condition_names`、`condition_root`，与 `configs/dataset/vae.yaml` 新字段不一致 |
| cached condition | 当前图像分支读取 `imgs`，不能只打开 cached flag 就假定特征缓存可替代图像 |

Diffusion loader 字段为 `raw_data_root`、`cached_latent_root`、`condition_data_root`、`train_dataset`、`val_dataset`、`test_dataset`、`epoch_scale_factor`。记录实际解析配置和实际消费字段；配置解析成功不代表数据语义正确。

## 修改与验证

保持行为归属：cache 在 `diffusion_latents.py`，padding 在 `diffusion_padding.py`，编码选择在 `diffusion_condition.py`，条件实现与 denoiser 在各自模块，装配／loss／采样在 `diffusion.py`。新增完整架构可建新模块与显式 config，避免继续膨胀旧类。

重设计交代训练／推理各自可得的信息、face correspondence、mask、相机约定、输出表示、checkpoint 迁移及最终评估。联合 geometry/topology 训练可以作为新方向，需明确损失语义和权重，不沿用旧指南中限制联合训练的默认立场。

相关检查包括导入、配置消费、真实小批量前向／反向、有限梯度、排列一致性与训练／推理接口。只运行与改动有关的测试；研究有效性由固定协议的下游结果判断。文档整理不需要启动训练或下载模型。
