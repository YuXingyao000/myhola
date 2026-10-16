# 单视图 Diffusion：架构核对与模块整理

日期：2026-10-16。状态：代码整理已实施；验证详情见 `VALIDATION.json`。

## 为什么

本主题负责图像条件 diffusion，数据集与 VAE 的优化分别在其他主题推进。
先建立可核对的网络结构、训练/推理信息边界及源码入口，后续才讨论表示和条件建模改动。
原 `diffusion_denoiser.py` 同时包含融合器、alignment、拓扑预测和去噪主干，
`condition_encoders.py` 同时包含多个模态，阅读和局部修改需要跨越无关逻辑。

本次是行为保持的拆分，不改变网络能力，不启动正式训练，也不将兼容性测试解释为精度提升。

## 具体实施

1. 以 `configs/model/diffusion.yaml` + `configs/condition/single_img.yaml` 核对默认层数、维度和数据流。
2. 将 diffusion 实现集中到 `src/brepnet/diffusion/`，按条件编码、融合、去噪、拓扑、latent、padding、损失、采样拆分。
3. 保留 `nn.Module` 的属性树、注册顺序和旧路径导出；训练配置与 `build_model` 接口保持兼容。
4. 为单视图主线绘制神经网络架构图和源码模块关系图。层内结构标出 residual、Q/K/V、维度和重复次数；不使用 Mermaid。
5. 验证旧导入、state_dict、前向/梯度/采样及真实 cache 接口，分别记录替代组件与真实资产的覆盖范围。

## 范围与后续讨论

本轮不修改数据读取、VAE、训练预算或 loss 定义。后续结构提案需要先说明解决的失败机制，
画清每层输入输出与张量形状，明确观测信息、隐藏几何先验及与通用 CAD agent 的比较协议。
新增架构或理论解释（例如 CLIP、ControlNet）尽量配图，并核对原始论文，不把已有实现名称当作理论保证。

仍需研究判断的问题包括：前端一次融合是否足够保留图像约束、全局 alignment 实际约束什么、
随机重复 slots 如何影响面数与对应关系、最终误差来自 diffusion 还是 VAE/后处理。
这些是待讨论的问题，不是已经确定的改动计划。

## 参考与证据

- 配置：`configs/model/diffusion.yaml`、`configs/condition/single_img.yaml`、`configs/train_diffusion_baseline_real.yaml`。
- 代码与架构说明：本目录 `README.md`。
- 兼容性结果与限制：本目录 `VALIDATION.json`。
- 原 HoLa 论文：[HoLa: B-Rep Generation using a Holistic Latent Representation](https://arxiv.org/abs/2504.14257)。本次未重新审阅论文；图描述仓库实现，不能视作论文原始架构。
