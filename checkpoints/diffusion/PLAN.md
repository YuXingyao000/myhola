# 单视图 Diffusion：逐层图像条件

日期：2026-10-16。状态：逐层条件架构实现；未启动正式训练，未评估重建效果。
本轮接口验证单独记录于 [LAYERWISE_VALIDATION.json](LAYERWISE_VALIDATION.json)。

## 为什么改

本主题负责图像条件 diffusion；数据集和 VAE 优化由其他主题推进。
旧路线先用 4 层完整 decoder fuser 融合图像，再加时间嵌入，随后 24 层 denoiser 主干无法直接重读 image tokens。
当前研究假设是：每一层主干处理后再次访问图像 memory，能给不同深度的面 latent 持续提供观测约束，改善条件利用。
这是待下游结果检验的假设；attention 存在不保证网络使用图像，更不保证几何一致或合法 B-Rep。

本轮改的是整体条件路由：删除前端 fuser，在 24 个原 EncoderLayer 后分别增加 image cross-attention；同时移除 alignment。
这不是只挪一个模块位置的纯消融，cross-attention 数量、宽度、参数和计算量也变化。
不把这一结构直接作为新理论贡献，也不预设它优于通用 CAD agent。

## 已确定的实现

1. 保留 DDPM、默认 epsilon L2、latent dim32、30 slots、random duplicate padding，以及冻结 VAE 的数据接口。
2. 保留冻结 DINOv2 和可训练图像 projection，输出共享 memory `[B,257,1024]`；维度直接读取 `condition.output_dim`。
3. 输入 latent `[B,30,32]` 经原 MLP 投影为 `[B,30,768]`，在原主干入口加时间 MLP 的 `[B,1,768]`。
4. 保留 24 个完整 TransformerEncoderLayer：pre-LN self-attention（12 heads）+ residual、pre-LN FFN（768→2048→768，ReLU/dropout0.1）+ residual。
5. 每个 EncoderLayer 后新增独立 pre-LN image cross-attention：Q768→768、K/V1024→768、12 heads，Wout768→768 + dropout + residual。24 组后执行最终 LN 和原输出 MLP。
6. 移除旧前端 fuser、alignment heads/loss 和未用 camera 参数。首个 image query 因主干入口加 t 而含时间信息，不新增逐层 t 或 adaLN。
7. 更新单图推理配置和 `[B,3,224,224]` 输入合同；checkpoint 严格加载，独立 VAE 不被 diffusion 权重覆盖。
8. 保留其他条件与可选 topology 接口，但本轮研究和实质验证只对单图路线负责；不借接口存在宣称完成其他模态验证。
9. 新训练入口 `train_diffusion_layerwise_real.yaml` 继承已有 real 配置与预算，仅设置独立的 `diffusion_layerwise_real` 实验名；未启动训练。

层次、张量形状、参数加载协议与模块入口见 [README.md](README.md)。
神经网络讨论继续使用标准层/张量/残差图，不使用 Mermaid；理论解释和后续结构提案优先配图。

## 历史 baseline 与迁移边界

2026-10-16 的行为保持模块拆分已保存到 `codex/diffusion-modular-baseline`，commit `b67c5d8e`。
当时从 `models/` 迁到 `diffusion/`，保留原模块属性树与数值语义；对应报告 [VALIDATION.json](VALIDATION.json) 保持原样。
当前分支为 `codex/diffusion-layerwise-conditioning`，不再承诺与旧架构 state_dict 等价。

旧 `condition_fuser` 配置显式报错，旧 diffusion checkpoint 不能直接 resume，也不能用 `strict=False` 忽略差异。
本轮没有自动旧权重迁移协议；复现旧模型使用 baseline 分支。VAE/cache 可以复用，但仍须确认实际来源同一。
正式研究训练前明确指定冻结 AE checkpoint、数据/split、seed、预算和输出 run；本次不启动训练。

## 后续判断方案：尚未运行

第一轮比较控制数据/split、AE/cache、DDPM 设置和训练预算：

- A：baseline 分支的前端 4 层 fuser + 24 层主干，将 alignment 关闭。
- B：当前逐层 cross-attention，alignment 已从实现移除。

先比较两种完整路线，而非把新路线称为参数量或算力严格匹配的单因素消融。
记录实际参数量、计算/采样耗时与预算差异；比较相同预定样本和采样/重试预算。
评估需要同时覆盖生成覆盖率、有效 STEP/solid 比例、几何与可见区域一致性；失败、缺失与超时保留在分母里。
训练 loss 和单元验证只用于实现判断，不决定研究路线优劣。

若后续比较通用 CAD agent，必须固定可用输入、工具和修复预算，单独报告成本/耗时。
不把合成真实感条件图当作实拍评估，不用单个 GT 排除单视图下合理的隐藏结构补全。

## 验证与证据范围

本轮验证检查新路由的层数/形状、图像与 t 对输出的影响、所有可训练分支的梯度、面排列语义、采样以及严格加载拒绝旧架构。
实际检查项目、真实资产与替代 backbone 的边界以 `LAYERWISE_VALIDATION.json` 为准。
没有训练结果，没有真实照片重建结论，也没有合法 solid 有效率改善结论。

参考与证据：

- 当前源码：`src/brepnet/diffusion/`；配置：`configs/model/diffusion.yaml`、`configs/condition/single_img.yaml`。
- 历史 baseline：`codex/diffusion-modular-baseline` / `b67c5d8e`，旧模块迁移报告 `VALIDATION.json`。
- [HoLa: B-Rep Generation using a Holistic Latent Representation](https://arxiv.org/abs/2504.14257)：表示的来源；本轮逐层条件路由属于仓库改动，不归为原论文架构或已证明的新贡献。
