# 新版无条件 Diffusion 训练

## 研究问题

不输入图片时，当前768维、24层主干能否保留旧模型的CAD生成能力？先固定原版VAE，比较完整生成链路；此次不增加条件模块。
旧模型基线见[1000件结果](../uncond_baseline_20261018/README.md)：原生合法solid为734/1000。

## 实施

- 入口：`configs/train_diffusion_uncond.yaml`，复用统一trainer、现有Diffusion和原版`AutoEncoder_light`。
- 原始v7几何在线经过冻结VAE，取mean，random duplicate padding到`[B,30,32]`；不读取图片或latent cache。
- 当前去噪器为768维、24层、12 heads、FFN2048；无图像encoder、无cross-attention参数、无topology条件。
- epsilon/L2、linear DDPM1000步、beta `[1e-4,0.02]`、fixed_small、clip_sample=false。
- Adam1e-4，无学习率调度，bf16-mixed、norm clip0.5。8卡×每卡192=全局1536，1,000,000步，名义1.536×10^9对象曝光；从随机主干开始，不加载旧diffusion权重。
- 训练集扩展100倍以控制验证/保存频率；验证不扩展且关闭旋转，不执行完整DDPM可视化。

## 数据与已知比较边界

当前train/validation为46,627/3,505个CAD，均存在于`/mnt/d/data/deepcad_v7`。split仍按CAD身份划分，训练时均匀采样cube24，验证使用原方向。
AE来自`/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt`，与旧diffusion内AE的704个tensor相同。

旧checkpoint记录4卡×384、1,000,000步、在线AE与mean；旧原始数据路径为已不存在的v6目录。
历史代码的增强实际为每轴0/90/180度的27组合，仅覆盖18方向且不均匀；旧AE虽然冻结梯度，整体train()仍会开启其dropout。
本轮保留当前cube24与冻结AE始终eval的正确行为，不恢复这些历史行为。当前清洗split也少于历史版本。
因此本轮对齐主要训练设置及对象曝光预算，不声称严格只改变主干宽度。

训练split SHA256：`bfd0f90428a40600c6f9f72658544c018e75f5576ad6f0af00dbe29224ac9417`。
验证split SHA256：`b6403d9b235db7ebeaac76ba3a9d076440f149863ac692ae938129e20404ee32`。

## 验证与判断

先验证真实每卡batch的形状、有限loss/梯度、冻结AE、无条件采样接口及短DDP运行；这些只检查实现。
正式训练由用户在tmux启动，agent不启动长训练。统一run目录保存配置、Git状态、seed和命令，续训使用resume。
训练后的生成比较沿用旧基线的1000个seed、1000步DDPM、相同后处理预算与完整分母，检查合法solid、生成形状和复杂程度，不单凭val loss选结论。
