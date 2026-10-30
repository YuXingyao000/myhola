# 新版无条件 Diffusion

训练当前768维、24层主干，固定原版1119 VAE，建立与[旧模型734/1000合法solid](../uncond_baseline_20261018/README.md)的比较。
本次只准备配置和验证实现，正式训练由用户启动。研究假设、历史差别及数据身份见[PLAN](PLAN.md)。

## 运行

在当前8×RTX4090源机的tmux中执行：

```bash
cd /mnt/d/python
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /mnt/d/anaconda3/envs/torch/bin/python \
  -m src.brepnet.train --config-name train_diffusion_uncond \
  trainer.devices=8 trainer.batch_size=192 trainer.seed=42 \
  trainer.max_steps=1000000
```

入口[train_diffusion_uncond.yaml](../../../configs/train_diffusion_uncond.yaml)已明确配置v7数据、train/validation/test清单与原版AE路径。
不加载任何旧diffusion或灰模权重；去噪主干从随机参数开始。条件为none，不构造DINO或图像cross-attention，也不读取图片和latent cache。
训练时：原始几何 → 冻结AE mean → 随机复制/排列到30个面token → 加噪 → 预测噪声。

| 设置 | 本轮 |
|---|---|
| 主干 | hidden768，24层，12 heads，FFN2048 |
| VAE | `1119_deepcad_aug1_11k.ckpt`，冻结且始终eval |
| 数据 | v7；train46627、validation3505；训练cube24、验证原方向 |
| 每卡／全局batch | 192／1536（8卡，无梯度累积） |
| 最大步数 | 1,000,000；名义1.536×10^9对象曝光 |
| 优化 | Adam1e-4，bf16-mixed，norm clip0.5，无学习率调度 |
| 噪声／目标 | linear DDPM1000步，epsilon/L2 |

改变设备数或batch会改变全局batch与训练量，不应只照搬100万步就称预算相同。
旧训练的v6数据、较大的历史split、非均匀18方向增强及冻结AE仍开启dropout，均与当前不同；此次对齐主要训练设置，不声称严格单变量比较。

## 输出与续训

每次正式运行创建`checkpoints/diffusion_uncond/<UTC时间戳>/`：`weights/`存best及last，`metadata/`存配置/Git/命令，日志与W&B独立保存。
训练集扩展100倍，一个epoch约3035个optimizer steps；完整验证每epoch一次，使用3505件清单且不重复扩展，不做1000步采样可视化。
现有trainer保留最低val loss的3份权重及last；生成能力仍要用固定1000次采样评估，不能只看val loss。

继续同一个run时使用实际的last路径，`max_steps`仍表示累计步数上限：

```bash
python -m src.brepnet.train --config-name train_diffusion_uncond \
  trainer.devices=8 trainer.batch_size=192 trainer.seed=42 \
  trainer.resume_from_checkpoint=/mnt/d/python/checkpoints/diffusion_uncond/<UTC时间戳>/weights/last.ckpt
```

## 改动与验证

只新增训练配置和`Diffusion_geometry_dataset`，后者复用原几何读取，只把非训练模式的epoch倍率设为1；现有VAE dataset的行为保持原样。
网络、loss和采样继续使用现有模块，没有新增模型加载器或检查框架。

[真实batch检查](VALIDATION.json)：每卡192，latent为`[192,30,32]`，所有可训练参数梯度有限；AE无梯度且保持eval，704个tensor与原checkpoint一致。
RTX4090单步检查峰值分配显存5.28GiB；完整无条件1000步采样与解码接口通过。单步耗时包含在线AE，未计DataLoader等待，不能据此推算正式训练总耗时。
双卡DDP短运行的结果另见`DDP_VALIDATION.json`；这些检查不代表正式训练已完成或生成能力已提升。
