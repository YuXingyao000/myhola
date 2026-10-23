# DPO 训练与选对事实核对

这份记录为后续完整 validation 结果提供训练背景，**不是失败根因或模型退化的因果结论**。
逐项来源、SHA256 和完整计数见 [TRAINING_CONTEXT.json](TRAINING_CONTEXT.json)。
本次只读核对 manifest、实际运行配置及训练时 Git 源码，没有加载大 checkpoint。

训练 run 为 `checkpoints/diffusion_preference_svr/20261023T030527Z`，评估使用其
`weights/steps/step=00005000.ckpt`。训练提交为 `7998cc8a`；核对的模型、dataset、trainer
源码与该提交逐字节一致。选对源码也与生成协议记录的哈希及归档副本一致。

## 数据与标签

直接扫描 `checkpoints/diffusion/preference_data/20261020T134930Z/pairs.jsonl`：

| 训练 pair | 数量 | pair 占比 |
|---|---:|---:|
| 合法优于确认非法 | 17,258 | 41.054% |
| 两者合法，按几何区分 | 24,779 | 58.946% |
| 总计 | 42,037 | 100% |

46,627 个训练 CAD，每图固定 8 个候选、最多一对，配对覆盖率为 90.156%。
无 pair 的 4,590 个 CAD 仍参加 GT 去噪监督。validation 的 95 对（48 validity、47 geometry）
**不参加训练**；训练中的 validation 只计算 100 个 GT 的去噪 loss。

实际冻结规则是 `svr_preference_v3`，**不是“三项投票，赢两项”**：

- 合法/非法：比较 raw 面 CD²、边 CD²、面 F1。前两项最多允许退步 2%，F1 最多降低 0.01；
  三项均守住上限即可，不要求显著改善。
- 两者合法：三轮 post 测量都必须满足上述退步上限，而且每轮至少一项达到显著改善：
  CD²降低 5% 或 F1 增加 0.02。没有额外的 raw 门槛。
- 优先选择合格 validity pair，否则选择 geometry pair；在合格对内按固定 seed 随机选一对。
  不是从全部候选中选择全局最优，也不追加采样。unknown 不自动作为负样本。

geometry pair 中有 2,833 对（11.433%）的 winner 按 raw 三项退步上限重新检查会不通过。
其中 raw 面 CD²数值更差 2,389 对，边 CD²更差 1,918 对，F1 更低 145 对；这些集合可重叠。
这是 post 标签不叠加 raw 条件的实际数据事实，不能直接推出它导致了某项下游变化。

## 实际训练目标与条件

```text
loss = GT_DDPM + lambda * DPO
lambda = 1.0
beta = 20.0
logit = -beta/2 * [(policy_winner_MSE - policy_loser_MSE)
                  - (reference_winner_MSE - reference_loser_MSE)]
DPO = -mean(logsigmoid(logit))
```

没有其他 auxiliary loss。没有直接的 OCC、几何、轮廓、可见面或深度损失。
正负候选共用随机 timestep 和噪声，每个完整 `[30,32]` latent 计算 mean MSE；
两类 pair 不另加类别权重，统一有放回均匀抽样。

模型确实读取各自对应的 `imgs.npz[svr_imgs][0]`：GT 与 pair 是独立对象流。
原 DINO 冻结并保持 eval；policy 的图像投影和原 1024 维、24 层 concat 去噪器可训练；
reference 的全部参数冻结。偏好分支关闭 dropout 但保留 policy 梯度。
原 VAE 用于生成阶段解码及重建独立 GT cache，训练 step 不运行 VAE/OCC。
候选 latent 不在后处理后重新编码；GT 使用同一原 VAE 的 mean，并做 random duplicate padding。

实际配置为 8 卡 × batch 8、FP32、Adam `1e-5`、gradient clip 1、seed 42、5,000 steps。
训练退出码为 0。标量 DPO 与 DDPM loss 的大小不能直接说明各自梯度影响。

几何标签比较完整 GT，包含隐藏部分，没有使用图片轮廓或可见区域分数。
因此“模型接收图片”和“损失直接约束可见证据”是两件事；后续可见区域分析用于检查这层差异，
不能把本次训练称为无条件训练，也不能将隐藏区域的不同补全自动判为错误。
