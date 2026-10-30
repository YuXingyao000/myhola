# 新观测标签 DPO：训练完成核验

北京时间 2026-10-24。用户报告训练可能完成后，只读核验日志、配置、数据摘要与 checkpoint；此次没有启动训练、重建评估或使用 GPU。完整证据见 [TRAINING_COMPLETION.json](TRAINING_COMPLETION.json)。

## 完成状态

Run：`checkpoints/diffusion_preference_svr_observation/20261024T040949.398883Z`。

- W&B 输出明确记录 `max_steps=5000` 达到后停止；内部退出记录 `complete=true`、`exit_code=0`。
- 运行于北京时间约 12:09–12:47，W&B runtime 2,260 秒（37 分 40 秒）。当前没有本轮训练进程。
- 实际读取 `weights/steps/step=00005000.ckpt` 和 `weights/steps/last.ckpt`，二者均为 global_step=5000、epoch=6，各 4,525,605,300 字节。
- checkpoint 均含 314 个 policy、314 个 reference、344 个 DINO state keys，以及 1 个 optimizer、314 个 optimizer state slots。这里核查可读性、结构和元数据，未声称逐权重数值或冻结参数一致性检验；推理入口仍进行严格加载检查。
- 1,000/2,000/3,000/4,000/5,000 步 checkpoint 均存在。

## 实际协议

与约定一致：从原 `Diffusion_svr_sq30_1500k.ckpt` 初始化，无 resume；8 卡 × batch8、FP32、Adam lr=1e-5、beta20、lambda1、clip1、seed42、5,000 optimizer steps。使用新 `observation_v1/pairs.jsonl` 和原生成 run 的 GT cache。

实际配置与旧 DPO 只有 pair manifest、run/W&B 名称和等价 cache 路径写法等预期差异。保存的 dataset provenance 与准备阶段记录完全相等，七类数据来源文件摘要核对通过。模型仍为原 1024 维、24 层 concat SVR，目标为 GT DDPM + DPO，尚未添加几何辅助项。

## 训练诊断及解释边界

本地 W&B 共 213 条 history，所有 train/validation/learning-rate 标量有限。

| 诊断量 | 旧标签 DPO | 新观测标签 DPO |
|---|---:|---:|
| 最后训练 epoch 段 DPO loss | 0.59692 | 0.62913 |
| 最后训练 epoch 段 GT DDPM loss | 0.04713 | 0.04341 |
| 最后训练 epoch 段 preference accuracy | 0.65398 | 0.63002 |
| 最近一次 validation GT loss | 0.07484 | 0.06233 |

最后训练量为 epoch6 末段平均值，最近 validation 对应 epoch5、日志零基 global_step=4367，并非 step5000 的重新评估。两个 pair 池及其标签分布不同，DPO loss/accuracy 不能直接跨池解释为模型优劣。较低 GT 去噪 loss 也不能替代合法率或几何重建指标。当前没有新模型完整重建效果结论。

## 后续完整重建评估

仍按原计划固定 3,505 件 validation、K=1、seed42、DDPM1000/FP32/TF32 off/batch1，原 VAE 和 recover_v1 后处理，比较旧 DPO 与新 DPO 的 step5000；同时保留原 SVR 作为参考。报告固定分母的生成覆盖、有效 solid、完整 raw/post 几何、输入轮廓和真正可见表面距离，以及共同可测 CAD 的配对差值与各自缺测率。

已只读确认旧完整评估 `preference_eval/20261023T090834Z` 和旧观测 audit `failure_analysis/runs/20261023T105637Z_full` 均完成，旧采样/后处理源码与固定 split 仍匹配。现已增加 `preference_compare --reuse-methods baseline` 的按方法复用能力，保留 CPU 输入/输出摘要校验；三阶段手动入口和跨 run 报告见 [README](README.md)。代码准备与 CPU 验证已完成，新模型正式评估尚未启动；不能提前给出提升结论或进入几何训练。
