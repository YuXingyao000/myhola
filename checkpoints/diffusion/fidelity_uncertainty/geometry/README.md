# 原 DPO + 冻结 decoder 几何监督

2026-10-25：两组正式训练均已完成额外5000步，checkpoint及Adam步数已核验。执行约定见 [SPEC.md](SPEC.md)，实现时的CPU数值证据见 [CPU_VALIDATION.json](CPU_VALIDATION.json)。本次完整validation评估的冻结协议见 [EVALUATION_SPEC.md](EVALUATION_SPEC.md)。

## 架构与梯度

没有删改原 1024 维、24 层 concat diffusion policy，没有新增可训练预测头。DINO 和 reference 仍冻结，原 surface/curve 预测头的结构与权重保持不变。训练时接回原 SVR 内置的完整冻结 VAE，使用其 decoder 对预测 latent 解码；没有替换成新版 UV/shared-curve VAE。

`image → frozen DINO → original policy(zt,t,image feature) → epsilon_hat → z0_hat → frozen original decoder → predicted XYZ → geometry loss`

几何梯度经 decoder 回到 policy；冻结只意味着 decoder 参数不求梯度，不对这条计算路径使用 no_grad/detach。可训练参数仍为原 policy 的 **206,716,192** 个，新增可训练参数 **0**。为保证独立严格恢复，geometry checkpoint 额外保存原冻结 VAE 的 181,583,229 个参数；原有 policy/reference/DINO state key 保留。

推理仍是原生 DDPM → 原 decoder 预测连接／点 → 后处理。严格推理 loader 只允许新增的冻结 VAE 参数逐 tensor 等于原 checkpoint；不忽略未知 state。原 classifier 未删除，也未添加新的 classifier／拓扑 loss。

## 实际 loss

训练中 GT latent 与偏好 pair 仍来自原先的两个独立数据流。

`L_total = L_ddpm + 1.0 * L_dpo + 0.0226 * L_geo`

- `L_ddpm`：GT 分支预测 epsilon 与真实 noise 的 MSE，先对每个 `[30,32]` latent 的所有元素取均值，再对 batch 平均。
- `L_dpo = mean(-logsigmoid(-10 * ((e_policy_win-e_policy_lose) - (e_ref_win-e_ref_lose))))`。`e` 为逐候选 epsilon MSE，beta=20，所以系数是 `-beta/2=-10`。同一 pair 的 win/lose 共用 t/noise，policy/reference 共用加噪输入，reference 无梯度；偏好比较关闭 policy dropout，GT 分支保留原训练行为。
- `z0_hat=(zt-sqrt(1-alpha_bar_t)*epsilon_hat)/sqrt(alpha_bar_t)`。使用 GT 去噪分支已有的 t／epsilon 预测，不额外生成完整 DDPM 轨迹，不 clip。
- `CD2(P,Q)=mean_p min_q ||p-q||² + mean_q min_p ||q-p||²`，**没有额外 1/2**。
- 每个 CAD 的 `L_surface` 是各真实面的 CD2 平均，`L_curve` 是各非 self 有向半边的 CD2 平均，二者相加。XYZ 为同一归一化全局坐标系的实际预测采样点；目标为真实 `data.npz` 的 GT 点，不是 `D(z_GT)` 的重建点，不监督 normal／bbox 参数本身。
- `L_geo = (1/B) sum_i 1[SNR(t_i)>=1] * (L_surface_i+L_curve_i)`。高噪声样本贡献 0，**分母仍为整个 GT batch B**，不是有效样本数。空边的 curve 项为 0。

GT padding 保持原随机复制和排列，只额外记录最初每个真实面的 slot。几何支路各取一次对应预测，不对所有复制 slot 重复计权，也不把多个预测 latent 平均成新 latent。曲线按 GT 有序 owner pair 监督，过滤 self-owner 与原 cache 编码一致；这是训练监督，不是推理 oracle 条件。面存在性、边存在性、闭合环／OCC solid 合法性没有新增显式 loss。

验证仍只计算原 GT DDPM loss；不运行几何项或 DPO。后续比较固定新 run 的 step5000，不按不同辅助项的训练 loss 挑最佳模型。

## 几何权重依据

使用固定 train 首 8 件，seed42，t=`[62,187,312,437,562,687,812,937]`，没有查看 validation/test 来调权重。对 epsilon 预测的梯度范数：DDPM **0.003822132**，未加权几何 **0.016881419**。按目标比例 10% 得到并固定 `lambda_geo=0.0226`，实际比例 **9.9819%**。

这是初始化尺度检查，不是最优超参数，也不代表 policy 参数空间的梯度比例恰好为 10%；训练中没有动态归一化、自动调权或网格搜索。

## 两组手动运行

两组均从旧标签 DPO 的 `20261023T030527Z/weights/steps/step=00005000.ckpt` 初始化，reference 仍为原 SVR。使用旧 42,037 对 pair、原 GT cache、8卡×batch8、FP32、seed42、Adam lr=1e-5、clip1，各**额外 5,000 步**。两组均新建 Adam 和 run，步数从 0 计；新 step5000 表示在原 DPO5000 上再训练了 5000 步，不是 resume 旧 run。

先运行控制组：

```bash
cd /mnt/d/python
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  /mnt/d/anaconda3/envs/torch/bin/python -u -m src.brepnet.train \
  --config-name train_diffusion_preference_continue
```

然后运行几何组：

```bash
cd /mnt/d/python
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  /mnt/d/anaconda3/envs/torch/bin/python -u -m src.brepnet.train \
  --config-name train_diffusion_preference_geometry
```

产物分别为 `checkpoints/diffusion_preference_continue/<UTC>/` 与 `checkpoints/diffusion_preference_geometry/<UTC>/`；每 1000 步保存 `weights/steps/step=...ckpt`。中断后使用相同 config 并显式传入新 run 的 `trainer.run_dir`、`trainer.resume_from_checkpoint=<新run>/weights/steps/last.ckpt`；恢复时从自身 checkpoint 严格加载，不重读最初两个权重、不重新初始化 policy。几何开关／权重／数据协议改变会拒绝 resume。

## 验证与边界

针对本次几何项、原 DPO、严格推理加载和已有 observation 数据兼容的 **47 项测试全部通过**。两份 Hydra 配置完整解析成功；除几何开关／权重和 run 名称外，解析后的设置一致。

CPU 真实数据检查使用源机 torch2.4.1+cu124：

- 原 decoder 与几何支路的对应 surface/curve XYZ 最大误差均 **0**。
- 真实两个 train CAD 为 7/14 面、30/72 条非 self 有向半边。完整 DDPM+DPO+geometry forward/backward 有限；几何梯度到 policy 第一个参数的范数 **0.0208323**，冻结参数均无梯度。
- 真实检查执行 **0 次 optimizer step**，未使用 GPU。小网络测试另验证 Adam step 后冻结权重不变、原损失与 RNG 保持一致、padding/面排列对应、空边、高噪声归零和严格初始化/恢复/推理加载。

CPU 检查可复现（选择尚不存在的输出文件）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /mnt/d/anaconda3/envs/torch/bin/python -u -m tools.check_preference_geometry \
  --output /tmp/preference_geometry_cpu_check.json --threads 8
```

上述CPU证据仅对应实现验收；随后用户已完成8卡正式训练，尚无基于这两组checkpoint的重建效果结论。等步数不等于等计算成本。原 decoder 能表达什么、GT owner-pair 监督与推理连接选择之间的差异仍是模型边界，点损失降低不保证最终 solid 合法率提高。

## 完整 validation 对照

北京时间2026-10-25 17:11已启动，后台监督进程PID为365820；实际进度以输出目录状态为准。输出根目录为 `checkpoints/diffusion/preference_eval/geometry_v1_evaluation`，当前尚无完整重建质量结果。

两组固定 checkpoint：

- 控制组：`checkpoints/diffusion_preference_continue/20261025T054547.727170Z/weights/steps/step=00005000.ckpt`。
- 几何组：`checkpoints/diffusion_preference_geometry/20261025T062543.945628Z/weights/steps/step=00005000.ckpt`。

评估入口的68项CPU测试通过。严格原生模型加载、训练来源、冻结权重、train/validation无identity重叠及测量实现核验记录在 [EVALUATION_PREFLIGHT.json](EVALUATION_PREFLIGHT.json)。只扩展模型选择与报告标签；采样、后处理及测量公式保持不变。

```bash
bash /mnt/d/python/checkpoints/diffusion/fidelity_uncertainty/geometry/run_evaluation.sh \
  /mnt/d/python/checkpoints/diffusion/preference_eval/geometry_v1_evaluation
```

新目录保存两组各3505件，8卡分成每组4卡；每卡batch1保持既有采样数值协议。`comparison/`包含raw/post和合法率，`observation/`包含可见一致性与失败阶段，两阶段各自生成配对统计和图表。内部目录名`baseline`对应继续DPO，`dpo_5000`对应DPO+几何项；差值始终是几何组减控制组。阶段状态见根目录`status.json`，日志见`logs/`。中断后在同一命令末尾加`--resume`，不改变源码、配置或输入；终态失败不补抽。原有历史评估目录保持不变。
