# 新版DPO数据与几何辅助训练

状态：北京时间 2026-10-25，用户已完成全量重选、新标签 DPO 5,000 步训练及完整 3,505 件 validation 评估。[训练完成记录](TRAINING_RESULTS.md)；[完整对照报告](../preference_eval/observation_v1_evaluation/report/README.md)。本轮新标签未呈现明确整体收益。后续继续DPO／增加几何项两组均已完成额外5000步，2026-10-25 17:11启动完整3505件validation对照，[本阶段记录](geometry/README.md)。数据验收见[数据结果与缺测分析](DATASET_RESULTS.md)。

1. 重评已有每图8候选：合法性、轮廓IoU、可见表面CD、轮廓边界距离；按保护/改善规则重选一对。
2. 新数据从原SVR起点训练5,000步，与旧DPO同预算比较。
3. 评估后从选定新DPO权重分叉，比较继续DPO与增加冻结decoder几何损失。

具体指标、win/lose阈值和训练设置见[PLAN.md](PLAN.md)。先前[宽泛草案](history/PLAN_v1_broad.md)仅作历史。

事实来源：[DPO结果](../failure_analysis/20261023/RESULTS.md)、[原训练契约](../failure_analysis/20261023/TRAINING_CONTEXT.md)。

## 完整对照结果与下一步建议（2026-10-25）

三阶段均为 complete，固定分母 3,505。报告完成于北京时间 2026-10-24 19:58:47；本次只读核验完成状态、阶段 receipt 的自身指纹与 workflow 绑定、最终 status/summary/provenance 文件摘要，没有重新运行生成或测量，也没有逐文件重新计算全部原始预测摘要。

旧标签/新标签 DPO 的有效 solid 为 3,093/3,083（88.245%/87.960%），新减旧 −0.285 个百分点，95% 配对区间为 [−1.198, +0.628] 个百分点。轮廓 IoU 为 0.905367/0.905249，可见表面 CD² 为 0.00771233/0.00793507；它们与完整 raw/post 主要几何指标的差值区间均跨 0。本轮不支持“新标签改善整体重建”的结论，也不证明两模型等价。

明确改善的是面数绝对误差 1.30927→1.19943。深度 MAE 0.042333→0.0448621，同时 overlap coverage 0.947097→0.951790；深度是在各自预测/GT mask 交集上计算，配对 CAD 相同不等于评分像素相同，因此不能把此差值直接解释为同一表面的深度退步。观测可测为旧 3,403、新 3,396，共同可测 3,320；三轮 post 完整为旧 2,982、新 2,972，共同 2,769。缺测继续保留在总体统计。

下一阶段保留旧 DPO step5000 与旧 pair 为共同起点，两臂分别继续 DDPM+DPO、增加冻结 decoder 的几何辅助项；reference 继续使用原 SVR，其他预算一致。GT 面/边对应、padding 来源映射与可微解码已实现，几何权重固定为 0.0226。两臂已完成训练，当前完整validation正在运行，尚无几何辅助训练的重建效果结论。不默认新标签成为主基线。K=1 仍不能给出同图多样性结论。

## 第二阶段：完整 validation 评估（手动启动）

入口已实现，CPU 验证见 [EVALUATION_VALIDATION.md](EVALUATION_VALIDATION.md)，固定口径见 [EVALUATION_SPEC.md](EVALUATION_SPEC.md)。下列命令现已由用户运行完成；保留作复现记录，已有目录应使用 `--resume` 核验，不能作为新 run 覆盖：

```bash
bash /mnt/d/python/checkpoints/diffusion/fidelity_uncertainty/run_evaluation.sh \
  /mnt/d/python/checkpoints/diffusion/preference_eval/observation_v1_evaluation
```

脚本依次完成新 DPO 生成/后处理、输入视角观测评估、旧新 DPO 对照报告。原 SVR 和旧 DPO 的已有结果只读复用；8 张卡全部用于新 DPO 的 3,505 件生成。每卡 batch1 保持旧评估的数值协议，CPU 后处理与观测各沿用 16 workers。GPU 阶段结束后 CPU 仍可能工作，不能凭显卡空闲判断全流程完成。

同一命令加 `--check-only` 仅检查权重、split、训练重叠、源协议及指标环境，不创建输出目录、不加载 GPU 模型。中断后加 `--resume`；配置、实现和来源摘要必须保持一致，已完成阶段通过 CPU 摘要校验后跳过，不再加载 GPU。输出目录有独占锁。启动后请保持这套评估代码和源数据不变；否则恢复会拒绝混合协议。

续跑只继续未完成工作，不增加候选预算：生成中断且已创建对象目录的样本记为 `generation_error`，不会补抽；观测中断已留下 `running` 记录的样本记为 `interrupted`。失败、超时、缺测始终留在固定分母。`complete` 表示所有预定样本已处理，不表示全部重建成功。

产物均位于指定输出目录：

- `status.json`：整个流程阶段与完成状态；`logs/` 为三阶段日志。
- `comparison/`：原 SVR 的只读引用、新 DPO 预测和 raw/post 几何/合法性。
- `observation/`：仅新 DPO 的轮廓、可见表面、深度及失败审计。
- `report/README.md`、`comparison.png` / `.svg`：三方法概览及旧新 DPO 配对比较；`summary.json` 和 `per_object.csv` 为完整数据。

重点读新减旧的有效 solid 比例、轮廓 IoU、可见表面 CD²、完整 raw/post CD²/F1，和每项各自/共同可测数量、缺测率、CAD 配对 95% CI。每图只有一个样本，不能由此次评估推断同图多解性或 best-of-K；几何训练仍等待本次结果。

## 第二阶段：新数据 DPO 对照

使用 [train_diffusion_preference_observation.yaml](../../../configs/train_diffusion_preference_observation.yaml)。从原 SVR 初始化 policy/reference，沿用原 VAE cache、8 卡 × batch8、FP32、Adam lr=1e-5、beta20、lambda1、seed42、clip1 和 5,000 步；训练目标仍为 GT DDPM + DPO。几何辅助训练留到该轮下游评估后。

```bash
cd /mnt/d/python
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  /mnt/d/anaconda3/envs/torch/bin/python -u -m src.brepnet.train \
  --config-name train_diffusion_preference_observation
```

该命令新建 `checkpoints/diffusion_preference_svr_observation/<UTC时间戳>/`，每 1,000 步保存固定 checkpoint。使用新 run 的 `weights/steps/step=00005000.ckpt` 与旧 DPO 的同一步数比较；不要给首次启动设置旧 DPO 的 resume 路径。新 run 中断后的恢复命令：

```bash
# 替换成实际新 run；max_steps=5000 是累计上限。
OBSERVATION_TRAIN_RUN='/mnt/d/python/checkpoints/diffusion_preference_svr_observation/<UTC时间戳>'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  /mnt/d/anaconda3/envs/torch/bin/python -u -m src.brepnet.train \
  --config-name train_diffusion_preference_observation \
  trainer.run_dir="$OBSERVATION_TRAIN_RUN" \
  trainer.resume_from_checkpoint="$OBSERVATION_TRAIN_RUN/weights/steps/last.ckpt"
```

共 320,000 次 GT 与 pair 抽样，4,965 个无 pair 的训练 CAD 仍参与 GT 去噪。pair 池大小变化使每对平均曝光从旧版约 7.61 次变为约 7.68 次；两类 pair 的比例也随重选变化。训练期间 validation 仍只用原 100 个 GT 计算去噪 loss；后续固定 3,505 件 validation 做 K=1 下游评估，同时报告有效率、可见一致性及完整 raw/post 几何。

[TRAINING_PREFLIGHT.json](TRAINING_PREFLIGHT.json) 记录：新旧实际配置除 pair 路径、run 名称/目录和等价的 cache 路径写法外一致；原 checkpoint SHA256 验证通过；训练目标/trainer 源码与旧训练提交一致，dataset 抽样/变换函数保持一致；`legacy.py` 仅新增下游 DPO 推理加载函数，原训练加载函数未变。真实 loader 清单包含 46,627 个 GT、41,662 个 pair、100 个 validation GT，并实际抽读一条训练样本，其 tensor 形状/有限值通过。此次检查不加载模型或运行 GPU。

## 第一阶段运行

执行契约见 [DATASET_SPEC.md](DATASET_SPEC.md)。入口是 `src.brepnet.data.preference_relabel`：复用原 run 的 K=8 候选，不重新运行 DDPM、VAE 或后处理。OCC 网格化和当前观测指标在 CPU 上执行；同 CAD 共用 GT 渲染/mask，跨 CAD 并行，每个数值库线程数固定为 1。减少重复计算比占满显存更符合这一阶段的实际工作。

在仓库根目录运行（目前环境为 `/mnt/d/anaconda3/envs/torch`）：

```bash
cd /mnt/d/python
CUDA_VISIBLE_DEVICES='' /mnt/d/anaconda3/envs/torch/bin/python \
  -m src.brepnet.data.preference_relabel \
  --source-run /mnt/d/python/checkpoints/diffusion/preference_data/20261020T134930Z \
  --output-dir /mnt/d/python/checkpoints/diffusion/preference_data/observation_v1 \
  --workers 32 --image-timeout 600
```

默认处理原 train 46,627 件和 validation 100 件，共 373,816 个候选位置，包括旧失败。`--workers 32` 是起始并发，不是已测得的最大吞吐；CPU/磁盘空闲时可用 `64`。建议在 tmux 内运行，退出 SSH 后任务仍保留。没有 GPU batch 参数，因为本阶段不加载 GPU 模型。

中断后使用同一命令末尾加 `--resume`。允许改变 `--workers`，但不允许混入不同代码、指标参数、超时预算或源文件。已经完成的候选仍会校验源摘要，再复用测量；测量失败/超时不会在续跑时自动反复重试。被手动中断、尚未落盘的候选可以继续计算。并发启动同一目录会被锁拒绝。

若只验证入口，另用新目录并加 `--splits validation --limit 2 --workers 2`。这类产物显式标记 `partial=true`，训练 loader 会拒绝；不要把小样本目录改名当作完整数据。

进度查看：

```bash
cat /mnt/d/python/checkpoints/diffusion/preference_data/observation_v1/status.json
```

完成后输出：

- `pairs.jsonl` / `validation_pairs.jsonl`：新版 pair，绝对路径引用旧候选的原始 `[30,32]` latent。
- `summary.json`：固定分母、可评分候选、pair 覆盖、旧新 pair 的保留/替换/翻转/弃权/新增数量。
- `cases/<split>/<id>/metrics.json` 和 `selection.json`：8 个候选状态、观测分数、源摘要及逐有序对的接受/拒绝理由。
- `metadata/protocol.json`、源码快照、launch/diff：输入与实现可追溯。旧 run 不写入任何内容。

新训练仍使用旧 run 的 `gt_cache`，不能把新输出目录当成新 GT cache。派生协议已接入 `PreferenceDataset`，完整生成并校验 manifest 后才允许训练。原生合法标签与“能否完整三角化并测量”分开，测量失败不会自动标成 loser。600 秒是每 CAD 的 8 候选总预算；预算耗尽时已完成结果保留，其余候选记录缺测。
