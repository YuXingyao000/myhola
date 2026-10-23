# 原版 SVR 的偏好数据与微调

本轮保持原 HoLa 的 **1024维、24层 Transformer、全局图像 concat**，使用
`Diffusion_svr_sq30_1500k.ckpt` 内的原始 VAE 和 DINO。
只处理灰模 `svr_imgs[0]`。规则定义以 [PLAN.md](PLAN.md) 为准，旧v2提议不生效。

## 本次运行

2026-10-23已核对生成完成：训练46,627件、373,016候选、42,037对；validation100件、
800候选、95对。训练pair含17,258对validity与24,779对geometry。清单、规则及引用文件
检查通过；保留评估报错、超时和未知状态，不将unknown直接用作negative。
用户已授权启动正式DPO微调，首轮预算与完整验证安排见
[训练记录](TRAINING_20261023.md)。以下保留生成阶段的启动记录。

2026-10-21用户要求暂停并扩到8卡。新配置为GPU0–7、每卡2个独立采样进程，
CPU后处理32个worker、队列64；原checkpoint、FP32、batch1、DDPM1000、K8及标签规则保持。
暂停前已有5,336件完整训练图片、42,695个完成候选、4,842对。已排空CPU后处理队列；
4个未完成GPU候选按原协议记unknown，不重抽。原输出和原始protocol保留。
性能和恢复验证见[报告](performance_20261021/README.md)。

八卡恢复于北京时间2026-10-21 13:53启动，继续使用原run目录；新tmux会话为
`diffusion-preference-8gpu-20261021T055326Z`，日志为
`/mnt/d/python/checkpoints/diffusion/preference_data/20261020T134930Z.resume_20261021T055326Z.log`。
调度提交`ed26c20b`，八卡命令如下：

```bash
/mnt/d/anaconda3/envs/torch/bin/python -u -m src.brepnet.data.preference_generate \
  --config configs/datagen/preference.yaml \
  --run-dir checkpoints/diffusion/preference_data/20261020T134930Z \
  --resume --accept-scheduling-update \
  'gpus=[0,1,2,3,4,5,6,7]' workers_per_gpu=2 cpu_workers=32 queue_size=64
```

2026-10-20 13:49:30 UTC启动正式数据生成：

- 代码提交：`0c5ff7346d86bb3194b4ea222f7db60b2032dc63`。
- GPU：4、5、6、7；CPU后处理worker 16，有界待处理队列32。
- Run：`/mnt/d/python/checkpoints/diffusion/preference_data/20261020T134930Z`。
- 日志：`/mnt/d/python/checkpoints/diffusion/preference_data/20261020T134930Z.log`。
- tmux：`diffusion-preference-20261020T134930Z`。
- 正式训练未启动；数据生成启动后不持续监看。

启动检查确认四个worker均完成旧checkpoint加载，已产生候选；随后结束监看。
本机提交成功，代码提交的后台同步被目标端同分支检查拒绝，两端分支不一致；
未切换目标端分支，也未强推。本机生成不依赖该同步。

实现验收：32项测试通过，2项可选GPU测试在普通测试命令中跳过，实际GPU检查另行执行并记录。
多进程真实接口测试使用2个validation对象、1个train对象，K=8共24候选；阶段转换、
两类pair、manifest及训练dataset只读加载均通过。这些小规模结果不用于推断全量pair率。

## 生成和训练的分工

| 模块 | 职责 |
|---|---|
| `src/brepnet/diffusion/legacy.py` | 从固定历史Git源码严格加载旧模型；原生DDPM1000采样、原AE解码与GT编码 |
| `src/brepnet/data/preference_generate.py` | GPU采样、有界CPU队列、阶段调度、每图选择与manifest |
| `src/brepnet/data/preference_io.py` | 协议、源码/输入摘要、运行溯源与恢复检查 |
| `src/brepnet/eval/preference.py` | 隔离OCC进程、recover_v1、native validity及raw/post指标 |
| `src/brepnet/data/preference_selection.py` | 明确的退步上限、改善阈值、有效性优先与固定seed选对 |
| `src/brepnet/data/preference_dataset.py` | 独立GT数据流与均匀pair数据流；核验生成协议 |
| `src/brepnet/diffusion/preference.py` | GT DDPM + Diffusion-DPO；冻结reference和共享DINO |

每图保存8组完整 `[30,32]` latent，然后解码、后处理和计算指标。训练直接读取这组未去重
latent；不将修复后的CAD重新编码。没有pair的训练图片仍参加GT DDPM监督。

同图最多一对；先选择合法优于确认无效的pair，否则选择两个合法结果间的几何pair。
CD²改善5%、容许退步2%；F1改善0.02、容许退步0.01。两个合法结果需要所有3轮
post测量都通过。无合适pair就记录原因，不增加采样次数。完整规则见PLAN第7节。

这是有GT的离线训练标签，不是部署时可以使用的候选选择器，也不是GT best-of-K的单次推理效果。

## 数据生成

```bash
# 当前八张卡，每卡两个独立进程；不会启动训练。
/mnt/d/anaconda3/envs/torch/bin/python -u -m src.brepnet.data.preference_generate \
  --config configs/datagen/preference.yaml \
  'gpus=[0,1,2,3,4,5,6,7]' workers_per_gpu=2 cpu_workers=32 queue_size=64
```

默认每图K=8，共800个validation候选、373,016个train候选。
validation若没有任何pair，程序记录 `stopped_no_validation_pairs` 并结束；不自动放宽规则。
其他情况下自动开始train，validation pair不进入训练manifest。

每个run包含：

```text
metadata/                 # 配置、源码快照、Git状态、固定split和逐图选择理由
candidates/{split}/{id}/c00/
  latent.npy              # 原始30×32，不重新排列或补齐
  data.npz                # 原AE原生输出
  mappings.npz            # latent/几何两次去重映射
  inference.json          # seed、耗时、输出hash
  evaluation/             # 指标、后处理日志、STEP和OCC诊断
gt_cache/{id}_0/           # 用同一旧AE重新编码的mean32/std32
gt_metric_samples/        # 同图同轮共用的GT STEP采样
validation_pairs.jsonl    # 不用于训练
pairs.jsonl               # train阶段完成后发布
status.json               # 当前阶段、数量、完成/失败状态
```

旧磁盘cache与三件FP32重编码GT存在小幅差异，当前没有充分证据解释来源。
因此本轮使用独立新cache，不把名字相同当作语义相同；原cache保留。

恢复必须显式指定原目录和 `--resume`，源码、权重、split、K、指标/预算保持一致。
已完成候选直接复用；中断且未写完的候选保留为unknown，不额外抽样。
运行错误、超时、缺失不当作negative。所有请求保留在分母里。

2026-10-21只迁移调度实现；恢复命令额外显式传`--accept-scheduling-update`，
记录在`metadata/runtime_revisions.json`及其源码快照目录。该选项仍拒绝模型、指标、
阈值、预算与数据变化。新运行过程中创建run下的`PAUSE`文件，会让各worker完成当前
图片的K个候选并排空后处理后退出；显式`--resume`清除此请求并继续。

## 训练代码

训练目标为：

```text
loss = GT_DDPM_loss + preference_weight * DPO_loss
logit = -beta / 2 * [(policy_winner_MSE - policy_loser_MSE)
                   - (reference_winner_MSE - reference_loser_MSE)]
DPO_loss = -mean(logsigmoid(logit))
```

winner/loser共用随机t和噪声；policy/reference共用冻结DINO输出。
偏好比较关闭dropout，policy仍保留梯度；GT监督保留原训练模式。
VAE不参与反传，OCC不进入训练step。reference连同policy、DINO一起保存到训练checkpoint。
恢复从该checkpoint严格加载，不再依赖初始旧SVR权重文件。

下面保留最初的命令模板；2026-10-23实际八卡、5,000步首轮命令与运行状态见
[训练记录](TRAINING_20261023.md)。`PREFERENCE_RUN`填生成run目录，`TRAIN_STEPS`显式给出预算。

```bash
# 示例预算与设备，启动前按实际计划填写。
PREFERENCE_RUN=/mnt/d/python/checkpoints/diffusion/preference_data/20261020T134930Z
TRAIN_STEPS=10000
CUDA_VISIBLE_DEVICES=4,5,6,7 /mnt/d/anaconda3/envs/torch/bin/python -m src.brepnet.train \
  --config-name train_diffusion_preference \
  dataset.pairs_manifest="$PREFERENCE_RUN/pairs.jsonl" \
  dataset.cached_latent_root="$PREFERENCE_RUN/gt_cache" \
  trainer.devices=4 trainer.batch_size=4 trainer.seed=42 \
  trainer.max_steps="$TRAIN_STEPS"
```

初始建议配置β=20、λ=1、lr=1e-5，不是已验证的最优设置。
当前validation只计算GT去噪loss，不能据此认定下游最优checkpoint。
后续效果比较仍需固定单次采样和原VAE/post协议，报告原始几何、STEP几何、
有效率及失败分母；本次实现测试不构成效果提升证据。

## 验证

原生DDPM完整采样与新接口的latent及raw输出逐bit相同；GT重编码与原AE posterior
逐bit相同。规则测试包含阈值边界、缺指标、unknown、零基准和重复测量翻转。
真实OCC链路已覆盖合法与无效结果。

真实GPU训练合同测试只执行单batch前后向，没有optimizer step。policy梯度全部有限，
DINO/reference无梯度，初始DPO为log(2)。实际生成候选读取保持完整原始latent。
详见 [TRAINING_VALIDATION.json](TRAINING_VALIDATION.json) 与 `VALIDATION.json`。

目标依据：[Diffusion-DPO论文](https://arxiv.org/abs/2311.12908)、
[官方实现](https://github.com/SalesforceAIResearch/DiffusionDPO/blob/main/train.py)。
本项目选样本阈值属于研究设计，并非这些文献已经证明有效的标签规则。
