# Joint shared-boundary decoder

本目录对应 [实现合同](PLAN.md)。冻结 encoder 的正式运行已完成 100,000 steps；2026-10-11 使用验证集最佳 step 97,559，与原 VAE 在相同 2,417 个 test 对象及共享 latent 下比较。**当前新 Decoder 的原始几何与自主拓扑均退步，尚不适合替换原 Decoder。** 完整协议、实体结果、固定选例与复现命令见 [比较报告](comparison_20261011/REPORT.md)和[比较 README](comparison_20261011/README.md)。[验证记录](NOTE.md)保留最初实现与短运行的历史状态，不能将其中“尚未训练”误读为当前状态。

## 当前源码入口

2026-10-13 将实验代码归并回原 `vae.py`／`blocks.py` 文件组织：`src/brepnet/models/vae.py:AutoEncoder_joint` 负责模型、encoder、初始化与导出，`vae.py:joint_boundary_loss` 及匹配函数负责监督；`src/brepnet/models/blocks.py:JointBoundaryDecoder/BoundaryInteraction` 负责共享边界解码和双向更新。原 `joint_vae.py`、`joint_decoder.py`、`joint_losses.py` 与同批六个 `intersection_*`、`holistic_encoder.py`、`local_intersection.py` 共 11 个文件已移除。历史 PLAN／NOTE 中的旧路径不重写。

本次文件归并保留 `AutoEncoder_joint` 注册名、配置、模块属性层级及已有 checkpoint state key；旧 encoder 的组件迁移、完整模型严格加载和 resume 语义不变，不增加 trainer 分支，也不改变 joint 网络计算。

原 refinement 的可选最终边几何头增量见 [curve_conditioning 说明](../intersection_refinement/curve_conditioning/README.md)。该配置属于 `AutoEncoder_intersection`，默认 `model.curve_conditioning.enabled=false`、`hidden_dim=128`，只调制 refined edge CNN／bbox 最终输出层前的特征，不施加于本页的 joint 模型。

## 表示与配置

保留旧 HoLa light encoder 的面、半边输入及归一化，将 decoder 改为面特征与共享边界 queries 的双向更新。当前数据表示是“每对不同面之间至多一条合并边界”，不是完整 native physical-edge / seam / UV-loop 表示。GT 边界只用于匹配与损失，导出时不以 GT edge count 或 GT query selection 筛选预测。

配置分别为：

| 配置 | encoder | 新 decoder | 默认预算 |
|---|---|---|---|
| [train_vae_joint_boundary_frozen](../../configs/train_vae_joint_boundary_frozen.yaml) | 旧权重迁移后冻结，保持 eval | 双向更新 | 8 GPU，每卡 batch 32，100,000 optimizer steps |
| [train_vae_joint_boundary_joint](../../configs/train_vae_joint_boundary_joint.yaml) | 可训练 | 双向更新 | 8 GPU，每卡 batch 4，100,000 optimizer steps |
| [train_vae_joint_boundary_one_way](../../configs/train_vae_joint_boundary_one_way.yaml) | 冻结 | 关闭 edge-to-face 更新 | 与 frozen 相同 |

`batch_size` 与 `num_workers` 都是每个 GPU 进程的设置；frozen 的全局 batch 为 `8 × 32 = 256`，默认共有 `8 × 4 = 32` 个 loader workers。W&B 只记录运行，不决定 GPU 数量。使用下面的 `python -m` 入口，由 Lightning 启动 DDP 子进程；目前未验证外部 `torchrun` 的运行目录合同。

2026-10-10 初始 frozen 配置曾为单卡 batch 4。现在同样 100,000 steps 对应的样本曝光量约为之前的 64 倍，不能作为等预算比较；若只对齐原先约 400,000 次样本曝光，可显式用 `trainer.max_steps=1563`，但这不等价于相同的优化过程。学习率仍为 `1e-4`；正式 frozen 运行已完成，质量判断见上述固定协议比较。joint 保留已做过复杂对象反向检查的每卡 batch 4，不能直接套用 frozen 的 batch 32。

DDP 的 face L1 先在每卡按本地面数平均，再跨卡平均；由于每卡面数不同，它不严格等于单卡 batch 256 的全局面平均。本次保留原损失归约方式。frozen 与 joint 的全局 batch 相差 8 倍，阶段间同 step 数也不是等样本预算。

单向版本仅用于之后隔离双向反馈的贡献，不等于已经建立与旧 HoLa decoder 的公平比较。共享边界曲线的存在只是表示性质，不作为性能指标。

旧 encoder 配置对应已检查的 `1119_deepcad_aug1_11k` light 权重：5 层 GAT、8 层 Transformer、32 维 face latent、768 维 backbone、每 stage 一个 primitive block。新 decoder 默认 128 个边界候选、hidden 256、3 轮更新；超过容量的 target 必须报错，不能静默截断。[边界数审计](validation/boundary_count_audit.json)用于说明容量选择。新配置训练默认使用 cube24 随机旋转，validation 使用 identity；下面的测试命令显式指定 `dataset.is_aug=0`，按每个 CAD 一个输出计数。

互反半边几何检查采用归一化 XYZ 距离阈值 `0.01`；明确允许闭合曲线的起点相位差，闭合判断阈值为 `1e-5`。原有代表曲线采样保持不变。这些数据与模型设置都在 resolved config 中记录。

## 运行目录

所有训练入口统一默认到 `./checkpoints`，不写入旧 `experiments/`：

```text
checkpoints/joint_boundary/
  PLAN.md                         # 受 Git 管理的合同
  README.md / NOTE.md              # 命令、结果与解释
  validation/                     # 紧凑的检查报告
  frozen/<UTC timestamp>/          # 一次独立训练
    weights/                      # best 与 last Lightning checkpoint
    logs/                         # 每次调用的日志及 TensorBoard
    wandb/                        # W&B 本地记录
    raw_predictions/<invocation>/  # 每对象 data.npz
    visualizations/               # 图像／几何预览
    metadata/
      run.json                    # 原 run 标识与 W&B ID
      hydra/                      # 最近调用的 Hydra 元数据
      invocations/<invocation>/
        config.yaml               # 解析后的实际配置
        git.json / git.diff       # Git revision、状态及 tracked dirty diff
        command.json              # 命令、seed、resume 来源
```

`trainer.output_dir` 是输出根目录；`trainer.run_dir` 是完整的显式 run 路径。默认时间戳区分新运行；已有 run 禁止当作新训练覆盖。恢复 `weights/*.ckpt` 时自动回到原 run；旧 `run/checkpoints/` 布局也能恢复。其他 checkpoint 布局需显式 `trainer.run_dir`。

W&B 默认启用，可用 `trainer.wandb.enabled=false` 关闭或 `trainer.wandb.offline=true` 离线运行；同一 run 续训保留 W&B ID。代码检查不需要线上 run。大型运行产物不提交，主题文档及汇总报告保留版本管理。

## 环境与共同数据参数

以下命令使用已有 `torch` 环境；数据路径只在命令中指定。每个新 shell 先设置：

```bash
conda activate torch
cd /mnt/d/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export BREP_DATA_ROOT=/mnt/d/data/deepcad_v7
export BREP_LEGACY_AE=/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt
BREP_DATA_ARGS=(
  "dataset.raw_data_root=$BREP_DATA_ROOT"
  dataset.train_dataset=checkpoints/joint_boundary/validation/train_contract_audit/accepted.txt
  dataset.val_dataset=checkpoints/joint_boundary/validation/val_contract_audit/accepted.txt
  dataset.test_dataset=checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt
)
```

完整原 split 的契约审计结果如下。拒绝对象的互反闭合边界几何超过 `0.01` 阈值；其中既有明显不一致，也有当前采样不足以确认的情况，因此没有自动放宽阈值。训练显式使用版本管理的 `accepted.txt`，dataset 遇到不合约对象仍报错，不在运行中静默过滤。

| split | 原始清单 | 契约接受 | 拒绝 |
|---|---:|---:|---:|
| train | 46,627 | 46,480 | 147 |
| val | 3,505 | 3,495 | 10 |
| test | 2,424 | 2,417 | 7 |

新模型与旧 baseline 比较必须使用相同 accepted 清单。测试报告同时列出 `2,417` 个可接受对象上的结果与原始 `2,424` 个对象的总体覆盖率；其余 `7` 个“当前契约不支持”的对象保留在原始分母中。

重新 preflight 时使用新的输出目录，保留全部 `samples.csv`、`rejected.csv`、`summary.json` 和带哈希的 accepted 清单。例如重建 test split 审计：

```bash
python -m tools.audit_shared_boundaries \
  --data-root "$BREP_DATA_ROOT" \
  --split-list src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt \
  --output-dir checkpoints/joint_boundary/validation/test_contract_audit_rerun \
  --reciprocal-tolerance 0.01 --closed-tolerance 1e-5 \
  --num-queries 128 --workers 8 --write-accepted-list
```

train、val 分别替换原始 split 清单与输出目录。退出码 `1` 表示存在拒绝对象；完整的报告与 accepted 清单仍已写出，不等于脚本没有完成。正式改用新清单前核对报告、计数与哈希，不覆盖已有审计记录。

首次冻结 encoder 训练必须给一个有效权重来源，防止误冻结随机 encoder。`model.init_encoder_checkpoint` 只迁移经过 key/shape 校验的旧 encoder 组件；新 decoder 随机初始化。

```bash
python -m src.brepnet.train --config-name train_vae_joint_boundary_frozen \
  "${BREP_DATA_ARGS[@]}" \
  "model.init_encoder_checkpoint=$BREP_LEGACY_AE" \
  trainer.seed=42 trainer.devices=8 trainer.batch_size=32 \
  trainer.max_steps=100000
```

需要明确输出路径时附加 `trainer.run_dir=/absolute/path/to/new-run`。设备数和预算应按实际研究协议修改，不把默认值当作已调优设置。

本配置设置 `trainer.cudnn_benchmark=false`：encoder 输入打包后的面数／边数每步变化，开启卷积算法搜索会产生额外开销。原始单卡短基准中，batch 增大、损失重复计算消除、线程数限制及关闭该搜索均有吞吐收益；详细口径见 [NOTE](NOTE.md)。CPU Hungarian 匹配及逐对象损失仍有同步开销，不保证 GPU 利用率持续满载。

## 续训与完整模型微调

**续训**恢复模型、优化器、步数与原 run；使用同一模型配置和数据协议。设 `BREP_RUN` 为实际 run 路径：

```bash
export BREP_RUN=/absolute/path/to/checkpoints/joint_boundary/frozen/run-timestamp
python -m src.brepnet.train --config-name train_vae_joint_boundary_frozen \
  "${BREP_DATA_ARGS[@]}" \
  "trainer.resume_from_checkpoint=$BREP_RUN/weights/last.ckpt" \
  trainer.max_steps=100000
```

续训／测试不需要再次传旧 AE；trainer 构造模型时跳过旧 encoder 迁移，再严格恢复完整新 checkpoint。`max_steps` 是累计上限，不是追加步数。若恢复旧单卡运行，显式保留其 `trainer.devices=1 trainer.batch_size=4` 才能保持原 batch 协议；改变设备或 batch 必须记录。保存发生在验证结束，首轮验证前中断的运行不一定有可恢复 checkpoint。

**从已训 frozen 模型做 joint fine-tuning**是新 run，使用严格的完整新模型权重初始化，不恢复旧 optimizer 或 step：

```bash
python -m src.brepnet.train --config-name train_vae_joint_boundary_joint \
  "${BREP_DATA_ARGS[@]}" \
  "model.checkpoint=$BREP_RUN/weights/last.ckpt" \
  model.init_encoder_checkpoint=null \
  trainer.seed=42 trainer.max_steps=100000
```

两种初始化字段互斥；`trainer.init_from_checkpoint` 的旧 permissive 入口被新模型拒绝。也可在 joint 配置中仅指定 `model.init_encoder_checkpoint`，表示从旧 encoder 开始、联合训练一个随机新 decoder；它与“完整 frozen 模型微调”是不同实验，必须分别记录。

## 测试与原始导出

测试必须明确 checkpoint，不回退到随机模型或模糊的“best”。固定选择规则后将 `last.ckpt` 换成相应 best 文件：

```bash
python -m src.brepnet.train --config-name train_vae_joint_boundary_frozen \
  "${BREP_DATA_ARGS[@]}" \
  "trainer.resume_from_checkpoint=$BREP_RUN/weights/last.ckpt" \
  trainer.evaluate=true trainer.devices=1 dataset.is_aug=0 \
  trainer.batch_size=4 trainer.wandb.enabled=false
```

当前正式测试／逐对象导出要求单卡，入口会拒绝多卡：默认 distributed sampler 会补齐样本，造成重复对象及输出路径冲突。DDP 训练期间的验证可以多卡；3,495 件 validation 在 8 卡下会补齐为 3,496 次采样，最终逐对象评估仍用单卡完整清单。

默认新建 `raw_predictions/<本次调用时间>/`，实际路径记录在日志与 config。可以显式 `eval.output_dir=/absolute/path/to/predictions`；重用该路径会覆盖同名对象，正式比较应使用独立目录。新模型按每对象 `items` 导出，保留表示标签、预测实体、连接和诊断；不以 GT 选择 query。旧 VAE 测试仍限制 batch=1，不能把拼接 batch 当作每个对象。

VAE 测试使用 posterior mean，并在同一次 forward 中计算 `test/loss` 和导出预测。训练／默认验证仍对 posterior 采样；因此 `val/loss` 与 `test/loss` 的 latent 取值协议不同，不能把二者数值直接当作重建质量变化。

此入口仅导出 raw NPZ，不自动执行 OCC。后处理必须采用声明的 shared-boundary adapter，不能把一条共享曲线直接冒充两条已有 reciprocal half-edges。正式比较固定样本清单、checkpoint、预算与后处理版本，保留失败／缺失／超时在分母内。

## 实现检查入口

模型／数据／导出的针对性测试和真实小批量结果见 [NOTE](NOTE.md)。CLI 短运行可采用下面的明确限制；这是实现检查，不是科研训练：

```bash
python -m src.brepnet.train --config-name train_vae_joint_boundary_frozen \
  "${BREP_DATA_ARGS[@]}" \
  "model.init_encoder_checkpoint=$BREP_LEGACY_AE" \
  trainer.wandb.enabled=false trainer.devices=1 trainer.num_workers=0 \
  trainer.batch_size=2 trainer.max_steps=2 \
  trainer.limit_train_batches=2 trainer.limit_val_batches=1 \
  trainer.num_sanity_val_steps=0 \
  trainer.exp_name=joint_boundary/implementation_check
```

没有正式训练结果时不得将 finite loss、可运行的导出或共享曲线身份写成有效性提升。
