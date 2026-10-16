# Local intersection 微调

用面 pair 的单个 Query 读取两面 CNN 局部特征与预测 XYZ，替换原 inter。保留投影、refiner、线性 classifier 和原损失，不引入 H/P。结构与比较协议见 [PLAN.md](PLAN.md)，实现验证见 [NOTE.md](NOTE.md)。

2026-10-12 容量修正：主配置已恢复旧 inter 的 **8层、512维、8heads、FFN2048、dropout0.1**，层内为相同的 pre-norm `TransformerDecoderLayer`（self-attention、cross-attention、ReLU FFN）。8层主干参数与旧版相同；完整 inter 为34,015,296参数，旧版为33,683,008，差额来自局部输入所需的 Query/memory 投影等。局部 token 的读取方式保持已讨论方案。修正后的正式训练由用户启动，输出独立写入 `full_capacity/`。验证见 [容量修正记录](NOTE.md#容量修正2026-10-12)。

2026-10-13 正式测试：`full_capacity/20261012T141134.581695Z` 已训练并由用户停止，按验证集固定epoch42/step15609（last为epoch43/step15972）。固定2417件与相同输入/后处理预算，优化后合法solid为 **1705/2417（70.54%）**，上一版refinement为2113/2417（87.42%）；27件救回、435件退步。本版明显优于历史2层结果，但尚未超过refinement。原始边CD均值为基线7.407倍、中位数2.748倍。见 [完整评估](evaluation_20261013/REPORT.md)、[研究解读](evaluation_20261013/NOTE.md)、[本轮失败分析](failure_census_20261013/NOTE.md)。

此前正式训练产物 `frozen/20261012T073904.575690Z` 是 **2层128维的历史版本**，验证最佳 epoch19 / step7260，与该 run 的 last 相同。其完整评估显示优化后合法 solid 从上一版2,113件降到482件；不能当作本次恢复容量版本的结果。见 [完整评估](evaluation_20261012/REPORT.md)、[研究解读](evaluation_20261012/NOTE.md)、[失败分析](failure_census_20261012/NOTE.md)和[交互统计页](failure_census_20261012/page/index.html)。

本轮712个失败与27个救回的双侧诊断已齐，独立复算通过。四组中文图和全部筛选/逐件证据可直接打开 [2026-10-13自包含统计页](failure_census_20261013/page/standalone.html)，不依赖临时HTTP服务。

## 开始微调

在仓库根目录运行。这里选择已经完成评估的 epoch10/step3993 最佳权重；不是后来 epoch17/step6534 的 `last.ckpt`。只重新初始化 local inter，其余模块严格迁移。**这是新实验初始化，不能改成恢复旧 checkpoint 的 optimizer。**

```bash
conda activate torch
cd /mnt/d/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

python -m src.brepnet.train --config-name train_vae_intersection_local \
  'model.init_refinement_checkpoint="/mnt/d/python/checkpoints/intersection_refinement/frozen/20261011T123836.485257Z/weights/epoch=0010-step=00003993-val_loss=0.039476.ckpt"' \
  dataset.raw_data_root=/mnt/d/data/deepcad_v7 \
  dataset.train_dataset=checkpoints/joint_boundary/validation/train_contract_audit/accepted.txt \
  dataset.val_dataset=checkpoints/joint_boundary/validation/val_contract_audit/accepted.txt \
  dataset.test_dataset=checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt
```

路径包含 `=`，上面的外层单引号、内层双引号均需保留，确保 Hydra 把整个路径当成一个值。

默认 8 卡、每卡 batch16、全局 batch128、bf16、Adam lr=2e-5，encoder/posterior 冻结；face Transformer、面／边 CNN 和 bbox、classifier、refiner 与新 inter 一起训练。W&B 继承现有配置开启。新 run 从 step0 开始，累计上限100万steps／1万epochs；可用 `trainer.max_steps=<预算>` 修改。

输出写入 `checkpoints/intersection_local/full_capacity/<UTC时间>/`，保留 weights/logs/wandb/raw_predictions/metadata 的现有目录合同，不覆盖旧 refinement 或2层local run。正式训练未由本次代码修改自动启动。初始权重来自上一版refinement；不从本次失败的2层local权重续训。新inter仍按既定迁移协议随机初始化，其他权重严格保留。

## 继续新 run 或测试

恢复本次8层新run时仍使用 `train_vae_intersection_local`，数据路径与上面相同，去掉初始化行并改成：

```bash
trainer.resume_from_checkpoint=/absolute/local/run/weights/last.ckpt
```

这才会恢复新架构的 optimizer 与累计步数。resume 会跳过全部初始化路径；保持新 run 的结构、设备、batch 与数据协议。测试在新 run 的续训命令后追加：

```bash
trainer.evaluate=true trainer.devices=1 trainer.batch_size=2 dataset.is_aug=0
```

导出仍为 `legacy_half_edges`，保持既有后处理输入接口。不能用 legacy 配置加载 local 权重；若仅从一个已训练的 local checkpoint 开始另一个新 run，使用 `model.checkpoint=/absolute/local.ckpt` 严格加载。

历史2层run的继续训练／测试使用 **`--config-name train_vae_intersection_local_small_legacy`**，再提供其原checkpoint和数据路径。该配置显式保留原128维／2层／4heads／FFN256／dropout0／GELU与原状态键；历史checkpoint自身未记录 `block_type` 时，也按原cross-attention实现严格加载。8层和2层权重不能混用，误配会报错。独立评估工具按checkpoint记录的配置构造，不会把历史2层权重自动扩成8层。

## 代码入口与分块

- `src/brepnet/models/local_intersection.py`：pair Query、两面局部 memory、cross-attention 与 E 输出。
- `src/brepnet/models/intersection_decoder.py`：提取 CNN 中间激活、复用 coarse face、连接旧几何头和 refiner。
- `src/brepnet/models/intersection_vae.py`：严格权重迁移；仅允许替换 `decoder.inter.*`。
- `configs/train_vae_intersection_local.yaml`：本轮配置。旧配置默认 `intersection.kind=legacy`，旧 checkpoint 继续严格加载。

`model.intersection.candidate_chunk_size=64` 和 `model.intersection.checkpoint_candidates=true` 控制新 attention；逐 chunk gather 两面的 memory，避免一次保存所有 pair 的512-token输入。这里的 checkpoint 表示反向重算。

`model.refinement.candidate_chunk_size=256`、`geometry_chunk_size=128` 和 `checkpoint_candidates=false` 继续控制旧几何解码／反馈路径；`model.loss_weights.geometry_chunk_size=128` 控制几何损失分块。两组分块含义不同，均不改变候选集合。
