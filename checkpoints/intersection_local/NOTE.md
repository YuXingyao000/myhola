# 实现与验证记录（2026-10-12）

2026-10-13补记：恢复容量后的正式run已完成44个验证epoch并由用户停止，本轮按验证集固定epoch42/step15609做全2417件测试。优化后合法solid为1705（70.54%），历史2层版482（19.94%），上一版refinement2113（87.42%）。当前权重仍未超过refinement；详见 [完整评估](evaluation_20261013/REPORT.md)、[研究解读](evaluation_20261013/NOTE.md)和[本轮失败分析](failure_census_20261013/NOTE.md)。下方为原实现与容量修正的历史验证记录，不是当前训练状态。

下方原始实现记录对应历史2层128维版本；该版本随后已完成训练和失败评估，见 [评估](evaluation_20261012/REPORT.md)。用户指出容量缩减超出讨论范围后，当前主配置恢复为8层512维，修正记录见本文末尾。验证结果仅说明实现合同，不代表研究效果。

## 改动

每个有序面 pair 通过 MLP 得到一个 Query，读取两面各256个 CNN 局部 token 与预测 XYZ；两层128维／4头 cross-attention 后得到 E64。初稿面提前解码并复用。局部特征与 XYZ 不 detach。classifier、投影、refiner、loss、有向候选和导出格式保持原协议。没有 H/P，也没有共享 E 融合。

新 inter 共311,616参数，随机初始化。因此初始边特征分布尚未匹配原几何头／classifier，不能期待第一步保持旧 checkpoint 的精度。训练仍使用已有 feature、几何及分类监督。

`init_refinement_checkpoint` 严格迁移已评估 epoch10/step3993 权重：保留581个非inter状态键，校验并丢弃149个旧inter键，初始化37个local inter键。GPU检查确认保留张量逐一完全相同。源文件SHA256与完整参数量见 [GPU记录](validation/gpu_smoke.json)。

完整local模型保存后，严格加载、训练resume及独立评估入口都会跳过历史初始化路径；评估不要求旧refiner源仍存在。独立评估工具的legacy基线仍用于其原有encoder审计协议。

## 单元与接口检查

81项通过：核心decoder／geometry／loss／VAE、新local与迁移、训练入口共74项；独立评估加载7项。覆盖：

- Query长度1、KV长度512；两面的局部特征与XYZ梯度；仅边loss也能反传到face CNN及bbox。
- coarse face只计算一次；有refiner时最终面再计算一次。
- 有向候选与双向logits、packed对象隔离、面置换一致性、singleton。
- 分块／非重入checkpoint的前向与梯度等价；bf16数值有限。
- 全部保留权重和posterior mean精确一致；冻结encoder；拒绝缺键、多键、shape不匹配和错误迁移来源。
- 新checkpoint严格恢复；resume清除初始化路径；删除旧迁移文件后独立评估仍可加载。

验证命令：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /mnt/d/anaconda3/envs/torch/bin/python -m pytest -q \
  src/brepnet/test/test_intersection_decoder.py \
  src/brepnet/test/test_intersection_geometry.py \
  src/brepnet/test/test_intersection_losses.py \
  src/brepnet/test/test_intersection_vae.py \
  src/brepnet/test/test_local_intersection.py \
  src/brepnet/test/test_local_intersection_migration.py \
  src/brepnet/test/test_training_device_contract.py \
  src/brepnet/test/test_intersection_evaluate_inference.py
```

## 真实数据与GPU检查

环境：Torch2.4.1+cu124，Lightning2.5.0.post0，RTX4090 24GB，seed42，bf16，cudnn_benchmark关闭。使用真实accepted train中的 `00001000`（30面）和 `00002505`（28面）。先batch2、再交替重复到batch16，各执行前向、反向、clip和一次Adam更新；这是容量／数值检查，临时模型丢弃，不用作训练成果。

batch16含13,008个有向候选，峰值allocated 10,196 MiB、reserved 10,972 MiB。新inter、face CNN/bbox、edge head、classifier和refiner均获得有限非零梯度；encoder/posterior无梯度。两个对象的推理导出检查通过。计时仅是预加载batch上的实现检查，不是与旧模型公平比较的吞吐结论，详见 [gpu_smoke.json](validation/gpu_smoke.json)。

8卡入口使用accepted train前128件、val前16件，每卡batch16，跑1步并验证／保存，再从该新checkpoint恢复至累计step2。resume时刻意给出不存在的初始化源，验证不会发生二次迁移。真实权重检查确认optimizer step为2，冻结encoder/posterior与原源一致。运行目录：

`validation/ddp/20261012T073250.503340Z/`

该目录是验证产物，不作为正式微调起点。每次调用的resolved config、Git、seed与精确命令保存在其 `metadata/invocations/`；汇总见 [summary.json](validation/summary.json)。正式命令见 [README.md](README.md)。

验证只出现现有PyTorch nested-tensor／autocast提示、DDP unused-parameter性能提示，以及恢复验证轮次边界时的一次val监测提示；随后实际验证、保存、step与optimizer恢复检查均通过。

## 容量修正（2026-10-12）

用户要求恢复旧inter的容量，只保留已确认的信息交换变化。此前未经充分说明把8层512维缩到2层128维是实现偏离；本次纠正，不用后续测试结果合理化这一改动。

| 规格 | 原refinement inter | 历史small local | 当前full-capacity local |
|---|---:|---:|---:|
| 层数／维度 | 8／512 | 2／128 | 8／512 |
| heads／FFN | 8／2048 | 4／256 | 8／2048 |
| dropout／层内激活 | 0.1／ReLU | 0／GELU | 0.1／ReLU |
| 层内结构 | self-attn＋cross-attn＋FFN | cross-attn＋FFN | self-attn＋cross-attn＋FFN |
| inter参数 | 33,683,008 | 311,616 | 34,015,296 |
| 8层Transformer主干参数 | 33,632,256 | 不适用 | 33,632,256 |

pair MLP仍产生单个Query，两张面仍各提供256个局部CNN＋XYZ token，memory共512个token；有向输出E64、classifier、refiner、几何头及全部loss不变。局部输入输出适配使完整inter比旧版多332,288参数（约0.99%）。8层本体采用与旧版相同的 `nn.TransformerDecoderLayer`、pre-norm和deepcopy堆叠方式。局部输入投影MLP的GELU沿用之前已实现形式，不与Transformer内部恢复的ReLU混淆。

初始化仍从已评估的refinement epoch10/step3993开始：严格保留全部581个非inter状态键，丢弃149个旧inter键，新inter的157个状态键随机初始化。没有从失败的2层local模型迁移权重；没有在本次修正中另外改学习率、batch、冻结范围、损失权重或refiner。

主配置显式选 `block_type=transformer_decoder`，hidden/layers引用旧顶层容量配置。历史未写block_type的配置继续构造旧cross-attention实现，防止改变已有checkpoint语义；历史继续训练单列 `train_vae_intersection_local_small_legacy`。新旧误配严格拒绝。

相关45项测试通过，覆盖两种block的双面局部输入、梯度、对象隔离、面置换、分块和bf16；另核对新层与旧层逐键形状及同权重输出一致、dropout0.1下checkpoint重算的RNG/梯度等价、历史状态严格加载和错误架构拒绝。真实GPU和训练入口检查记录在 [validation_full_capacity](validation_full_capacity/)。正式训练由用户启动，验证中的临时optimizer更新不作为训练产物。

真实GPU验证使用RTX4090、bf16和原30/28面训练对象，batch2之后重复到batch16，包含13,008个有向候选。每次完成前向、反向、梯度裁剪和Adam更新；batch16峰值allocated 10,972.9 MiB、reserved 12,448 MiB，所有检查模块梯度有限且非零。243个encoder/posterior状态张量在更新后仍与源完全相同。旧2层实际checkpoint严格加载通过；新完整状态恢复后权重逐张量完全相同，GPU输出最大绝对差4.77e-7，在atol1e-6/rtol1e-5内。详情及可复现脚本见 [gpu_smoke.json](validation_full_capacity/gpu_smoke.json) 和 [gpu_smoke.py](validation_full_capacity/gpu_smoke.py)。初次验证对GPU输出采用逐位相等过于严格，随后仅修正验证容差，模型和权重处理没有为此改变。

8卡入口验证也通过：每卡batch16，从新配置初始化并执行1step、验证、保存，再恢复到累计2step。最终checkpoint记录512维8层Transformer、157个inter状态键；495个Adam状态的step均为2，243个冻结状态张量与源完全相同。恢复时刻意给出不存在的初始化源，已确认不会二次迁移。两次进程均exit0；恢复轮次边界出现既有val/loss监测警告，之后实际验证与保存正常完成。完整命令、split和invocation元数据见 [ddp_summary.json](validation_full_capacity/ddp_summary.json)，汇总见 [summary.json](validation_full_capacity/summary.json)。这两步只验证执行与恢复，不用作正式重训起点。
