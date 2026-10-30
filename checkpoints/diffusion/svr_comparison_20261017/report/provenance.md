# SVR 比较的权重、源码与输入溯源

本记录只说明比较对象是谁、加载是否严格、输入与 AE 是否一致，不把接口验证当作生成质量结论。
审计于 2026-10-17 在源机以 CPU 完成；没有启动训练或 GPU 推理。
完整比较协议见 [PLAN](../PLAN.md)，紧凑证据见 [audits](audits/manifest.json)。

## 已保存 checkpoint 的身份

下表中的 step 和 epoch 均直接读取 checkpoint，未从文件名或训练日志推断。

| 对象 | 文件 | global step | epoch | 完整 state key 数 |
|---|---|---:|---:|---:|
| 历史 HoLa SVR | `/mnt/d/ckpt/HoLa/Diffusion_svr_sq30_1500k.ckpt` | 1,500,000 | 82 | 1375 |
| 当前 best | 当前 run 的 `epoch=0002-step=00054639-val_loss=0.042497.ckpt` | 54,639 | 2 | 2353 |
| 当前 last | 当前 run 的 `last.ckpt` | 182,130 | 9 | 2353 |

当前 run 为 `checkpoints/diffusion_layerwise_svr/20261016T143450.891057Z/weights/`。
历史 checkpoint 保存的实验名为 `1224_cond_svr_sq30_aug`，Lightning 版本为 2.4.0。
其 callback 记录的最小 `Validation_Loss` 是 step 700,000 的 0.0320410，所提供 step 1,500,000 的记录为 0.0328852。
因此本次称其为“用户指定的历史 SVR checkpoint”，不称为已核实的最低验证 loss 或最佳重建版本。
详见 [历史 checkpoint 元数据](audits/historical_checkpoint.json)。

## 源码事实：历史网络和“CLIP”含义

历史模型以固定 Git commit `c2660c00c6a9c8f9f82bd5b9f4d3c774fa38d361` 的
`src/brepnet/diffusion_model.py:Diffusion_condition` 和
`src/brepnet/model.py:AutoEncoder_1119_light` 恢复。
这是已经严格验证兼容的推理源码；checkpoint 没有保存 Git commit，不能据此认定训练时恰好使用该 commit。

该源码的图像主线使用 torch.hub `dinov2_vitl14_reg` 的全局 CLS 特征：1024 维经可训练投影变为 256 维，
复制到各 face slot 后与 768 维 noisy latent 特征拼接。24 层 TransformerEncoder 的总宽度为 1024，
末端输出每面 32 维 epsilon。它不是当前的 24 层逐层 image cross-attention，也不是后来的 patch-memory alignment 架构。
saved config 的 `cached_condition=true` 属于训练数据读取；原生 `extract_condition` 在没有 `img_features` key 时会直接编码 `imgs`，无需改模型配置。

历史 commit `c2b5bdf6b564946014a21273d4240c05d0bfc9dc` 的标题是 `adding clip`，
其实际改动包括训练器的 `gradient_clip_val=0.5`，不能把这一标题解释为加入 CLIP 图像编码器或对比学习。
后来的 commit `be94bbea8a6eadc49ef7539e2ea0c13525b71a72` 才能明确读到
`clip_style_symmetric_infonce`、图像/几何 alignment heads 与前端 cross-attention。
指定的 1375-key SVR checkpoint 不匹配这一后期架构。
单凭没有额外 state key 仍不能排除未记录的无参数训练损失；用户记忆中的“CLIP 最佳版本”需另用对应权重和训练记录确认。
另外找到的 alignment 候选及其训练记录见 [独立候选审计](audits/alignment_checkpoint_audit.json)，不与这里的 1,500,000-step checkpoint 混称同一模型。

## 运行证据：严格加载与接口检查

[legacy_runtime.py](../legacy_runtime.py) 从固定 Git 版本提取必要源码，并使用本地 DINO 源码以 `pretrained=False` 构造；
DINO 和 AE 权重全部来自完整 diffusion checkpoint，没有下载或替换预训练权重。
构造时只取消外部 AE 初始化路径，并补齐历史构造器需要、但不影响单图分支的 `is_aug` 字段。
使用 `strict=True, assign=True` 加载全部 1375 个 tensor，missing/unexpected 均为零。
Scheduler 按原配置重新构造 CPU 常量，检查全部参数与 buffer 已离开 meta device。
历史 `src/shared` 导入上下文结束后，当前模块对象、`sys.path` 和历史工具曾修改的 Tensor repr 均恢复。

CPU 接口检查输入只有 `conditions.imgs=[1,1,3,224,224]` 与 `img_id=[1,1]`，没有图像缓存、GT latent 或拓扑。
原图编码得到 `[1,1,256]`，一次去噪得到 `[1,30,32]`；均为有限 FP32 数值。
这验证加载和单步接口可运行，不代表完成 1000 步采样或证明重建质量。
证据见 [加载检查](audits/legacy_runtime_validation.json) 与 [CPU 前向检查](audits/legacy_cpu_forward_validation.json)。

## 运行证据：五份来源中的 AE 权重相同

比较直接读取实际保存 tensor，按完整名称映射检查全部 704 个 AE 参数/缓冲区，未根据配置路径推断相等：

- 历史 SVR 的 `model.ae_model.*` 与 `/mnt/d/ckpt/HoLa/AE_deepcad_1100k.ckpt` 完全相同。
- 历史 SVR 的同一组 tensor 与 `/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt` 完全相同。
- 当前 best、last 的 `model.autoencoder.*` 和 `model.latent_provider.autoencoder.*` 各 704 个 tensor，均与上述指定 AE 完全相同。

当前两份 checkpoint 的两处别名都分别检查了 key 集、shape、dtype、`torch.equal` 与内容 SHA256；缺失、额外、形状/dtype/内容不一致均为零。
共同 AE 内容摘要为 `1d3a601b126bde86ac22cb876d4c4c021e006c72681814b90be688649d67d6b4`。
两个独立 AE 文件的实际 step 都是 1,045,500、epoch 8499；文件名中的 `1100k` 不替代这一元数据。
证据见 [历史 AE 比较](audits/historical_ae_equality.json) 与 [当前 best/last AE 比较](audits/current_ae_equality.json)。

权重相同不等于原生解码流程完全相同。本次仍保留各版本的 latent/geometry 去重和输出行为。
特别是历史 AE 的 `pred_face_adj_prob` 实际被写成展平的 connectivity，不能据其字段名当作 adjacency 概率使用。

## 运行证据：v6/v7 的视角 0 与坐标

对冻结的全部 100 个 validation CAD ID，直接比较了当前磁盘上的旧、新资产：

| 检查 | 结果 |
|---|---:|
| v6 `svr_imgs[0]` 与 v7 `svr_imgs[0]` 像素完全一致 | 100/100 |
| `data.npz` 的全部共同几何/拓扑字段完全一致 | 100/100 |
| 旧 cache 与 `_24` cache 的 rotation-0 `features.npy` 完全一致 | 100/100 |
| 缺失或读取失败 | 0/100 |

v6 图像数组为 64 views，v7 为 24 views；几何比较包含面/边采样点、connectivity、face adjacency 和 zero positions，
只排除 v6 `data.npz` 额外携带的图像字段。原数组逐项 equality 和再次计算的内容摘要均一致。
完整数据根目录、聚合 hash 算法与值见 [v6/v7 审计](audits/v6_v7_view0.json)。
该证据支持本次视角 0、原坐标比较；不证明其余 64/24 旋转编号等价，也不证明当前磁盘内容从历史训练至今从未变化。

## 运行证据：可恢复旧 split 的 ID 交集

历史 checkpoint 记录的 `data/list/new/` 路径在所选推理源码 commit 中已经移除，因此从更早的 Git 快照恢复核对：

| 旧快照 | 旧 train ID 数 | 旧 val ID 数 | train ∩ 当前 val 3505 | train ∩ 冻结 100 | 旧 val 包含冻结 100 |
|---|---:|---:|---:|---:|---:|
| `591528ca` | 53,947 | 3549 | 0 | 0 | 100 |
| `5aa93205` | 47,284 | 3517 | 0 | 0 | 100 |
| `a6d9bffb^`，删除路径前 | 47,284 | 3517 | 0 | 0 | 100 |

三份旧 validation 都包含当前全部 3505 个 validation ID。
证据记录了完整 commit、split 原始字节 SHA256，以及当前/冻结清单 hash，见 [split 审计](audits/historical_splits.json)。
这是 CAD ID 交集检查，不是新增的独立几何去重；checkpoint 未保存 split 内容 hash，仍无法排除历史训练机上的未提交清单改动。
本审计没有读取 test 样本或用 test 调整协议。

## 源码事实：原生随机数路径与比较边界

历史 `inference` 的起始噪声由 `torch.randn(...).to(device)` 先在 CPU 生成；
当前 `diffusion/model.py:inference` 使用 `torch.randn(..., device=device)` 直接在目标设备生成。
本次逐 ID 固定相同 seed，并保留各自原生采样。因此同一 seed 提供可复现条件，但不保证两模型的初始噪声 tensor 完全相同。
这是相同输入/采样预算下的 checkpoint 系统比较，不是共同初始噪声实验或等训练预算的架构消融。

审计 JSON 的哈希字段均注明范围：state schema hash 不包含权重值，AE 内容 hash 不等于整个 checkpoint 文件 hash，
数据聚合 hash 只覆盖冻结 100 件及明确字段。报告保留原始临时审计文件的 SHA256，省略重复的大量 tensor key 和逐件重复结构；
归档文件自身的大小和 SHA256 列于 [manifest](audits/manifest.json)。
