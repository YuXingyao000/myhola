# UV 边界监督

本轮保留已采用的 CNN encoder、8 层 512 维 legacy intersection、refinement、curve conditioning、classifier 和 XYZ 几何头。在最终 refined face/edge feature 后增加辅助 UVPathHead，使用原始 CAD 提取的边—面对应关系监督。设计与数据判定规则见 [PLAN.md](PLAN.md)。

这不同于此前失败的 UV **面编码器**实验。新训练从 epoch 157 / step 57354 的 conditioned CNN checkpoint 初始化，冻结 encoder/posterior，创建新 optimizer/run。新 UV 头增加 **286,532 参数、34 个 state keys**，其余 736 个权重项严格保留；完整新 checkpoint 和 resume 仍严格匹配。

## 本次新增的位置

| 文件 | 改动 |
|---|---|
| `src/brepnet/data/extract_edge_uv.py` | 从旧 NPZ 与对应 normalized STEP 构造 UV sidecar，前台进度、版本检查、可恢复运行 |
| `src/brepnet/data/edge_uv.py` | 不依赖 OCC 的标签格式、摘要与对应关系检查 |
| `src/brepnet/dataset.py` | 可选 `edge_uv_root`，连接行同步过滤与 batch 拼接；刚体旋转不改变 UV |
| `src/brepnet/models/blocks.py` | 辅助 `UVPathHead` 与可选输出 |
| `src/brepnet/models/vae.py` | UV 监督、固定对应几何监督、严格 checkpoint 初始化与可选 UV 导出 |
| `src/brepnet/train.py` | resume 清空组件初始化路径，检查标签根目录配置 |
| `configs/model/vae_intersection.yaml`、`configs/dataset/vae.yaml` | 默认关闭的新开关和标签路径 |
| `configs/train_vae_uv_boundary.yaml` | 本次训练入口，仍使用统一 trainer |

## 张量与损失

输入为 refined `E[P,64]`、owner `F_i/F_j[P,32]`。concat 后经 Linear(128,128)+SiLU，与 `t=k/15` 的 Linear(1,32)+SiLU+Linear(32,128) 相加，得到 `[P,16,128]`。经过两个独立的 pre-LN TransformerEncoderLayer（4 heads、FFN256、GELU、dropout0），再经 LayerNorm+Linear(128,4)，reshape/permute 为 `[P,2,16,2]`。原 inter 不缩层、不缩宽。

UV 只作为辅助预测和监督，不替换最终 XYZ 曲线，也不向推理输入 GT。`pred_edge_uv` 仅随已启用模型导出；原 post 不消费此字段。预测曲线的改善仍需下游实测，不能把这个支路说成精确曲面求交。

新增两个 loss，原有各项保持：

- `uv`：同一参数图内、有效点的归一化 UV L1，初始权重 **0.01**。
- `uv_anchor`：在固定 GT UV 读取预测曲面，约束预测边的相同采样点，初始权重 **0.1**。

具体为 `C_pred - I(S_pred,U_gt) - stopgrad(C_gt - I(S_gt,U_gt))` 的有效点 XYZ L1。`I` 使用 bilinear、align_corners=True，网格轴为 `[v,u]`。扣除 GT 插值残差，避免要求 16×16 网格的插值精确等于原始 B-spline。原 XYZ 重建 loss 继续约束绝对位置。

投影或周期分支无法可靠确定的位置标为无效；保留它们的原几何训练目标，不静默删 CAD。`uv_valid_fraction` 记录当前 batch 的监督覆盖率。周期曲线使用已经确定的原参数图分支，包括闭环两端可能分别为0和1；本轮不将两端无条件取模合并。

## 数据位置

```text
/mnt/d/data/deepcad_v7/{model_id}/data.npz
/mnt/d/data/deepcad_v7/{model_id}/normalized_shape.step
/mnt/d/data/deepcad_v7_uv/v1/dataset.json
/mnt/d/data/deepcad_v7_uv/v1/manifest.jsonl
/mnt/d/data/deepcad_v7_uv/v1/summary.json
/mnt/d/data/deepcad_v7_uv/v1/{model_id}/edge_uv.npz
```

sidecar 按原 `edge_face_connectivity` 的**行**保存 `[R,2,16,2]` UV；边点数组仍按显式 edge_id 索引。两者不能混用。保存两侧面参数范围、STEP面对应、源 NPZ/STEP 摘要、有效掩码、原生/投影方法、回代距离及无效原因。

先用完整16×16坐标匹配 STEP 面与旧 NPZ 编号。匹配16个边点及方向后优先读取 native pcurve；合并/拟合曲线使用明确标记的投影。匹配容差初值1e-5，投影标签回代阈值初值1e-3，均为现有归一化 XYZ 单位。它们是标签筛选规则，不是网络精度或最终 solid 合法率。

## 在 172.31.224.142 直接运行：生成

以下均在服务器登录后的终端**前台**执行。不使用 SSH 包装、nohup 或自动 hook。

```bash
cd /mnt/d/python
conda activate torch

python -u -m src.brepnet.data.extract_edge_uv \
  --data-root /mnt/d/data/deepcad_v7 \
  --output-root /mnt/d/data/deepcad_v7_uv/v1 \
  --split checkpoints/joint_boundary/validation/train_contract_audit/accepted.txt \
  --split checkpoints/joint_boundary/validation/val_contract_audit/accepted.txt
```

这会处理既有 train 46,480 件和 validation 3,495 件，保留 split 归属；本轮不生成或分析 test。程序逐件打印完成、有效点、native与projection数量；最终打印并保存 summary。默认单进程，几何生成使用 CPU，不占用训练 GPU。

可先在同一命令末尾加 `--limit 8`，之后去掉 limit 重跑会检查并复用已完成标签。按 Ctrl+C 可停止，重新执行相同命令继续；已发布 NPZ 原子保存。源摘要、生成代码、OCC版本、参数或split清单变化时拒绝混写，使用新的版本目录。缺失 NPZ/STEP、无法读取 STEP 等源错误记录为 `source_failed`，退出码为1；需要处理后重跑，不生成全无效文件掩盖源文件缺失。`partial/invalid` 表示源文件已读到，但几何标签有不确定位置，仍保留对象。

生成结束先检查 `summary.json` 的 pending/source_failed、有效覆盖率和 `manifest.jsonl` 的原因统计。判断或调整方案只使用 validation；不要看 test 个例调参数。

## 在 172.31.224.142 直接运行：训练

在标签生成并检查完成后启动。不要从上次 UV face encoder 的 last.ckpt resume。

```bash
cd /mnt/d/python
conda activate torch

python -u -m src.brepnet.train --config-name train_vae_uv_boundary \
  dataset.raw_data_root=/mnt/d/data/deepcad_v7 \
  dataset.edge_uv_root=/mnt/d/data/deepcad_v7_uv/v1 \
  dataset.train_dataset=checkpoints/joint_boundary/validation/train_contract_audit/accepted.txt \
  dataset.val_dataset=checkpoints/joint_boundary/validation/val_contract_audit/accepted.txt \
  dataset.test_dataset=checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt \
  'model.init_uv_boundary_checkpoint="/mnt/d/python/checkpoints/intersection_refinement/curve_conditioning/20261013T083729.740439Z/weights/epoch=0157-step=00057354-val_loss=0.038204.ckpt"' \
  trainer.devices=8 trainer.batch_size=16 trainer.seed=42 \
  trainer.learning_rate=2e-5 trainer.max_steps=100000 \
  trainer.wandb.enabled=true trainer.wandb.offline=false
```

checkpoint 参数的内层双引号需保留，避免 Hydra 将文件名中的 `=` 当作配置语法。训练读取test清单配置但本命令不运行test；以后正式test需要另行构造冻结标签版本，或直接使用无GT的decode接口导出几何。

新 run 在 `checkpoints/uv_boundary/candidate/<UTC>/`，权重、日志、W&B、配置/Git记录沿用统一目录。终端显示实际 run 路径与 W&B URL。8卡×每卡16、bf16、学习率2e-5；100,000步为本次配置预算，可显式调整，不代表保证在一晚内结束。

Ctrl+C 中断训练后先确认日志是否已保存 last.ckpt。续训使用同入口，增加 `trainer.resume_from_checkpoint=/绝对路径/weights/last.ckpt` 并设置 `model.init_uv_boundary_checkpoint=null`；保留原数据版本和训练配置，不能再次进行组件初始化。

新增 loss 会改变 total loss 的数值口径，不能直接把它与旧实验的 total loss 比大小来判断提升。后续仍比较同一 validation 清单上的几何误差、合法 solid 和失败统计。

## 当前状态

实现完成，正式数据集未生成、训练未启动。本次交付前台命令，由用户手动运行。

- 相关数据、loader、模型、loss、迁移和导出共130个测试用例通过：初次联合运行127项，源文件错误处理修复后重跑数据/loader31项（其中新增3项）。
- validation 前8件构造和断点恢复通过：17,152个有效标签位置，17,024个来自 native pcurve、128个来自投影；最大回代残差约5.36e-4。[数据验证记录](validation/labels.json)。这8件不代表全量标签覆盖率。
- 加载真实 adopted checkpoint，在7面和20面两个对象、124条GT连接上完成 FP32/BF16 前反向；loss及527个梯度张量有限，UV头和refiner梯度非零，encoder/posterior保持冻结。验证无GT UV输入的推理及导出。[GPU验证记录](validation/real_smoke.json)。无optimizer step，不代表训练收益。
- 最后源文件错误处理修复只改变报错路径；用最终生成器重新构造8件，全部sidecar字段与GPU验证版本逐数组一致。
- 2026-10-18，代码提交 `3d177c27` 已部署到142的 `/mnt/d/python`、`codex/vae-uv-boundary` 分支。在其 Python3.10.22 / PyTorch2.8.0+cu128 / PythonOCC7.8.1.1 / RTX5090 上，58项相关测试、相同8件生成与恢复、真实checkpoint的FP32/BF16前反向和无GT推理均通过。[远端验证记录](validation/remote_smoke.json)。原32个未跟踪文件逐个SHA核对保留；未启动正式生成或训练。
