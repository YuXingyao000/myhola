# 2026-10-11：实现、运行验证与首次测试

实现阶段状态（正式训练前）：已实现、已验证前反向/迁移/多卡/续训/导出。后续训练和完整测试结果见本页末节。分支 `codex/intersection-refinement`，起点 `5c9a2d27`。本次按已讨论的方案保留原 `inter` 与线性 classifier，仅增加一轮几何反馈和面/边特征残差；没有独立拓扑预测器。

## 实际检查结果

- 新增 52 项测试通过，覆盖连续几何距离及梯度、候选枚举、对象隔离、排列一致性、双向 logits、零残差、损失/GT 对应、旧权重迁移与冻结。
- 24 项现有相关测试通过：训练设备合同、run 管理和 joint decoder。合计 76 项通过。
- 使用真实旧 checkpoint 与两个 29/28 面对象、1568 个有向候选，单卡 bf16 完成两步优化。残差头第一步有梯度，几何读取模块在第二步获得梯度，冻结 encoder 无梯度，所有 loss 有限。峰值 allocated 显存约 2388 MiB。这是该小批量的测量，不是正式训练吞吐或最大显存保证。
- 8 张 RTX 4090、每卡 batch=2，真实 CLI 完成初始两步训练、验证和保存，然后续训到 step 4、step 6。
- 最新 `weights/last.ckpt` 的 `global_step=6`，已保存优化器状态的 step 均为 6。
- 从此 checkpoint 单卡测试两个 batch，导出 4 个对象，`legacy_half_edges` 表示、整数局部索引、有限几何和对称连接概率均通过检查；`tools.vae_compare_post.validate_raw_prediction` 接受这些文件。没有运行 OCC，因此不代表这些对象已形成 solid。

紧凑报告：[summary.json](validation/summary.json)、[gpu_smoke.json](validation/gpu_smoke.json)。所有短运行 seed=42，W&B 关闭；正式配置 W&B 开启。测试环境为 `/mnt/d/anaconda3/envs/torch/bin/python`，Torch 2.4.1+cu124、PyTorch Lightning 2.5.0.post0。

## 资产与可复现输入

- 数据：`/mnt/d/data/deepcad_v7`。
- 初始化：`/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt`，严格加载完整旧 VAE，再转移模块权重。
- 正式配置使用 `checkpoints/joint_boundary/validation/{train,val,test}_contract_audit/accepted.txt`，数量分别为 46480/3495/2417。训练 split 原定 46627 件，其中 147 件被合同审计拒绝；测试 split 的 7 个合同拒绝项也不能从后续全体比较的原定分母中悄悄删除。
- 单卡大面数检查：验证集 `00011043`、`00003999`。
- 多卡检查：已保存的 [smoke_train.txt](validation/smoke_train.txt) 32 件和 [smoke_val.txt](validation/smoke_val.txt) 16 件；文件哈希见 summary。
- 运行目录：`validation/ddp8/20261011T115022.529848Z/`；权重、逐对象几何、日志、每次调用的配置/Git/命令元数据保留本机，不提交 Git。

## 验证中修复的续训问题

最初一次续训继承了已到达的 epoch 上限，未执行优化；之后显式提高 `max_epochs=10`，才确认从 step 2 继续到 4。带 `=` 的 best checkpoint 路径需要给 Hydra 的整个值加引号；README 使用 `last.ckpt` 避免这一命令行歧义。

从 best 权重续训时，Lightning 原行为把最近权重写成 `last-v1.ckpt`，使旧 `last.ckpt` 停留在 step 2。新增 `trainer.checkpoint_enable_version_counter`，全局默认仍为 true；仅本实验设为 false。随后从 step 4 恢复到 6，确认固定路径 `last.ckpt` 已更新，同时带 epoch/step 的 top-3 文件仍保留。

## 短运行复现

在 README 的环境下执行；每次初始运行新建独立时间目录。

```bash
python -m src.brepnet.train --config-name train_vae_intersection \
  model.init_legacy_checkpoint=/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt \
  dataset.raw_data_root=/mnt/d/data/deepcad_v7 \
  dataset.train_dataset=checkpoints/intersection_refinement/validation/smoke_train.txt \
  dataset.val_dataset=checkpoints/intersection_refinement/validation/smoke_val.txt \
  dataset.test_dataset=checkpoints/intersection_refinement/validation/smoke_val.txt \
  trainer.exp_name=intersection_refinement/validation/ddp8 \
  trainer.devices=8 trainer.batch_size=2 trainer.num_workers=0 \
  trainer.max_steps=2 trainer.max_epochs=2 \
  trainer.limit_val_batches=1 trainer.num_sanity_val_steps=0 \
  trainer.wandb.enabled=false
```

续训移除 `model.init_legacy_checkpoint`，增加 `trainer.resume_from_checkpoint=/absolute/run/weights/last.ckpt`，设置 `trainer.max_steps=4 trainer.max_epochs=10`；再将累计上限改为 6。单卡导出使用相同 split，追加 `trainer.evaluate=true trainer.devices=1 trainer.limit_test_batches=2 dataset.is_aug=0`。正式训练、测试和 loss-only 对照命令见 [README](README.md)。

## 尚待研究验证

默认冻结 encoder/posterior，保留原 latent 定义；这不意味着已有 diffusion 对新 decoder 的输出质量已经验证。先比较固定 posterior-mean VAE 重建，再检查 diffusion 生成 latent 的解码，二者输入分布和结论分开。

本轮没有修改面去重、环排序或 vertex 预测，也没有保证闭环/solid。新旧比较须固定 accepted split、原定分母、face 去重策略、后处理版本和优化预算。`model.refinement.enabled=false` 可隔离反馈模块的作用，但仍包含新损失和对称化，不等同于未经修改的 legacy baseline。正式对照尚未运行。

## 后续性能诊断

用户启动正式配置后报告吞吐较低。检查到日志为每轮 2905 步、约 0.80 step/s；检查时该进程已经停止。保持网络与损失定义，测量更大分块、关闭候选 activation checkpoint 与更大 batch；8 卡每卡 batch=16 已完成短运行。建议覆盖参数、实测吞吐与限制见 [性能记录](validation/performance/README.md)，没有修改训练配置默认值。

## 用户中断后：首次完整测试

用户因验证集 loss 基本不再下降而中断训练，并要求测试。本轮没有续训或修改模型，固定选择验证最优 epoch 10 / step 3993；last 为 epoch 17 / step 6534。验证最优之后 7 轮未刷新。训练审计、协议、输入审计及具体命令见 [评估目录](evaluation_20261011/README.md)。

在相同 2417 件、相同 posterior mean、相同后处理算法和预算下：面 XYZ MAE 下降 2.11%，边平方 CD 下降 12.90%。直接拟合后的原生合法 solid 为 1058→1062，基本持平；加入几何优化后为 2048→2113（84.73%→87.42%），107 件救回、42 件退步。原始清单另外 7 件合同不支持样本保留在报告的 2424 总体分母。

固定旧 diffusion latent 重放时，面/边平方 CD 分别增加 0.30%/1.24%。因此这轮证据支持重建与优化后 solid 有所改善，不能宣称生成端已提升，也不能把效果单独归因于反馈块：微调、新损失及候选规则同时变化。尚未运行正式 loss-only 对照。

完整表格、逐对象成功转换及图表见 [REPORT](evaluation_20261011/REPORT.md)。两件 optimized 任务在几何采样阶段超时，其中一件已通过原生 solid 检查；实体有效性保留已确认结果，几何覆盖率如实记为不可计算。GPU mean 复核门槛修订与保留的首轮记录见同目录 numerical_audit.json，不改变两模型实际使用的同一份输入 latent。
