# Encoder：让边内容直接进入 GAT 消息

本轮检验：在原有 attention 权重之外，直接把边特征写入 face 消息，能否改善边几何和最终 solid。完整固定协议见 [PLAN.md](PLAN.md)。实现与 smoke test 通过不代表研究假设成立；结果需候选与对照训练、固定下游评估后判断。

## 新增内容与保留部分

- `src/brepnet/models/blocks.py` 的 `EdgeValueGATv2Conv` 保留原 GATv2 score 和 face value；新增 `MLP([F_target,F_source,E])`，按同一个 alpha 聚合后加到原输出。MLP 为 `128→64→SiLU→32`，末层零初始化。5 层共新增 **51,680** 参数。
- 原 5 层／32 维／单头 GAT、自环与 mean edge attribute、外部 residual／activation 保留；原 intersection、conditioning、refinement、几何头及 loss 权重保留；point-feedback 关闭。
- `src/brepnet/models/vae.py` 负责开关和严格权重迁移；`configs/model/vae_intersection.yaml` 默认关闭 `encoder_edge_value`。本轮配置是 `configs/train_vae_encoder_edge_value.yaml`，启用开关并设 `hidden_dim: 64`。
- 候选使用 `model.init_encoder_edge_value_checkpoint`，只允许新增 `value_net` 参数；对照关闭开关并通过 `model.checkpoint` 加载同一 curve-conditioning 权重。两者都全量解冻 Encoder、posterior、Decoder，KL sum×1e-6 重新参与梯度；使用新 Adam。

两组均为 8 GPU×batch16、lr2e-5、bf16、seed42、40 epoch／14,520 optimizer steps，每 epoch 完整验证。先候选再对照，不并抢八张卡，无自动重试。原始 32 维 latent 形状不变，但训练后语义会改变，**旧 latent cache 和 diffusion checkpoint 不能默认兼容**。

## 启动、查看与停止

在仓库根目录、已有 torch 环境中运行。`start` 在前台管理整个顺序任务；正式后台启动由终端／任务管理器独立保留该进程。启动前核对 GPU 空闲、真实 batch 验证和固定协议。

```bash
/mnt/d/anaconda3/envs/torch/bin/python checkpoints/encoder_edge_value/run_pair.py start
/mnt/d/anaconda3/envs/torch/bin/python checkpoints/encoder_edge_value/run_pair.py status
/mnt/d/anaconda3/envs/torch/bin/python checkpoints/encoder_edge_value/run_pair.py stop
```

`stop` 核对 Linux boot ID、PID、进程启动时间，向当前训练进程组转发 SIGINT，并取消尚未启动的对照。中断候选后不会自动开始对照；失败也不会重试或继续下一组。退出码为零仍需 `last.ckpt` 的 `global_step` 达到14,520，才能开始下一组。不要依赖历史 PID。`last.ckpt` 只保证最近已保存 epoch 的状态，不保证包含中断前最后一步。

记录在 `checkpoints/encoder_edge_value/metadata/active_pair.json`，每次启动另保留 `<UTC>.json`；含两组完整 argv、源 checkpoint 与 split SHA-256、Git、PID／进程组、退出码与输出目录。启动器只调用既有 `python -m src.brepnet.train`，不新增 trainer。目录：

```text
checkpoints/encoder_edge_value/{candidate,control}/<UTC>/
  weights/       checkpoint
  logs/          console.log 与训练日志
  wandb/         W&B 运行文件
  metadata/      统一训练入口的配置、Git 与命令溯源
```

启动器中的数据与 accepted split 路径明确固定，并校验源 checkpoint 哈希及 split 数量／重复项；每组开始前重新核对 Git、源权重及 split 与启动时一致，改变则中止队列。运行期间应保持本 worktree 的 commit 和未提交文件状态不变；不记录环境变量或凭证。运行产物及 metadata 已由仓库规则忽略，研究文档与配置受版本管理。

## 恢复单组训练

新增的 `resume` 管理单组续训，继续使用上述 `status` / `stop`，不会排队启动另一组。必须显式填写组名与**累计**步数、epoch 上限；默认读取本次 `evaluation_20261015/training_audit.json`，也可用 `--audit` 指定后来完成的审计。

```bash
/mnt/d/anaconda3/envs/torch/bin/python checkpoints/encoder_edge_value/run_pair.py resume \
  --arm candidate --max-steps 29040 --max-epochs 80
```

这条命令从审计中已固定的 `last_saved` 续训；当前为step14,520/epoch39，目标29,040步表示追加14,520步。它读取该run第一次调用保存的完整resolved config，保留八卡、batch、学习率、loss、架构与W&B run ID，并通过Lightning完整恢复optimizer和训练计数。它不根据test结果改选checkpoint，也不提供从best自动回退。

启动前会核对审计权重SHA、模型/数据配置、optimizer/fit-loop状态以及训练源码，并将本轮已评估的best和last独立复制到同run的`weights/evaluated_snapshots/<SHA>.ckpt`。随后从固定last副本恢复；新记录保存原路径→归档路径→SHA的映射，旧评估协议不改写。这样后续覆盖`last.ckpt`或更新top-k不会丢失已评估权重。新调用单独保存启动元数据、stop marker、`console_resume_<UTC>.log`和Hydra配置，不覆盖旧启动记录。

如果审计里的权重、配置或源码已经变化，启动器会拒绝继续。预算不足或未明确指定组名也会报错；不会静默开始一个新run。

从该组的 `metadata/invocations/<UTC>/config.yaml` 恢复完整原配置；将下方路径替换为实际组目录。候选／对照的开关由各自保存的配置保留，不要互换权重。统一 trainer 在 resume 时清理初始化字段，恢复同一 run 与 optimizer，不再从源权重初始化。

```bash
/mnt/d/anaconda3/envs/torch/bin/python -m src.brepnet.train \
  --config-path /absolute/run/metadata/invocations/<UTC> --config-name config \
  'trainer.resume_from_checkpoint="/absolute/run/weights/last.ckpt"' \
  'trainer.run_dir="/absolute/run"' \
  'hydra.run.dir="/absolute/run"' \
  hydra.output_subdir=metadata/hydra hydra.job.chdir=false \
  trainer.max_epochs=10000 trainer.max_steps=14520
```

恢复命令放宽 epoch 上限以避开 Lightning 恢复时的 epoch 计数提前结束，仍以 **14,520 总 optimizer steps** 限制原预算；已到该步数的 run 不会再训练，增加预算需另行确定。恢复单组不会恢复启动器的候选→对照队列；后续对照是否需要继续，由中断位置与剩余预算决定。checkpoint 只按同预算窗口内最小 `val/loss` 选取，不能用 test 选择。最终需比较边几何尾部、owner 对应误差、原生 solid 与失败类型，不凭平均训练 loss 宣布有效。
