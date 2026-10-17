# 灰模 Diffusion 训练吞吐

2026-10-16：本轮修正数据读取和 Lightning/DDP 开销，不改网络、loss、图像归一化、latent、batch 或训练预算。
协议见 [PLAN.md](PLAN.md)，性能和行为验证见 [VALIDATION.json](VALIDATION.json)。
旧代码基准为 `f9ebe960`；正式训练没有自动重启。

## 找到的问题与改动

用户 run `checkpoints/diffusion_layerwise_svr/20261016T122801.320776Z` 的日志中，
长停顿反复出现在第 1/5/9/13/17/21/25 步附近，与每卡 4 个数据 worker 的供数节奏吻合。
这支持输入等待的判断，但 stdout 时间戳不是 GPU profiler，不能单凭它量化计算时间。

1. **单张图却解压 24 张。** `imgs.npz["svr_imgs"][0]` 原先先解压整个数组。
   新 `image_loading.py` 对 C-order uint8 的首视图只读取必要字节，像素和后续预处理相同。
   其他视角保留完整读取；其他格式明确警告后回退；已读文件头/数据损坏继续报错。
   部分读取不扫描剩余视图，因此不提供未读取尾部的完整 ZIP CRC 检查；关闭开关可恢复全读。
2. **无消费者的原图被传到 GPU。** SVR 配置省略 `conditions["ori_imgs"]`；模型仍读取原归一化 `imgs`。
   调试工具需要原图时可显式设置 `dataset.return_original_images=true`。
3. **每步日志产生同步。** 旧总 loss 每 batch 都 `sync_dist=True` 且进入进度条，
   `log_every_n_steps=50` 无法消除它的每步 all-reduce/标量读取。
   新入口保留全局 `train/loss_epoch`，每 50 步记录 rank-0 的 `train/loss_step_local`；后者不能当作全局 step loss。
   子 loss 的全局 epoch 汇总、`val/loss`、选模和 checkpoint 规则不变。
4. **不必要的 DDP 图遍历。** 仅已核验的 SVR/无 topology/random padding 入口关闭 unused-parameter 搜索，
   并使用梯度 bucket view。其他模型默认仍保留原设置，不能据此假定其参数都参与训练。
5. **每次验证都由 rank 0 完整采样。** 原 sanity validation 也触发 1000 步 DDPM，用户 run 中约 19 秒，其余卡等待。
   `trainer.diffusion_validation_inference` 改为显式开启，且始终跳过 sanity。正常 validation loss 保留。
6. **首次 HDD 随机读盘。** 数据位于 `/mnt/d` 旋转磁盘。128 个分散训练样本首次观察读取中位数 23.43ms，
   立即重读 6.91ms；其中首遍 latent 文件自身约 11.44ms。没有清系统缓存，不能把首遍称为严格冷盘测量。
   因此提供原字节复制到显式 SSD/内存盘的工具，同时暂存图片和 latent。

DataLoader 的 pin memory、persistent workers、prefetch，以及 Lightning CUDA non-blocking 传输原本已经开启。
当前仍为 4 workers/卡、prefetch=4，不把增加 worker 当成机械盘吞吐的保证。主训练进程 CPU 线程显式设为 4，
同时设置 `OMP_NUM_THREADS`，防止 Lightning 子进程启动时覆盖该预算。
优化器和梯度裁剪保留：本机 Lightning 2.5 的 mixed-precision clipping 与 fused Adam 不兼容，不能直接切换。

## 实测边界

8×RTX4090，真实训练数据、冻结 DINO/VAE、每卡 B64、bf16、seed42；每组 20 步，前 4 步 warmup。
两组都关闭 logger 输出、进度条、checkpoint、sanity 和 validation，隔离普通训练 step；模型内部 loss 记录逻辑仍执行。
两次均为暖系统缓存，优化组在基准组之后运行。各 rank 的全部 batch CAD 次序一致，首个计量 batch 的
图像、latent/padding、rotation 哈希逐 rank 完全一致。原 run 的 0.40step/s 不用于计算本表提速。

| 短窗口指标 | 旧实现 | 新 SVR 配置 |
|---|---:|---:|
| 诊断显式 CPU 线程数/训练进程 | 64 | 4 |
| 全局吞吐（最慢 rank 的 host 窗口） | 约 1,615 samples/s | 约 1,818 samples/s |
| optimizer step/s | 3.15 | 3.55 |
| CUDA event 中位耗时，各 rank | 303.5–305.0ms | 271.8–274.5ms |
| 峰值 allocated 显存/卡 | 10.62–10.63GiB | 9.82GiB |

普通 step 短窗口吞吐提高约 13%。这不是完整长跑的速度保证：仅有 16 个计量 step，
未清系统缓存，warmup 边界未做设备同步，host 窗口可能含前一步未完成工作。
旧 run 未记录每个训练进程的实际 CPU 线程数；表中旧实现的 64 是本次诊断显式设置，不据此推断旧 run。
CUDA event 包含通信依赖/提交空隙，不能解释为纯 GPU 算力时间；loader wait proxy 也可能与 GPU 重叠，不能直接相加。
脚本提供可选 warmup 边界同步供后续严格比较；不逐 step 加 barrier。

首视图 helper 的同进程暖缓存中位读取时间为 5.048→0.348ms（64 次各）；这只是图片读取，不是端到端训练倍数。
显存下降来自去掉无用输入与减少梯度副本，并不表示训练变慢。当前保留全局 B512；加大 batch 会改变训练与样本曝光预算，
本轮没有借填满显存改变它。接口/吞吐验证不代表重建质量改善。

## 输入暂存与启动

暂存工具 [stage_svr_inputs.py](../stage_svr_inputs.py) 只使用标准库，原样复制 train/validation 指定的
`imgs.npz` 与 `{ID}_0/features.npy`，保留 split 原字节，记录每个文件来源、size/mtime 和 SHA-256。
缺文件或复制期间源文件变化会报错；已有目标目录拒绝覆盖；只有 `INCOMPLETE` 消失且 `manifest.json` 完成才可使用。
源变更检测使用复制前后的 size/mtime 检查；该工具不提供源数据并发写入时的原子快照，应从稳定的数据资产准备暂存。
目标盘需有足够空间和 inode。以下目标为当前机器显式内存盘路径，重启/容器重建后需要重新准备：

本次已准备 `/dev/shm/hola-svr-20261016`：46,627 个训练 CAD、3,505 个验证 CAD，无交集；
共 100,266 个文件（含两份 split），payload 3,845,664,374 bytes（约 3.58GiB），复制约 194 秒。
全部目标文件已重新读取并核对 SHA-256；实际 loader 的训练 8 个和验证 4 个样本，完整 batch、padding 与 RNG 状态均与原根完全一致。
这次只准备输入并做短诊断，没有启动正式训练；目录仍在时直接执行下方训练命令，不要重复暂存到已有目标。

```bash
python checkpoints/diffusion/stage_svr_inputs.py \
  --train-split src/brepnet/data/list/deduplicated_deepcad_training_7_30.txt \
  --val-split src/brepnet/data/list/deduplicated_deepcad_validation_7_30.txt \
  --condition-root /mnt/d/data/deepcad_v7_cond \
  --latent-root /mnt/d/data/ae_cache/1119_deepcad_aug1_11k_24 \
  --dst-root /dev/shm/hola-svr-20261016 \
  --workers 4
```

已完成暂存后，从仓库根目录使用现有 torch 环境开始一个新 run：

```bash
/mnt/d/anaconda3/envs/torch/bin/python -m src.brepnet.train \
  --config-name train_diffusion_layerwise_svr \
  trainer.devices=8 trainer.batch_size=64 trainer.seed=42 trainer.max_steps=10000000 \
  dataset.raw_data_root=/mnt/d/data/deepcad_v7 \
  dataset.condition_data_root=/dev/shm/hola-svr-20261016/conditions \
  dataset.cached_latent_root=/dev/shm/hola-svr-20261016/latents \
  dataset.train_dataset=/dev/shm/hola-svr-20261016/splits/train.txt \
  dataset.val_dataset=/dev/shm/hola-svr-20261016/splits/validation.txt \
  model.autoencoder.checkpoint=/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt
```

仍由 Lightning 建立分布式 sampler、播种 worker、自动优化和多卡通信；不要在外层再启动 8 份相同命令。
默认 run 为 `checkpoints/diffusion_layerwise_svr/<UTC 时间>/`，权重和日志遵循统一目录合同。
该暂存只包含灰模 train/validation 的 rotation0；开启 photo、rotation augmentation 或 test 前需显式换回原始根或准备相应数据，
不要用缺失文件过滤的结果冒充完整 test。直接用 HDD 也能运行上述配置，但首次寻道停顿未必消失。

当前改动不改变任何神经网络 state key，已有本架构 checkpoint 仍可 resume；旧 fuser 架构的限制仍然适用。
关闭验证采样会减少随机数消耗，故即使 seed 相同，也不承诺与旧完整 run 的逐步训练轨迹 bitwise 一致；输入与训练目标的定义不变。
用户此前短 run 未保存 checkpoint，因此本机命令开始新 run，没有假装续训。

2026-10-17 更新：按用户要求，上面的新 run 命令现采用 10,000,000 总步数，由用户手动停止。后续正式运行已完成此前设置的 100,000 steps，
其 `last.ckpt` 可续训；不要为增加预算重新初始化，续训命令与曲线记录见 [训练预算说明](../training_budget/README.md)。

## 复现诊断

[benchmark_training.py](../benchmark_training.py) 用 `torchrun` 启动，每 rank 保存时序与样本哈希，不保存训练权重。
`--source-root` 必须指向所测源码和配置，支持独立 snapshot，避免基准误导入当前代码。
原始 rank traces 位于被 Git 忽略的 `checkpoints/diffusion/performance/`；紧凑验证报告保留版本管理。
关联检查共 100 passed、1 skipped（CPU 模式跳过 CUDA 专属检查）；另已完成真实 8 卡前后各 20 步。
两 rank 的轻量 CPU DDP 检查确认 epoch loss 全局聚合，而 step local loss 只在 rank 0 记录。

```bash
/mnt/d/anaconda3/envs/torch/bin/torchrun --standalone --nproc_per_node=8 \
  checkpoints/diffusion/benchmark_training.py \
  --source-root /mnt/d/python --source-commit "$(git rev-parse HEAD)" \
  --report-dir /tmp/hola-svr-throughput-new \
  --steps 20 --warmup 4 --batch-size 64 --workers 4 --torch-threads 4 \
  --strategy config --cache-state 'warm OS cache; not evicted'
```

相关实现依据：本机 PyTorch 2.4.1 / Lightning 2.5.0.post0 源码及
[PyTorch performance tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)。
