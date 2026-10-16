# Diffusion 训练吞吐优化

## Task

解决灰模 SVR 训练间歇停顿；优化输入流水线和 Lightning/DDP 配置，保持当前网络、损失与训练数据语义。

## Assumptions

用户观察是 8 卡不能持续工作、显存未占满。当前进程已结束，可使用空闲 GPU 做短诊断。
源码显示 pin_memory、persistent workers、预取和 CUDA non_blocking 传输已启用，不重复实现。
吞吐诊断使用当前随机初始化网络，不声称训练或重建质量改善。正式训练不自动重启。

## Inputs

- 基准源码：`f9ebe960`；用户 run：`checkpoints/diffusion_layerwise_svr/20261016T122801.320776Z/`。
- 真实训练 split、`imgs.npz:svr_imgs`、32 维 mean/std cache、冻结 VAE/DINO；不以 test 个例调整实现。
- 本机 PyTorch 2.4.1、Lightning 2.5.0.post0、8×RTX4090；数据位于旋转盘 `/mnt/d`。

## Outputs

可复现短基准脚本、紧凑性能记录、经过验证的 loader/训练配置，以及新的启动命令。
临时日志和大型产物不提交；不覆盖用户 run。

## Processing Protocol

保留 B64/卡、8 卡、bf16、seed42、同一训练 split、padding、视角0、优化器与梯度裁剪。
先记录旧路径，再比较新路径；普通 step 基准均关闭 validation/sanity/采样、logger 和 checkpoint，单独记录旧采样停顿。
每次20步、前4步warmup；记录实际样本、缓存冷热限制、线程、workers与源码版本。大 batch 仅在实测后作为另一个明确预算比较。

## Metrics

各 rank 的 host step 间隔、数据等待代理、CUDA event 时间、峰值显存、全局 samples/s；报告中位数及 p95。
GPU utilization 快照只作辅助证据；NCCL 等待也可能显示高占用，不等同有效计算。

## Decision Rule

图像/latent 合同正确且 DDP 连续训练通过后，采用减少等待、改善实际吞吐的实现。
不以显存占满作为目标，不以吞吐推断收敛、质量或最佳全局 batch。

## Edge Cases

缺失/损坏文件明确报错，不丢样本或静默换图；保留其他视角、real-photo、sketch 与旧调用的有效合同。
无 workers 时不能传 multiprocessing-only 参数。其他 VAE/learned-topology 不继承未经核验的 DDP unused-parameter 假设。
若出现 OOM/非有限数/通信异常，保留失败记录并停止对应诊断，不覆盖旧结果。

## Validation / Sanity Checks

精确对比所取 SVR 像素、归一化、缓存统计与随机padding；检查 DDP 梯度、日志全局聚合及采样开关。
验证 CPU、num_workers=0 与正式多卡配置。基准计时标明是否含数据等待/启动/日志，不跨不同协议混比。

## Implementation Constraints

仅改 diffusion 数据热路径与必要共享训练配置；默认兼容其他训练主题。保持 nn.Module/state_dict 合同。
不切换图像插值、不引入新的视觉特征或训练目标。不为速度关闭原梯度裁剪。
保留 Lightning 已有的分布式 sampler、worker seed、异步 CUDA 传输和自动优化。

## Deliverables

代码/config、验证和性能结果、原因与限制说明、用户可执行的新命令；提交仅本任务改动。

## 2026-10-16 实测后的范围补充

原 run 的 4-batch 周期停顿与 4 workers 的供数节奏吻合。128 个分散训练样本首次观察读取中位数 23.43ms，
立即重读 6.91ms；没有清空系统缓存，因此不是受控冷盘实验。首遍 latent 文件本身的中位读取耗时为 11.44ms。
短训练对照同时表明旧代码在暖缓存窗口已经能达到约 3.15step/s，不能与原 run 的混合时序直接比较加速倍数。

因此补充显式 SVR 输入暂存工具，处理 CPU 解压优化不能消除的首次 HDD 随机读取。
仅按 training/validation 原 split 原字节复制 imgs.npz 和 rotation-0 features.npy；不改变图像、latent、split 或采样。
目标根目录必须显式指定并且不存在，逐文件记录 size/mtime/SHA-256，缺失或复制期间发生变化则失败并保留不完整标记。
暂存目录包含原 split 和完整 manifest，不纳入 Git；只做当前灰模视角0训练/验证，后续开启 augmentation、photo 或 test 时不能直接沿用。
本机内存盘容量充足，抽样估计 payload 为 3.44GiB；先验证工具，再全量暂存供训练命令使用，记录实际大小与结果。
