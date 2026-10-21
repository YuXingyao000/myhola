# 单卡独立采样进程吞吐实测（2026-10-21）

## 问题与固定条件

原生成每卡一个进程，虽然显存仅约3.3 GB，但GPU利用率约45%。本次仅改变同卡
独立采样进程数，判断能否填充小batch前向间的执行空隙；不改变网络、采样协议或精度。

- 初轮GPU0、CPU线程1；补测GPU1、CPU线程2，与正式生成配置一致。均为RTX 4090，
  测量时所用卡无其他任务。
- 原版 `Diffusion_svr_sq30_1500k.ckpt`，每进程独立模型与随机状态。
- 同一validation图片 `00938923` 的 `svr_imgs[0]`，原版DINO、VAE。
- 原生DDPM 1000步，batch=1，FP32，matmul/cudnn TF32关闭。
- 每进程先完整生成一个warmup候选；所有进程就绪后同时计时，各生成3候选。
- 不包含加载、warmup、磁盘输出和CPU后处理时间。本次候选不进入正式数据。

## 正式线程配置实测（GPU1，CPU线程2）

| 每卡进程 | 计时候选数 | 耗时（秒） | 候选/秒 | 相对单进程 | 峰值显存（MiB） | 平均GPU利用率 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 15.193 | 0.1975 | 1.000× | 3,225 | 47.2% |
| 2 | 6 | 16.674 | 0.3599 | 1.822× | 6,491 | 99.3% |

正式线程配置下，2进程相对单进程吞吐提升82.2%。没有在CPU线程2下重测3进程。

## 初轮探索实测（GPU0，CPU线程1）

| 每卡进程 | 计时候选数 | 耗时（秒） | 候选/秒 | 相对单进程 | 峰值显存（MiB） | 平均GPU利用率 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 15.390 | 0.1949 | 1.000× | 3,225 | 45.5% |
| 2 | 6 | 16.683 | 0.3596 | 1.845× | 6,491 | 99.6% |
| 3 | 9 | 23.439 | 0.3840 | 1.970× | 9,756 | 98.0% |

此组2进程相对单进程吞吐提升84.5%；3进程相比2进程再提升6.8%。短测未做多轮统计，
不能将差值当作稳定收益置信区间，也不能直接当作包括OCC的端到端速度。

建议先采用**每卡2个独立worker**。此时GPU已接近满载；第三个worker的收益小，
同时增加模型显存、调度与CPU资源。按正式线程2的实测，8卡×2worker纯采样吞吐
外推约2.88候选/秒，相对旧4卡×1worker约3.64倍；实际总速度还取决于CPU后处理
和有界队列。

## 一致性与边界

单进程与2/3进程使用相同的3个rank0 seed，得到的原始 `[30,32]` latent及全部raw
字段逐bit相同。2进程与3进程中重复的另外3个seed也相同。共9次重复seed对照全部
通过；哈希包含数组dtype、shape及原始字节，见 [GPU_WORKERS.json](GPU_WORKERS.json)。

另在GPU1按正式配置CPU线程数2复现同图seed931000，其latent和全部raw字段也与
GPU0、CPU线程数1的结果逐bit相同；检查结果见同一JSON的 `cross_gpu_check`。

GPU1、CPU线程2完整补测的9次候选，其latent和全部raw字段也都与GPU0线程1下
相同seed结果逐bit相同；补测计时、资源占用和对照见JSON的
`production_cpu_threads_2_benchmark`。

独立进程不共享模型与RNG，正式任务应按图片分片，避免同一图片GT cache及同一候选
目录被多个worker同时写入。模型加载时的历史源码缓存、日志和provenance文件也需
采用独立worker名。上述基准候选独立于正式生成结果。

## 正式调度迁移与恢复验证

用户授权四卡扩八卡后，选择8卡×2个独立worker，CPU后处理16→32、队列32→64。
旧任务先暂停GPU、排空CPU候选，再退出；原有5,336张完整训练图片和4,842对保留。
4个正在生成且未写完的候选沿用原协议记unknown，不重抽，见
[RESUME_PRESERVATION.json](RESUME_PRESERVATION.json)。

新入口为每个GPU worker独立命名日志、源码缓存和provenance。恢复先核验已有输入、
GT缓存、生成输出hash与评价输入，再重建计数/manifest；完整图片不再加载模型或重放OCC。
累计运行时间保留，另计本次恢复耗时。`PAUSE`文件支持在图片边界暂停并排空CPU。

原`metadata/protocol.json`字节与fingerprint保持不变。显式
`--accept-scheduling-update`仅接受`preference_generate.py`/`preference_io.py`
这两个经审阅的编排文件变化；其余源码与全部科学参数仍严格匹配。每次恢复追加保存
runtime配置、Git、完整source hashes、源码快照与diff，而不是把新源码伪装成原版本。

35项相关测试通过。两张真实validation图、每图K8的同卡双worker完整流程通过；
16个候选的latent、raw、映射文件hash与原双GPU运行完全相同。恢复耗时5.16秒，
计数、累计时间、pair选择及原protocol字节不变，见
[PIPELINE_VALIDATION.json](PIPELINE_VALIDATION.json)。小规模结果仅验证实现。

将多个候选直接合并成大batch可能有额外收益，但会改变本轮已固定的随机数流及
浮点计算路径；本次未测试或应用。半精度、减少DDPM步数同样未采用。

## 复现

```bash
PYTHONPATH=/mnt/d/python /mnt/d/anaconda3/envs/torch/bin/python \
  checkpoints/diffusion/preference/performance_20261021/benchmark_gpu_workers.py \
  --gpu 0 --output /tmp/hola-gpu-workers-20261021-repeat

# 正式生成的CPU线程配置，只比较1与2个独立采样进程。
PYTHONPATH=/mnt/d/python /mnt/d/anaconda3/envs/torch/bin/python \
  checkpoints/diffusion/preference/performance_20261021/benchmark_gpu_workers.py \
  --gpu 1 --cpu-threads 2 --workers 1 2 \
  --output /tmp/hola-gpu-workers-threads2-20261021-repeat
```

脚本每轮独立启动进程，输出保存于指定临时目录，不覆盖已有正式候选。
