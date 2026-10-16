# 局部注意力 intersection：与上一版 refinement 比较

固定测试清单 2,417 件（原 2,424 件，7 件数据合同不支持），两种归档 latent 输入保持逐值相同。主对照是上一版 `intersection`，本版标为 `local`；原 HoLa 只提供已审计的 latent 来源。协议见 [PLAN](PLAN.md)，训练审计见 [training_audit.json](training_audit.json)，最终统计见 [REPORT](REPORT.md)，失败全集见 [失败诊断](../failure_census_20261012/README.md)。

权重按验证集 callback 固定为 epoch19 / step7260，`val/loss=0.5074835419654846`；`last.ckpt` 与它字节相同。没有根据测试表现重选权重。实际记录 20 个 epoch，不等于配置训练预算已耗尽。训练曲线见 [PNG](visualizations/training_curves.png)，不能仅凭部分几何分项回升认定整体过拟合。

## 复现

在仓库根目录使用已有 `torch` 环境。下面是本次实际入口；完整重跑使用新的输出目录，保留原运行。两种 mode 分别执行，不能把 archived latent 的无对应评估混入 posterior mean 的面对应指标。

```bash
conda activate torch
cd /mnt/d/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

python -m tools.intersection_evaluate_inference \
  --method local \
  --checkpoint 'checkpoints/intersection_local/frozen/20261012T073904.575690Z/weights/epoch=0019-step=00007260-val_loss=0.507484.ckpt' \
  --legacy-checkpoint /mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt \
  --reference-root checkpoints/joint_boundary/comparison_20261011 \
  --data-root /mnt/d/data/deepcad_v7 \
  --split-list checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt \
  --output-root checkpoints/intersection_local/evaluation_20261012 \
  --mode posterior_mean --num-shards 8 --shard-index 0 --device cuda:0
```

每个 mode 对 `shard-index=0…7` / `cuda:0…7` 各运行一次，等待全部分片结束；另一个 mode 为 `archived_latents`。分片不通过 DDP sampler 补齐 ID。实际命令与日志在 `metadata/orchestration/`。FP32、关闭 TF32、seed42，输入使用原归档数组；mean 重新编码仅作审计。

`raw_predictions/<mode>/intersection` 只读指向上一版对应目录；`post/<mode>/intersection` 同理，来源记录在 [reference_links.json](reference_links.json)。结果复用保留原方法名、协议、逐对象状态和时间，核对算法、预算、清单与 raw SHA；它不是本次重新执行的上一版耗时。

两个 raw mode 分别计算：

```bash
python -m tools.vae_compare_metrics \
  --baseline-method intersection --candidate-method local \
  --comparison-root checkpoints/intersection_local/evaluation_20261012 \
  --data-root /mnt/d/data/deepcad_v7 \
  --split-list checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt \
  --mode posterior_mean --workers 16
```

只对 posterior mean 执行两种后处理，mode 分别为 `fit_only` 和 `optimized`：

```bash
python -m tools.vae_compare_post \
  --baseline-method intersection --candidate-method local --methods local \
  --reference-root checkpoints/intersection_refinement/evaluation_20261011 \
  --comparison-root checkpoints/intersection_local/evaluation_20261012 \
  --data-root /mnt/d/data/deepcad_v7 \
  --split-list checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt \
  --mode optimized --workers 32 --timeout 180 \
  --max-optimize-iter 200 --original-test-count 2424
```

生产 post 未修改，CPU 每进程单线程，`drop_num=0`，优化名义 200 次、原算法自适应上限 600 次。180 秒是整个对象子进程预算，包含后续几何测量；若 STEP 已确认为合法 solid 后测量超时，不抹去 solid 成功。中断续跑只在协议相同时加 `--resume`，不删除或重算失败终态来提高成功率。

全部完成后生成报告：

```bash
python -m tools.local_evaluate_report \
  --comparison-root checkpoints/intersection_local/evaluation_20261012

python checkpoints/intersection_local/evaluation_20261012/build_independent_review.py
```

独立复核脚本从 CSV、NPZ 和原来源重算，不调用评估汇总函数。结果在 [independent_review.json](independent_review.json)。原始几何均值为主指标；`diagnostics/` 中的分布、配对改善比例、极端值敏感性只作解释，不替换全集结果。

## 范围

后验均值重建用于检查 decoder；固定 diffusion latent 只检查这批既有输入的解码变化，没有重新采样 diffusion，也不是完整照片条件重建测试。新模型增加了训练曝光且 intersection 重新初始化，本次不能视为等预算架构消融。所有失败、缺失和超时留在固定分母；STEP 几何同时报告各自与共同可计算集合，不能仅看成功子集。
