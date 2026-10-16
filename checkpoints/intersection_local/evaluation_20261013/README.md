# 完整容量局部注意力：固定输入下的 decoder 评估

当前完整容量版 `local` 对比上一版 Intersection refinement `intersection`。固定 accepted test 2,417 件，原 2,424 件中另有 7 件数据合同不支持。2 层 small local 的历史输出保留在 `../evaluation_20261012`，不是本轮主 baseline，也不作为当前模型结果。

协议见 [PLAN](PLAN.md)，训练审计见 [training_audit.json](training_audit.json)，汇总见 [REPORT](REPORT.md)，失败诊断见 [本轮失败统计](../failure_census_20261013/README.md)。报告及诊断只有在全部终态完成后生成；运行中日志不能当作最终结论。

权重只按验证集 callback 选择 epoch42 / step15609、`val/loss=0.09879826009273529`，SHA256 为 `c16d9ebe829659f3f4b8eb631e115fb10d03a28e9319bdb6bb5968b8d4bdb8f3`。最后权重 epoch43 不另做 test 选优。实际 inter 为 512 维、8 层、8 heads、FFN2048、dropout0.1、ReLU 的 TransformerDecoderLayer，同时保留 face pair 单 query 与两面局部 CNN/XYZ memory。严格加载及冻结 encoder 243 个张量相等的证据见 [validation_summary.json](validation_summary.json)。

## 运行与复现

本次真实执行命令及退出码完整记录在 `metadata/orchestration/*_jobs.json`；各子任务日志同目录。调度脚本仅调用既有工具，不改变模型、几何算法或判据。已有结果禁止覆盖；完整复现应换新输出目录并重新冻结协议。

```bash
cd /mnt/d/python
conda activate torch
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

python checkpoints/intersection_local/evaluation_20261013/run_evaluation.py smoke
python checkpoints/intersection_local/evaluation_20261013/run_evaluation.py inference
```

Smoke 只在独立 validation 子目录处理原 split 的两件，用于严格加载、输入和几何合同检查，不纳入全集。正式推理每个 mode 各 8 个 GPU 分片；先 posterior_mean，再 archived_latents。FP32、TF32 off、seed42，逐件重放原归档 float32 latent，不重新采样 diffusion。各分片先核对前三件 mean 重编码误差在 atol/rtol `1e-5` 内，实际 decoder 输入仍使用逐值相同的归档数组。

posterior_mean 全部 2,417 件导出后启动后处理与该 mode 的 raw 指标；archived_latents 完成后计算另一组 raw：

```bash
python checkpoints/intersection_local/evaluation_20261013/run_evaluation.py post
python checkpoints/intersection_local/evaluation_20261013/run_evaluation.py metrics_posterior_mean
python checkpoints/intersection_local/evaluation_20261013/run_evaluation.py metrics_archived_latents
```

两种 post mode 为 fit_only 与 optimized，仅处理 posterior mean，各 32 并发、CPU 单线程、每对象全流程 180 秒，`drop_num=0`，优化名义 200 / 原算法自适应上限 600。超时覆盖几何测量；如果超时前已确认原生合法 solid，该成功不会被抹去。失败终态不自动重试以筛选更好结果。

`raw_predictions/<mode>/intersection` 和 `post/<mode>/intersection` 只引用上一版对应目录；来源见 [reference_links.json](reference_links.json)。生产 post 的 11 个算法文件 SHA 与原协议完全相同，见 [算法核验](baseline_algorithm_audit.json)。后处理 runner 在调度前另核验全部历史终态、raw SHA、split、采样、预算和 STEP 存在性，保留原记录 fingerprint 与历史耗时。STEP SHA 在本次复用时补记，不声称有历史哈希证明。

两种 raw 指标与两种 post 终态齐后生成诊断及报告：

```bash
python checkpoints/intersection_local/evaluation_20261013/diagnostics/raw_distribution_analysis.py
python checkpoints/intersection_local/evaluation_20261013/diagnostics/adjacency_subset_analysis.py
python checkpoints/intersection_local/evaluation_20261013/diagnostics/build_raw_distribution_report.py
python checkpoints/intersection_local/evaluation_20261013/build_report.py
python checkpoints/intersection_local/evaluation_20261013/build_independent_review.py
```

本轮报告 wrapper 仅适配“完整容量版”标签与 20261013 失败链接，使用既有 2417 件完整性及配对汇总校验。生成来源和产物 SHA 记录在 `report_generation.json`。原始均值为主指标；[分布诊断](diagnostics/RAW_DISTRIBUTION.md) 的中位数、尾部、改善率及去尾均值用于解释，去尾不改变全集主分母。

## 解释范围

posterior mean 评估检查 decoder 重建；archived latent 是相同旧 diffusion 输入的 decoder 重放，不是重新采样或完整照片条件流程。当前模型有额外训练曝光且 inter 重新初始化，不能视为等预算架构消融。主有效性使用原读取精度 TopAbs_SOLID + BRepCheck；兼容容差 0.1 单列。几何各自可计算集合与共同可计算集合分别报告，不等同于 solid 成功子集。训练曲线现象、raw 几何、最终实体有效性及失败多标签各有范围，不用单一 loss 趋势替代下游结论。
