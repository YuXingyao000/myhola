# VAE 解码器固定协议对比

先读 [结果与解释](REPORT.md)、[冻结协议](PLAN.md)、[资产审计](ASSET_AUDIT.md)。实现基准提交为 **`d72ecb08`**；三个针对性测试文件共 **28 tests PASS**。运行环境记录见 [environment.json](environment.json)，实际推理数值设置和导入版本以各 mode 的 `metadata/inference/*/protocol.json` 为准。

本次比较固定 Legacy 1119 与 Joint epoch 538／step 97,559 checkpoint：主组共享 posterior mean，第二组重放相同历史 diffusion latent。两 Decoder 均自主预测拓扑、不额外删面；这不是同训练预算消融，也不重新训练或运行 diffusion 采样。

## 复现命令

以下在仓库根目录用 **Bash** 执行。完整重跑选择**未存在的新输出根**，保留本次结果；所需数据和 checkpoint 必须已经挂载。固定选例文件直接复制，不重新按结果挑选。

```bash
cd /mnt/d/python
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
comparison_python=/mnt/d/anaconda3/envs/torch/bin/python
comparison_root="$PWD/checkpoints/joint_boundary/comparison_20261011_replay"
comparison_data=/mnt/d/data/deepcad_v7
comparison_split=checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt
comparison_legacy=/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt
comparison_joint=checkpoints/joint_boundary/frozen/20261010T084310.423740Z/weights/epoch=0538-step=00097559-val_loss=0.028098.ckpt
comparison_archive=experiments/baseline/diffusion_baseline_real_v7_cube24/test_outputs
test ! -e "$comparison_root"
mkdir -p "$comparison_root/logs"
cp checkpoints/joint_boundary/comparison_20261011/selected_examples.json "$comparison_root/"
```

八卡按 ID 分片，无 distributed sampler 补齐。每张卡同时驻留两个 VAE；两个输入模式依次执行，每个模式内部八个分片并行。FP32、阈值、权重与源码 hash 由 runner 记录。

```bash
for comparison_mode in posterior_mean archived_latents; do
  comparison_extra=()
  if [[ "$comparison_mode" == archived_latents ]]; then
    comparison_extra=(--latent-root "$comparison_archive")
  fi
  comparison_pids=()
  for comparison_shard in {0..7}; do
    "$comparison_python" -m tools.vae_compare_inference \
      --mode "$comparison_mode" \
      --legacy-checkpoint "$comparison_legacy" \
      --joint-checkpoint "$comparison_joint" \
      --data-root "$comparison_data" --split-list "$comparison_split" \
      --output-root "$comparison_root" \
      --num-shards 8 --shard-index "$comparison_shard" --device "cuda:$comparison_shard" \
      "${comparison_extra[@]}" \
      > "$comparison_root/logs/inference_${comparison_mode}_${comparison_shard}.log" 2>&1 &
    comparison_pids+=("$!")
  done
  for comparison_pid in "${comparison_pids[@]}"; do
    wait "$comparison_pid"
  done
done
```

中断后对相同输出根、相同分片参数的 inference 命令增加 `--resume`；只有协议、输入来源 hash 及两份 latent 一致的完整输出对才会跳过。改变权重、输入或源码需换新根。

Raw 指标遍历固定 2,417 个 intended ID；归档组不计算依赖 GT face order 的面 MAE／拓扑 F1。脚本**拒绝覆写**已有 `per_object.csv` 或 `summary.json`。

```bash
for comparison_mode in posterior_mean archived_latents; do
  "$comparison_python" -m tools.vae_compare_metrics \
    --mode "$comparison_mode" --comparison-root "$comparison_root" \
    --data-root "$comparison_data" --split-list "$comparison_split" --workers 16
done
```

CPU 后处理只处理 **posterior_mean** 组，两种预算独立运行。每对象／方法／模式的整个子进程硬超时 180 秒；optimized 名义 200 次迭代，原算法可自适应延长至 600。每进程单线程，`drop_num=0`；算法仍包含原有适配、修补和缝合。

```bash
for comparison_post_mode in fit_only optimized; do
  "$comparison_python" -m tools.vae_compare_post \
    --mode "$comparison_post_mode" --comparison-root "$comparison_root" \
    --data-root "$comparison_data" --split-list "$comparison_split" \
    --methods legacy joint --workers 32 --timeout 180 \
    --max-optimize-iter 200 --original-test-count 2424
done
```

Post 重复调用会按**相同协议／代码指纹和 raw 文件 hash**继续尚未执行的任务，无需 `--resume`。失败、超时和中断终态保留，不自动重试；改变指纹或已处理的 raw 输入会拒绝混用。可给相同命令增加 `--summary-only`，仅由现有逐件记录重建汇总。

固定选例画图与汇总图：raw 显示未裁剪面网格及全部导出曲线；STEP 图本身不代表有效 solid。Post 图明确使用 posterior 组，不能把它标成归档 latent 的后处理结果。

```bash
for comparison_mode in posterior_mean archived_latents; do
  "$comparison_python" -m tools.vae_compare_visualize \
    --comparison-root "$comparison_root" --data-root "$comparison_data" \
    --mode "$comparison_mode"
done
for comparison_post_mode in fit_only optimized; do
  "$comparison_python" -m tools.vae_compare_visualize \
    --comparison-root "$comparison_root" --data-root "$comparison_data" \
    --mode posterior_mean --post-mode "$comparison_post_mode"
done
"$comparison_python" -m tools.vae_compare_plot_metrics --comparison-root "$comparison_root"
```

定向验证命令：

```bash
"$comparison_python" -m pytest -q \
  src/brepnet/test/test_vae_compare_inference.py \
  src/brepnet/test/test_vae_compare_metrics.py \
  src/brepnet/test/test_vae_compare_post.py
```

## 产物导航

| 路径（相对于 comparison root） | 用途 |
|---|---|
| [PLAN.md](PLAN.md)、[ASSET_AUDIT.md](ASSET_AUDIT.md)、[REPORT.md](REPORT.md) | 固定协议、资产证据、结果解释 |
| [selected_examples.json](selected_examples.json) | 查看预测前按稳定 hash／GT 面数分层固定的选例 |
| `raw_predictions/<input_mode>/<method>/<id>/data.npz` | 两 Decoder 原始预测、相同输入 latent 及来源 hash |
| [raw_metrics/posterior_mean/summary.json](raw_metrics/posterior_mean/summary.json)、[raw_metrics/archived_latents/summary.json](raw_metrics/archived_latents/summary.json) | Raw 指标汇总；同目录 `per_object.csv` 是逐件记录 |
| `post/{fit_only,optimized}/summary.json`、`samples.csv` | 固定分母的有效性、覆盖、几何和失败汇总 |
| `post/<post_mode>/<method>/<id>/` | STEP、success marker 和逐对象后处理产物；不能仅凭文件存在判成功 |
| `metadata/inference/<input_mode>/`、`metadata/post/<post_mode>/` | 协议／代码／权重 hash、调用信息和逐件状态 |
| `visualizations/<input_mode>/{raw,fit_only,optimized}/contact_sheet.png` | 固定选例总览；post 子目录仅存在于 posterior_mean |
| `visualizations/{raw_comparison,diffusion_replay,post_comparison}.png` | 已记录指标的汇总图；post 图需两个 post summary 都存在 |

权重、NPZ、STEP、日志与逐件大文件留在本地；协议、审计、紧凑汇总和报告受 Git 管理。失败和缺失保留于 intended 分母，原始 2,424 test 总体覆盖率单列。

本次执行曾修复 raw 汇总 JSON 的 NumPy 标量类型序列化问题，修复只影响汇总类型转换及写出检查，不改变几何／拓扑公式或逐件指标；重跑两个 mode 的 `per_object.csv` 与修复前逐 byte 一致。原 CSV 和失败 manifest 保留在 `metadata/raw_metrics_serialization_failure_20261011T051859.659962Z/`，未覆盖或丢弃。
