# 原版 VAE 全部失败的溯源统计

本目录覆盖 comparison_20261011 的 legacy / posterior_mean / optimized 全部 369 个失败，来自 2417 个 accepted test identity。原 test 的 7 个合同不支持对象另列；本次不是 diffusion 生成测试。

统计口径见 [PLAN.md](PLAN.md)。结果以 [REPORT.md](REPORT.md)、[统计页面](page/index.html)、[summary.json](summary.json) 和 [cases.json](cases.json) 为准。页面只展示统计，逐件 3D 沿用 [50 件工作台](../legacy_failures_20261011/page/index.html)。

## 复现

从仓库根目录执行。完整重算应选一个新输出根，保留本目录与原比较结果。

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
census_python=/mnt/d/anaconda3/envs/torch/bin/python
census_root=checkpoints/joint_boundary/legacy_failure_census_replay
census_comparison=checkpoints/joint_boundary/comparison_20261011
census_data=/mnt/d/data/deepcad_v7

"$census_python" -m tools.legacy_failure_census_manifest \
  --comparison-root "$census_comparison" --data-root "$census_data" --output-root "$census_root"
"$census_python" -m tools.legacy_failure_provenance \
  --manifest "$census_root/manifest.json" --output-root "$census_root" --workers 8 --timeout 180
"$census_python" -m tools.legacy_failure_upstream \
  --manifest "$census_root/manifest.json" --root "$census_root" --workers 8
"$census_python" -m tools.legacy_failure_saved_step \
  --manifest "$census_root/manifest.json" --root "$census_root" --workers 8 --timeout 180
"$census_python" -m tools.legacy_failure_census_report --root "$census_root"
```

8 个原有 fit-only 有效、optimized 无效的对象，额外读取相同 fit-only 路径的内部轨迹：

```bash
"$census_python" -m tools.legacy_failure_provenance \
  --manifest "$census_root/manifest.json" --output-root "$census_root/fit_only_counterfactual" \
  --post-mode fit_only --workers 8 --timeout 180 \
  --only-ids 00007551 00071931 00212409 00326520 00423247 00677369 00730675 00933890
```

诊断中断后同参数增加 `--resume`。provenance / saved-step 要求输入和源码指纹不变；upstream 对新增阶段输入可按 hash 重新计算并存档旧诊断。修改测量定义应使用新根，不混合不同协议。原推理、GT、STEP 和比较表只读。

页面可由旧工作台同一个本地服务访问：`http://127.0.0.1:8765/legacy_failure_census_20261011/page/`。服务未启动时：

```bash
python -m http.server 8765 --bind 127.0.0.1 --directory checkpoints/joint_boundary
```

## 解释边界

- 原对照明确关闭面 latent 去重，面数与身份来自编码输入；统计不能据此判断 diffusion 去重好坏。
- 正常 reciprocal 半边、不同 pair 的几何接近和 STEP 内尚未共享拓扑的近重合自由边分别统计。
- 最终失败面的持续阻断与修复尝试中曾出现但可能已解决的异常分开。`BRepCheck` 通过不能认证 GT 孔/边界被保留。
- 多标签频次可以重叠。接近阈值为既有工程容差的观察刻度，不能直接当实体失败判据；特别是面指标是未裁剪采样网格的接近代理。
- STEP 重放 DATA 相等才将该轨迹作为原结果的精确重放；分歧、超时、缺测留在原 369 分母并标明未知。

## 产物

`manifest.json` / `failed_ids.txt` 固定全集；`traces/` / `replay/` 记录过程和阶段几何；`upstream/` 记录 GT 对照的上游异常；`saved_step/` 记录原生检查与自由边接近。`cases.csv`、`summary.json`、报告、页面源码和工具受 Git 管理；完整 `cases.json` 含逐面/边拟合测量，可由汇总工具再生，不提交大型逐实体明细。图表 PNG/SVG、页面 data.js、STEP/NPZ、长轨迹与运行元数据留在本地。

全量完成：369/369 重放与原 STEP DATA 一致，369/369 上游及原 STEP 诊断完成；1,845 个源资产 SHA256 保持不变。相关 39 项测试通过。没有修改生产模型或后处理算法。

`optimization_regressions.json` 是 8 件优化路径退化的补充审计，包含输入/选择相同的检查、GT 误差、逐面拟合与裁剪、逐容差 sewing 对照和来源哈希；它不自动并入常规汇总器。阅读该文件时以其 `definitions` 中的整体路径干预限制为准。
