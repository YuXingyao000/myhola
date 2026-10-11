# Intersection refinement 全集失败溯源

先看[通俗解释与下一步方向](NOTE.md)，再看[统计页面](page/index.html)或[完整统计报告](REPORT.md)。页面可以切换304失败全集、262持续失败、42退步和107救回组，并按0.002/0.006/0.01/0.08归一化长度阈值查看问题与对应ID。这里没有新增3D案例图库；链接仍指向原来的50件。

## 输入和协议

固定[首次评估](../evaluation_20261011/README.md)的epoch10/step3993 checkpoint与posterior mean输出，复用[原版比较](../../joint_boundary/comparison_20261011/README.md)。接受集2417，原始2424中7件合同不支持，仍单列。精确执行规范见[PLAN](PLAN.md)。本任务没有改训练模型、checkpoint或生产后处理。

主统计304＝262持续失败＋42退步。额外107救回作为对照。新侧重放411；旧失败369经资产、生产算法、采样与协议核验后复用，旧成功对照补充42；另作6个新模型fit-only路径对照。优化使用原评估CPU路径、seed、drop_num=0，名义200/自适应600迭代，逐件180秒上限。

## 复现

以下从仓库根目录运行，已存在的诊断产物保留，不重新覆盖原评估目录。Python环境为 `/mnt/d/anaconda3/envs/torch/bin/python`。可令 `PY`、`ROOT` 为下列任务变量：

```bash
PY=/mnt/d/anaconda3/envs/torch/bin/python
ROOT=checkpoints/intersection_refinement/failure_census_20261011
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

冻结输入清单（本次已生成；不要在正在运行的诊断中重建）：

```bash
"$PY" -m tools.intersection_failure_manifest \
  --comparison-root checkpoints/intersection_refinement/evaluation_20261011 \
  --reference-root checkpoints/joint_boundary/comparison_20261011 \
  --baseline-census-root checkpoints/joint_boundary/legacy_failure_census_20261011 \
  --data-root /mnt/d/data/deepcad_v7 \
  --split-list checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt \
  --output-root "$ROOT"
```

工具保留历史文件名，但方法由显式manifest字段确定；不会把intersection结果标成legacy。

```bash
"$PY" -m tools.legacy_failure_provenance --manifest "$ROOT/manifest.json" --output-root "$ROOT" --workers 8 --timeout 180
"$PY" -m tools.legacy_failure_upstream --manifest "$ROOT/manifest.json" --root "$ROOT" --workers 8
"$PY" -m tools.legacy_failure_saved_step --manifest "$ROOT/manifest.json" --root "$ROOT" --workers 8

"$PY" -m tools.legacy_failure_provenance --manifest "$ROOT/legacy_control_manifest.json" --output-root "$ROOT/legacy_controls" --workers 8 --timeout 180
"$PY" -m tools.legacy_failure_upstream --manifest "$ROOT/legacy_control_manifest.json" --root "$ROOT/legacy_controls" --workers 8
"$PY" -m tools.legacy_failure_saved_step --manifest "$ROOT/legacy_control_manifest.json" --root "$ROOT/legacy_controls" --workers 8

"$PY" -m tools.legacy_failure_provenance --manifest "$ROOT/manifest.json" \
  --output-root "$ROOT/fit_only_counterfactual" --post-mode fit_only --workers 8 --timeout 180 \
  --only-ids 00021937 00043034 00533990 00827871 00869029 00991826
"$PY" -m tools.legacy_failure_upstream --manifest "$ROOT/manifest.json" \
  --root "$ROOT/fit_only_counterfactual" --workers 8 \
  --only-ids 00021937 00043034 00533990 00827871 00869029 00991826
```

上游诊断需要同组provenance先完成。六件fit-only的upstream为partial，因为该路径没有优化后坐标；不代表原始/选择后数据或裁剪轨迹缺失。

生成统计与页面；`--overwrite`只用于重新生成派生报告，不重跑或修复几何。中文图表需要CJK字体，可用 `--font-path /absolute/font.otf` 显式指定。

```bash
"$PY" -m tools.intersection_failure_report \
  --census-root "$ROOT" \
  --legacy-census-root checkpoints/joint_boundary/legacy_failure_census_20261011 \
  --font-path /root/.cache/hola-browser-libs/fonts/NotoSansCJKsc-Regular.otf \
  --overwrite

python -m http.server 8765 --bind 127.0.0.1 --directory /mnt/d/python/checkpoints
```

浏览地址：`http://127.0.0.1:8765/intersection_refinement/failure_census_20261011/page/`。页面须通过HTTP打开，以便链接本地报告和图表。

## 可复核产物

- `manifest_summary.json`、`validation/source_preservation.json`：固定分母与6433个源文件的哈希审计。
- `compact_summary.json`：受Git管理的紧凑统计；完整 `summary.json`、`cases.json/csv`、`page/data.js` 在本地保留，可重新生成。
- `traces/`、`replay/`、`upstream/`、`saved_step/`：完整阶段事件与测量；原输入不改动。
- `legacy_controls/`、`fit_only_counterfactual/`：配对对照，与主304失败分母分开。
- `optimization_regressions.json`：6件整条路径的逐件精细核验、坐标变换反推及原始测量；主要判断已写入NOTE。
- `validation/independent_recount.json`：直接读原事件的独立重计，不依赖报告聚合结果。
- `validation/measurement_code_audit.json`、`validation/summary.json`：测量代码保持不变的审计及验证汇总。
- `visualizations/`：4组中文PNG/SVG统计图；生成图可单独导出。

浏览器验证脚本 `tools/intersection_failure_browser_check.cjs` 使用Node原生WebSocket连接Chrome DevTools，无需安装Playwright。它核对16种组/阈值组合、表格计数、ID展开、成对详情、资源链接和移动端布局，结果及截图写入 `validation/`。
