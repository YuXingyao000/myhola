# 曲线条件调制：全集失败溯源

比较 `conditioned`（曲线条件调制）与 `intersection`（上一版 Intersection refinement）。候选按验证集选 epoch157/step57354，权重 SHA256 `4c7910b8169206c74fcf38c358cf171359149e0429c8debbc227d391aa56613f`，训练审计见 [评估目录](../evaluation_20261014/training_audit.json)。固定2417件；原2424中7件合同不支持另列。不复用 local 的候选输出。

执行口径见 [冻结计划](PLAN.md)。全部本轮失败及全部救回做双侧诊断；主失败分母只含本轮失败，救回为独立对照。原 refinement 历史诊断按 ID 与协议、资产 SHA 核验复用，其余同 ID baseline 对照完整补齐。所有 fit-only 成功而 optimized 失败的对象均加入路径对照。

当前状态：原始测试、两类后处理和全集失败诊断已完成。306件失败、43件救回的双侧轨迹均完整复现；统计与研究判断见 [REPORT](REPORT.md)、[NOTE](NOTE.md)。运行记录保存在 `metadata/orchestration/state.json`，最终独立复算与页面检查另存紧凑JSON。

## 执行

在独立 worktree 运行，科学 runner 及依赖在正式 smoke 前冻结。调度器等待本轮两种 post 表全部终态，生成 manifest、每个非空 cohort 的少量真实 smoke，通过后继续全集诊断与报告。每个 runner 最多8并发，每进程1线程、每件180秒；最多4个 runner 同时运行。不同侧 trace/saved STEP 独立；upstream 等同侧 trace 完成。

```bash
cd /mnt/d/python_curve_eval
/mnt/d/anaconda3/envs/torch/bin/python \
  checkpoints/intersection_refinement/curve_conditioning/failure_census_20261014/run_census.py
```

同一 root 不重复启动调度器。已冻结 manifest 后如需恢复中断，按状态文件中原命令逐阶段 `--resume`；禁止修改 runner 源码后静默续接。命令、起止时间、退出码、进度保存在 `metadata/orchestration/`。调度器不生成研究结论，最终 `NOTE.md` 需基于完整结果填写。

手工冻结入口：

```bash
PY=/mnt/d/anaconda3/envs/torch/bin/python
CENSUS_ROOT=checkpoints/intersection_refinement/curve_conditioning/failure_census_20261014
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

"$PY" -m tools.paired_failure_manifest \
  --candidate-method conditioned \
  --comparison-root checkpoints/intersection_refinement/curve_conditioning/evaluation_20261014 \
  --reference-root /mnt/d/python/checkpoints/intersection_refinement/evaluation_20261011 \
  --baseline-census-root /mnt/d/python/checkpoints/intersection_refinement/failure_census_20261011 \
  --data-root /mnt/d/data/deepcad_v7 \
  --split-list checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt \
  --output-root "$CENSUS_ROOT"
```

调度器调用既有 `legacy_failure_provenance`、`legacy_failure_upstream`、`legacy_failure_saved_step`，方法身份由 manifest 显式决定。没有修改生产后处理、采样、阈值或失败判据。candidate 输出在本目录，baseline 缺失对照在 `baseline_controls`，全部优化退步的 fit-only 对照在 `fit_only_counterfactual`。historical baseline 原 method/路径/指纹保持原样。

## 报告与图表

所有诊断终态后执行：

```bash
"$PY" -m tools.paired_failure_report \
  --census-root "$CENSUS_ROOT" \
  --baseline-census-root /mnt/d/python/checkpoints/intersection_refinement/failure_census_20261011
```

生成 `REPORT.md`、`compact_summary.json`、`cases.csv`、四组 PNG/SVG、`page/index.html`。静态四图分别为失败共现观察、相同 ID 旧新变化、独立拟合/周期替换后的边面贴合、最终失败面的候选检查。页面沿用四组×四阈值筛选与双侧逐 ID 详情；正/负/未知保留既有定义。仅重新生成核验完毕的报告时加 `--overwrite`。

研究 `NOTE.md` 写好后打包自包含页面：

```bash
"$PY" "$CENSUS_ROOT/build_standalone.py"
```

`page/standalone.html` 内嵌数据、脚本、四图和下载文件，直接本地打开即可。`standalone_packaging.json` 记录打包输入及输出 SHA。打包只改展示，不重新计算科学测量。

## 展示核验

普通页面可在仓库根目录运行 `python -m http.server 8767 --bind 127.0.0.1`。检查：

```bash
FONTCONFIG_FILE=/root/.cache/hola-browser-libs/fonts.conf \
LD_LIBRARY_PATH=/root/.cache/hola-browser-libs/root/usr/lib/x86_64-linux-gnu \
node tools/paired_failure_browser_check.cjs \
  http://127.0.0.1:8767/checkpoints/intersection_refinement/curve_conditioning/failure_census_20261014/page/ \
  "$CENSUS_ROOT/browser_validation" \
  /root/.cache/ms-playwright/chromium-1155/chrome-linux/chrome

FONTCONFIG_FILE=/root/.cache/hola-browser-libs/fonts.conf \
LD_LIBRARY_PATH=/root/.cache/hola-browser-libs/root/usr/lib/x86_64-linux-gnu \
node "$CENSUS_ROOT/check_standalone.cjs" \
  file:///mnt/d/python_curve_eval/checkpoints/intersection_refinement/curve_conditioning/failure_census_20261014/page/standalone.html \
  "$CENSUS_ROOT/standalone_validation" \
  /root/.cache/ms-playwright/chromium-1155/chrome-linux/chrome
```

遍历四组四阈值的全部观察和 paired 转换、ID详情、资源、390px窄屏与 JS；离线检查阻断 HTTP/HTTPS 并要求四张内嵌图全部解码成功。普通页面的原HoLa图库仅在本 worktree 实际存在时提供链接；不把历史50件解释为本轮结果。

## 解释边界

最早观察位置与最后阻断位置分开；OCC码和距离超阈值不是唯一根因，多标签不能相加成失败总数。合法 solid 不保证几何准确，闭合 wire 不保证裁剪成功。额外训练曝光与条件调制同时变化，因此当前比较不能单独确定结构的因果收益。
