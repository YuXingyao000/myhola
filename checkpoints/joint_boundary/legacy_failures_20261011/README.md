# 原版 VAE 失败案例工作台

打开 [工作台](page/index.html#00001926)，或先读 [分析报告](REPORT.md)。本地浏览器可直接打开 `page/index.html`；无需网络或后台服务。页面提供 50 件 GT / raw / post 同步 3D 视图、固定四视角截图、原始数值、阶段 trace、筛选和浏览器本地备注。

当前本机也可用 `http://127.0.0.1:8765/legacy_failures_20261011/page/#00001926`。服务若已停止，在仓库根目录执行：

```bash
python -m http.server 8765 --bind 127.0.0.1 --directory checkpoints/joint_boundary
```

建议先查看 `00001926`（局部裁剪）、`00009683`（全部面裁剪仍有开放边）、`00480996`（原生检查／兼容容差差异）、`00038614`（整体尺度与误差）。这只是固定清单中的阅读入口，不是重新筛选。

## 范围与显示约定

- 来自 comparison_20261011 的 **legacy / posterior_mean / optimized**；不是照片条件生成。
- 369 个原生无效 solid 中冻结 50 件；50 件的分层构成不用于估计全体发生率。
- 成功需要 STEP 原生读取后的 root `TopAbs_SOLID` 加 BRepCheck 通过，保留原评估；不能用 success.txt 或截图代替。
- Raw 是未裁剪采样网格和全部 directed half-edges；post 是原始保存 STEP，不是重放结果。
- 红线是 native 拓扑 free edge；紫线是非流形 incidence。seam、退化边分别统计，不混作孔洞。
- 三栏共用坐标和相机。点击面显示的编号只在当前栏有效；不暗示 GT/raw/post 面对应。
- 13 件部分面无法三角化；页面显示缺失数并保留可读部分，不以 raw mesh 冒充 post。
- 页面备注只保存在当前浏览器，可导出 JSON；切换 file/HTTP 打开方式属于不同浏览器存储来源。

## 复现

已有结果保持只读；完整复现使用新根。以下 Bash 命令在仓库根目录执行，既有模型、数据和原比较产物需要已挂载。

```bash
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
atlas_python=/mnt/d/anaconda3/envs/torch/bin/python
atlas_original=checkpoints/joint_boundary/legacy_failures_20261011
atlas_root=checkpoints/joint_boundary/legacy_failures_replay
atlas_comparison=checkpoints/joint_boundary/comparison_20261011
atlas_data=/mnt/d/data/deepcad_v7
test ! -e "$atlas_root"
mkdir -p "$atlas_root/page"
cp "$atlas_original/page/index.html" "$atlas_original/page/style.css" "$atlas_original/page/app.js" "$atlas_root/page/"
cp "$atlas_original/PLAN.md" "$atlas_original/AUDIT.md" "$atlas_original/REPORT.md" "$atlas_original/README.md" "$atlas_root/"

"$atlas_python" -m tools.legacy_failure_select \
  --comparison-root "$atlas_comparison" --data-root "$atlas_data" --output-root "$atlas_root"
"$atlas_python" -m tools.legacy_failure_trace \
  --manifest "$atlas_root/selected_cases.json" --output-root "$atlas_root" --workers 8 --timeout 180
"$atlas_python" -m tools.legacy_failure_geometry \
  --root "$atlas_root" --comparison-root "$atlas_comparison" --data-root "$atlas_data" --workers 8 --timeout 180
"$atlas_python" -m tools.legacy_failure_extent \
  --root "$atlas_root" --comparison-root "$atlas_comparison" --data-root "$atlas_data"
"$atlas_python" -m tools.legacy_failure_page --root "$atlas_root"
```

trace、geometry 中断后可用原参数增加 `--resume`；只跳过相同协议的终态，不增加失败对象的修复预算。修改工具或协议应新建输出根。仅重新打包已有页面数据可重跑 `legacy_failure_page`，它不会修改科学测量；首次构建下载固定 SHA256 的 Three.js 0.160.1，此后查看离线运行。静态 HTML/CSS/JS 是受 Git 管理的源文件。

测试：

```bash
"$atlas_python" -m pytest -q \
  src/brepnet/test/test_legacy_failure_select.py \
  src/brepnet/test/test_legacy_failure_geometry.py \
  src/brepnet/test/test_legacy_failure_trace.py
```

浏览器检查使用 Playwright/Chromium（已有安装可通过 `PLAYWRIGHT_MODULE_PATH` 和可选 `PLAYWRIGHT_CHROMIUM_EXECUTABLE` 显式指定）：

```bash
node tools/legacy_failure_browser_check.cjs "$atlas_root"
```

## 文件地图

| 路径 | 内容 |
|---|---|
| `PLAN.md` / `AUDIT.md` / `REPORT.md` | 冻结协议、选择与总体审计、阶段和视觉观察 |
| `selected_cases.json` / `selected_ids.txt` | 50 件身份、分层、原始指标和资产路径 |
| `selection_summary.json` | 369 失败与 2,048 成功的描述性对照及面数标准化 |
| `case_summary.json` | 受 Git 管理的 50 件紧凑数值、子形状错误与阶段摘要 |
| `diagnostics/{id}.json` | 本地详细原生检查、子形状错误、wire 端点、半边差异、网格覆盖 |
| `diagnostics_extent.json` | 整体轴向采样跨度及对应误差比例；不是壁厚 |
| `traces/{id}.json` / `trace_summary.json` | 本地逐阶段详细重放及受版本管理的汇总；优化 loss_checkpoints 为空，不宣称保存 loss 轨迹 |
| `trace_step_data_audit.json` | 50 件原／重放 STEP DATA 段逐字节相同的核对 |
| `visualizations/` | 逐件网格 JSON、四视角图、5 张十件总览；本地资产，不提交 |
| `page/{index.html,style.css,app.js}` | 受版本管理的浏览页源码 |
| `page/{data.js,geometry/,vendor/}` | 构建生成的离线数据与 Three.js；不提交 |
| `replay/` / `metadata/` | 独立重放 STEP、阶段 NPZ、运行协议、开发问题归档；不提交 |
| `validation/` / `page_build.json` | 定向测试和浏览器验证、输入指纹与构建记录 |

原始数据、几何、checkpoint、截图、逐面详细诊断／trace 与运行日志均不提交；选择清单、逐件紧凑摘要、诊断汇总、工具和报告保存版本，便于围绕相同 CAD ID 复查。复制的 REPORT 保留本次已完成分析，不自动代表新运行的结论。
