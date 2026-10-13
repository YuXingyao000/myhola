# 完整容量局部注意力 intersection：全集失败溯源

本目录比较恢复为512维8层后训练完成的local与**上一版 Intersection refinement**。机器方法名为`local`、`intersection`。本轮固定epoch42/step15609验证最佳权重，不复用10月12日的小容量local结果。

当前状态：完整评估、全集诊断、独立复算和普通/离线页面验证均已完成。固定2417件中上一版合法2113件，本版1705件；本版失败712件＝持续失败277件＋退步435件，另有27件救回。当前诊断739件；baseline复用375件历史诊断，补齐364件同ID对照；全部11件fit-only成功但optimized失败的路径对照都纳入。执行依据是[冻结计划](PLAN.md)和[评估协议](../evaluation_20261013/PLAN.md)。

打开[自包含离线页面](page/standalone.html)，或阅读[研究判断](NOTE.md)、[中文报告及四图](REPORT.md)、[紧凑统计](compact_summary.json)。已保留四组中文PNG/SVG、四组×四阈值交互页面和逐ID双侧详情。自包含HTML为约26.7MB，核心统计、数据、脚本和四图直接内嵌，可在本地浏览器打开，不依赖临时HTTP服务。原50件三维图库仅为独立的历史参考，不作为本轮结果。

本版739件、新增baseline364件、fit对照11件的trace均与原终态及STEP DATA一致，无instrumentation错误；本版和新增baseline的upstream/saved-step全部ok。11件fit-only upstream为预期的结构性partial（无optimized快照），不是轨迹失败。10393个来源资产零变更；独立审计通过1478条双侧轨迹、35472次判据、448项展示统计及11件完整路径对照。

普通页面与离线版均通过4组×4阈值、28项观察及成对转换、ID详情、390px窄屏与JS检查。离线检查中HTTP/HTTPS请求被阻断，外部请求为0，四张内嵌图片全部解码成功。见[验证记录](validation_summary.json)、[打包来源与哈希](standalone_packaging.json)。

具体命令、进程与UTC起止时间见`metadata/orchestration/events.jsonl`；科学runner及生产算法在正式smoke之前冻结。只复用历史refinement的411件诊断中经身份与哈希审核通过的部分；其他同ID对照在本目录补齐。

## 输入冻结

先完成本轮2417件的raw、fit-only和optimized表，全部终态后运行。原2424中7件合同不支持对象单列；不能按已完成目录缩分母。

从仓库根目录执行：

```bash
PY=/mnt/d/anaconda3/envs/torch/bin/python
CENSUS_ROOT=checkpoints/intersection_local/failure_census_20261013
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

"$PY" -m tools.paired_failure_manifest \
  --comparison-root checkpoints/intersection_local/evaluation_20261013 \
  --reference-root checkpoints/intersection_refinement/evaluation_20261011 \
  --baseline-census-root checkpoints/intersection_refinement/failure_census_20261011 \
  --data-root /mnt/d/data/deepcad_v7 \
  --split-list checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt \
  --output-root "$CENSUS_ROOT"
```

此命令只读/校验来源，拒绝覆盖已有冻结manifest。输出：

- `manifest.json`：本版所有失败＋救回对照；主失败分母始终仅为本版失败。
- `baseline_manifest.json`：完全相同ID的上一版结果。
- `baseline_reuse.json`：按ID核验已有411件诊断，保留原method、路径、协议指纹。
- `baseline_control_manifest.json`：缺少可复用诊断的上一版同ID对照；可能含成功或失败，不假定全为成功。
- `source_asset_audit.json` 和各组ID列表：冻结输入、生产算法与诊断资产哈希。

cohort数量全部由这次评估派生，不沿用上一轮304/262/42/107。

## 阶段重放与测量

既有工具名保留 `legacy_`，但方法由manifest显式决定，测量与生产后处理决策不变。每个runner最多8并发、每进程1CPU线程、每件180秒。正式执行前，使用正式manifest、正式root、`--workers 8 --only-ids <少量真实ID>` 做重放核验；通过后相同命令去掉 `--only-ids` 并追加 `--resume`，继续全量。

**同一root、同一类型runner不能并发运行两次。** smoke到全量期间冻结runner及其依赖文件，尤其 `tools/vae_compare_post.py`、`tools/failure_census_contract.py`；源码哈希改变时不得以resume静默混用。

本版优化路径：

```bash
"$PY" -m tools.legacy_failure_provenance \
  --manifest "$CENSUS_ROOT/manifest.json" --output-root "$CENSUS_ROOT" \
  --workers 8 --timeout 180

"$PY" -m tools.legacy_failure_upstream \
  --manifest "$CENSUS_ROOT/manifest.json" --root "$CENSUS_ROOT" --workers 8

"$PY" -m tools.legacy_failure_saved_step \
  --manifest "$CENSUS_ROOT/manifest.json" --root "$CENSUS_ROOT" \
  --workers 8 --timeout 180
```

若 `baseline_control_manifest.json` 的 `cases` 非空，补齐上一版对照：

```bash
"$PY" -m tools.legacy_failure_provenance \
  --manifest "$CENSUS_ROOT/baseline_control_manifest.json" \
  --output-root "$CENSUS_ROOT/baseline_controls" --workers 8 --timeout 180

"$PY" -m tools.legacy_failure_upstream \
  --manifest "$CENSUS_ROOT/baseline_control_manifest.json" \
  --root "$CENSUS_ROOT/baseline_controls" --workers 8

"$PY" -m tools.legacy_failure_saved_step \
  --manifest "$CENSUS_ROOT/baseline_control_manifest.json" \
  --root "$CENSUS_ROOT/baseline_controls" --workers 8 --timeout 180
```

依赖关系：两侧provenance互不依赖，可以并行；两侧saved_step只读原评估STEP，也可并行，不需要等provenance。每侧upstream须等该侧provenance完整结束，避免读取仍在变化的trace/geometry。建议第一阶段最多32CPU并发，第二阶段再并行upstream；不因增加机器并行而改变单件超时和优化预算。

## 全部fit-only成功、optimized失败路径对照

这部分没有采样；`optimization_regression_ids.txt` 中每一件都要处理。为空则跳过runner，报告会显式输出零。仅需provenance和upstream，不重复saved_step。

下面Python入口安全地传递全部ID，避免shell分词；如已有smoke，给两个命令各追加 `--resume`。

```bash
"$PY" - <<'PY'
import json
from pathlib import Path
import subprocess
import sys
root = Path('checkpoints/intersection_local/failure_census_20261013')
ids = json.loads((root / 'manifest.json').read_text())['optimization_regression_ids']
if ids:
    subprocess.run([sys.executable, '-m', 'tools.legacy_failure_provenance',
        '--manifest', str(root / 'manifest.json'),
        '--output-root', str(root / 'fit_only_counterfactual'),
        '--post-mode', 'fit_only', '--workers', '8', '--timeout', '180', '--only-ids', *ids], check=True)
    subprocess.run([sys.executable, '-m', 'tools.legacy_failure_upstream',
        '--manifest', str(root / 'manifest.json'),
        '--root', str(root / 'fit_only_counterfactual'), '--workers', '8', '--only-ids', *ids], check=True)
else:
    print('No fit-only-valid / optimized-failed cases in the frozen population.')
PY
```

fit-only重放独立于optimized重放，可在预算允许时并发；它自己的upstream仍需等待fit-only trace结束。报告核对相同raw，以及选择后的面/边/connectivity逐元素一致性。此对照包含后续拟合裁剪变化，不能解释为单一优化loss的因果消融。

## 汇总、来源保护与页面检查

上述所有应运行诊断结束后：

```bash
"$PY" -m tools.paired_failure_report \
  --census-root "$CENSUS_ROOT" \
  --baseline-census-root checkpoints/intersection_refinement/failure_census_20261011
```

报告拒绝未结束、错method、错协议或变化的诊断来源；历史复用直接核对原manifest与原诊断。自动写 `source_preservation.json` 检查冻结来源未修改。终态timeout/error仍保留分母，其阶段证据标未知，不改原成功/失败。仅重新生成已核验的报告时使用 `--overwrite`。

优先打开[自包含离线页面](page/standalone.html)。它内嵌数据、四图与下载文件，不依赖HTTP服务。打包命令为 `python checkpoints/intersection_local/failure_census_20261013/build_standalone.py`；须在最终NOTE写完后运行，`standalone_packaging.json`保存所有打包源和输出哈希。

普通多文件页面的可选HTTP地址为 <http://127.0.0.1:8766/checkpoints/intersection_local/failure_census_20261013/page/>；临时服务不会在对话外永久运行，需要时从仓库根目录执行 `python -m http.server 8766 --bind 127.0.0.1`。浏览器检查命令：

```bash
FONTCONFIG_FILE=/root/.cache/hola-browser-libs/fonts.conf \
LD_LIBRARY_PATH=/root/.cache/hola-browser-libs/root/usr/lib/x86_64-linux-gnu \
node tools/paired_failure_browser_check.cjs \
  http://127.0.0.1:8766/checkpoints/intersection_local/failure_census_20261013/page/ \
  "$CENSUS_ROOT/browser_validation" \
  /root/.cache/ms-playwright/chromium-1155/chrome-linux/chrome
```

检查涵盖四组×四阈值的所有观察与成对转换计数、ID详情、资源、窄屏溢出与JS错误，保存桌面/手机截图及JSON。自包含版另用本目录 `check_standalone.cjs` 检查同一交互矩阵，并禁止HTTP/HTTPS请求、确认全部四张图内嵌且能够解码；输出到 `standalone_validation/`。

## 解释边界

距离超阈值和OCC错误码是观察，不是唯一根因；多标签不能相加成总失败数。闭合环不保证可裁剪，近边不证明应该焊合，丢wire不证明丢孔，合法solid也不保证完整准确重建。已有50件原HoLa三维图库只作历史参考，本轮不伪造新的3D对照。
