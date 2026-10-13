# 局部注意力 intersection：全集失败溯源

本目录比较本版局部注意力与**上一版 Intersection refinement**，沿用[冻结评估协议](../evaluation_20261012/PLAN.md)及本目录[诊断计划](PLAN.md)。机器方法名分别为 `local`、`intersection`；不将上一版误标为原HoLa。

正式全集诊断已完成。接收的2417件中，上一版合法实体2113件，本版482件；本版失败1935件，包含共同失败301件和退步1634件，另有3件救回。完成本版1938件诊断，复用上一版407件已有诊断并补齐1531件同ID对照；全部5件fit-only成功但optimized失败的对象均已完成路径重放。原始2424件中的7件合同不支持对象仍单列。

打开[交互统计页面](http://127.0.0.1:8766/checkpoints/intersection_local/failure_census_20261012/page/)，或阅读[中文报告](REPORT.md)、[研究判断](NOTE.md)和[紧凑统计](compact_summary.json)。页面保留本版失败全集、两版都失败、退步、救回四组，以及 `.002/.006/.01/.08` 归一化长度阈值、ID展开和同ID双列详情。四组PNG/SVG为固定全集统计图，不随页面筛选变化。

阶段证据核验：本版1937/1938件重放与原STEP的DATA完全一致；`00748659`保留原`Transfer failed`异常，仍计入失败分母，其依赖完整重放的观察记为未知。新增上一版1531件和fit-only路径5件均通过同一核验，无instrumentation错误。24909项来源资产哈希核对无变化。具体命令、PID与起止时间见[执行日志](metadata/orchestration/events.jsonl)，自动检查及人工图像检查见[验证记录](validation_summary.json)。下文保留完整复现命令。

## 输入冻结

先完成本轮2417件的raw、fit-only和optimized表，全部终态后运行。原2424中7件合同不支持对象单列；不能按已完成目录缩分母。

从仓库根目录执行：

```bash
PY=/mnt/d/anaconda3/envs/torch/bin/python
CENSUS_ROOT=checkpoints/intersection_local/failure_census_20261012
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

"$PY" -m tools.paired_failure_manifest \
  --comparison-root checkpoints/intersection_local/evaluation_20261012 \
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
root = Path('checkpoints/intersection_local/failure_census_20261012')
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

静态页面地址为 <http://127.0.0.1:8766/checkpoints/intersection_local/failure_census_20261012/page/>。当前服务从仓库根目录启动；需要重新启动时执行 `python -m http.server 8766 --bind 127.0.0.1`。浏览器检查命令：

```bash
FONTCONFIG_FILE=/root/.cache/hola-browser-libs/fonts.conf \
LD_LIBRARY_PATH=/root/.cache/hola-browser-libs/root/usr/lib/x86_64-linux-gnu \
node tools/paired_failure_browser_check.cjs \
  http://127.0.0.1:8766/checkpoints/intersection_local/failure_census_20261012/page/ \
  "$CENSUS_ROOT/browser_validation" \
  /root/.cache/ms-playwright/chromium-1155/chrome-linux/chrome
```

检查涵盖四组×四阈值的所有观察与成对转换计数、ID详情、资源、窄屏溢出与JS错误，保存桌面/手机截图及JSON。

## 解释边界

距离超阈值和OCC错误码是观察，不是唯一根因；多标签不能相加成总失败数。闭合环不保证可裁剪，近边不证明应该焊合，丢wire不证明丢孔，合法solid也不保证完整准确重建。已有50件原HoLa三维图库只作历史参考，本轮不伪造新的3D对照。
