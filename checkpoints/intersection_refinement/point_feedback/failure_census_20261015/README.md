# 逐点曲线反馈：全集失败溯源

比较 `point_feedback`（逐点曲线反馈）与直接基线 `conditioned`（曲线条件调制）。按验证集选择候选 epoch34 / step12705；权重及训练曝光以 [训练审计](../evaluation_20261015/training_audit.json) 为准。固定2417件，原2424中7件合同不支持另列。原 refinement 的三级历史比较在评估报告呈现，不冒充本轮直接baseline。

当前状态：原始评估、两类后处理及完整双侧诊断已完成。313失败、50救回共363件完成双侧精确重放；321份baseline历史诊断核验复用，补跑42份，另有19件优化路径对照。来源资产变化0；观察和解释见 [REPORT](REPORT.md)、[NOTE](NOTE.md)。展示验证结果在下方路径记录。执行口径见 [冻结计划](PLAN.md)，进度见 `metadata/orchestration/state.json`。

## 执行

科学 runner 及依赖在真实 smoke 前记录到 `metadata/source_freeze.json`，调度器在每个阶段启动前检查哈希。等本轮两种 post 表全部终态，冻结全部候选失败和全部救回；先核验少量真实重放的原终态、STEP DATA 一致，再 resume 全集。

```bash
cd /mnt/d/python_curve_eval
/mnt/d/anaconda3/envs/torch/bin/python \
  checkpoints/intersection_refinement/point_feedback/failure_census_20261015/run_census.py
```

同一 root 不重复启动调度器。中断后按状态文件中原命令逐阶段 `--resume`；禁止改科学 runner 后静默续接。每 runner8并发、每进程1线程、每件180秒，最多4个 runner并行。同侧 upstream 等 trace 完成，saved STEP 独立。原始和历史文件只读。

冻结入口的方法必须显式：

```bash
/mnt/d/anaconda3/envs/torch/bin/python -m tools.paired_failure_manifest \
  --baseline-method conditioned --candidate-method point_feedback \
  --comparison-root /mnt/d/python_curve_eval/checkpoints/intersection_refinement/point_feedback/evaluation_20261015 \
  --reference-root /mnt/d/python_curve_eval/checkpoints/intersection_refinement/curve_conditioning/evaluation_20261014 \
  --baseline-census-root /mnt/d/python_curve_eval/checkpoints/intersection_refinement/curve_conditioning/failure_census_20261014 \
  --data-root /mnt/d/data/deepcad_v7 \
  --split-list /mnt/d/python_curve_eval/checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt \
  --output-root /mnt/d/python_curve_eval/checkpoints/intersection_refinement/point_feedback/failure_census_20261015
```

候选诊断在本目录；baseline 缺失对照在 `baseline_controls`；候选所有 fit-only 成功而 optimized 失败的路径对照在 `fit_only_counterfactual`。复用的 conditioned 历史路径、method 和协议指纹原样保留。

## 报告与离线页面

调度器在全部诊断终态后生成 `REPORT.md`、`compact_summary.json`、`cases.csv`、四组 PNG/SVG 和 `page/index.html`。四图沿用多标签观察、同 ID 成对变化、独立拟合/周期替换贴合、最终失败面候选检查。交互页保留四组×四阈值、正/负/未知及双侧 ID 详情。只有经来源检查的展示重建可加 `--overwrite`。

```bash
/mnt/d/anaconda3/envs/torch/bin/python -m tools.paired_failure_report \
  --census-root /mnt/d/python_curve_eval/checkpoints/intersection_refinement/point_feedback/failure_census_20261015 \
  --baseline-census-root /mnt/d/python_curve_eval/checkpoints/intersection_refinement/curve_conditioning/failure_census_20261014
```

最终 `NOTE.md` 完成后执行 `build_standalone.py`，把四图、统计、脚本及下载文件内嵌到 `page/standalone.html`。本地直接打开，不依赖 HTTP；`standalone_packaging.json` 记录所有打包输入和输出 SHA。打包不重算测量。

## 展示检查

普通页面只用临时 `127.0.0.1:8767` 本地服务，调用 `tools/paired_failure_browser_check.cjs`；离线页面调用本目录 `check_standalone.cjs`。浏览器环境：

```bash
FONTCONFIG_FILE=/root/.cache/hola-browser-libs/fonts.conf \
LD_LIBRARY_PATH=/root/.cache/hola-browser-libs/root/usr/lib/x86_64-linux-gnu \
node tools/paired_failure_browser_check.cjs \
  http://127.0.0.1:8767/checkpoints/intersection_refinement/point_feedback/failure_census_20261015/page/ \
  checkpoints/intersection_refinement/point_feedback/failure_census_20261015/browser_validation \
  /root/.cache/ms-playwright/chromium-1155/chrome-linux/chrome
```

离线检查使用 `file:///mnt/d/python_curve_eval/checkpoints/intersection_refinement/point_feedback/failure_census_20261015/page/standalone.html`，输出到 `standalone_validation`。两者保存 `browser_check.json`，遍历四组四阈值及全部观察转换，检查详情、资源、390px布局和 JS；离线阻断 HTTP/HTTPS 并要求四张内嵌图可解码。

## 解释边界

多标签可以重叠；未知留在分母。最早取得证据的位置不一定是错误起点，末端阻断不是唯一根因。闭环不保证可裁剪，合法solid不保证几何精确，没有候选可检查不算检查通过。额外训练曝光与新模块同时变化，不作为等预算因果消融。
