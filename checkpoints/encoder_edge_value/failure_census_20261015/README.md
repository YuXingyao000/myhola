# Encoder 三个固定快照的全集失败溯源

本目录比较 `encoder_edge_value` 的验证最优step1089与历史 `conditioned`；`control_comparison/` 比较同为step1089的 `encoder_gat_control` 与相同历史基线；`endpoint_14520_comparison/` 单独分析新结构的固定末次step14520。主结构比较需要合并同曝光两组相同ID结果，不能将任何一组相对历史基线的差异全归因于新消息MLP。协议见 [PLAN](PLAN.md)、[端点计划](endpoint_14520_comparison/PLAN.md) 与 [整体评估计划](../evaluation_20261015/PLAN.md)。

当前状态：三组全部失败及救回双侧重放、fit-only退步路径、独立统计审计、四张中文图、普通与自包含离线页面均完成。每组普通/离线浏览器检查均通过4组×4阈值、28项观察转换、逐ID详情、四图及390px布局；离线检查阻断HTTP/HTTPS，外部请求与脚本错误均为0。

固定2417件、历史基线2111个optimized合法solid。下表仅表示各快照相对同一历史基线的配对结果；三个失败集合不同，不能直接用各自失败标签百分比解释机制变化。

| 快照 | optimized合法solid | 新增失败 / 救回 | 双侧诊断ID | 结果笔记 | 离线页 |
|---|---:|---:|---:|---|---|
| 新GAT，验证最优step1089 | 2118 | 47 / 54 | 353 | [NOTE](NOTE.md) | [打开](page/standalone.html) |
| 原GAT解冻对照，验证最优step1089 | 2110 | 52 / 51 | 358 | [NOTE](control_comparison/NOTE.md) | [打开](control_comparison/page/standalone.html) |
| 新GAT，固定末次step14520 | 2095 | 64 / 48 | 370 | [NOTE](endpoint_14520_comparison/NOTE.md) | [打开](endpoint_14520_comparison/page/standalone.html) |

三组共2162条双侧轨迹均复现原终态和STEP DATA，无重放未知；独立审计分别核对79,072、80,192、82,880个主观察判定。fit-only合法而optimized失败分别13、15、12件，完整路径对照均已核验。判定所需的派生几何未知仍保留分母。用户已选择从末次状态按原配置继续训练，本目录只记录续训前固定快照，不写入后续权重或结果。

两个 Encoder 已训练，使用同一 GT CAD 与规范面顺序分别取得各自 posterior mean；`same_geometry_source` 明确验证源 `data.npz` SHA 与各自 latent SHA。旧 `identical_latents` 默认校验保持不变。逐 ID 诊断几何公式、采样、四阈值及生产后处理均不变。

## 冻结与运行

源码适配与35项相关测试完成后，已分别冻结三组来源；以下为实际使用入口，重现时不要覆盖现有产物。每个group只冻结一次，禁止覆盖已有source freeze。

```bash
cd /mnt/d/python_curve_eval
/mnt/d/anaconda3/envs/torch/bin/python checkpoints/encoder_edge_value/failure_census_20261015/run_census.py --group candidate --freeze-sources
/mnt/d/anaconda3/envs/torch/bin/python checkpoints/encoder_edge_value/failure_census_20261015/run_census.py --group control --freeze-sources
/mnt/d/anaconda3/envs/torch/bin/python checkpoints/encoder_edge_value/failure_census_20261015/run_census.py --group endpoint_14520 --freeze-sources
```

确认各组 post 两模式均完成后运行：

```bash
/mnt/d/anaconda3/envs/torch/bin/python checkpoints/encoder_edge_value/failure_census_20261015/run_census.py --group candidate
/mnt/d/anaconda3/envs/torch/bin/python checkpoints/encoder_edge_value/failure_census_20261015/run_census.py --group control
/mnt/d/anaconda3/envs/torch/bin/python checkpoints/encoder_edge_value/failure_census_20261015/run_census.py --group endpoint_14520
```

各自有进程锁；中断后核对来源与子进程状态再用 `--resume`。任务保存 `metadata/orchestration/state.json`、阶段命令/日志与smoke原终态和STEP DATA核验。每runner8并发，每组最多4 runner；主比较两组最高64并发，端点独立调度，与所有其他后处理/训练CPU worker合计不超过128。不要并行启动相同group。

全部候选失败+救回双侧诊断；历史 `conditioned` 文件只读核验，缺失在组内 `baseline_controls/` 补跑。所有 fit-only合法但optimized失败在组内 `fit_only_counterfactual/` 跑完整路径对照。空集合显式0，未知留分母。

## 四图与离线页面

诊断完毕生成中文 `REPORT.md`、`compact_summary.json`、`cases.csv`、四张 PNG/SVG、交互页；互斥末端阻断与可重叠几何/拓扑观察分开。四个筛选组为本版失败全集、持续失败、新增失败、救回，各有 .002/.006/.01/.08 四个归一化长度阈值。

在最终 `NOTE.md` 写完后，`build_standalone.py --root <census_root>` 把数据、四图、脚本和报告下载全部内嵌进 `page/standalone.html`。不依赖网页服务；打包记录 `standalone_packaging.json` 来源和输出 SHA。该步骤不重算测量。

普通页面用现有 `tools/paired_failure_browser_check.cjs` 检查，离线页面用本目录 `check_standalone.cjs` 检查；两者遍历4组×4阈值及全部观察转换，核对 ID详情、四图和390px布局。浏览器环境沿用 `FONTCONFIG_FILE=/root/.cache/hola-browser-libs/fonts.conf`、`LD_LIBRARY_PATH=/root/.cache/hola-browser-libs/root/usr/lib/x86_64-linux-gnu`、Chrome `/root/.cache/ms-playwright/chromium-1155/chrome-linux/chrome`。离线检查屏蔽HTTP/HTTPS并要求无外部依赖。

独立统计审计入口为 `review_census.py --root <census_root>`。它复用并记录此前独立审计谓词的SHA，不导入报告管线的判定函数；重新核对来源、原终态、STEP DATA、28观察×4阈值、成对转换、最终失败面和fit路径选择数组。这里不重新计算几何采样。各组 `independent_review.json`、`browser_validation/browser_check.json`、`standalone_validation/browser_check.json` 与 `completion.json` 保存核验记录。

高频贴面告警不是失败充分条件；闭合环不保证裁剪、没有候选可检查不代表通过。解冻/KL/曝光与新架构的影响应结合原GAT同曝光对照解释，不外推旧cache/diffusion或照片效果。
