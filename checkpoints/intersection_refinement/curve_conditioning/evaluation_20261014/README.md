# 曲线条件调制：与原 refinement 的完整对比

在固定 2417 件 accepted test、同一组 float32 latent 和同一后处理预算下，比较原 `intersection` 与本轮 `conditioned`。原2424件中7件合同不支持样本另列。科学协议见 [PLAN](PLAN.md)，训练审计见 [training_audit.json](training_audit.json)，独立复算见 [independent_review.json](independent_review.json)。

2026-10-14完整测试及失败诊断已完成：[指标报告](REPORT.md)、[研究判断](NOTE.md)、[失败统计](../failure_census_20261014/REPORT.md)、[四组图表与逐ID离线页面](../failure_census_20261014/page/standalone.html)。优化后合法solid为2113→2111；面原始精度和修复后边几何有收益，原始边中位误差与最终有效性没有改善。

当前使用验证回调选出的 epoch157/step57354，SHA256 `4c7910b8169206c74fcf38c358cf171359149e0429c8debbc227d391aa56613f`。模型加载严格匹配实际配置；243个冻结 encoder/posterior 张量与基线完全一致。最佳权重之后的 last epoch184/step67155不用于测试选优。候选在原baseline之上额外训练57354步，因此不能将结果视为等预算结构消融。

评估在 `/mnt/d/python_curve_eval` 独立 worktree 进行；原 `/mnt/d/python` 当前为另一研究任务的分支，不切换或修改其未提交工作。历史 raw、STEP 与诊断只读复用，原生产post的11个文件SHA均一致，见 [baseline_algorithm_audit.json](baseline_algorithm_audit.json)。

## 执行与复现

运行状态与所有实际命令、退出码、日志位于 `metadata/orchestration/`。脚本拒绝覆盖已有阶段；完整复现需先设置新的输出目录，并使用冻结的checkpoint、split及协议。不要把同一路径重复执行当作新实验。

```bash
cd /mnt/d/python_curve_eval
PY=/mnt/d/anaconda3/envs/torch/bin/python
RUN=checkpoints/intersection_refinement/curve_conditioning/evaluation_20261014/run_evaluation.py
"$PY" "$RUN" smoke
"$PY" "$RUN" inference_posterior_mean
"$PY" "$RUN" inference_archived_latents
"$PY" "$RUN" metrics_posterior_mean
"$PY" "$RUN" metrics_archived_latents
"$PY" "$RUN" post
```

两个mode各8个独立GPU分片，FP32、TF32 off、seed42，严格逐值复用旧归档latent。mean重编码只作容差1e-5的数值审计，不改变实际decoder输入。posterior mean的后处理分为fit_only/optimized，每mode32个单线程worker，每对象全流程180秒，drop_num=0，名义优化200/原算法自适应600。原生TopAbs_SOLID+BRepCheck为主有效性，兼容0.1口径单列；原历史耗时不作为受控速度基线。

正式结果齐后，运行同目录 `diagnostics/raw_distribution_analysis.py`（如需重建，请先核对其输出覆盖保护），再通过既有报告入口生成图表：

```bash
"$PY" -m tools.local_evaluate_report \
  --comparison-root checkpoints/intersection_refinement/curve_conditioning/evaluation_20261014 \
  --candidate-method conditioned \
  --failure-census-link ../failure_census_20261014/REPORT.md
```

分布诊断见 [diagnostics/RAW_DISTRIBUTION.md](diagnostics/RAW_DISTRIBUTION.md)。失败报告及离线图表页面的生成和核验见 [全集失败溯源](../failure_census_20261014/README.md)。诊断依赖正式后处理终态，不用半完成样本估计全测试集成功率。

## 解释范围

posterior mean属于VAE重建测试；archived_latents属于同一批既有diffusion输入的decoder重放，没有重新采样模型，不证明完整实拍重建能力改善。平方CD使用归一化XYZ距离，不能直接当作线性CAD tolerance；均值、分位数、逐件改善和最终实体有效性必须结合阅读。失败距离/OCC错误是证据，不是唯一因果标签；多标签不相加为失败总数。
