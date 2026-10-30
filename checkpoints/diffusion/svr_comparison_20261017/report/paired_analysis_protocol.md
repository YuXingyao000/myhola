# 逐对象配对 bootstrap

`analyze.py` 是独立的报告统计脚本，只读取已完成的 CSV/JSON，不加载模型、几何或 GT，也不修改推理与评估结果。
当前状态：实现与合成数据 self-check 通过；已对完整终态报告运行正式分析，结果见 `paired_analysis.json`，结论及限制见上级主题 README。

比较固定为 `hola_1500k`、`current_best_54639`、`current_last_182130` 三个主模型的全部三组两两配对。
每个差值均为 **right − left**；JSON 同时保存两侧在该配对集合上的原均值、差值均值、共同样本数和 ID、逐对象差值及 95% percentile CI。
差值数组与 `ids` 数组逐项对应，样本顺序使用冻结清单，配对不依赖 CSV 行顺序。

| 指标 | 分母 | right 更好的方向 |
|---|---|---|
| raw face CD² | 两模型该指标均有限的共同 CAD ID | 差值为负 |
| raw face primitive F1 | 两模型该指标均有限的共同 CAD ID | 差值为正 |
| native valid solid binary | 全部冻结 100 个 CAD ID | 差值为正 |

连续指标各自选择共同可计算集合，不给缺失数值填零，也不要求边指标或整份 raw export 完全成功。
因此其 CI 条件于该共同集合，不能替代全集覆盖率。零个共同样本时均值/CI 为 null，并明确记录状态。

有效性使用 `native_valid_solid` 阶段的实际结果：明确 True 记 1，False 或空字段记 0；
已有 True 不因后续 `geometry_error` 或 `timeout` 被清零。这与原报告的 native validity 口径一致，不是整条流水线全部成功率。
缺失输出/失败样本保留在 100 件分母中，但必须存在对应的终态记录；缺行或 `not_run`/`running` 不能当作失败零值吞掉。
另输出双方都有效、双方都无效、left 有效/right 无效、left 无效/right 有效四种转换计数，以及有效率差的百分点表示。

每一对模型、每一个指标用 NumPy `default_rng(42)` 从配对 CAD ID 中有放回抽取同样的 n 个 ID，重复 10,000 次，
每次计算该配对差值的均值。取 2.5% 和 97.5% 百分位，使用 linear quantile interpolation。
每个比较重置 seed 42，新增其他指标不会改变已有区间；不会分别重采样两个模型从而破坏配对。

这是 **validation100、K=1 下对 CAD 对象重采样的不确定性**，条件于已经保存的那次生成。
区间不包含生成 seed 波动、重训波动或选择 checkpoint 带来的不确定性，也未对多模型/多指标比较作多重校正。
它不是等训练预算架构消融、test 结果或真实照片泛化结论。

## 输入与完成条件

同主题目录必须已有冻结 `validation100.txt` 和完整报告：

- `report/raw_per_object.csv`
- `report/post_per_object.csv`
- `report/inference_per_object.csv`
- `report/summary.json`

脚本检查冻结清单 SHA256、summary 的 inference/post 全终态标志，以及三主模型在三个 CSV 中各 100 个唯一 ID。
inference 只接受 `ok/error`，post 接受当前固定 runner 的终态集合；未识别状态显式报错。
报告文件在读取到分析完成之间发生变化也会报错，避免混合两次报告。
输出 `report/paired_analysis.json` 包含所有输入文件 hash、脚本 hash 和 NumPy 版本。

## 调用

合成数据检查不读取已有实验输出：

```bash
/mnt/d/anaconda3/envs/torch/bin/python \
  checkpoints/diffusion/svr_comparison_20261017/analyze.py --self-check
```

在全部正式报告结束并确认开始分析后执行：

```bash
/mnt/d/anaconda3/envs/torch/bin/python \
  checkpoints/diffusion/svr_comparison_20261017/analyze.py
```

自检覆盖恒定配对差的退化 CI、平衡二元差值、无共同数值、指标各自分母、CSV 乱序、
100 件有效率及 timeout 后已完成的有效性、缺行/重复/非终态拒绝，以及固定 seed 的结果可重复性。
