# BREP Evaluation 说明

这个目录只负责实验最后一步 evaluation。后处理阶段的输入约定是：

- 预测目录：`pred_root/<sample_name>/`
- 主要预测文件：`recon_brep.step`
- solid 成功标记：`success.txt`
- 可选 mesh：`recon_brep.stl`
- GT 目录：`gt_root/<sample_name>/`
- 主要 GT 文件：`normalized_shape.step`

统一入口：

```bash
python -m src.brepnet.eval.run \
  --pred-root /path/to/pred_post \
  --gt-root /path/to/deepcad_v6 \
  --split-list src/brepnet/data/list/test.txt \
  --metrics condition,validity
```

旧入口已经移到 `legacy/compat/`，只用于临时兼容，例如：

```bash
python -m src.brepnet.eval.legacy.compat.eval_condition --eval_root PRED --gt_root GT --list SPLIT
python -m src.brepnet.eval.legacy.compat.check_valid --data_root PRED
```

## 视角和旋转协议

新协议默认只评估视角 0，也就是 identity rotation：

```bash
--rotation-policy none
```

历史数据曾用 `x/y/z` 三轴各 4 个 90 度 Euler 旋转，得到 64 个 view id；这些矩阵实际只对应立方体旋转群的 24 个唯一姿态，并且重复次数不均匀。公共旋转协议在：

```text
src/brepnet/data/rotations.py
```

旧数据修复时才使用：

```bash
--rotation-policy search24
```

如果已知某个样本的 canonical rotation id，可使用：

```bash
--rotation-policy known --rotation-id 7
```

## Metric Families

当前整理成 6 类 metric family。

### 1. Validity

文件：`metrics/validity.py`

检查预测 STEP 是否存在、是否是 valid solid，并统计基础拓扑数量：

- `has_step`
- `is_valid_solid`
- `has_success_marker`
- `num_faces`
- `num_edges`
- `num_vertices`

输出：

- 每样本：`eval_validity.npz`
- 全局：`eval_validity.csv`
- 全局：`eval_validity_summary.json`
- 图：`reports/validity_faces_edges.png`

### 2. Condition / BREP Reconstruction

文件：`metrics/condition.py`

比较预测 BREP 和 GT BREP 的几何与拓扑一致性：

- face/edge/vertex Chamfer
- face/edge/vertex precision、recall、F-score
- FE topology precision、recall、F-score
- EV topology precision、recall、F-score
- recon/GT 的 face、edge、vertex 数量
- rotation policy 和实际使用的 rotation id

输出：

- 每样本：`eval_condition.npz`
- 旧兼容可选：`eval.npz`
- 失败：`eval_error.txt`
- 全局：`eval_condition.csv`
- 全局：`eval_condition_summary.json`
- 图：`reports/condition_face_chamfer.png`

### 3. Complexity

文件：`metrics/complexity.py`

描述生成 solid 自身复杂度，不比较 GT：

- `num_faces`
- `num_edges`
- `num_vertices`
- `cyclomatic_complexity`
- `mean_curvature`

输出：

- 每样本：`eval_complexity.npz`
- 全局：`eval_complexity.csv`
- 全局：`eval_complexity_summary.json`

### 4. Unique / Novel

文件：`metrics/uniqueness.py`

把样本转成 face graph：节点是 face geometry，边是 face adjacency。

- Unique：生成集中互相不重复的比例
- Novel：生成结果是否不在训练集中

Novel 默认沿用历史加速协议：先在 `fake_post/<sample>/nearest.txt` 里提供候选训练样本，再只对候选做 graph identical 检查。

输出：

- `eval_unique_summary.json`
- `eval_unique_novel_summary.json`
- 重复 component / non-novel prefix 列表

### 5. Point Cloud Set

文件：`metrics/point_cloud_set.py`

在采样点云集合上比较生成分布和 GT 分布：

- `MMD-CD`
- `COV-CD`
- `JSD`
- 点云均匀性 helper：NND / Clark-Evans R

`sample_points.py` 和 `sample_points_gt.py` 是点云采样预处理，不是 metric 本体。

### 6. LFD

文件：`metrics/lfd.py`

LFD 是外部工具链，不和普通 Python metric 混在一起。外部代码保留在：

```text
src/brepnet/eval/lfd/evaluation_scripts/
```

一般流程：

1. 对生成结果提取 LFD feature
2. 对 GT 提取 LFD feature
3. 计算 LFD matrix pickle
4. 用 `metrics/lfd.py` 或 `legacy/compat/viz_lfd.py` 汇总和画图

输出：

- `eval_lfd_summary.json`
- `reports/lfd_distribution.png`
- 可选 nearest pair 可视化：`reports/lfd_nearest_pairs.ply`

## 目录说明

```text
run.py                  # 统一入口
protocol.py             # 样本枚举、split list、标准文件名
io.py                   # npz/json/csv/error 写入
parallel.py             # 单进程/Ray 调度
metrics/                # 每类 metric 一个文件
legacy/                 # 旧大脚本归档，保留参考和回滚依据
lfd/evaluation_scripts/ # 外部 LFD 工具链
```

`eval` 根目录不再保留旧脚本名。新逻辑应写到 `metrics/` 或 `adapters/`；旧命令需要迁移到 `run.py`，临时兼容脚本放在 `legacy/compat/`。
