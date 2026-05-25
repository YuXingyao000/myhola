# Evaluation 问题记录（2026-05-25）

## 结论

今天发现：之前一批 conditional diffusion / topology bias 实验的 evaluation 结果不能直接信。指标异常差的主要原因不是模型 loss 低但几何真的差，而是 evaluation 协议和训练 / 推理使用的视角协议不一致，尤其是 `eval_brep.py` 这个 legacy 入口本身也不适合当前 pipeline。

最关键的事实：当前代码里 `cube24` 的 `view 0` 不是 identity pose。训练/测试 dataset 在 `is_aug=0` 时仍会使用 `cube_id=0`，再映射到 cached latent 的 Euler-64 id；这个 id 是 `4`，不是 `0`。而默认 evaluation 用 identity GT 比较，所以预测和 GT 处在不同姿态，Chamfer / F-score 会系统性变差。

## 直接证据

本地打印得到：

```text
dataset cube24_to_euler64[0] = 4
dataset cube24 matrix 0 =
[[ 0.  0.  1.]
 [ 0.  1.  0.]
 [-1.  0.  0.]]
is identity? False
```

这说明“输入图片都是 0 号视角”不等于“预测是 identity 坐标系”。在统一 eval 的 24 个旋转表里：

```text
cube0 same matrix -> rotation_id = 12
cube0 inverse/T   -> rotation_id = 4
```

抽样验证 `00007186`，同一份预测只改 evaluation rotation：

```text
identity / rotation_id=0:
face_cd   0.03899
edge_cd   0.08551
vertex_cd 0.14544
face_f    0.70
edge_f    0.42
vertex_f  0.25

known / rotation_id=12:
face_cd   0.00223
edge_cd   0.00453
vertex_cd 0.00579
face_f    ~1.00
edge_f    ~1.00
vertex_f  ~1.00

search24:
best_rotation_index = 12
指标与 known rotation_id=12 基本一致
```

所以至少这类样本的低指标是姿态错配导致的，不是模型结果本身差到这个程度。

## `eval_brep.py` 的具体问题

文件：`src/brepnet/post/legacy/eval_brep.py`

### 1. 它是 legacy 入口，不匹配当前 `construct_brep` 输出

当前后处理主要输出：

```text
<post_root>/<sample>/recon_brep.step
<post_root>/<sample>/recon_brep.stl  # solid 成功时才有
<post_root>/<sample>/success.txt     # valid solid 标记
```

但 `eval_brep.py` 的 face 指标读取的是：

```text
<post_root>/<sample>/recon_face/*.stl
```

如果没有 `recon_face` 目录，它不会从 `recon_brep.step` 抽 face 来评估，`num_recon_face` 会保持 0。后面的 `compute_statistics()` 又把 `num_recon_face == 0` 当作 exception 跳过。也就是说它天然不适配现在的 pipeline。

相关位置：

```text
src/brepnet/post/legacy/eval_brep.py:174-189  # face metric 只看 recon_face/*.stl
src/brepnet/post/legacy/eval_brep.py:318-322  # num_recon_face == 0 被当 exception
```

### 2. 没有 rotation policy

`eval_brep.py` 总是直接比较：

```text
pred recon_brep.step/stl  vs  gt normalized_shape.step / mesh.ply
```

它没有 `--rotation-policy known/search24`，也不会知道训练时使用的是 `cube_id=0 -> euler64_id=4` 的 latent pose。对于当前 single-view FLUX / cube24 数据，这会把旋转后的预测和 identity GT 硬比，指标必然差。

### 3. GT mesh / STEP / predicted STEP 混合使用，协议不统一

`eval_brep.py` 的 stl CD 用 GT `mesh.ply`，edge/vertex CD 用 GT `normalized_shape.step`，face CD 又依赖 pred `recon_face/*.stl`。这和新的 `src.brepnet.eval.run` / `metrics.condition` 不是同一套协议。不同 metric family 混在一起，容易出现 face、edge、vertex 的口径不一致。

### 4. 硬编码 CUDA

`eval_brep.py` 里直接：

```python
device = torch.device('cuda')
```

没有 CPU fallback。环境/机器不同会直接失败或行为不稳定。

### 5. STEP 类型判断也偏 legacy

`eval_brep.py` 里用 `gen_shape.ShapeType() == TopoDS_Shell` 判断 shell，但 `TopoDS_Shell` 是类，不是 `TopAbs_SHELL` 枚举。这个分支很可疑，至少不能作为当前 validity 结论的可靠来源。新的 validity 应该使用 `src.brepnet.eval.metrics.validity`。

## `src.brepnet.eval.run` / `condition` 的问题

统一入口比 `eval_brep.py` 更接近当前 pipeline，但默认参数仍然会踩坑：

```bash
python -m src.brepnet.eval.run --metrics condition
```

默认是：

```text
--rotation-policy none
```

这等价于只用 `rotation_id=0`，而 `rotation_id=0` 在 `src/brepnet/data/rotations.py` 里是 identity。它和 `dataset.py` 的 cube24 view-0 不是同一个概念。

因此对于当前 0 号 FLUX/single-view 条件，如果目标 latent 使用的是 dataset 的 `cube_id=0`，更合理的临时评估是：

```bash
python -m src.brepnet.eval.run \
  --pred-root /path/to/post_root \
  --gt-root /mnt/d/data/deepcad_v6 \
  --metrics condition,validity,complexity \
  --rotation-policy known \
  --rotation-id 12 \
  --from-scratch \
  --use-ray --num-cpus 16
```

如果不确定某批预测到底是哪一个姿态，先用诊断模式：

```bash
python -m src.brepnet.eval.run \
  --pred-root /path/to/post_root \
  --gt-root /mnt/d/data/deepcad_v6 \
  --metrics condition \
  --rotation-policy search24 \
  --from-scratch \
  --use-ray --num-cpus 16
```

如果 `best_rotation_index` 大量集中在 12，说明这批基本就是 dataset cube0 pose。若分布很散，则说明数据/推理还有更大的 pose 或样本配对问题。

## 受影响范围

需要重新审计所有使用以下入口或默认参数得到的结果：

- `src/brepnet/post/legacy/eval_brep.py`
- `python -m src.brepnet.eval.run --metrics condition` 但没有显式设置 `--rotation-policy`
- 使用 `dataset.is_aug=0`、`single_img` / FLUX / cube24 view0 条件训练或测试的实验
- 所有把 `view 0` 当成 identity 来解释的指标和可视化结论

特别是这些结果目前不能直接用于论文或最终对比：

- topology bias vs baseline 的 condition 指标
- real photo / FLUX single-view 的 face/edge/vertex CD 与 F-score
- 任何基于 `eval_brep.py` 输出的 face 指标或 valid-solid 分组统计

## 临时可靠流程

1. 后处理仍用当前入口：

```bash
python -m src.brepnet.post.construct_brep \
  --data_root /path/to/raw_npz \
  --out_root /path/to/post_root \
  --use_ray --use_cuda --num_cpus 16
```

2. 对当前 single-view FLUX view0 结果，用 known rotation：

```bash
python -m src.brepnet.eval.run \
  --pred-root /path/to/post_root \
  --gt-root /mnt/d/data/deepcad_v6 \
  --metrics condition,validity,complexity \
  --rotation-policy known \
  --rotation-id 12 \
  --from-scratch \
  --use-ray --num-cpus 16
```

3. 做方法对比时，为避免人为指定 rotation 带来争议，可以同时报告：

```text
identity eval      # 历史默认，但当前不公平
known rotation 12  # 与 dataset cube0 pose 对齐
search24           # 上界/诊断，用来确认 pose 错配
```

## 后续必须整理的事项

- 统一 rotation protocol：`dataset.py` 的 cube24 order、`src/brepnet/data/rotations.py` 的 OCTAHEDRAL order、Blender render order 必须只有一个权威定义。
- 在 inference 输出中写入 metadata，例如 `rotation_source=cube24`、`cube_id=0`、`euler64_id=4`、`eval_rotation_id=12`，避免 eval 靠人工猜。
- 废弃或重命名 `src/brepnet/post/legacy/eval_brep.py`，至少在 README/脚本里明确“不要用于当前 construct_brep 输出”。
- `eval.run` 应增加自动 pose 选项：从 metadata 读 rotation；没有 metadata 时要求用户显式选择 `none/known/search24`。
- 所有历史实验需要按相同协议重新跑 evaluation，尤其是 topology bias 和 baseline 对比。
- 最终报告里要明确 train loss 是 latent pose 下的 loss，condition metric 是 CAD geometry pose 下的 metric；两者只有在 rotation 对齐后才可比较。

## 今日临时代码修复记录

为了让 eval 入口能跑，今天还发现并临时修了几个依赖问题：

- `shared/occ_utils.py` 当前分支缺少 `get_curve_length`、`get_points_along_edge`、`get_triangulations` 等 helper，已从 `main` 恢复。
- `src/brepnet/eval/run.py` 原本顶层 import 所有 metric，导致只跑 `unique` 也会被 `condition` 的 OCC helper import 卡住；已改为按 metric 延迟 import。
- `src/brepnet/eval/metrics/point_cloud_set.py` 顶层 import `chamfer_distance`，导致 `unique` 只想用 `normalize_pc` 也失败；已改为在 `_pairwise_cd()` 内延迟 import。

这些只是让脚本可运行，不等于 evaluation protocol 已经彻底整理完成。
