# BRepNet 后处理说明

`src/brepnet/post` 负责把模型预测出的面、边和拓扑连接关系重建成 BRep，并导出 STEP/STL 等几何文件。这个目录里最重要、最常用的入口是：

```bash
python -m src.brepnet.post.construct_brep
```

当前重构目标不是改变算法，而是先把后处理阶段的输入、输出、文件职责和风险边界讲清楚。尤其要注意：这里大量代码直接依赖 `pythonocc`，不同版本的 API 参数、返回类型和对象行为可能不一致，因此后续重构必须非常保守。

## 主入口

常用命令形式：

```bash
python -m src.brepnet.post.construct_brep \
  --data_root /path/to/prediction_root \
  --out_root /path/to/post_root \
  --use_ray \
  --num_cpus 100 \
  --drop_num 2 \
  --from_scratch
```

单样本 debug：

```bash
python -m src.brepnet.post.construct_brep \
  --data_root /path/to/prediction_root \
  --out_root /path/to/post_root \
  --prefix sample_id \
  --from_scratch
```

`construct_brep.py` 中当前最核心的函数是 `construct_brep_for_sample(...)`。旧函数名 `construct_brep_from_datanpz(...)` 保留为兼容入口，因为仓库内仍有代码直接调用它。

## 输入协议

默认输入目录结构：

```text
data_root/
  sample_id/
    data.npz
```

`data.npz` 当前支持三类历史格式。

第一类：

```text
sample_points_faces
sample_points_lines
edge_face_connectivity
```

第二类：

```text
pred_face
pred_edge
pred_edge_face_connectivity
```

第三类：

```text
pred_face
pred_edge
face_edge_adj
```

第三类会从 `face_edge_adj` 中恢复 `edge_face_connectivity`。这一步会影响后续 face、edge 和 connectivity 的索引关系，不能随意改变。

读取后统一构造成：

```python
Shape(face_points, edge_points, edge_face_connectivity, False)
```

后续数据流依赖这个 `Shape` 对象中的：

```text
recon_face_points
recon_edge_points
edge_face_connectivity
face_edge_adj
pair1
is_end_point
recon_geom_faces
recon_topo_faces
recon_geom_curves
recon_topo_curves
```

`Shape` 当前不拆成多个类。它是单个样本的唯一状态容器，因为 face、edge、connectivity、vertex 约束和 OCC edge 替换之间存在强索引耦合。为了提高可读性，`shape.py` 中新增了更清楚的方法名，并保留旧方法名作为兼容入口：

```text
select_valid_half_edges()      # 旧名 remove_half_edges()
detect_edge_openness()         # 旧名 check_openness()
build_face_edge_adjacency()    # 旧名 build_fe()
build_corner_constraints()     # 旧名 build_vertices()
replace_periodic_face_edges()  # 旧名 build_geom()
```

`Shape` 生命周期：

```text
初始化:
  recon_face_points
  recon_edge_points
  edge_face_connectivity
  interpolation_face

拓扑预处理后:
  openness
  face_edge_adj
  pair1
  is_end_point
  remove_edge_idx_src
  remove_edge_idx_new

OCC 拟合后:
  recon_geom_faces
  recon_topo_faces
  recon_geom_curves
  recon_topo_curves

周期面 edge 替换后:
  clinder_face_idx
  replace_edge_idx
  recon_topo_curves
```

## 输出协议

默认输出目录结构：

```text
out_root/
  sample_id/
    separate_faces.ply
    recon_brep.step
    recon_brep.stl
    success.txt
    debug_face_loop/
```

文件含义：

- `separate_faces.ply`：独立面片的可视化结果，用于检查拟合出的面。
- `recon_brep.step`：重建出的 BRep/compound STEP 文件。
- `recon_brep.stl`：solid 成功时导出的 STL。
- `success.txt`：只有成功构造 solid，并且 `check_step_valid_soild(...)` 通过时才写入。
- `debug_face_loop/`：单样本 debug 或保存中间数据时使用。

需要特别区分：

- 有 `success.txt`：表示 solid 构造成功且 STEP 合法性检查通过。
- 有 `recon_brep.step` 但没有 `success.txt`：通常表示 solid 失败后输出了 compound fallback。
- 没有 `recon_brep.step`：表示该样本后处理失败或被跳过。

## `construct_brep.py` 数据流

当前主流程可以理解为以下阶段：

1. 读取 `data.npz`，构造 `Shape`。
2. 删除或合并不可靠的 half edges。
3. 检查 edge 是否闭合，构造 face-edge 邻接关系。
4. 根据三面环估计 vertex 约束。
5. 可选地优化 face/edge 几何位置。
6. 用点阵拟合 OCC surface。
7. 用边点拟合 OCC curve/edge。
8. 针对周期面等情况替换部分 edge。
9. 导出 `separate_faces.ply` 作为中间可视化。
10. 根据 `drop_num` 枚举候选 face 组合。
11. 为每个 face 构造 wire 和 trimmed face。
12. 尝试 sewing trimmed faces 并构造 solid。
13. solid 成功后写 `recon_brep.step`、`recon_brep.stl`、`success.txt`。
14. solid 失败时，用 mixed faces 尝试 compound fallback，并写 `recon_brep.step`。

这个顺序是后处理的核心数据流。后续重构可以改名字、拆函数、加注释，但不应随意改变阶段顺序。

## 当前文件职责

根目录只保留主流程和仍被主流程直接依赖的基础模块：

- `construct_brep.py`：主入口。负责从 `data.npz` 到 STEP/STL 的完整后处理流程。
- `constants.py`：后处理中使用的拟合、连接、修复容差等常量。容差顺序会影响结果，不能随意调整。
- `shape.py`：`Shape` 类和面点加密采样逻辑。`Shape` 保存预测 face/edge/connectivity 以及拓扑处理中间状态。
- `geometry_optimization.py`：torch/chamfer 几何优化逻辑，包含 `optimize(...)`。
- `occ_fit.py`：点阵到 OCC surface/curve 的拟合逻辑，包含 `create_surface(...)` 和 `create_edge(...)`。
- `occ_topology.py`：OCC wire、trimmed face、solid、compound、validity 相关构造函数。
- `mesh.py`：OCC shape/face 的 triangulation 和分离面片导出辅助函数。
- `debug.py`：debug 可视化、颜色和 edge OBJ 导出。
- `distance.py`：`chamferdist` 与 PyTorch fallback 的统一选择。
- `utils.py`：兼容层。旧代码仍可从这里导入 `Shape`、`create_surface`、`get_solid` 等符号，但实际实现已经移动到上述模块。
- `chamferdist_torch.py`：`chamferdist` 不可用时的 PyTorch fallback。

辅助工具放在 `tools/`：

- `tools/construct_face.py`：单独检查 trimmed face 构造有效性的工具。
- `tools/move.py`：历史整理/检查脚本，用于移动、统计、筛选失败样本。

历史代码放在 `legacy/`：

- `legacy/eval_brep.py`：历史评价脚本。它更接近 eval 逻辑，保留在 post legacy 中作为旧实验入口。
- `legacy/brepgen_with_our_post.py`：历史批处理入口，直接调用 `construct_brep_from_datanpz(...)`。
- `legacy/check_brep.py`：简单统计 `success.txt` 的工具。
- `legacy/geom_optimization.py`：面和边的几何优化实验代码，当前主流程不直接依赖。
- `legacy/optimization.py`：旧优化相关代码，当前不是主入口。
- `legacy/move_test.py`：历史测试/临时脚本。

## 重构边界

允许做的事：

- 给函数改更清楚的名字。
- 保留旧函数名作为兼容别名。
- 把长函数按连续逻辑块拆成少量浅层函数。
- 写中文说明和注释。
- 把 CLI 解析、样本枚举、单进程/Ray 调度整理清楚。

禁止或需要单独确认的事：

- 随意增加或删除公开函数参数。
- 随意改变函数返回值数量、类型和组织结构。
- 改变 `Shape` 中 face、edge、connectivity 的索引关系。
- 改变 `CONNECT_TOLERANCE` 等容差尝试顺序。
- 改变 `pythonocc` API 调用的参数数量和顺序。
- 把调用链拆得过深，导致人工排查困难。
- 同时大规模重构 `utils.py` 和 `construct_brep.py`。

## 建议的后续重构顺序

第一步只整理 `construct_brep.py`：

1. 保留 `construct_brep_from_datanpz(...)` 兼容入口。
2. 使用 `construct_brep_for_sample(...)` 作为更清晰的主函数名。
3. 旧函数只转发到新函数。
4. 把主流程拆成少量浅层 helper，但每个 helper 只包住当前连续代码块。
5. 保持 CLI 参数不变。

第二步已经将 `utils.py` 拆为多个职责明确的模块，并保留 `utils.py` 作为兼容 re-export。后续如果继续整理，应优先替换外部的 `from src.brepnet.post.utils import *`，再考虑删除无用历史函数。
