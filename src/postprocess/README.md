# BRepNet 后处理说明

`src/postprocess` 负责把模型预测出的面、边和拓扑连接关系重建成 BRep，并导出 STEP/STL 等几何文件。这个目录里最重要、最常用的入口是：

```bash
python -m src.postprocess.construct_brep
```

当前重构目标不是改变算法，而是先把后处理阶段的输入、输出、文件职责和风险边界讲清楚。尤其要注意：这里大量代码直接依赖 `pythonocc`，不同版本的 API 参数、返回类型和对象行为可能不一致，因此后续重构必须非常保守。

## 后处理研究的数据边界（2026-10-16）

用户的最新约定：当前DeepCAD继续开发，但后处理分析和方案调整只使用validation；test在方案冻结后评估数据。后续ABC最终评估进一步保持隔离。

- DeepCAD 的规则、容差、修复顺序、候选排序和重试预算在validation上筛选，失败原因分析也在validation进行。test用于冻结方案的数据比较，不据test个例继续调整。历史test曾参与开发，如实记录，不将其表述为从未见过的独立检验。
- 固定开发比较的 CAD 清单、split 来源、checkpoint、原始预测和选择准则。比较覆盖完整预定集合，同时报告救回、退步、几何保真、边界保留和耗时；失败、缺失、超时不移出分母。GT 可用于诊断与评估，不能进入实际修复或候选选择。
- 旧后处理与候选使用相同输入信息和预定总预算，新增 fallback、容差尝试及重试均计入成本。保留原始 split 和历史结果。
- ABC 最终评估子集在方案、代码／配置、checkpoint 与协议冻结后使用，不据其结果或失败样本调规则、阈值或挑版本。开展 ABC 工作时明确具体子集，并核对与 DeepCAD 等开发数据的 CAD／几何重叠；不同数据集名称本身不证明独立性。

本轮已按上述规则完成原 Intersection refinement VAE 的3495件validation对照：`recover_v1`原生合法率88.01%→90.04%，新增71件、原有成功件无退步。它保留为显式可选策略，默认仍为`legacy`。几何退步、较大内部容差与组环结构差异均单列，详见[图文总览](../../../checkpoints/postprocess_refinement/README.md)与[结果报告](../../../checkpoints/postprocess_refinement/validation_20261016/REPORT.md)。

固定同一方案的[2417件test评估](../../../checkpoints/postprocess_refinement/test_20261016/REPORT.md)也已完成：87.34%→89.16%，新增44件、零退步；未按test结果修改规则，后续失败原因分析仍使用validation。

## 圆柱面表示与特殊处理（2026-10-16 核对）

用户补充的历史研究知识：项目中的圆柱重建采用样条曲面，用户观察到自相交，并已通过实验确定需要按一个完整 loop 处理其边界。现有周期面特殊处理有这一表示背景，不能仅按标准解析圆柱的直觉删除或替换。

当前源码能确认的约定：

- [提取器](../data/extract_brep_from_step_half_structure.py) 对 face 的 UV 参数范围使用包含两端的 `linspace`，采样为 `16×16` 矩形网格；不按孔或裁剪边界逐点过滤。覆盖完整周期时，首末采样可在空间中重合。此事本身不能证明或否定重建曲面的自相交。
- 提取器主动跳过同一面两侧的 seam，因此预测边并非完整 OCC 边界的直接副本。
- [拟合](occ_fit.py) 生成一般 B-spline 曲面，并尝试设置周期性，没有重新建立解析圆柱；[构面](occ_topology.py) 的特殊分支采用该拟合曲面的完整自然 wire。
- 修改相关分支前，应先保留并验证已有完整 loop 行为；不能只凭失败子集中的残差变化断定该设计应被移除。具体自相交形态和历史实验效果仍属于用户报告，当前源码核对未独立复现它们，也不认证所有已有数据均由此版提取器生成。

## 主入口

常用命令形式：

```bash
python -m src.postprocess.construct_brep \
  --data_root /path/to/prediction_root \
  --out_root /path/to/post_root \
  --use_ray \
  --num_cpus 100 \
  --drop_num 2 \
  --from_scratch
```

单样本 debug：

```bash
python -m src.postprocess.construct_brep \
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

第二步已经将 `utils.py` 拆为多个职责明确的模块，并保留 `utils.py` 作为兼容 re-export。后续如果继续整理，应优先替换外部的 `from src.postprocess.utils import *`，再考虑删除无用历史函数。
