# 解码后处理指南

本目录接收模型**已经解码出的面／边 XYZ 采样点及连接关系**，完成预测整理、几何优化、OCC 拟合和 B-Rep 构造。latent 与神经网络 Decoder 属于 `src/models/vae/`，不在这里计算。仓库工作流、代码卫生和研究纪律以根 [AGENTS.md](../../AGENTS.md) 为准；使用方式见 [README.md](README.md)。

## 文件职责

| 文件 | 职责 |
|---|---|
| `decoded_prediction.py` | NPZ 读取、表示适配、半边筛选／去重与边重编号、邻接与角点软关系；`Shape` 保存单样本各阶段状态 |
| `geometry_optimization.py` | 拟合前采样点优化：每边三轴缩放和平移、每面平移；可选顶点簇项 |
| `endpoint_repair.py` | 恢复路径里的定向端点修复：按面配对端点、并成顶点簇、精确贴合、剔除两头悬空的多余边 |
| `occ_fit.py` | 点阵拟合 B-spline 曲面／曲线、既有周期转换、周期面自然边替换 |
| `occ_topology.py` | curve → wire → trimmed face → sewing／solid，含自然环修复和 compound fallback |
| `construct_brep.py` | 调度上述阶段、候选／恢复路径、预算检查及文件输出；主入口 `construct_brep_for_sample` |
| `constants.py` | 拟合／连接等容差及其尝试顺序 |
| `distance.py`、`chamferdist_torch.py` | Chamfer 距离后端及 PyTorch fallback |
| `mesh.py`、`debug.py` | 网格化与中间几何可视化 |
| `cpu_runtime.py` | CPU 评估进程与线程资源控制 |
| `utils.py` | 尚有调用者的旧导入入口；既有分层债务，不在此新增实现或兼容导出 |

保持调度顺序可直接阅读。新增算法按计算职责放置，不把读取、索引处理、拟合与裁剪揉成一个大函数，也不为整理增加转发层。

## 实际执行顺序与数据

```text
模型导出的 data.npz（不在后处理中读取 latent）
    |
    v
decoded_prediction.load_prediction_shape
    Shape: face_points [F,U,V,3], edge_points [P,L,3]
           connectivity [R,3] = [edge_id, face_i, face_j]
    |
    v
construct_brep._prepare_shape_topology → Shape 的筛选／邻接方法
    legacy_half_edges: 无向面 pair 内选择一条预测边
    shared_boundary_v1: 几何筛选后按 confidence 等选择共享 query
    重编 edge_id → [E,L,3] / [E,3]
    openness、face_edge_adj、pair1 / is_end_point
    |
    +-- optimized: geometry_optimization.optimize（拟合前）
    |                整条边缩放／平移，整张面平移
    +-- fit-only: 跳过该优化
    |
    v
construct_brep._fit_occ_geometry → occ_fit
    create_surface / create_edge → OCC B-spline / TopoDS 对象
    replace_periodic_face_edges → 可替换共享的 recon_topo_curves[e]
    |
    v
occ_topology
    每面取共享 edge → 组 wire → 裁剪 surface
    （满足既有条件时使用完整自然 wire）
    → 修复／缝合 → solid → 写 STEP → 重新读取并检查
    |
    +-- 失败时可能写 compound；不能据此计为有效 solid
```

这是单条构造路径。`legacy` 还可按 `drop_num` 枚举面候选；正式 `recover_v1` 要求 `drop_num=0`，恢复顺序见下文。

### 索引与半边

- `load_prediction_shape` 只读取和建立状态，不提前筛选、不修改磁盘原始预测。无标签按历史 `legacy_half_edges` 解释；显式未知表示报错，不能把双侧 UV 直接当成 XYZ 输入。
- 输入连接行通过第一列引用边点数组；不要假定原始连接行号就是边 ID。筛选后边点与连接第一列同步重编，面 ID 保持不变；后续 drop 候选另行重映射面 ID。
- legacy 按无向面 pair 聚合候选，依边到双面的距离选一条，**没有平均两条半边**。现实现每对面最多保留一条；超过两个 legacy 候选时跳过该 pair。这是当前表示／适配限制，不是普适的物理边去重算法。
- `shared_boundary_v1` 先做几何门限筛选，再按 confidence、残差、source index 选一条；仍限制每对面一条。原始 query ID 与筛选后的边 ID 分开记录，不能混用 `remove_edge_idx_src` 和 `remove_edge_idx_new`。
- 去重后的两张面引用同一个共享 3D edge ID。OCC 中的有向使用、pcurve 和环路方向是后续构造问题；不能因训练使用半边而要求最终每侧保留一条独立 3D 边。
- `pair1 / is_end_point` 是基于三面关系和端点距离生成的角点软约束，不是可靠的 GT 顶点 ID。当前没有启用后来实验中的共享顶点聚类。
- 常用面采样为 `16×16`、边采样为 16 点。现有优化器仍把末点索引写为 15；读取支持某形状不代表全部优化步骤已经支持任意采样数。

### 几何优化、周期面与自然边界

- 当前 `geometry_optimization` 在拟合前调整整条边的缩放／平移和整张面的平移。端点随整条边变化，并非独立优化变量；这也不是曲线／曲面控制点优化。
- 当前提取器按 face 的完整矩形 UV 范围采样，包含参数域两端，未逐点按裁剪环或孔过滤，并跳过同面两侧的 seam。不能把预测边集当作完整 OCC 边界的逐项副本；复用旧数据还需核对生成版本。
- `create_surface` 拟合一般 B-spline，并用原有闭合启发式尝试周期转换；它不是解析圆柱分类器。周期性不等于圆柱类型，局部圆柱面也不必周期闭合。
- **保留两处不同的自然边界修复。** `replace_periodic_face_edges` 在拟合后替换相似的共享 TopoDS edge；`create_trimmed_face_from_wire` 在既有条件下用拟合面的完整自然 wire 构面。它们都受周期性、边数和自然 wire 等条件控制，不能在“整理代码”中静默关闭。
- 周期接缝可以在一个自然环中出现两次有向使用；wire 数量变化不能单独判为孔丢失。普通面的缺边／开环检查仍需保留。
- **split-face 与 seam（2026-11-07）。** 周期曲面上两个各自绕周期方向一圈的闭合环（圆柱侧面的两端）必须由 seam 边连成一个 wire 才能成面。`ShapeFix_Face` 默认的 `FixSplitFace` 会在 `Perform()` 里先把这种面拆成两张，随后显式的 `FixMissingSeam()` 无事可做；历史代码因此把一端边界丢掉。`create_trimmed_face_from_wire(..., split_face=False)` 关闭拆面，让 `FixMissingSeam` 补 seam，诊断记 `seam_added`；此时环数 2→1 是完整边界，不按 `wire_count_decreased_during_face_fix` 拒绝。默认 `split_face=True` 保持历史行为，`recover_v2` 的两条 seam 路径用 `False`。失败机制与 169 件重放见 [失败阶段记录](/mnt/d/hola_research/postprocess_failure_stages_20261105/README.md)。
- `complete` 构造路径会复制 OCC 对象以隔离逐面修复，再进行缝合。共享输入边 ID 不保证中途所有 TopoDS 对象身份相同；最终需要核对几何、边界与 solid 合法性。
- 后续若研究拟合后的连续优化，应显式说明如何传递优化后的曲线、曲面、UV／pcurve及自然修复接口。不能只因“保留优化曲线”就关闭修复，也不能把间隙下降直接写成可组环／可裁剪的保证。OCC API 变化须在现有环境验证实际调用链。

## 恢复策略、结果与评估

函数和 CLI 的默认路径仍是 `legacy`。`recover_v1` 显式依次尝试：

1. 原 `legacy` optimized 路径。
2. `complete` optimized：完整面覆盖、STEP roundtrip 检查。
3. `complete` fit-only：从原始预测重新拟合，跳过拟合前优化。
4. `fit_score_v2` 的 `complete` optimized。

`recover_v2` 在上述四条之后再加两条：`seam_roundtrip_optimized`、`seam_roundtrip_fit_only`，即 `complete` 路径加 `split_face=False`。

`recover_v3` = `recover_v2` 去掉 `fit_score_v2_roundtrip_optimized`（validation3495 上零救回，只消耗共享预算），再加两条定向端点修复路径 `repair_roundtrip_fit_only`、`repair_roundtrip_optimized`（`endpoint_repair.py`，`split_face=False`）：在半边选择之后、拟合之前，(1) 按面把端点互为最近、距离 ≤ 0.2、第二候选至少远 0.02 的配对并成顶点簇；(2) 仍悬空的端点挂到已成立的簇上，条件是该簇在这条边的两张面里各恰好有一个端点、距离 ≤ 0.2、次优簇至少远 0.02，按距离从近到远处理，同一条边两端不进同一簇，簇心不因挂入而移动；(3) 簇内端点精确挪到簇心、边内部按弧长线性过渡；(4) 两张面里都两头悬空的开边视为多余边从连接中剔除。optimized 变体把簇传给 `geometry_optimization.optimize(vertex_clusters=...)`，用簇心项替换历史三面角点项和逐步最近邻项，边面 Chamfer 项不变。配不上的端点不动，不读 GT。

各路径从同一原始预测和相同 Python／NumPy RNG 状态开始，不是把上一路结果继续优化。首个满足对应路径验收条件的结果即停止：都须通过原生 solid 检查，`complete` 还要求 roundtrip 后面数相同、wire 数不减少。恢复路径共用预算；生产评估的进程超时还由外层执行器约束，不把额外 fallback 当作免费操作。容差顺序、随机尝试、drop、拟合评分和自然修复均属于算法行为。

`success.txt`、可读取 STEP、兼容检查、原生有效 solid、边界完整性是不同状态。正式原生评估在 fresh mode 0 读取后检查 SOLID 与 `BRepCheck_Analyzer`；不能把 compound、仅有标记文件或图连通算作有效实体。

2026-11-05 整理所依据的是已恢复的完整流程。历史 `roles_best7885` 在固定 validation3495 上为 **3189/3495（91.24%）**；3165/3495（90.56%）是后来关闭 optimized 路径自然修复的对照，不能替代前者。来源与回退核对见 [结果报告](/mnt/d/hola_research/curve_head_roles/evaluation_20261101/REPORT.md)、[回退记录](/mnt/d/hola_research/continuous_postprocess/ROLLBACK_20261105.md)。该数字不是默认 CLI 或其他权重的保证；代码整理没有重跑完整3495件。

连续控制点优化、共享顶点推断、严格周期判定及关闭自然修复的实验已经撤回，不是当前包内实现。历史实验的 UV 用作残差对应／诊断，没有导出 OCC pcurve。研究记录保留，但不能据旧笔记自动恢复这些分支。

规则设计、容差选择、失败分析使用 validation；test 用于冻结方案后的评估。固定输入预测、清单、checkpoint、预算及合法性口径，同时报告救回、退步、几何保真和耗时，失败／缺失／超时保留在分母中。修复和候选选择不能读 GT。ABC 最终评估隔离遵守根指南。

## 修改与验证

- 移动实现一次更新仓库调用点，不新增旧路径别名或兼容包装。既有 `utils.py` 等债务不在本次扩张。
- module 改名在 `tools/migrate_checkpoint.py` 登记元数据路径迁移；本次 `shape.py` → `decoded_prediction.py` 不改变神经网络参数、optimizer 或 latent cache。旧评估有源码指纹时应新开 run，不改写历史协议／哈希。
- 组织重构核对阶段顺序、候选选择及索引等价，再运行相关测试：`tests/test_shared_boundary_post.py`、`tests/test_occ_fit.py`、`tests/test_post_policies.py`、`tests/test_vae_compare_post.py`。尤其保留周期面完整自然环、真实拟合与普通开环拒绝用例。
- 一次性等价性脚本和结果写 `/mnt/d/hola_research/<主题>/`；小样本／单元检查不冒充完整合法率复现。当前整理记录在 [postprocess_organization_20261105](/mnt/d/hola_research/postprocess_organization_20261105/)。
- 维护本文件与 README；`CLAUDE.md` 只是 Claude Code 的入口摘要，算法约定写在这里。不在整理任务中混入拟合、容差、端点或裁剪算法实验。
