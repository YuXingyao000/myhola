# 后处理（src/postprocess/）：Claude Code 入口

权威指南是本目录的 [AGENTS.md](AGENTS.md)（文件职责、索引约定、两处自然边界修复、恢复顺序、验证要求），使用方式见 [README.md](README.md)。两者冲突时以 `AGENTS.md` 为准；本文件是 Claude 的工作摘要，补充历史尝试的结论和当前优化方向，不复述算法细节。根指南见仓库 [CLAUDE.md](../../CLAUDE.md) 与 [AGENTS.md](../../AGENTS.md)。

## 这一层在做什么

模型（`src/models/vae/`）解码出每面 `16×16×3` 采样点、每条有向半边 `16×3` 采样点和 `[edge_id, face_i, face_j]` 连接；本目录把它们变成可写 STEP 的合法 solid。核心思路是**先做一个简单的几何优化，再交给 PythonOCC 7.8.1 的拟合／裁剪／缝合**，后处理本身不读 latent、不读 GT。

```text
data.npz                      decoded_prediction.load_prediction_shape → Shape
   |
   v  半边 → 共享边（见下）    Shape.select_valid_half_edges / select_shared_boundaries
   |                          detect_edge_openness, build_face_edge_adjacency,
   |                          build_corner_constraints(0.2) → pair1 / is_end_point
   v
geometry_optimization.optimize   AdamW 200 步（角点不收敛时最多 600）
   变量：每边 scale[3]+shift[3]、每面 shift[3]；端点不是独立变量
   loss ：边→两张 owner 面的单向 Chamfer + 三面角点距离 + 面内端点最近邻 + 1e-8 正则
   |
   v
occ_fit.create_surface / create_edge   GeomAPI_PointsToBSpline(Surface)，C2，变分光顺
   容差序列 FACE [1e-3,1e-2,3e-2,5e-2,8e-2]、EDGE [1e-3,5e-3,8e-3,5e-2]，评分选最好
   set_face_uv_periodic：首末采样距离 < 平均间距 → SetUPeriodic/SetVPeriodic
   replace_periodic_face_edges：周期面且恰 2 条边 → 用自然边替换相似的共享 TopoDS edge
   |
   v
occ_topology  create_wire_from_unordered_edges（ConnectEdgesToWires，随机 3 次取 wire 数最少）
              create_trimmed_face_from_wire（ShapeFix_Wire/Face；周期面 2 等长边 → 整条自然 wire）
              get_solid（Sewing → ShapeFix_Shell → MakeSolid → ShapeFix_Solid）
              CONNECT_TOLERANCE [2e-3 … 8e-2] 逐级放宽
   |
   v
construct_brep  legacy 路径写 STEP 后重读检查；recover_v1 再依次 complete-optimized /
                complete-fit-only / fit_score_v2，共用 180 s；失败写 compound（不算合法）
```

## 半边与共享边：必须保住的约定

- VAE 对每个有序面对 `(i,j)` 预测一条有向半边，所以同一条物理边通常有两份互反预测。`select_valid_half_edges` 按**无向面对**聚合，用边到两面的 Chamfer 距离选一条（阈值 0.3），**不平均**；超过两个候选直接跳过该面对。`shared_boundary_v1` 表示改为按 confidence／残差选，同样每面对一条。
- 选完后边点与 `edge_face_connectivity[:,0]` 同步重编号为 `[E,L,3]`，面 ID 不变；两张面引用同一个共享 edge_id。`remove_edge_idx_src`（原始 ID）与 `remove_edge_idx_new`（重编后 ID）不能混用。
- 后续 OCC 中的有向使用、pcurve、环路方向由 ShapeFix 自己处理；代码里没有显式 pcurve，也没有可靠的顶点 ID。`pair1/is_end_point` 只是三面关系＋端点最近邻的软角点，末点索引写死为 15。
- 一个面对只能有一条边：当前表示无法表达两张面共享两条物理边（如半圆柱的两条竖边），这是模型侧限制，后处理补不回来。

## 当前基线与失败画像（validation 3495，固定口径）

| 项 | 数值 | 来源 |
|---|---|---|
| 最好流程 | `roles_best7885` + `recover_v3` + 180 s：**3327/3495（95.19%）**，对 recover_v1 基线 3189（91.24%）救回 138、退步 0 | `/mnt/d/hola_research/postprocess_failure_stages_20261105/README.md` |
| 本目录代码状态 | 2026-11-05 回退到该 91.24% 流程的 11 个文件哈希（见 `continuous_postprocess/ROLLBACK_20261105.md`），再做了纯搬动整理（`postprocess_organization_20261105/`）；2026-11-07 加入 `recover_v2`（见下） | 全量结果见研究目录 README |
| 失败 306 件 | trimming 299、solid 构造 4、native/boundary 拒绝 3；其中 222 件 face-pair 拓扑全对 | 同上 REPORT |
| 合法输出来源 | legacy optimized 3114、roundtrip_optimized 30、roundtrip_fit_only 45（recover_v1 基线逐件 `chosen_path`） | 同上运行的 `post_diagnostics.json` |

口径：STEP 以 `read.precision.mode=0` 重读后为 `TopAbs_SOLID` 且 `BRepCheck_Analyzer` 通过；compound／shell／`success.txt` 都不算；失败和超时留在 3495 分母。另有 tolerance 0.1 的兼容口径单列。

### 近 400 件失败的共性（三次失败普查的交集）

来源：`joint_boundary/legacy_failures_20261011/`（原版 VAE test 369 件、50 件逐阶段重放）、`intersection_local/failure_census_20261013/`（712 件多标签观察）、`curve_head_roles/evaluation_20261101/visuals/`（当前权重 5 件人工曲线对照）。

1. **拓扑大多是对的，几何不相容。** 369 件里 268 件无向 face-pair 与 GT 完全一致；当前 306 件里 222 件 face-pair 全对。失败集中在"边贴不到它所属的两张面"（712/712）和"相邻边端点对不上"（658/712）。
2. **一条边只能照顾一侧。** 选中的半边常贴近一张 owner 面、离开另一张（00425503、00024146、00022149）。单条边对 GT 更近，不等于两侧同时相容；00022149 曲线对 GT 误差降 86% 却从合法变成 trimming 失败。
3. **"点靠近"不等于"环成立"。** 337/712 件所有候选环都闭合并被接纳，面仍裁不出来；最终失败面的 BRepCheck 以 `UnorientableShape`、`NotClosed`、`InvalidSameParameterFlag/InvalidCurveOnSurface` 为主，即三维贴合之外的 pcurve／方向／参数一致性问题。
4. **优化不单调。** 每轮都有 8–11 件 fit-only 合法、optimized 反而失败；旧优化会把短边拉短、把近直线弄弯（00003999 edge23：端距 0.236→0.19，双侧投影 RMS 3.1e-4→3.7e-3）。
5. **周期面是独立子问题。** 自然边替换／自然 wire 决定大量周期面能否裁出；00000894 同一优化结果仅切换自然 wire 开关就从拒绝变为七个容差全部接受。
6. **sewing 通过也可能不是 solid。** 50 件重放里 11 件各面裁剪都通过、free edge 为 0 仍构不成 solid；1 件内存 solid 通过、保存后原生读回失败。

### 已尝试并有全量数字的方案（不要重复做无对照版本）

| 方案 | 结果（对 3495） | 结论 |
|---|---|---|
| `roundtrip_retry` / `fit_score_v2` / `recover_v1`（2026-10-16） | 3076→3147，+71，零退步 | 已并入；收益来自更严验收＋多路径恢复，不是单次算法提升 |
| 连续样条交替优化替代旧优化（2026-11-02，OSQP） | 3189→3150，−39；优化组边 CD² −79% | 几何更准但合法率降；主因是**错误端点软配对把边拉回折**＋关闭自然 wire 替换 |
| 旧优化→连续串联（cascade） | 3165→3147（同裁剪策略）−18 | 更强求解器把错误端点关系满足得更好，不推广 |
| 三面顶点组推断＋严格周期判定（2026-11-05，142 服务器） | 3165→2679，−486（491 退步中 453 件对应自然 wire 替换失效） | 严格周期判定关掉了周期补救；顶点约束本身效果未分离 |

这些代码都已回退，只在 stash `ff7f55f3`（16 文件）与研究目录里；按哈希识别，不要按 stash 序号恢复。

### 2026-11-07：失败阶段表与缺 seam 修复（`recover_v2`）

306 件失败按 `post_diagnostics.json` 分成四类（`/mnt/d/hola_research/postprocess_failure_stages_20261105/`）：环闭合但 `ShapeFix_Face` 合并了环 169 件、边接不上 104 件、面检查不过 20 件、缝不上 12 件。第一类的机制：圆柱侧面这类周期曲面两端各一个闭合环，需要 seam 边连成一个 wire；`ShapeFix_Face` 默认 `FixSplitFace` 在 `Perform()` 里先把面拆成两张，显式 `FixMissingSeam()` 再也补不上。`create_trimmed_face_from_wire(..., split_face=False)` 关掉拆面即可；重放在 169 件里救回 98 件、另救回 3 件 open_wire。`recover_v2` = `recover_v1` 四条路径之后追加 `seam_roundtrip_optimized`、`seam_roundtrip_fit_only`，只对前四条都失败的样本生效。边接不上的 104 件里，25 件是角点缺口 0.08–0.2（优化可拉拢）、15 件是面里多了一条错边（剔边规则）、64 件边缺失或严重错误（模型侧）。

`recover_v3`（11-07/08）= `recover_v2` 去掉零救回的 fit-score-v2 路径，再追加两条定向端点修复路径（`endpoint_repair.py`：面内互为最近且无歧义的端点配对 → 跨面顶点簇 → 悬空端点挂到拓扑一致的已成立角点（两张面里各恰好一个端点、≤ 0.2、无歧义）→ 精确贴合；两头悬空的多余边剔除；optimized 变体用簇心项替换旧角点项和最近邻项）。在 recover_v2 剩余 201 件上救回 33 件。11-07 对悬空端点的检查否定了"二价端点"假设：70 个悬空端点全在某个角点 0.2 以内，形态是边够不到已成立的角点、微小边把角点挤满、或面太小缺口与面同量级；前两种由挂角点规则处理，第三种不碰。

## 现在的优化方向（研究判断，尚未实现）

以 91.24% 流程为基线，新规则在 validation 上设计、固定清单与预算；每个改动单独对照，报告救回／退步／几何保真／耗时。

1. **端点与顶点关系要先可靠，再谈求解器。** 最近邻端点配对是前两轮退步的共同根因。可靠的顶点推断需要结合三面关系、边长尺度和歧义保留，不做距离传递聚类；无法确定的端点就不施加约束。
2. **优化目标直接写"边同时贴两张连续面、端点共用、环闭合"，而不是采样点 Chamfer。** 拟合后的曲线／曲面才是 OCC 真正裁剪的对象；若做连续优化，位移盒按边长缩放，形状保护（回折、弧长）不通过整体回退。
3. **把裁剪失败按 OCC 错误类型拆开。** `UnorientableShape`／`NotClosed`／`SameParameter` 各对应不同修复（方向、缺边、pcurve 重建＋`SameParameter`），需要在 `create_trimmed_face_from_wire` 里显式检查和记录，而不是盲加 ShapeFix。
4. **周期面单独处理。** 周期判定、seam、自然 wire 替换是一组耦合规则；改任何一个都要先在周期面子集做对照，保留现有两处自然修复直到有替代证据。
5. **拟合前优化处理了 97.6% 的成功件，不是主要矛盾**；fit-only 恢复只救 45 件。历史记录里"fit-only 占近一半"（1491/3165）来自 2026-11-02 cascade 对照臂，那一臂对 optimized 路径关闭了自然边替换，不能当基线。"旧优化 / 仅拟合 / 新优化"三路对照仍是每轮标配，但要用相同裁剪策略。

## 工作规则（本目录特有）

- 修复与候选选择不能读 GT；GT 只用于事后几何评估。test 在方案冻结后只评一次。
- 不要在"整理"PR 里改容差、端点、周期或裁剪规则；算法实验与搬动分开提交。
- 新开关默认关闭、写明删除条件；全量无净收益就删路径。
- 改签名前在实际环境验证 PythonOCC 7.8.1 调用链；已知坑：状态枚举迭代返回 int，不能取 `.name`。
- 每件评估在独立子进程、单线程（`cpu_runtime.py`），超时与崩溃留在分母。
- 相关测试：`tests/test_shared_boundary_post.py`、`test_occ_fit.py`、`test_post_policies.py`、`test_vae_compare_post.py`（环境 `/mnt/d/anaconda3/envs/torch/bin/python`，CPU）。

## 数据与记录位置

- 当前权重的 raw 预测：`/mnt/d/hola_runs/curve_head_roles/evaluation_20261101/raw_predictions/posterior_mean/roles_best7885`；逐件结果 `per_cad.csv.gz`、配对 ID 在 `SUMMARY.json`。
- 固定清单：`src/preprocess/splits/deepcad_v7_contract_validation.txt`（3495）。
- 失败案例浏览页：`joint_boundary/legacy_failures_20261011/page/index.html`、`intersection_local/failure_census_20261013/page/index.html`。
- 连续优化原型与 OCC 适配（单文件，无包依赖）：`/mnt/d/hola_research/continuous_postprocess/alternating_spline_refinement.py`。
- 论文：HoLa（SIGGRAPH 2025，arXiv 2504.14257）。论文后处理只描述 C2 B-spline 拟合、共享端点组环、最大环为外边界、OCC 缝合；拟合前几何优化、容差序列、周期自然边替换和恢复路径都是本仓库后加的，论文 VAE 合法率 94.18% 与本仓库口径不同。
