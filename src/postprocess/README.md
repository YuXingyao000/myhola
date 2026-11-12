# Img2Brep 解码后处理

本目录把**已经解码的面／边 XYZ 采样点和连接关系**变成 B-Rep／STEP。神经网络 Decoder 位于 `src/models/vae/decoders/`；这里不解码 latent。

修改代码前阅读 [AGENTS.md](AGENTS.md)：其中记录半边筛选与共享索引、拟合前优化、周期面自然修复、恢复顺序及验证要求。

## 从哪里看代码

```text
construct_brep.py                       调度入口：construct_brep_for_sample
    |
    +-- decoded_prediction.py           读取 → 半边筛选／边重编号 → 邻接／角点关系
    |       load_prediction_shape       Shape 保存本样本的数组与中间状态
    |
    +-- endpoint_repair.py              恢复路径可选：端点配对 → 顶点簇 → 精确贴合／剔多余边
    |
    +-- geometry_optimization.py        可选拟合前优化：整条边缩放／平移、整张面平移
    |
    +-- occ_fit.py                     B-spline surface / curve 拟合
    |       replace_periodic_face_edges 周期面自然边替换
    |
    +-- occ_topology.py                curve → wire → trim → sewing → solid
            create_trimmed_face_from_wire 含既有完整自然环修复
```

预测读取和索引处理集中在 `decoded_prediction.py`，原 `shape.py` 已移除。拟合和自然边替换集中在 `occ_fit.py`，裁剪／缝合在 `occ_topology.py`。这次文件整理保留计算规则、执行顺序和阈值；没有加入新的连续优化或顶点算法。

## 输入

每个样本为 `data_root/<sample_id>/data.npz`。当前读取以下格式：

| 表示 | 主要字段 |
|---|---|
| 历史采样点 | `sample_points_faces`、`sample_points_lines`、`edge_face_connectivity` |
| 历史预测 | `pred_face`、`pred_edge`、`pred_edge_face_connectivity` |
| 历史邻接格式 | `pred_face`、`pred_edge`、`face_edge_adj`；读取时恢复连接行 |
| 显式 `representation="shared_boundary_v1"` | `pred_face`、`pred_edge`、`pred_edge_face_connectivity`、`pred_edge_confidence` |

无表示标签按 `legacy_half_edges` 处理。统一数组为面 `[F,U,V,3]`、边 `[P,L,3]`、连接行 `[edge_id,face_i,face_j]`。读取后再做筛选与重编号，磁盘上的原始预测保持不变。现有适配最多保留每对面一条边；不能当作任意 B-Rep 拓扑的通用去重器。

## 运行

单样本、CPU、默认 `legacy` 路径：

```bash
python -m src.postprocess.construct_brep \
  --data_root /path/to/predictions \
  --out_root /path/to/new_post_run \
  --prefix sample_id \
  --drop_num 0
```

此 CLI 未暴露 `post_policy`。需要明确的 `recover_v1` 恢复策略时，由评估入口或 Python 调用传入：

```python
from src.postprocess.construct_brep import construct_brep_for_sample

construct_brep_for_sample(
    prediction_root, output_root, sample_id,
    v_drop_num=0, use_cuda=False, is_optimize_geom=True,
    v_max_optimize_iter=200, post_policy="recover_v1",
    time_budget_seconds=180,
)
```

该示例说明接口，不替代正式评估的 seed、清单、进程超时和溯源配置。`recover_v1` 从同一原始预测依次尝试 legacy optimized、complete optimized、complete fit-only、fit-score-v2 optimized；`recover_v2` 在其后追加两条关闭 `ShapeFix_Face` 拆面、让 `FixMissingSeam` 为周期面补 seam 的 complete 路径（optimized、fit-only），只对前四条都失败的样本生效；`recover_v3` 去掉零救回的 fit-score-v2 路径，再追加两条定向端点修复路径（`endpoint_repair.py`：无歧义的端点配对并成顶点簇，悬空端点挂到拓扑一致的已成立角点，精确贴合，剔除两头悬空的多余边），只对前五条都失败的样本生效；`recover_v4` 把这两条修复路径换成先紧后松的两档共四条（紧档：两个悬空端点配对的距离上限取 0.05，挂到已成立角点不设距离上限；松档：都不设上限），其余路径与 `recover_v3` 相同；首个满足该路径验收条件的结果即停止，共用预算。都须通过原生 solid 检查，complete 路径还检查 roundtrip 后面数相同、wire 数不减少。各路径不会在前一路结果上接着优化。输出使用新目录。

## 输出与判定

输出位于 `out_root/<sample_id>/`：

- `separate_faces.ply`：拟合面可视化。
- `recon_brep.step`：构造结果；失败时也可能写出 compound，文件存在不等于有效 solid。
- `success.txt`：对应路径通过验收后写入，仍须按评估协议独立核验。
- `recon_brep.stl`：legacy 路径成功后的网格导出；complete 恢复路径不保证生成 STL。
- `post_diagnostics.json`、`policy_attempts/`：显式策略的路径、耗时与验收记录。
- `debug_face_loop/`：启用中间数据保存时的诊断产物。

正式原生合法性使用 fresh mode 0 读取 STEP 后的 SOLID 与 `BRepCheck_Analyzer` 检查。原生有效、边界完整及与预测／GT 的几何保真分别报告。

## 基线与兼容性

本次整理以2026-11-05恢复的完整后处理为起点。该流程配合 `roles_best7885` 的历史 validation3495 结果为 **3189/3495（91.24%）**，见 [报告](/mnt/d/hola_research/curve_head_roles/evaluation_20261101/REPORT.md)和[回退记录](/mnt/d/hola_research/continuous_postprocess/ROLLBACK_20261105.md)。2026-10-16 的90.04%是旧权重／旧轮次记录，保留在[历史报告](/mnt/d/hola_research/postprocess_refinement/validation_20261016/REPORT.md)，不是当前最优数字。

目录整理不改变模型权重或数据格式；模块路径元数据迁移登记为 `2026-11-05-postprocess-decoded-prediction`。源码指纹会随文件移动变化，旧评估记录不改写，续跑要求指纹一致的任务需新开 run。既有 `utils.py` 导入债务保留，后续迁移调用点时不新增兼容层。

本次验证与差异记录见 [整理记录](/mnt/d/hola_research/postprocess_organization_20261105/)。没有在本次整理中重新跑完整3495件评估。
