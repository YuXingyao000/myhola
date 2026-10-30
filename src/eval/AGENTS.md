# 评估指南

## 研究与评价边界

评价应回答照片到 B-Rep 的几何准确性、可见证据一致性、实体有效性以及计算成本，不只追踪旧实验的一项均值。
区分灰模条件、生成照片条件和真实拍摄条件；合成域成绩不能直接作为真实照片泛化结论。
强通用模型可以生成 CAD 程序作为基线；比较时记录相同输入、工具权限、推理预算、重试和人工干预。
单图隐藏几何可能不唯一；分别报告完整 GT 重建和可见证据一致性，不能把二者互相替代。

## 当前入口与模块

统一几何评估入口为 `python -m src.evaluate`，从仓库根目录运行。

```bash
python -m src.evaluate \
    --pred-root /path/to/post_output \
    --gt-root /path/to/ground_truth \
    --split-list /path/to/frozen_model_ids.txt \
    --metrics condition,validity,complexity
```

- `protocol.py`：样本 ID 与预测/GT 目录定位。
- `src/evaluate.py`（原 `eval/run.py`）、`parallel.py`、`io.py`：指标调度、并行和结果保存。
- `metrics/condition.py`：面/边/点 CD、匹配 F1、面边/边点拓扑指标。
- `metrics/visible_condition.py`：使用 GT 可见性缓存的条件指标。
- `metrics/validity.py`：STEP 读取、solid 类型和内核有效性。
- `metrics/complexity.py`、`uniqueness.py`、`lfd.py`：结构统计、图重复性与 LFD。
- `adapters/`、`baseline_preprocess/`：历史 baseline 输出适配；协议必须按具体方法核对。

`metrics/point_cloud_set.py` 存在不代表 `src/evaluate.py` 已提供相应 CLI 指标；以入口 dispatch 为准。
旧 `legacy/` 评估副本与硬编码路径的 shell 脚本已删除；旧 `train_diffusion.py`、`model.py` 路径不再代表当前架构。原生 STEP 合法性、采样 CD 与 OCC 诊断在 `step_metrics.py`、`occ_diagnostics.py`。

## 输入与阶段

推理原始输出位于 `raw_root/{model_id}/data.npz`，保存面、边和连接预测。
后处理入口为 `python -m src.postprocess.construct_brep --data_root ... --out_root ...`。
后处理目录的 `recon_brep.step` 是评估输入；GT 通常为 `gt_root/{model_id}/normalized_shape.step`。
`recon_brep.step` 存在不等于有效 solid：后处理失败后仍可能输出 compound fallback。
`success.txt` 是后处理成功标志，不能替代重新读取 STEP 的独立有效性检查。
保留 raw/post 两阶段指标和失败数量；拟合、几何优化、删面与 sewing 都可能改变预测结果。
对比时固定后处理参数与预算，包括 `drop_num`、优化次数、容差、超时和 CPU/GPU 路径。
PythonOCC API 与容差敏感；修改 post 的公开接口或几何阶段须验证完整调用链。

## 共享边界输出

新 NPZ 必须明确标记标量 `representation="shared_boundary_v1"`，并提供 `pred_edge_confidence[Q]`。
每个 active query 只有一条预测曲线；`pred_edge_face_connectivity[Q,3]` 引用预测面的局部编号，不是旧 reciprocal half-edges。
Raw 导出保留重复 pair、同面 assignment 和全部 active queries，不能按 GT 边数、连接或匹配结果筛选。
`data/shared_boundaries.py:shared_prediction_diagnostics` 统计这些原始错误；它不修复或减少预测。
Post 显式适配：检查预测几何与两个预测面距离，在同一无向 pair 的全部可用候选中按置信度、几何残差、原 edge ID 选一条。
选择与移除原因写入 `shared_boundary_post.json`，原始 NPZ 不覆盖；当前 post 仍只支持不同 face pair 至多一边，不支持 native 多边/seam。
GT 几何 roundtrip 只能验证接口和内核调用；它不能证明模型预测准确、拓扑有效率提高或训练成功。

## 坐标、指标与 Oracle 信息

当前 condition 固定使用 `data/rotations.py` 中 identity-first cube24 的 0 号 GT；不是最优旋转搜索。
不要用旧 Euler64 或 Blender 原始数字序号解释当前输出；相机、尺度和坐标变换必须来自记录。
CD 当前在同类 primitive 的合并点集上计算；Hungarian 匹配用于 primitive 检测 F1 及拓扑对应，不能把 CD 描述成匹配后均值。
报告 CD 的实际实现、归一化、采样密度及 F1 阈值；展示单位换算与原始数值分开。
有效 solid 仅表示当前内核/容差检查通过，不证明形状正确、可加工或满足工程约束。
可见性来自 GT；训练监督和评估标签可以使用，若推理时额外输入给模型、筛选候选或指导修复，必须标为 Oracle 设置。
当前 visible condition 把全部预测面与可见 GT 面比较；它并不是仅比较相机可见的预测表面，也不是图像重投影误差。
没有配对 GT 的实拍样例可做可见证据与有效性分析，不能把 loader 占位几何当真实 GT。

## 固定分母与已知汇总限制

评估前冻结 intended model IDs；以该清单作为尝试总数，保留每个 ID 的阶段状态。
至少分别报告：请求总数、推理成功、后处理产出 STEP、有效 solid、可计算几何指标，以及各失败原因。
准确率或成功率必须注明分母；成功样本上的 CD/F1 均值与全样本成功率并列报告。
当前 `condition.prepare_eval_data` 对缺失或不可读预测使用原点退化几何 fallback；因此有数值结果不意味着预测成功。
当前 `compute_statistics` 使用 split 与已有预测目录交集，并跳过没有结果的目录；不能将其 `num_total` 自动视为 intended 总数。
默认无 split 时仅扫描已有目录，会遗漏未产生目录的样本；`--only-valid` 又进一步筛选 `success.txt`，必须明确披露。
跨方法配对比较既报告共同成功集合，也报告各方法在固定全集上的失败情况，避免只比较幸存样本。
失败、超时、NaN/Inf、缺 GT 和适配错误应分别记录，不能静默删除或当作普通重建误差。

## 缓存、数据质量与验证

结果缓存当前优先读取 `eval_condition.npz`，也兼容旧 `eval.npz`；默认跳过已有结果。
改变预测、GT、split 或指标协议时使用新输出目录或明确重算，并记录版本；旧缓存不能混作新结果。
生成图片的质量筛查（SAM2／DINO／CLIP）须区分轮廓分割误差与几何失真；这些分数不能认证孔槽保持或照片真实性。旧 `quality_metrics.py` 已随旧 datagen 删除。
生成数据的验收不应仅依赖待评估重建模型，以免筛选出该模型偏好的测试分布。
修改指标实现时验证已知几何差异、缺失输出及坐标错误等有区分力的案例；文档修改不触发批量 CAD 评估。
冻结清单、指标设置、后处理设置和逐件结果；记录写入 `/mnt/d/hola_research/<主题>/`、运行产物写入 `/mnt/d/hola_runs/`，遵循根目录 `AGENTS.md`。
