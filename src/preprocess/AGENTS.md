# 预处理指南（src/preprocess/）

本目录专指对 DeepCAD／ABC 原始 CAD 的预处理：STEP → `data.npz`、可见性缓存、cube24 旋转约定、共享边界与 UV 监督目标、固定 split 清单。图片数据集生成在 `src/datagen/`（见其 `AGENTS.md`），训练读取在 `src/datasets/`。研究目标、代码卫生与路径对照见根目录 [AGENTS.md](../../AGENTS.md)。

## 模块

| 位置 | 职责 |
|---|---|
| `extract_brep.py` | `python -m src.preprocess.extract_brep`：显式数据根、输出根、清单与排除清单；拒绝覆盖已有输出和重叠路径；`--dry-run` 只出计划 |
| `extract_brep_from_step_half_structure.py` | STEP → 面／半边采样与连接关系的提取实现（由 `extract_brep` 调用） |
| `extract_visibility.py` | 逐视图可见性缓存 `visibility_root/{model_id}/view_{rotation_id:03d}/visibility.npz` |
| `rotations.py` | cube24 旋转约定：`rotation_id=0` 是 identity；latent、条件图、adjacency、visibility 共用 |
| `shared_boundaries.py` | 共享边界监督目标与预测诊断（`AutoEncoder_joint_dataset`、评估都依赖） |
| `edge_uv.py`、`extract_edge_uv.py`、`uv_curve_targets.py` | 边 UV 监督数据（`edge_uv_root`）与 UV 曲线目标；生成命令见 `tools/generate_uv_curve_targets.py` |
| `model_lists.py` | 读取 8 位 ID 清单（去重保序、拒绝非法 ID） |
| `splits/` | 固定清单：`deduplicated_deepcad_{training,validation,testing}_7_30.txt`、`deepcad_v7_contract_{train,validation,test}.txt`（VAE 合同审计 accepted，validation 3495 件）、`deepcad_v7_validation100.txt` |

相关工具：`tools/audit_deepcad_v7_dataset.py`（资产与字段审计）、`tools/filter_deepcad_v7_lists.py`（完整性筛单，先 `--dry-run`）、`tools/generate_model_lists.py`。工具的报告目录显式传入，写到 `/mnt/d/hola_research/<主题>/`；它们不认证图像与 CAD 几何配对。`docs/rotation_cleanup.md` 含 identity 在 18 的旧约定，不能照单执行，当前语义以 `rotations.py` 为准。

## 运行时数据契约

以 `src/datasets/` 的 loader 与所用 Hydra 配置为准，不能依据历史 README 或文件名推断格式。

| 资产 | 当前协议 |
|------|----------|
| 原始几何 | `raw_data_root/{model_id}/data.npz`；含面、边、连接关系等数组 |
| 灰模图 | `condition_data_root/{model_id}/imgs.npz`，训练读取键 `svr_imgs`（文件中另有历史键 `sketch_imgs`，不再使用） |
| 照片风格条件 | `condition_data_root/{model_id}/real_photo.npz`，训练读取键 `flux` |
| 旋转 latent | `cached_latent_root/{model_id}_{rotation_id}/features.npy`，通常 `[num_faces, 64]` = mean(32) + std(32)，不是 log-variance |
| 可选可见性 | `visibility_root/{model_id}/view_{rotation_id:03d}/visibility.npz` |

- latent、条件图、face adjacency 与 visibility 必须使用相同对象、旋转及面索引。
- 单张 `flux` `[H,W,3]` 仅支持 identity；旋转增强需对应的 `[24,H,W,3]`。
- `real_photo_ratio>0` 不支持 `cached_condition=True`；验证／测试的照片比例核对实际 loader 配置。
- loader 拒绝旧 `natural_img`／`natural.npz`；不要改称 `svr.npz` 或静默增加格式猜测。
- `real_photo.npz` 是历史接口名；FLUX 编辑图是**合成真实感图像**，不证明来自实拍。

## 共享边界监督契约

`AutoEncoder_joint_dataset` 保留旧 encoder 的面顺序、双向边、归一化和旋转，只额外返回逐对象 `boundary_targets`。

- 每组不同面的无向 pair 必须恰有两条 reciprocal rows；目标保留较小原始 edge ID 的采样，不平均、不重采样。这代表旧提取器合并后的 face-pair boundary，不是完整 native physical edge，也不提供 coedge、UV/pcurve、loop 或 vertex 监督。
- `curves[K,L,3]`、局部 `face_pairs[K,2]` 与 `is_closed[K]` 在 collate 后仍逐对象保存；不能使用批次全局面编号。
- 闭合曲线保留原始首末采样，端点距离用 `boundary_closed_tolerance` 判断；匹配可循环对齐前 `L-1` 点。互反检查先比较正／反序；配置允许时，对闭合曲线再用双向顶点到折线距离。阈值均为输入全局 XYZ 单位。
- 缺失、seam、覆盖不全及不一致的目标带 sample ID 报错；零边界合法，容量溢出由模型报错，禁止静默截断或过滤。
- 训练清单使用经完整审计的 accepted IDs（`splits/deepcad_v7_contract_*.txt`），审计记录在 `/mnt/d/hola_research/joint_boundary/validation/`；对照方法使用同一清单。
- VAE 同时支持标准与旧字段名，但矛盾配置必须报错；新配置使用 `raw_data_root`、`epoch_scale_factor`、`condition_name`。

## Split 与数据纪律

- 拆分以 CAD identity／去重组为单位，之后才派生视角、外观与生成变体；同源变体不能跨 split。当前 filter 的去重只移除同一清单里的重复 ID，几何近重复需另行处理。
- 不依据测试集重建成绩反向清洗测试集；保留过滤原因、原清单、版本和完整分母。GPU／机器分片不得替代研究 split。
- 新派生数据写入新位置，记录源文件摘要、实际生效参数、代码版本、view／rotation 映射和过滤原因；源数据、缓存不原地改写。
- 存在性检查与数组读取检查分别报告；通过存在性检查不代表可解压、视图正确或几何配对可靠。
- ABC 最终评估子集在查看结果前冻结方案与协议（见根 `AGENTS.md`）；核对其与开发数据的 CAD／几何重叠，不因数据集名称不同就认定独立。

修改数据读取时用具有代表性的格式与异常输入验证；完整提取、训练和全库清理不作为文档检查的副作用。
