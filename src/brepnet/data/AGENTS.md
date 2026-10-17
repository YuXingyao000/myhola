# 数据准备与生成指南

## 核心目标与范围（2026-10-12）

本目录的核心工作是**配对图像数据生成与数据质量评估**，服务于从单张真实照片重建 B-Rep CAD component 的研究。
从已知 CAD 的白模／灰模或受控材质渲染出发，生成真实感图片；目标是扩大几何对应可靠、来源可追溯、对实拍重建有用的数据覆盖。
数量、照片风格和下游效果分别验证，不能以图片总数或视觉好看替代标签正确性。
用户报告已有约 5 万余张合成真实感图片，这是现有规模背景，不是本轮已核验库存或未来规模上限。

FLUX 编辑图统一称为**合成真实感图像**，与相机实拍照片分别命名、划分和报告；`real_photo.npz` 是历史接口名，不证明图片来自真实拍摄。
有配对 CAD 的实拍定量集与无 GT 的实拍定性集分开。单张照片通常无法唯一确定隐藏几何，不能把合理但不同的隐藏补全全部视为图像生成错误。
当前 Blender → FLUX.1-Kontext → NPZ 是可复用资产，不是未来生成器、条件表示或筛选方法的架构约束。
通用模型可辅助生成、提议和评价；其推测的隐藏结构或主观打分不能自动成为几何真值。

## 工作方法：先建立评价，再按证据调整

1. 先审阅当前输入、生成机制、数据契约、实际产物和已有报告，区分源码事实、历史结果、用户观察与研究假设。
2. 围绕研究目标讨论评价维度及可测性，再确定样本、指标定义、失败处理、人工核验方式与判定规则。旧脚本里的默认阈值不自动成为验收标准。
3. 冻结评价协议和对照数据版本，检查指标能否区分已知失败，同时核对测量工具自身的错误；据此建立当前管线的基线。
4. 从逐件结果和失败类型提出可证伪的改动假设，说明预期改善、可能损害及取舍，再调整生成、后处理或筛选。
5. 使用相同 CAD／视图清单、采样与重试预算、评价版本进行比较，报告分布、失败案例与覆盖变化，再决定是否扩大生成规模。

这是持续迭代的方法，不是看到问题便立刻修改 prompt、guidance、材质或生成模型的指令。
2026-10-12 本轮只审阅、更新指南并讨论指标；不据此启动生成、训练、全库过滤或修改生成逻辑。
小规模运行用于验证实现和测量可靠性；研究方向由明确假设与下游证据判断，不能退化为无止境的局部调参。

## 脚本定位：维护功能与历史入口分开

以下分类是后续维护政策。“保留功能”不等于入口已跑通；“弃用”指不作为新实验入口，不代表删除文件、历史产物或所有函数都已失效。
本目录以外的评估模块仍保持原职责，不为聚焦数据生成而复制一套评估实现到 `data/`。

### 核心功能：继续维护，先核对接口

| 位置 | 职责与当前状态 |
|---|---|
| `datagen/run_blender.py`、`datagen/run_flux.py`、`datagen/run_pack.py` | 分阶段编排；保留职责，但配置、CLI 和打包合同尚未闭合，见下方问题清单 |
| `datagen/blender_scripts/render_single_view.py`、`datagen/blender_scripts/render_cube24.py` | 受控相机、网格归一化、材质与视图渲染；cube24 文件编号不是运行时编号 |
| `datagen/flux_scripts/`、`datagen/external/generation/` | FLUX 编辑 worker 与底层加载／推理；单视图委派入口和动态 prompt 传递存在漂移 |
| `datagen/prompts.py`、`datagen/config.py`、`datagen/launcher.py`、`datagen/utils.py`；根目录 `configs/datagen/` | 提示构建、路径、进程与分片、配置；不能仅凭 YAML 推断实际生效行为 |
| `datagen/tools/pack_single_view_npz.py` | 写入 `real_photo.npz` 的功能归属主线；当前导入失效，不能直接作为可用命令推荐 |
| `../eval/quality_metrics.py` | 当前合成图片质量评价入口，读取 `imgs.npz` 与 `real_photo.npz`；已有 SAM2／DINO／CLIP 实现，但评价合同仍需审阅 |
| 根目录 `tools/audit_deepcad_v7_dataset.py`、`tools/filter_deepcad_v7_lists.py` | 资产与字段审计、可选数组检查、完整性筛单；不认证图像与 CAD 几何配对 |
| `../dataset.py`、`list/` | 消费端数据契约与源 split 清单；GPU／机器分片不得替代研究 split |

### 按需使用的功能性辅助

- `rotations.py`、`shared_boundaries.py`：现有训练／评估直接依赖的旋转协议与边界监督适配，继续保留；不能因本轮聚焦图片而废弃。
- `extract_visibility.py`、`extract_brep_from_step_half_structure.py`：可见性缓存与几何提取资产；按需检查路径、相机、面顺序与提取语义，不是默认图像生成入口。
- `datagen/tools/sample_lists.py`、`datagen/tools/prepare_render_lists.py`、`datagen/split_model_lists.py`、`sample_random_data_folders.py`：抽样或计算任务分片。记录源清单和 seed；合并清单或剔除缺资产对象必须保留源 split 与排除记录。
- `datagen/mask_flux_with_blender.py`：按参考渲染亮度阈值将背景置白的派生处理，不是几何修复或质量认证；保留未经 mask 的 FLUX 原图，不能用施加的参考轮廓证明原生成图保持了轮廓。
- `datagen/real_photo/prepare_real_photo_dataset.py`：实拍推理输入准备；默认裁剪含颜色／背景启发式，需要检查适用对象。loader 占位几何不是照片 GT。
- `../eval/run.py` 与 `../eval/metrics/`：下游 B-Rep 重建评价，按 [评估指南](../eval/AGENTS.md) 使用；与生成图片本身的质量评价分开。

### 弃用为新实验入口，保留历史资产

| 位置 | 弃用原因／后续边界 |
|---|---|
| `datagen/run_pipeline.py` | 不完整转发配置，阶段合同未闭合；停用其“一键可复现”入口定位，先核对分阶段合同 |
| `datagen/run_lists.py`、`datagen/compat.py` | 旧 list 编排与兼容对象；`split_model_lists.py` 等现有辅助工具依赖的兼容对象暂保留，不继续扩展为新配置层，也不通过恢复旧全局对象掩盖合同冲突 |
| `datagen/tools/pack_natural_npz.py`、`compress_natural_img.py`、`pack_conditional_data.py`、`combine_data.py` | 旧 `natural.npz`／条件组装流程，不是当前照片条件格式；迁移时显式说明输入输出与视图映射 |
| `extract_imgs.py`、`prepare_img_feature.py` | 旧 OCC Euler64 渲染／特征缓存入口；不能直接替代当前 identity-first cube24 和条件编码合同 |
| `fidelity_natural.py`、`fidelity_sketch.py`、`feature_similarity.py`、`feature_similarity_lightning.py` | 旧条件格式、草图任务或搜索最佳参考视图的协议；不作为当前已知相机配对图片的验收入口。Lightning 版本还导入不存在的 `fiedility` 模块 |
| `prepare_custom_6_demo.py`、`def_real_data.py`、`def_real_data_icp.py` | 旧 demo／重建数据准备；不以其中的占位或配准结果定义新数据集 GT |
| `extract_brep_face.py`、`extract_txt_data.py`、`extract_txt_features.py` | 旧单独面／文本任务；不属于当前配对图片主线 |
| `batch_copy.sh`、`convert_split_from_deepcad.py`、`filter_out_faces.py`、`check_data_from_lfd.py`、`check_data_from_lfd_debug.py`、`viz_dataset.py` | 历史复制、划分、筛选或调试资产；不自动执行其中机器路径、清理或重划 split 行为 |

需要复用历史算法时可显式适配，不把旧入口恢复为默认协议。`datagen/README.md`、`datagen/commands_reference.sh` 的旧命令仅作线索。
`ROTATION_CLEANUP.md` 仍含 identity 在 18 的旧约定和删除计划，不能照单执行；当前语义以 `rotations.py` 与消费端为准。
训练入口是 `python -m src.brepnet.train`；不要恢复不存在的 `train_diffusion.py`、`model.py` 或 `diffusion_model.py` 路径。

## 当前运行时契约

以 `dataset.py` 和所用 Hydra 配置为准，不能仅依据历史 README 或文件名推断格式。

| 资产 | 当前协议 |
|------|----------|
| 原始几何 | `raw_data_root/{model_id}/data.npz`；含面、边、连接关系等数组 |
| 灰模与草图 | `condition_data_root/{model_id}/imgs.npz`，键为 `svr_imgs`、`sketch_imgs` |
| 照片风格条件 | `condition_data_root/{model_id}/real_photo.npz`，训练读取键 `flux` |
| 旋转 latent | `cached_latent_root/{model_id}_{rotation_id}/features.npy`，通常为 `[num_faces, 64]` 的 latent stats |
| 可选可见性 | `visibility_root/{model_id}/view_{rotation_id:03d}/visibility.npz` |

`rotation_id=0` 是 identity；latent、条件图、face adjacency 与 visibility 必须使用相同对象、旋转及面索引。
单张 `flux` 的形状 `[H,W,3]` 仅支持 identity；旋转增强需对应的 `[24,H,W,3]` 图像。
`real_photo_ratio>0` 不支持 `cached_condition=True`；验证/测试的照片比例也要核对实际 loader 配置。
当前 loader 明确拒绝旧 `natural_img` / `natural.npz` 路径。不要将其改称 `svr.npz` 或默默增加格式猜测。
实拍准备脚本复制任意有效 `data.npz` 作为 loader 占位；该文件不是照片对应的 GT，不能据此计算重建准确率。

## 共享边界监督契约

`AutoEncoder_joint_dataset` 保留旧 encoder 的面顺序、双向边、归一化和旋转，只额外返回逐对象 `boundary_targets`。
每组不同面的无向 pair 必须恰有两条 reciprocal rows；目标保留较小原始 edge ID 的采样，不平均、不重采样。
这代表旧提取器合并后的 face-pair boundary，不是完整 native physical edge，也不提供 coedge、UV/pcurve、loop 或 vertex 监督。
`curves[K,L,3]`、局部 `face_pairs[K,2]` 与 `is_closed[K]` 在 collate 后仍逐对象保存；不能使用批次全局面编号。
闭合曲线保留原始首末采样，端点距离用 `boundary_closed_tolerance` 判断；匹配可循环对齐前 `L-1` 点。
互反检查先比较正/反序；配置允许时，对闭合曲线再用双向顶点到折线距离。所有阈值均为输入全局 XYZ 单位。
缺失、seam、覆盖不全及不一致的目标带 sample ID 报错；零边界合法，容量溢出由模型报错，禁止静默截断或过滤。
先用 `tools/audit_shared_boundaries.py` 完整审计清单，记录 accepted IDs 和逐件 rejection；对照方法使用同一训练清单。
VAE 同时支持标准与旧字段名，但矛盾配置必须报错；新配置使用 `raw_data_root`、`epoch_scale_factor`、`condition_name`。

## 当前生成方向与已核对的合同问题

2026-10-17 迁移时已修复 Hydra 配置组 package、FLUX 完整 prompt / 负提示 / 预编码参数转发与单视图 CLI 接口，并新增显式路径、清单和输出保护的 `python -m src.brepnet.data.extract_brep`。修复及验证边界见 [记录](../../../checkpoints/server_migration/deployment_20261016/data_entrypoints/README.md)。这些入口修复不代表 ABC 全库预处理完成，也不代表下列打包、旋转映射和来源记录问题均已解决。

当前设计从网格渲染 RGB，再以 Kontext 和文字描述编辑为桌面上的工业零件照片，最后打包成条件输入。
虽然 README 使用“白模”表述，Blender 默认配置实际指定 `Metal010_1K-JPG` PBR 材质；不能假定现有输入都是纯灰模。
当前 FLUX 调用没有显式传入 CAD、深度、法线或拓扑约束；几何保持依赖编辑模型行为，需用数据验证。
以下是当前源码问题，不证明历史数据由此版本生成，也不证明已有图片全部错误；本轮记录而不修复。

- `datagen/tools/pack_single_view_npz.py` 仍从 `datagen/config.py` 导入已移除的 `PATHS`／`RUNTIME`；list 工具已改从 `compat.py` 导入。`compat.py` 也不具备 pack 所需的完整字段，不能只改 import 来修复。
- `run_pack.py` 的 cube24 分支仍产出 `natural.npz`，与当前 loader 契约不一致。
- 单视图 pack 默认要求 `blender`、`flux`、`flux_masked` 三种输入；`run_pipeline.py` 未包含 mask 阶段。
- `run_pipeline.py` 只转发 mode 与四个路径，train list、输出位置、prompt 和生成参数不能假定完整传入子进程。
- `run_flux.py` 已完整转发 prompt 配置；动态 prompt 按 CAD / seed 生成，固定模式可预编码，动态模式配合预编码明确报错。直接调用 worker 且不传完整配置时仍保留历史行为，不能与新 launcher 配置混为一谈。
- 单视图 worker 委派给 `external/generation/generate.py`，现已接受 `--prompt-base`／`--neg-prompt` 及完整 prompt 配置；实际权重加载和目标 GPU 验证单独记录。
- Blender cube24 使用自身枚举顺序，raw identity 为 `18.png`；不能直接用 PNG 序号索引运行时 identity-first latent／条件图。旧 cube24 pack 没有完成这项映射。
- Blender 相机与归一化定义在源码中，但没有随每件 PNG 保存相机／归一化／旋转 sidecar；FLUX 有 prompt 和 seed 日志，但缺完整逐图版本记录。launcher 重跑会覆盖同路径日志，`skip_existing` 仅检查文件存在，不能保证同一生成版本。
- datagen 默认配置仍含 v6 路径；现有 v7 资产不能由这些默认值推断可直接再生。
- Hydra defaults 的 package 层级已显式对齐 `cfg.paths`／`cfg.blender`／`cfg.flux`／`cfg.prompt`，并通过真实 compose 与入口 fixture 检查；这不证明底层 pack 已可运行。

底层 GGUF 加载和推理参数有版本敏感性，变更应记录版本并验证调用链；历史 README 的“不可修改”不构成研究架构限制。

## 已有评价证据与边界

- [2026-5-30 图片质量报告](../../../experiments/2026-5-30/dataset_evaluation/deepcad_v7_cond_quality_report.json) 保存 52,556 个不同 CAD ID 的逐件分数，另有 521 skipped；不是本轮重评结果。
  旧 SAM2 silhouette IoU 均值 0.6079、中位数 0.7327，28,100 件达到旧阈值 0.70；53.47% 的 pass rate 以可评价样本为分母，不是已认证的几何合格率。
  报告未完整保存生成版本、输入摘要和失败 ID／原因，不能直接升级为冻结的新基线，也不能从低 IoU 单独判定 FLUX 形变。
- [2026-06-07 资产审计](../../../experiments/2026-06-07/dataset_audit/summary.json) 记录 53,225 个 CAD ID、52,556 passed，且未启用数组读取与 latent shape 检查；这是当时资产审计，不是现有图片计数或配对认证。
- `../eval/quality_metrics.py` 当前会略过缺失／读取失败对象；SAM2 使用参考 mask 质心作单点提示，质心可能落在孔洞或背景。
  `clip_realism` 实际是图文余弦相似度，文本含生成正／负提示；DINO 使用全局 CLS 相似度。名称不能替代指标语义。
- 具体新指标、阈值、组合与筛选策略须在讨论后单独制定；指南不预设“高 IoU／CLIP／DINO 即可扩库”的结论。

## 生成与清理原则

当前 Kontext 接收 RGB 图与提示词；负提示不构成相机、轮廓、孔洞或拓扑保持保证。
Blender 已知相机不意味着生成照片仍严格服从该相机；保存相机/归一化变换、源 CAD ID、渲染版本、提示、模型版本与 seed。
新派生数据还需记录源文件摘要、实际生效参数、代码版本、view／rotation 映射、裁剪缩放／mask 等处理、失败与过滤原因；原图与派生图可相互追溯。
SAM2 轮廓 IoU、DINO 相似度和 CLIP 分数是辅助筛查信号，不能独立认证几何对应或真实感。
新数据设计应分别约束外观变化和结构保持，明确内部边界、孔槽、遮挡与相机变化的处理方式。
按源 CAD 或几何家族划分数据，再派生视图、材质和生成变体；同源变体不能跨 split。
区分重复 ID、几何近重复和外观近重复；当前 filter 的去重仅移除同一清单里的重复 ID。
不要依据测试集重建成绩反向清洗测试集；保留过滤原因、原清单、版本和完整分母。
分别统计 CAD identity／去重组、视图、外观变体、生成尝试与最终接受图片数；不能把 24 个视图当作 24 个独立几何样本。
评价前固定预定清单，缺失、超时、生成失败和测量失败均保留状态；成功子集分数与全清单覆盖率并列，不能静默缩小分母。

## 修改与验证

路径、生成参数与数据模式通过 Hydra 或显式 CLI 提供；不要在新源码中写入机器路径。
源数据、缓存和旧实验产物保持可追溯，派生数据写入新位置；历史实验不自动成为新运行计划。
过滤前可用 `python tools/filter_deepcad_v7_lists.py --dry-run`；实际审计/过滤时显式指定清单、资产根目录和报告目录。
审计与过滤工具的默认报告目录仍指向 `experiments/2026-06-07/`，新报告不要覆盖到该历史目录。
数组读取检查与存在性检查分别报告；通过存在性检查不代表文件可解压、视图正确或几何配对可靠。
修改数据读取时验证具有代表性的格式与异常输入；完整生成、训练和全库清理不作为文档检查的副作用。
后续图像生成主题的计划、协议、审计和结果统一放在 `checkpoints/image_generation/` 的独立目录，记录“为什么、实施、状态／结果、参考链接”；尚未实施的讨论明确标注为候选方案。
其他主题同样遵循根目录 `AGENTS.md`，新记录写入 `checkpoints/`，不要向历史 `experiments/` 追加产物。
