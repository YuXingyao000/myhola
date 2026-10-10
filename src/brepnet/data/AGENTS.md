# 数据准备与生成指南

## 研究定位

本目录服务于两个目标：从 CAD 渲染生成保持几何对应的照片风格数据，以及支持真实照片到 B-Rep 的重建。
先明确数据生成、几何对应、跨域泛化之间的因果假设，再决定清理规则或生成机制；不要默认继续旧实验的局部调参。
FLUX 合成照片与相机实拍照片必须分别命名和报告；文件名 `real_photo.npz` 不代表内容一定来自真实拍摄。
通用模型生成的 CAD 可以成为基线或方法组件；不能把其推测的隐藏结构自动作为几何真值。

## 代码入口与职责

- `../dataset.py`：训练实际读取的数据契约；主类为 `Diffusion_dataset`、`AutoEncoder_dataset3` 及严格加载的 `AutoEncoder_joint_dataset`。
- `shared_boundaries.py`：旧 reciprocal rows 到共享边界监督的显式适配，以及未筛选预测的诊断计数。
- `rotations.py`：训练、推理、评估共用的 identity-first cube24 旋转协议。
- `datagen/run_blender.py`、`run_flux.py`、`run_pack.py`：现存的分阶段编排入口，运行前核对配置与下游接口。
- `datagen/blender_scripts/`：Blender 相机、材质、渲染及模型旋转。
- `datagen/flux_scripts/`、`datagen/external/generation/`：FLUX.1-Kontext 图像编辑、提示与生成。
- `datagen/tools/`：图像打包；`datagen/real_photo/prepare_real_photo_dataset.py`：实拍图片的推理输入准备。
- 仓库根目录 `tools/audit_deepcad_v7_dataset.py`：文件、NPZ 字段与可选数组/latent 检查。
- 仓库根目录 `tools/filter_deepcad_v7_lists.py`：根据资产完整性过滤清单。
- `list/`：模型清单；生成任务的 GPU 分片不能代替 train/val/test 划分。

训练入口是 `python -m src.brepnet.train`。不要恢复不存在的 `train_diffusion.py`、`model.py` 或 `diffusion_model.py` 路径。

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

## 已观察到的生成链路漂移

以下是待修复问题，不代表现有资产均有错误；重新生成前必须核对目标路径和数据版本。

- `datagen/config.py` 已无 `PATHS` / `RUNTIME`，部分 pack/list 工具仍导入这些旧对象。
- `run_pack.py` 的 cube24 分支仍产出 `natural.npz`，与当前 loader 契约不一致。
- 单视图 pack 默认要求 `blender`、`flux`、`flux_masked` 三种输入；`run_pipeline.py` 未包含 mask 阶段。
- `run_pipeline.py` 只转发 mode 与少数路径，不能假定 train list、提示和生成参数完整传入子进程。
- Blender cube24 脚本使用自身枚举顺序；数字 PNG 序号不能直接当成运行时 identity-first 序号。
- datagen 默认配置仍含 v6 路径；现有 v7 资产不能由这些默认值推断可直接再生。

因此不把 `run_pipeline` 当作已验证的一键入口；先修复并检查所用分支的明确契约，再运行数据生成。

## 生成与清理原则

当前 Kontext 接收 RGB 图与提示词；负提示不构成相机、轮廓、孔洞或拓扑保持保证。
Blender 已知相机不意味着生成照片仍严格服从该相机；保存相机/归一化变换、源 CAD ID、渲染版本、提示、模型版本与 seed。
SAM2 轮廓 IoU、DINO 相似度和 CLIP 分数是辅助筛查信号，不能独立认证几何对应或真实感。
新数据设计应分别约束外观变化和结构保持，明确内部边界、孔槽、遮挡与相机变化的处理方式。
按源 CAD 或几何家族划分数据，再派生视图、材质和生成变体；同源变体不能跨 split。
区分重复 ID、几何近重复和外观近重复；当前 filter 的去重仅移除同一清单里的重复 ID。
不要依据测试集重建成绩反向清洗测试集；保留过滤原因、原清单、版本和完整分母。

## 修改与验证

路径、生成参数与数据模式通过 Hydra 或显式 CLI 提供；不要在新源码中写入机器路径。
源数据、缓存和旧实验产物保持可追溯，派生数据写入新位置；历史实验不自动成为新运行计划。
过滤前可用 `python tools/filter_deepcad_v7_lists.py --dry-run`；实际审计/过滤时显式指定清单、资产根目录和报告目录。
审计与过滤工具的默认报告目录仍指向 `experiments/2026-06-07/`，新报告不要覆盖到该历史目录。
数组读取检查与存在性检查分别报告；通过存在性检查不代表文件可解压、视图正确或几何配对可靠。
修改数据读取时验证具有代表性的格式与异常输入；完整生成、训练和全库清理不作为文档检查的副作用。
新 joint-boundary 运行、审计与记录写入 `checkpoints/` 的独立目录；遵循根目录 `AGENTS.md`，不要向历史 `experiments/` 追加本轮产物。
