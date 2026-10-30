# 图片数据集生成指南（src/datagen/）

本目录是我们自己的图片数据集生成代码：从已知 CAD 制作场景模板与房间场景、拍摄（Blender 5.2.2）、Codex 审阅返修、照片风格修图。流程、命令与文件职责见同目录 `README.md` 及各 `step*/README.md`。DeepCAD／ABC 原始数据预处理在 `src/preprocess/`。研究目标、代码卫生与路径对照见根目录 [AGENTS.md](../../AGENTS.md)。

## 运行约定

- 脚本以脚本方式运行（`python src/datagen/<script>.py`、`blender --python src/datagen/build.py`）或 `python -m src.datagen.step2`；同目录模块用脚本式导入（`from geometry import ...`），不要改成依赖仓库其他包的导入。import-linter 禁止 `src.datagen` 与训练栈互相导入。
- 模型读取的协议与指南在 `prompts/`（含 `prompts/room/REFERENCE.md` 质量参考）；它们随代码版本管理。运行产物、逐轮证据、渲染图写到 `/mnt/d/hola_runs/` 或数据目录，研究记录写到 `/mnt/d/hola_research/image_generation/<主题>/`，不写进仓库。
- step1／step4 在运行记录里冻结脚本路径用于续跑校验；2026-10-30 从 `packages/cad_scene_factory/` 迁到此处之前启动的 run 不能在这里续跑。
- `step4/retouch/` 的分类器结构改编自 Image-Adaptive-3DLUT（Apache-2.0，`LICENSE` 随附）；预训练权重不进仓库。

## 数据研究原则

- FLUX 等编辑模型输出的是**合成真实感图像**，与相机实拍照片分别命名、划分和报告。有配对 CAD 的实拍定量集与无 GT 的实拍定性集分开。单张照片通常无法唯一确定隐藏几何，不能把合理但不同的隐藏补全全部视为生成错误。
- 先建立评价再调整：审阅输入、生成机制、数据契约与实际产物；讨论评价维度与可测性；冻结协议与对照数据版本并检查指标能否区分已知失败；从逐件结果提出可证伪的改动假设；用相同 CAD／视图清单、预算与评价版本比较后再扩大规模。不要一看到问题就立即改 prompt、材质或生成模型。
- 数量、照片风格和下游效果分别验证；图片总数或视觉好看不能替代标签正确性。SAM2 轮廓 IoU、DINO、CLIP 分数只是辅助筛查信号，不能独立认证几何对应或真实感。
- 已知 Blender 相机不意味着生成照片仍严格服从该相机。每张图保存相机／归一化变换、源 CAD ID、渲染版本、提示、模型版本与 seed；原图与派生图（裁剪、mask、修图）可相互追溯。
- 按源 CAD 或几何家族划分数据，再派生视图、材质和生成变体；同源变体不能跨 split。分别统计 CAD identity、视图、外观变体、生成尝试与最终接受图片数；24 个视图不是 24 个独立几何样本。
- 评价前固定预定清单；缺失、超时、生成失败和测量失败均保留状态，成功子集分数与全清单覆盖率并列。
- 通用模型（Codex 等）可辅助生成、提议和评价；其推测的隐藏结构或主观打分不能自动成为几何真值。

历史证据（六月 FLUX 管线的质量报告与资产审计）在 `/mnt/d/hola_research/experiments/2026-5-30/`、`2026-06-07/`；它们未完整保存生成版本与失败原因，不能直接作为冻结基线。旧 Blender → FLUX.1-Kontext → NPZ 管线代码已于 2026-10-30 删除，需要时从 tag `archive/pre-cleanup-20261030/main` 取回。
