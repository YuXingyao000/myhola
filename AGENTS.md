# HoLa-BRep Research Guidelines

## 研究目标与协作方式（2026-10-09 重启）

当前有两个交付目标：

1. 从已知 CAD 的白模／灰模渲染生成真实感图片，建立几何对应可靠、来源可追溯的数据集。
2. 从真实照片重建合法且与观测一致的 B-Rep，明确单视图下不可辨识部分的不确定性。

FLUX 编辑图属于**合成真实感图像**，不能直接称为实拍数据。真实照片能力需要独立实拍评估；有配对 CAD 的定量集与无 GT 的定性集分开。

用户希望 agent 参与研究判断和模型设计。先讨论任务假设、表示能力、观测信息、训练目标与推理接口，再讨论结构改动。主动提出有依据的架构方案、反例和取舍；不要把研究协作退化为持续建议小测试或调超参数。小规模运行用于验证实现，不代替研究方向判断。

已有代码、HoLa 论文和六月实验是可复用资产，不是必须延续的架构约束。允许重新设计表示、条件建模、解码器及训练目标；说明为何值得改变及对数据／checkpoint 的影响。不要默认继续 topology CVAE、attention bias、adapter 或旧 ROADMAP 的实验队列。

用户认为 GPT-6 Astra / Claude Opus 5.5 已显著改变 Img2CAD 的能力格局，并报告自己用 GPT-6 对约 100 个零件进行了代码重建测试，也在 `/mnt/d/agent` 做过探索。这是重要的用户实测背景；未经读取完整协议和结果，不改写成已复现的全领域 SOTA 结论。通用模型应作为当前竞争基线，也可作为数据生成、候选提议或评估辅助工具。提出专用模型时说明预期价值及如何验证，不能预设其精度、成本或速度更优。

讨论与笔记默认使用中文，代码标识符使用英文。区分源码事实、运行结果、文献结论、用户观察和研究假设。直接完成已授权的检查与整理，保留用户的未提交工作。

## 从哪里恢复上下文

- 原论文：[HoLa: B-Rep Generation using a Holistic Latent Representation](https://arxiv.org/abs/2504.14257)。论文的 surface-only holistic latent 与仓库新增的 condition fuser、alignment、topology bias 分开描述。
- 实验历史：[ROADMAP](experiments/ROADMAP.md)、[REFERENCES](experiments/REFERENCES.md)、日期目录中的 `NOTE.md`。其中“当前主线／下一步”截至六月，**重启后不自动代表当前计划**。
- 恢复状态不能只读笔记：核对 checkpoint 元数据、实际输出及评估报告，区分“已实现、已训练、已评估”。
- [2026-06-28 NOTE](experiments/2026-06-28/NOTE.md) 仍写待训练，但该日 `outputs_oracle_topology_log_multiplier_real/checkpoints/last.ckpt` 已到 100,000 steps（该运行设定上限）；`last-v1.ckpt` 到 892,437 steps / epoch 48（上限 1,000,000）。两者均为 real-photo、`log_multiplier`、`max_multiplier=2`。本次未在该输出目录找到完整下游评估报告；不能称为尚未训练，也不能宣称有效性已提升。
- 六月末旧汇总混有不同 checkpoint、后处理和评估口径，不作为已固定的公平基线。`experiments/baseline/` 是历史复现资产，使用前确认协议。
- 原 `skills/brepnet-experiment-{notes,planning}/SKILL.md` 及部分安装副本在工作区已删除。不要依赖失效路径或擅自恢复；下面的记录规则保持自包含。

## 当前实现地图

主包是 `src/brepnet/`，其他 `src/` 项目和历史脚本不是默认入口。

| 位置 | 当前职责 |
|---|---|
| `src/brepnet/train.py` | Hydra + Lightning 统一 VAE / diffusion 入口 |
| `src/brepnet/dataset.py` | 当前 `Diffusion_dataset`，以及仍有旧字段的 VAE / legacy 数据路径 |
| `src/brepnet/models/vae.py` | 面／半边编码、拓扑特征融合、每面 latent、neural intersection、几何解码 |
| `src/brepnet/models/joint_vae.py`、`joint_decoder.py`、`joint_losses.py` | 新 shared-boundary VAE 与面／边双向解码；保持旧 encoder 合同，具体实现见 `checkpoints/joint_boundary/PLAN.md` |
| `src/brepnet/run_management.py`、`prediction_export.py` | 独立 run 目录、配置／Git 溯源、逐对象原始预测导出 |
| `src/brepnet/models/diffusion.py` | diffusion 装配、损失、采样与 VAE 解码 |
| `src/brepnet/models/diffusion_latents.py` | cached mean/std 或 VAE 编码 |
| `src/brepnet/models/diffusion_padding.py` | face token padding、有效面筛选、重复 latent 去除 |
| `src/brepnet/models/diffusion_condition.py`、`condition_encoders.py` | 条件工厂、图像／点云编码 |
| `src/brepnet/models/diffusion_denoiser.py` | 图像融合、alignment、denoiser、可选 oracle / learned topology bias |
| `src/brepnet/data/datagen/`、`configs/datagen/` | Blender 渲染、FLUX 编辑、打包；存在入口契约漂移，见局部指南 |
| `src/brepnet/data/rotations.py`、`extract_visibility.py` | cube24 旋转与可见性缓存 |
| `src/brepnet/eval/run.py`、`eval/metrics/` | 统一评估调度与指标 |
| `src/brepnet/post/` | 曲面／曲线拟合、裁剪、缝合、STEP 构造与检查 |

典型路径：图片 → 冻结 DINOv2 patch tokens → 前端 cross-attention fuser → face-latent DDPM → 冻结 VAE 的面几何与成对 intersection 解码 → OCC 后处理。默认每面 32 维、最多 30 个 token、random duplicate padding。这是已有实现，不是未来架构要求。

细节见 [源码指南](src/brepnet/AGENTS.md)、[数据指南](src/brepnet/data/AGENTS.md)、[评估指南](src/brepnet/eval/AGENTS.md)。

## 架构讨论必须处理的问题

- **观测与先验**：区分可见几何证据与隐藏补全。单张照片通常不能唯一确定完整 CAD；每张训练图只有一个 GT 本身并不否定 conditional diffusion / CVAE 的概率学习。分析唯一输出假设、潜在变量和评估是否错误惩罚合理补全。
- **面集合对应**：face latent 的排列没有天然语义。独立预测的 adjacency 不能未经对应关系处理就作用于随机排列／复制的 latent slots。推理时额外输入 GT ordering、GT face count 或 GT topology 必须标明 oracle 条件；训练时使用 GT 监督不属于此限制。
- **几何与有效性**：图连通、没有孤立点不等于可裁剪、流形、闭合的 STEP solid。分析 VAE 表示／解码和 OCC 后处理的限制，不能只看 diffusion loss 或 adjacency F1。
- **真实感与几何保真**：照片风格更真不代表配对标签正确。相机、轮廓、孔槽、内部边界、薄壁与遮挡变化都可能破坏监督；SAM/CLIP/DINO 分数只是质量信号。
- **新方向的证据**：写明架构解决哪个失败机制、和通用 CAD agent 的关系、决定去留的下游证据。查阅近期原始论文，不能把已有的 validity-aware decoder 等泛化思路直接当作新贡献。

## 数据契约

以下是已有 DeepCAD v7 资产的位置，不保证当前机器挂载完整；新代码通过 Hydra overrides 或环境变量传路径。

| 资产 | 已用位置 |
|---|---|
| 原始 B-Rep | `/mnt/d/data/deepcad_v7` |
| 条件图像 | `/mnt/d/data/deepcad_v7_cond` |
| 24-view latent cache | `/mnt/d/data/ae_cache/1119_deepcad_aug1_11k_24` |
| AE checkpoint | `/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt` |

- `imgs.npz`：`svr_imgs`、`sketch_imgs`；当前照片条件是 `real_photo.npz` 的 `flux`，不是 `svr.npz`、`single_view.npz` 或 `natural.npz`。
- latent cache：`{model_id}_{rotation_id}/features.npy`，通常 `[num_faces, 64] = [mean(32), std(32)]`，不是 log-variance；默认 diffusion 使用 mean。
- 图像、latent、B-Rep face order、visibility 必须使用一致的 model/view/rotation 约定。不要仅凭数字文件名认定不同工具的 cube24 顺序一致。
- 拆分以 CAD identity／去重组为单位，之后才扩展视角与外观。保留源 split、过滤理由及版本，避免相同几何的不同照片跨 split 泄漏。
- 新数据记录源 CAD、相机／变换、图像模型版本、prompt、seed、参数与筛选原因；生成器升级单独追踪。

## 开发与验证

环境定义为 `environment.yml`，常规安装为 `conda env create -f environment.yml`、`conda activate hola-brep`、`pip install -e .`。先检查现有环境，不因阅读或文档任务自动重装。

训练使用 `python -m src.brepnet.train --config-name <config>`。准备 real-photo baseline 时优先检查 `configs/train_diffusion_baseline_real.yaml`，显式设置数据路径、设备数与 checkpoint。不要把 `train_diffusion_real.yaml` 名字当作协议保证：它继承的 overfit / validation 条件有陷阱，详见源码指南。VAE loader 已接受规范字段及显式旧别名，冲突值必须报错；仍需核对实际数据。新联合边界 decoder 的命令与迁移协议见 [README](checkpoints/joint_boundary/README.md)。

保持既有模块职责；有理由改变边界时说明迁移。用 Python 3.10+、四空格、snake_case、显式配置与适当类型注解。不要把研究行为藏在路径猜测、checkpoint 名称或静默 fallback 中。

验证与改动相称：文档检查路径和差异；代码检查相关导入／单元行为；模型或数据接口检查真实小批量形状、loss、梯度与推理语义。实现可靠后才启动正式研究运行。不要把整个 `src/brepnet/test/` 当作无外部依赖的统一套件，也不要用 smoke test 宣称研究结论。

## 记录与评估纪律

**从 joint-boundary 实现开始，新实验、计划及结果记录统一放在 `checkpoints/`，不再写入 `experiments/`。** 后者保留为历史资产，不移动或删除。每个研究主题维护受 Git 管理的 `PLAN.md`、`README.md`／`NOTE.md`，记录“为什么、具体实施、结果／状态、参考链接”；事实、计划和已完成结果分开。

统一训练默认目录为 `checkpoints/<trainer.exp_name>/<UTC 时间戳>/`，权重 `weights/`、日志 `logs/`、W&B `wandb/`、逐对象预测 `raw_predictions/`、可视化 `visualizations/`、配置与 Git 元数据 `metadata/` 分开。`trainer.output_dir` 是根目录；`trainer.run_dir` 显式指定完整 run。续训使用 `trainer.resume_from_checkpoint` 回到原 run；不要将 init 当作 resume。每次调用保存 resolved config、Git commit／dirty diff、seed 和命令，已有 run 不允许无续训标记覆盖。

运行权重、原始几何及临时日志由 `.gitignore` 排除；研究文档与紧凑的验证报告保留版本管理。记录数据／split、checkpoint、seed、预算及输出路径。先建立可判断方向的研究问题，不机械延续旧实验编号。

比较固定输入信息、样本列表、采样／重试预算、checkpoint、后处理版本及指标协议。分别报告生成覆盖率、有效 STEP／solid 比例、几何与可见区域一致性，以及成本／耗时。失败、缺失、超时留在预定分母里；仅成功子集的几何分数单列。

GT best-of-K 是 oracle 指标，不能当作可用的单次输出质量；同时说明实际选择策略。原始解码质量与修复后的质量分开。比较 agent 与神经模型时，记录各自可用的工具和修复预算。

不要提交数据集、checkpoint、生成模型、W&B token 或机器凭证。PythonOCC 7.8.1 API 具有版本敏感性，改签名必须验证调用链。保留历史产物，简短提交说明目标；实验 PR 附数据假设、协议、结果与笔记链接。
