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

报告与界面的字体偏好：代码首选 Commit Mono（字体家族名 CommitMono），Geist Mono 为备选；标题／设计 UI 用 Jost，正文／内容用 Archivo，中文用思源黑体。用户当前 Windows Cursor 代码已选 Commit Mono，Codex 界面／正文用 Archivo、代码用 Commit Mono。远程客户端字体由 Windows 本地配置决定，服务器字体用于生成产物；命令行配色和布局等待用户另行指定，P10k 保持 ASCII 模式。

## 从哪里恢复上下文

- 第二台 RTX 5090 服务器的环境、数据迁移进度和已知缺口见 [部署记录](checkpoints/server_migration/deployment_20261016/README.md)。目标仍使用 `/mnt/d/python`；新环境采用 PyTorch 2.8/cu128、PythonOCC 7.8.1 API 系列，不能把这些版本反套到源机。实际完成状态以记录中的验收结果为准，暂缓迁移的 diffusion 权重不能假定已存在。
- 原论文：[HoLa: B-Rep Generation using a Holistic Latent Representation](https://arxiv.org/abs/2504.14257)。论文的 surface-only holistic latent 与仓库历史 condition fuser / alignment、当前逐层 image cross-attention 及可选 topology bias 分开描述。
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
| `src/brepnet/models/vae.py` | 原 AutoEncoder／Light，HolisticEncoder／FacePosterior，AutoEncoder_intersection／AutoEncoder_joint 的装配、权重迁移、损失与导出 |
| `src/brepnet/models/blocks.py` | 原网络 block，IntersectionDecoder／IntersectionRefiner／LocalPairIntersection、几何投影与状态结构、JointBoundaryDecoder；新增 FacePairConditioning 仅用于可选的最终边几何头调制 |
| `src/brepnet/run_management.py`、`prediction_export.py` | 独立 run 目录、配置／Git 溯源、逐对象原始预测导出 |
| `src/brepnet/diffusion/model.py`、`sampling.py`、`losses.py` | diffusion 装配、训练损失、DDPM 采样与 VAE 接口 |
| `src/brepnet/diffusion/latents.py`、`padding.py` | cached mean/std 或 VAE 编码，face token 打包、筛选与去重 |
| `src/brepnet/diffusion/conditioning.py`、`image_encoder.py` | 条件工厂与图像编码；其他条件编码分文件保留 |
| `src/brepnet/diffusion/denoiser.py`、`fusion.py`、`topology.py` | 24 层去噪主干、每层后的独立 image cross-attention、可选 oracle / learned topology bias |
| `src/brepnet/data/datagen/`、`configs/datagen/` | Blender 渲染、FLUX 编辑、打包；存在入口契约漂移，见局部指南 |
| `src/brepnet/data/rotations.py`、`extract_visibility.py` | cube24 旋转与可见性缓存 |
| `src/brepnet/eval/run.py`、`eval/metrics/` | 统一评估调度与指标 |
| `src/brepnet/post/` | 曲面／曲线拟合、裁剪、缝合、STEP 构造与检查 |

当前单图路径：图片 → 冻结 DINOv2 的 CLS / patch tokens → 可训练图像投影；带时间嵌入的 noisy face latent → 24×（完整 TransformerEncoderLayer → 独立 image cross-attention residual）→ DDPM → 冻结 VAE → OCC 后处理。默认每面 32 维、最多 30 个 token、random duplicate padding。前端 4 层 fuser 与 alignment 已移除；这是当前实现，不是未来架构要求，也不代表已训练或已提升重建质量。

Diffusion 新代码从 `src/brepnet/diffusion/` 导入；保留的旧路径导出不保证旧架构配置兼容。旧模块化 baseline 位于归档 tag `archive/2026-10-16/codex/diffusion-modular-baseline`（`b67c5d8e`）；逐层条件实现已合入 `main`，显式拒绝 `condition_fuser` 配置，旧 diffusion checkpoint 不可直接 resume 或宽松加载。当前结构、加载协议及历史验证边界见 [说明](checkpoints/diffusion/README.md)，分支归档与恢复见 [Git 整理记录](checkpoints/repository_maintenance/20261016/README.md)。本主题讨论神经网络架构时优先可视化每层、数据流、张量形状及输入输出，不使用 Mermaid 画神经网络。

细节见 [源码指南](src/brepnet/AGENTS.md)、[数据指南](src/brepnet/data/AGENTS.md)、[评估指南](src/brepnet/eval/AGENTS.md)。

2026-10-13 按原文件组织归并 VAE 实验代码：移除 `models/` 下原 `intersection_*` 六个文件、`joint_*` 三个文件及 `holistic_encoder.py`、`local_intersection.py`，实现集中到 `vae.py`／`blocks.py`。已有模型名称、模块属性层级和旧 checkpoint state key 保留，trainer 继续共用 config 装配。历史 PLAN／NOTE 的旧路径保留为当时记录，当前入口以本地图及主题 README 为准。

可选最终 curve 条件调制见 [配置与迁移说明](checkpoints/intersection_refinement/curve_conditioning/README.md)：`model.curve_conditioning.enabled` 默认 `false`、`hidden_dim=128`；启用后仅在 refined edge CNN 与 center/scale MLP 的最终输出层前调制。`init_refinement_checkpoint` 对 local 分支只替换旧 inter，对此条件分支保留所有旧参数、仅初始化新增 conditioning；两者均为新 run 初始化，完整模型加载／resume 仍严格匹配。

2026-10-14 新增可选 [最终曲线逐点反馈](checkpoints/intersection_refinement/point_feedback/README.md)，仍写在 `blocks.py`／`vae.py`。`model.curve_point_feedback` 默认关闭；启用时在 final 曲线点读取 final 预测 owner 曲面，不池化，以零初始化 XYZ 增量同时更新 local/global 几何。新 run 使用 `init_curve_feedback_checkpoint` 严格保留已训练 conditioned refinement；原 inter、loss 和 encoder 合同不变。通用 `trainer.max_time` 可设运行时间预算，resume 会恢复 Timer 累计时间。

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

训练使用 `python -m src.brepnet.train --config-name <config>`。当前先用灰模的逐层条件训练入口为 `configs/train_diffusion_layerwise_svr.yaml`，训练/验证 photo ratio 均为 0，仅读取 `imgs.npz["svr_imgs"]`，不要求或读取 `real_photo.npz`；保留原 `is_aug=false` 视角 0、VAE/cache 和 split。`train_diffusion_layerwise_real.yaml` 保留为生成图条件入口；正式运行显式设置数据/split、设备数、每卡 batch、seed 与 AE checkpoint。配置名含 `baseline` 不保证仍是旧架构，旧模型要回 baseline 分支复现。不要把 `train_diffusion_real.yaml` 名字当作协议保证：它继承的 overfit / validation 条件有陷阱，详见源码指南。VAE loader 已接受规范字段及显式旧别名，冲突值必须报错；仍需核对实际数据。新联合边界 decoder 的命令与迁移协议见 [README](checkpoints/joint_boundary/README.md)。

保持既有模块职责；有理由改变边界时说明迁移。用 Python 3.10+、四空格、snake_case、显式配置与适当类型注解。不要把研究行为藏在路径猜测、checkpoint 名称或静默 fallback 中。

验证与改动相称：文档检查路径和差异；代码检查相关导入／单元行为；模型或数据接口检查真实小批量形状、loss、梯度与推理语义。实现可靠后才启动正式研究运行。不要把整个 `src/brepnet/test/` 当作无外部依赖的统一套件，也不要用 smoke test 宣称研究结论。

## 记录与评估纪律

**从 joint-boundary 实现开始，新实验、计划及结果记录统一放在 `checkpoints/`，不再写入 `experiments/`。** 后者保留为历史资产，不移动或删除。每个研究主题维护受 Git 管理的 `PLAN.md`、`README.md`／`NOTE.md`，记录“为什么、具体实施、结果／状态、参考链接”；事实、计划和已完成结果分开。

统一训练默认目录为 `checkpoints/<trainer.exp_name>/<UTC 时间戳>/`，权重 `weights/`、日志 `logs/`、W&B `wandb/`、逐对象预测 `raw_predictions/`、可视化 `visualizations/`、配置与 Git 元数据 `metadata/` 分开。`trainer.output_dir` 是根目录；`trainer.run_dir` 显式指定完整 run。续训使用 `trainer.resume_from_checkpoint` 回到原 run；不要将 init 当作 resume。每次调用保存 resolved config、Git commit／dirty diff、seed 和命令，已有 run 不允许无续训标记覆盖。

运行权重、原始几何及临时日志由 `.gitignore` 排除；研究文档与紧凑的验证报告保留版本管理。记录数据／split、checkpoint、seed、预算及输出路径。先建立可判断方向的研究问题，不机械延续旧实验编号。

比较固定输入信息、样本列表、采样／重试预算、checkpoint、后处理版本及指标协议。分别报告生成覆盖率、有效 STEP／solid 比例、几何与可见区域一致性，以及成本／耗时。失败、缺失、超时留在预定分母里；仅成功子集的几何分数单列。

2026-10-16 用户进一步明确后处理研究的数据分工：**规则设计、失败原因分析与方案调整使用 validation；test用于方案冻结后的数据评估，不依据test个例调规则、容差或挑版本。** 当前DeepCAD仍是开发与探索基准，允许按这一分工继续研究。固定比较清单、checkpoint、输入信息、修复预算与验收口径，完整报告救回、退步和几何保真。历史DeepCAD test曾参与开发，需如实记录，不能追溯声称完全未见。用于实际推理的修复与候选选择不能读取 GT。

**后续 ABC 的最终评估子集采用更严格的隔离。** 在查看其结果或失败样本前冻结方案、代码／配置、checkpoint 和评估协议，不根据该子集调规则、阈值或挑版本。具体子集与数据使用方式在开展 ABC 工作时明确，并核对其与开发数据的 CAD／几何重叠；不因数据集名称不同就认定完全独立。保留原始 split 和历史结果，不通过改名或删样本制造独立性。

GT best-of-K 是 oracle 指标，不能当作可用的单次输出质量；同时说明实际选择策略。原始解码质量与修复后的质量分开。比较 agent 与神经模型时，记录各自可用的工具和修复预算。

不要提交数据集、checkpoint、生成模型、W&B token 或机器凭证。PythonOCC 7.8.1 API 具有版本敏感性，改签名必须验证调用链。保留历史产物，简短提交说明目标；实验 PR 附数据假设、协议、结果与笔记链接。
