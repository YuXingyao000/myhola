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

## 仓库、外部目录与工作流（2026-10-30 整理后）

本仓库只放代码、配置、测试和少量协议文件（split 清单、prompt）。其余内容在仓库外：

| 位置 | 内容 | 写入规则 |
|---|---|---|
| `/mnt/d/python` | 代码仓库（GitHub `YuXingyao000/myhola`），唯一长期分支 `main` | 只经 PR 合入 |
| `/mnt/d/hola_research/` | 研究记录（独立 git 仓库）：每个主题一个目录，`PLAN.md`、`README.md`／`NOTE.md`、`ARCHITECTURE.md`、一次性分析脚本；原 `checkpoints/` 记录与 `experiments/` 历史都在这里 | 新计划、审计、结果都写这里，不写进代码仓库 |
| `/mnt/d/hola_runs/` | 训练与评估运行目录（权重、日志、缓存），原 `checkpoints/<exp>/...` 同路径迁入 | `trainer.output_dir` 默认 `${HOLA_RUNS_ROOT:-/mnt/d/hola_runs}` |
| `/mnt/d/hola_archive/20261030/` | 整理时迁出的旧产物：`outputs_brepnet_diffusion/`、`claude_try/`、旧 datagen 输出、Depth Anything 等 | 只读 |
| `/mnt/d/data/` | 数据集与派生缓存 | 代码通过 Hydra override 或环境变量传路径 |

整理前的完整状态在各分支 tag `archive/pre-cleanup-20261030/*`，整理计划与过程见 `/mnt/d/hola_research/repo_cleanup_20261030/`。

**分支与审阅。** `main` 是唯一长期分支，5090 同步只跟 `main`。每个任务一个短期分支和 GitHub PR，1–3 天内 squash 合并；纯文件移动与逻辑修改分成不同 PR；PR 按 `.github/pull_request_template.md` 写清为什么、改了什么、如何验证、删了什么、新增开关及其删除条件、审阅指引。worktree 只给并行 agent 用：一个 agent 一个 worktree 一个 PR，合并后删除。训练从打了 tag 的提交启动（`run/<exp>/<date>`），不为跑实验保留长期分支。

**5090 同步。** 主工作区 `/mnt/d/python` 有提交后后台同步，Git 增量推送且只允许目标当前同名分支快进；详情见 `/mnt/d/hola_research/server_migration/deployment_20261016/code_sync/README.md`。提交完成不等于同步成功，报告前核对 `.git/hola-sync/status.json` 与 HEAD。目标有修改、分叉或分支不一致时保留并处理原因，不 force push/reset，不整目录覆盖 `.git`。数据、权重走独立迁移。

## 从哪里恢复上下文

- 研究记录：`/mnt/d/hola_research/<主题>/`。恢复状态不能只读笔记：核对 `/mnt/d/hola_runs/` 下 run 的 `metadata/`、实际输出及评估报告，区分“已实现、已训练、已评估”。
- 外部论文数字放进对比表前，先读 `/mnt/d/hola_research/baselines/README.md` 核对口径。
- 第二台 RTX 5090 服务器的环境与迁移状态：`/mnt/d/hola_research/server_migration/deployment_20261016/README.md`。目标使用 PyTorch 2.8/cu128、PythonOCC 7.8.1 API 系列；服务器上的 `checkpoints/` 尚未迁到 `hola_runs`。
- 原论文：[HoLa: B-Rep Generation using a Holistic Latent Representation](https://arxiv.org/abs/2504.14257)。论文的 surface-only holistic latent 与当前逐层 image cross-attention、可选 topology bias 分开描述。
- 六月实验历史在 `/mnt/d/hola_research/experiments/`（`ROADMAP.md`、`REFERENCES.md`、日期目录 `NOTE.md`）；其中“当前主线／下一步”截至六月，**不代表当前计划**。六月末旧汇总混有不同 checkpoint、后处理和评估口径，不作为固定的公平基线。

## 当前代码地图

```text
src/
  train.py  inference.py  evaluate.py   入口：Hydra + Lightning 训练；单图推理；评估调度
  common/       run 目录与溯源、逐对象预测导出、OCC 工具、io（sha256/json）、step_io
  preprocess/   DeepCAD／ABC 原始数据 → npz、可见性、cube24 旋转、shared boundary、UV 目标；splits/ 为固定清单
  datagen/      图片数据集生成：场景模板、room workflow、step1–4（Blender 5.2.2 + Codex 审阅）、retouch
  datasets/     训练读取：brep.py（VAE 与 diffusion 数据集）、photo、spatial_photo、preference、photo_preference
  preference/   DPO 偏好对生成管线（采样、评分、选择、观测重标注）
  models/       vae/（VAE 包）、diffusion/（diffusion 包）、blocks.py（共享算子与 VAE 兼容导出）
  postprocess/  曲面／曲线拟合、裁剪、缝合、STEP 构造与检查
  eval/         指标、单视图／偏好／照片评估、step_metrics（原生 STEP 合法性与采样 CD）
configs/        Hydra 配置；实验旋钮在这里暴露
tests/          单元与接口测试（pytest testpaths）
tools/          仍在用的命令行工具与仓库检查（check_repo.py、migrate_checkpoint.py 等）
thirdparty/     仅 Pointnet2_PyTorch：models/diffusion/legacy.py 构建原版 SVR 时旧源码顶层导入它
```

当前单图路径：图片 → 冻结 DINOv2 的 CLS / patch tokens → 可训练图像投影；带时间嵌入的 noisy face latent → 24×（完整 TransformerEncoderLayer → 独立 image cross-attention residual）→ DDPM → 冻结 VAE → OCC 后处理。条件只剩 `single_img` 与 `none`；点云、草图、文本、多视图、Depth Anything 已删除。这是当前实现，不是未来架构要求，也不代表已训练或已提升重建质量。照片 baseline／DPO／spatial 模型通过 `models/diffusion/legacy.py` 从历史提交 `c2660c00` 解出旧源码构建原版 SVR 再初始化。

VAE 与 diffusion 内部结构的进一步整理另行讨论；目录内说明见 `src/AGENTS.md`、`src/preprocess/AGENTS.md`、`src/eval/AGENTS.md`、`src/models/SKILL.md`。

## 代码卫生（机器检查，2026-10-30）

**检查即规范**：pre-commit（`.pre-commit-config.yaml`）与 CI 运行下列检查；失败时修代码，不跳过 hook、不加忽略来绕过。每个 clone 执行一次 `pre-commit install`。

| 检查 | 规则 |
|---|---|
| `tools/check_repo.py layout` | 顶层目录白名单；`npz/npy/ckpt/pth/png/jpg/zip/blend/ply/step/jsonl/csv` 等数据或二进制文件不进 git（仅 vendored LFD 评估脚本与 `docs/` 例外） |
| `tools/check_repo.py dead` | `src` 中每个模块必须能从入口、带 `__main__` 的 CLI、`tools/` 或 `tests/` 到达；没人用的模块直接删除，靠 git 历史找回 |
| import-linter（`pyproject.toml`） | `src` 不得导入 `tools/`、`tests/`；分层：入口 > eval／preference > postprocess > models > datasets > preprocess > common；`datagen` 与训练栈互不导入。`ignore_imports` 只列整理时遗留的已知债务，修一条删一条，不许新增 |
| ruff | 目前只开真实错误（语法、未定义名）；扩大规则或全仓格式化单独开 PR |
| check-added-large-files | 单文件不超过 1MB |

检查之外，agent 写代码时遵守（这些是此前代码里反复出现的问题）：

- **不留兼容层**：改名、移动模块时一次改完所有调用点，不新增 re-export 包装或别名模块；checkpoint 兼容只走 `tools/migrate_checkpoint.py` 的版本化映射。
- **一次性脚本不进包**：诊断、审计、对照、可视化脚本放在 `/mnt/d/hola_research/<主题>/`，与记录在一起；只有被反复使用、有测试的工具才进 `tools/`。
- **不新增 shell 包装与机器路径**：实验差异写成 config 或 Hydra override；不在代码里写 `/mnt/...`、代理地址等机器相关常量（默认路径用环境变量或 config）。
- **实验开关有删除条件**：新增开关的 PR 写明默认值与“什么证据出现时删除”；主题结束时一并清理，不让 `enabled: false` 的分支长期堆积。
- **数据不进仓库**：图片、权重、缓存、逐对象结果写到 `/mnt/d/data`、`/mnt/d/hola_runs` 或研究记录；仓库内只留固定 split 清单与 prompt 这类协议文件。
- **测试不写仓库目录**：临时文件用 `tmp_path`／`tempfile`；不依赖研究记录里的脚本或绝对数据路径。

## 路径对照与 checkpoint 迁移（2026-10-30）

旧路径只在历史记录与旧 run 的 `metadata/` 中出现；新代码一律用新路径。module 名为 `src.<目录>.<文件>`。

| 旧路径 | 新路径 |
|---|---|
| `src/brepnet/train.py`、`inference.py` | `src/train.py`、`src/inference.py` |
| `src/brepnet/eval/run.py` | `src/evaluate.py` |
| `src/brepnet/run_management.py`、`prediction_export.py` | `src/common/` 同名 |
| `shared/common_utils.py`、`shared/occ_utils.py` | `src/common/` 同名 |
| `src/brepnet/dataset.py` | `src/datasets/brep.py` |
| `src/brepnet/data/{photo,photo_preference,preference,spatial_photo}_dataset.py` | `src/datasets/{photo,photo_preference,preference,spatial_photo}.py` |
| `src/brepnet/data/preference_{generate,io,selection,source,observation_selection,observation_worker,relabel}.py` | `src/preference/{generate,io,...}.py` |
| `src/brepnet/data/{extract_*,edge_uv,rotations,shared_boundaries,model_lists,uv_curve_targets}.py` | `src/preprocess/` 同名 |
| `src/brepnet/data/list/` | `src/preprocess/splits/` |
| `src/brepnet/diffusion/` | `src/models/diffusion/` |
| `src/brepnet/models/{vae/,blocks.py,__init__.py}` | `src/models/` 同名 |
| `src/brepnet/post/` | `src/postprocess/` |
| `src/brepnet/eval/` | `src/eval/` |
| `src/brepnet/test/` | `tests/` |
| `packages/cad_scene_factory/` | `src/datagen/` |
| `tools/vae_compare_post.py` 中的 `sha256_file`、`fingerprint`、`atomic_json`、`read_json` | `src/common/io.py` |
| 同上 `read_step_at_precision` | `src/common/step_io.py` |
| 同上 `configure_cpu`、`preload_cpu_dependencies`、`terminate_process` | `src/postprocess/cpu_runtime.py` |
| 同上 `SAMPLING`、`native_validity`、`sample_native_geometry`、`squared_chamfer`、`validate_raw_prediction` 等 | `src/eval/step_metrics.py` |
| `tools/legacy_failure_geometry.py:native_diagnostics` | `src/eval/occ_diagnostics.py` |
| `checkpoints/<主题>/`（记录） | `/mnt/d/hola_research/<主题>/` |
| `checkpoints/<exp>/<run>/`（运行） | `/mnt/d/hola_runs/<exp>/<run>/` |
| VAE 合同审计 accepted 清单、diffusion validation100 | `src/preprocess/splits/deepcad_v7_contract_{train,validation,test}.txt`、`deepcad_v7_validation100.txt` |

**Checkpoint 迁移。** `python tools/migrate_checkpoint.py OLD.ckpt NEW.ckpt [--dry-run]` 按 `MIGRATIONS` 顺序应用并记入 `checkpoint["hola_migrations"]`，只写新文件。spatial／photo／preference 的 resume 会比较保存的 `dataset.provenance` 与路径，整理前的 checkpoint **必须先迁移才能 resume**（已在 `diffusion_spatial_photo/20261028T104221Z/weights/last.ckpt` 上验证：原文件 resume 检查失败，迁移后通过且严格加载）。

| 版本 | 内容 |
|---|---|
| `2026-10-30-layout` | `dataset.name` 等 module 路径、split 清单路径、`/mnt/d/python/checkpoints/` → `/mnt/d/hola_runs/`；state_dict 键不变 |

之后每次改 module／属性名：在 `MIGRATIONS` 追加一个版本（state_dict 前缀映射 + 字符串映射），在上表登记，并在 PR 的“Checkpoint／数据兼容”一栏说明。纯改名保持参数注册顺序，optimizer state 才能继续使用。其他已知限制：preference 生成、photo baseline 等记录了源码路径哈希的旧运行，在新路径下不能续跑，需新开 run；datagen step1／step4 记录脚本路径，旧位置启动的 run 同样不能在此续跑。

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

训练使用 `python -m src.train --config-name <config>`。当前先用灰模的逐层条件训练入口为 `configs/train_diffusion_layerwise_svr.yaml`，训练/验证 photo ratio 均为 0，仅读取 `imgs.npz["svr_imgs"]`，不要求或读取 `real_photo.npz`；保留原 `is_aug=false` 视角 0、VAE/cache 和 split。`train_diffusion_layerwise_real.yaml` 保留为生成图条件入口；正式运行显式设置数据/split、设备数、每卡 batch、seed 与 AE checkpoint。配置名含 `baseline` 不保证仍是旧架构，旧模型要回 baseline 分支复现。不要把 `train_diffusion_real.yaml` 名字当作协议保证：它继承的 overfit / validation 条件有陷阱，详见源码指南。VAE loader 已接受规范字段及显式旧别名，冲突值必须报错；仍需核对实际数据。新联合边界 decoder 的命令与迁移协议见 `/mnt/d/hola_research/joint_boundary/README.md`。

保持既有模块职责；有理由改变边界时说明迁移。用 Python 3.10+、四空格、snake_case、显式配置与适当类型注解。不要把研究行为藏在路径猜测、checkpoint 名称或静默 fallback 中。

### 实验代码与架构记录（2026-10-18 用户约定）

- 这是上下游强耦合的研究 pipeline。模型代码优先展示网络架构、向量变换和信息职责；`forward()` 直接呈现计算，不写成输入验证流程。
- 不在 encoder、decoder、网络 block、loss 中逐层重复检查张量 shape、rank、dtype、device、Python 类型、字段集合或有限值；不为这些检查新增 `validate_*`、包装层或“严格模式”。让索引、张量算子和 PyTorch 层自然处理接口错误。类型注解、简短形状注释可以保留。
- 不把一次实验采用的配置组合全部硬编码成禁止其他组合的 `raise`。组合、前提和版本写在实验架构记录里。保留真正影响实验含义的少量判断，例如互斥权重来源、严格权重加载、无法容纳全部 GT 的匹配容量；不要重复实现 PyTorch 已有的类型/形状检查。
- 实际计算中的 mask、变长 batch、空边/空候选分支、混合精度转换、数值稳定项、GT 有效点筛选不是防御性检查，按算法需要保留。数据生成/读取时的源文件和样本有效性检查、OCC 合法性判定、避免覆盖已有实验产物的规则也不属于此次删除范围。
- 每次讨论并确定一套完整实验架构，在 `/mnt/d/hola_research/<主题>/ARCHITECTURE.md` 记录：实现 Git hash、配置入口及关键开关、模块路径、训练/冻结范围、训练与推理输入输出、坐标/索引约定、loss 对应关系；用 Markdown fenced ASCII 图展开主要 PyTorch 层、层数、宽度、张量形状与数据流。不要使用 visualize 或 Mermaid 代替文档内可读的结构图。
- 架构记录是可独立阅读的设计快照，代码不解析它、不依赖它运行。新方案明确增加日期/版本记录或独立快照；历史运行仍以各自 `metadata/` 中的 resolved config、Git commit/dirty diff 和 checkpoint 为准，不用后来的文档追认旧运行架构。实际运行应引用对应快照；没有训练过的实现不要记为已训练或有效。

当前示例：`/mnt/d/hola_research/uv_boundary/ARCHITECTURE.md`。本约定优先于历史笔记中要求在模型内部反复校验“合同/契约”的描述。

验证与改动相称：文档检查路径和差异；代码检查相关导入／单元行为；模型或数据接口检查真实小批量形状、loss、梯度与推理语义。实现可靠后才启动正式研究运行。不要把整个 `tests/` 当作无外部依赖的统一套件，也不要用 smoke test 宣称研究结论。

## 记录与评估纪律

**新实验、计划及结果记录统一放在 `/mnt/d/hola_research/<主题>/`（独立 git 仓库），不写进代码仓库。** `/mnt/d/hola_research/experiments/` 是六月历史，不再追加。每个研究主题维护受 Git 管理的 `PLAN.md`、`README.md`／`NOTE.md`，记录“为什么、具体实施、结果／状态、参考链接”；事实、计划和已完成结果分开。

统一训练默认目录为 `${HOLA_RUNS_ROOT:-/mnt/d/hola_runs}/<trainer.exp_name>/<UTC 时间戳>/`，权重 `weights/`、日志 `logs/`、W&B `wandb/`、逐对象预测 `raw_predictions/`、可视化 `visualizations/`、配置与 Git 元数据 `metadata/` 分开。`trainer.output_dir` 是根目录；`trainer.run_dir` 显式指定完整 run。续训使用 `trainer.resume_from_checkpoint` 回到原 run；不要将 init 当作 resume。每次调用保存 resolved config、Git commit／dirty diff、seed 和命令，已有 run 不允许无续训标记覆盖。

运行权重、原始几何及临时日志在 `/mnt/d/hola_runs`；研究文档与紧凑的验证报告在 `/mnt/d/hola_research` 中做版本管理。记录数据／split、checkpoint、seed、预算及输出路径。先建立可判断方向的研究问题，不机械延续旧实验编号。

比较固定输入信息、样本列表、采样／重试预算、checkpoint、后处理版本及指标协议。分别报告生成覆盖率、有效 STEP／solid 比例、几何与可见区域一致性，以及成本／耗时。失败、缺失、超时留在预定分母里；仅成功子集的几何分数单列。

2026-10-16 用户进一步明确后处理研究的数据分工：**规则设计、失败原因分析与方案调整使用 validation；test用于方案冻结后的数据评估，不依据test个例调规则、容差或挑版本。** 当前DeepCAD仍是开发与探索基准，允许按这一分工继续研究。固定比较清单、checkpoint、输入信息、修复预算与验收口径，完整报告救回、退步和几何保真。历史DeepCAD test曾参与开发，需如实记录，不能追溯声称完全未见。用于实际推理的修复与候选选择不能读取 GT。

**后续 ABC 的最终评估子集采用更严格的隔离。** 在查看其结果或失败样本前冻结方案、代码／配置、checkpoint 和评估协议，不根据该子集调规则、阈值或挑版本。具体子集与数据使用方式在开展 ABC 工作时明确，并核对其与开发数据的 CAD／几何重叠；不因数据集名称不同就认定完全独立。保留原始 split 和历史结果，不通过改名或删样本制造独立性。

GT best-of-K 是 oracle 指标，不能当作可用的单次输出质量；同时说明实际选择策略。原始解码质量与修复后的质量分开。比较 agent 与神经模型时，记录各自可用的工具和修复预算。

不要提交数据集、checkpoint、生成模型、W&B token 或机器凭证。PythonOCC 7.8.1 API 具有版本敏感性，改签名必须验证调用链。保留历史产物，简短提交说明目标；实验 PR 附数据假设、协议、结果与笔记链接。
