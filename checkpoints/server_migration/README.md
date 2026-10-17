# 第二台服务器迁移清单

清点日期：2026-10-17（北京时间；原始测量时间保留 UTC）。目标为 `root@172.31.224.142:12222`，代码仍放在 `/mnt/d/python`。用户选择“必要资产优先，历史大文件另列”。

**当前状态（2026-10-17 北京时间）：已进入部署和传输。** 用户更新密码后 SSH 已成功，目标为 8×RTX 5090。新环境已安装并通过基础 GPU / OCC / NCCL 检查，Node/npm 和 Codex 已配置；必要数据仍在分组传输。最新执行与验收见 [部署记录](deployment_20261016/README.md)。下方容量表保留首轮清点口径：按用户后续要求，取消旧环境约 23.98 GB 的复制，并暂缓活跃 diffusion 目录约 8.8 GB，实际首批资产规划约 **1.66 TB**，另加新安装环境占用。尚未宣称全部完成。

## 容量与范围

下表使用十进制 GB / TB。数据与大文件容量由实际目录或文件元数据测量；尚未对全部数据内容计算校验和。源机训练和其他工作树仍可能更新，因此这是清点快照，正式传输前需固定文件清单。

| 首批内容 | 容量 GB | 选择与说明 |
|---|---:|---|
| 项目代码、Git 历史、thirdparty 和相关工作树 | 6.52 | 保留原路径和未提交工作；扣除与下方近期实验目录重叠的跟踪文件 |
| DeepCAD v7 几何 | 4.26 | /mnt/d/data/deepcad_v7 |
| DeepCAD v7 完整条件包 | 85.02 | /mnt/d/data/deepcad_v7_cond，含灰模、合成照片及现有其他条件 |
| 24-view latent features | 11.55 | **抽样估算**；文件 payload 约4.24 GB，表中包含目录和块分配开销 |
| 可见性预处理 | 62.24 | **抽样估算**；约62.24 GB，现存3个CAD缺口见下文 |
| DeepCAD v6 源几何、STL、场景素材及其他小型预处理 | 63.93 | 其中v6约55.44 GB、STL约2.19 GB；其余目录见data.json |
| ABC 原始 STEP | 972.76 | /mnt/d/data/data_step，100万个对象目录 |
| ABC 已准备图像和解包条件 | 257.87 | 保留new_imgs与三个解包目录；不把归档重复计入首批 |
| 近期正式权重、实验记录与后处理缓存 | 170.52 | 两处checkpoints主体，扣除29份临时验证权重 |
| DeepCAD 与 ABC 原始 AE 权重 | 4.77 | /mnt/d/data/ae_checkpoints |
| Anaconda base、torch 与 generate 环境 | 23.98 | base不含envs/pkgs/conda-bld；保留同一prefix |
| DINOv2、两版 Blender 与报告字体 | 3.83 | 模型缓存不带登录文件 |
| 当前 FLUX Q8 所需模型组件 | 22.66 | 按源码精选；目标侧加载验证后才能确认生成可用 |
| 旧生成来源日志及当前重放仍需的baseline输出 | 0.42 | 避免遗漏历史目录中的现用依赖 |
| **首批规划合计** | **约 1690 GB，即 1.69 TB / 1.54 TiB** | 已区分测量与估算；不是完成传输的字节数 |

建议目标 `/mnt/d` 至少有 **2 TB 可用空间**，实验增长另留余量，并检查空闲 inode；当前规划涉及数百万文件和目录，预留约一千万空闲 inode 较稳妥。热缓存另需约4 GB内存盘空间。

已测组件合计约 1.617 TB，另外约 73.79 GB 为两个大缓存的抽样估算。固定seed为20261017：latent抽2048个视角目录，visibility抽128个CAD完整子树。visibility总量的近似95%抽样均值误差约±4.48 GB，不覆盖缺失文件等系统性问题。没有继续运行耗时数小时的全量冷目录扫描。

容量计算见 [summary.json](20261017/summary.json)，抽样依据见 [sampling.json](20261017/sampling.json)。

详细测量分别见 [数据](20261017/data.json)、[权重与实验缓存](20261017/weights.json)、[环境包清单](20261017/environment.json)、[生成模型与旧产物](20261017/generation.json)、[代码统计](20261017/code.json)、[上下文](20261017/context.json)。各明细包含父子目录及可选项，不能把所有行直接相加。

近期实验资产采用两处 `checkpoints` 主体，排除清单列出的 29 份临时验证权重：全量约 200.951 GB，临时权重约 30.427 GB，保留约 **170.524 GB**。其中约 **102.506 GB** 为原始预测、后处理结果、evaluation / comparison 和失败追踪，已经包含在上述容量内。这些资产便于重复后处理研究，不能因 `.gitignore` 排除就漏迁。`validation` 内的有效 split、合同审计、脚本和报告均保留。

## 历史大文件另列

以下先留在源机，后续复现相应历史实验或重新打包图像时再追加。归档与解包目录尚未全量内容比对，不能据此认定可删除。表中整目录含少量已经选中的来源日志或重放输入，后续追加时需扣除重叠，不能直接将本表相加作为额外精确容量。

| 内容 | 源路径 | 容量 GB |
|---|---|---:|
| ABC 原始归档 | `/mnt/d/data/data_step.tar` | 967.87 |
| ABC 条件压缩包，可能包含尚未解包内容 | `/mnt/d/data/abc_v2_cond/*.tar.gz`，20份 | 323.38 |
| 旧 diffusion 大量训练权重 / 日志 | `/mnt/d/python/outputs_brepnet_diffusion` | 488.05 |
| 另一管线预处理资产 | `/mnt/d/data/processed_deepcad` | 468.20 |
| 两套旧生成目录中的原图 / 输出 | `src/brepnet/data/DataGenerationRefactored` 与 `DataGeneration` | 435.11 |
| 旧 v6 条件及其组合 / cube24 副本 | `/mnt/d/data/deepcad_v6_cond*`，此处仅三套主目录 | 427.11 |
| 24-view cache 中伴生解码结果 | `ae_cache/1119_deepcad_aug1_11k_24/*/data.npz` | 约130.95，抽样估算 |
| 旧 experiments 大文件 | `/mnt/d/python/experiments` | 124.25 |
| 另一 organized DeepCAD 快照 | `/mnt/d/data/DeepCAD` | 82.50 |
| 原 HoLa 等旧权重 | `/mnt/d/ckpt` | 37.11 |
| 旧 Euler64 latent / 解码缓存 | `ae_cache/1119_deepcad_aug1_11k`，3,471,808个目录 | 未全量统计 |

旧 FLUX 其他量化版本、额外 transformer 权重、其他生成副本等见 `generation.json`。首批不搬旧生成 PNG 的依据是当前训练使用打包条件库；重新做图像质量研究或重打包时仍可能需要源图，不能把“当前不搬”写成“没有价值”。

## 数据路径与现存缺口

- DeepCAD 使用 `/mnt/d/data/deepcad_v7`、`deepcad_v7_cond`、`deepcad_v7_visibility` 和 `ae_cache/1119_deepcad_aug1_11k_24`。split、CAD identity、face order、cube24 顺序及相机约定原样保留。
- `deepcad_v7_cond` 元数据清点确认 52,556 个对象，每个均有六个约定文件；`imgs.npz` 和 `real_photo.npz` 各 52,556 份。这只验证存在性，未以本次清点替代数组键、形状和内容验证。
- 24-view cache 首批选择 `features.npy` 及其元数据，保留原目录布局。当前 diffusion / audit / filter 读取该文件；未发现当前调用者使用同目录的 `data.npz`。后者作为可复用解码缓存另列，留在源机，不删除。
- 完整目录名核对确认 52,556 个 split CAD 均有 0..23 的 latent 目录。可见性库只有52,553个CAD，相对条件库缺 `00221518`、`00447887`、`00579276`；迁移保留该缺口记录，不静默删除样本或调整 split。目录覆盖不等于每个文件完整、每个rotation语义正确。
- 旧 baseline 的 `experiments/baseline/diffusion_baseline_real_v7_cube24/test_outputs` 仍是 archived latent 重放的来源，约 53 MB，虽在历史目录中仍纳入首批。不能把整份 `experiments` 一概排除而漏掉它。
- 当前 `/dev/shm/hola-svr-20261016` 是可重建热缓存，只覆盖 train / validation 的灰模条件和 rotation0 latent。payload 为 3,845,664,374 B，另有 manifest；重建方法见 [现有暂存脚本说明](../diffusion/throughput/README.md)。目标机重启后需要重建，不将这份缓存当成完整照片 / 24-view / test 数据。
- ABC 首批包含 `/mnt/d/data/data_step` 的 100 万个原始对象目录、`abc_v2_cond` 的现有三份解包条件以及 `abc_v2_new_imgs`。**这些文件目前不构成可直接交给现有 HoLa loader 的完整 ABC 数据包**：`abc_v2_new_imgs` 是图像，`ae_cache/1225_abc_aug1` 当前为空，部分旧提取脚本引用已失效路径。迁移保留现存资产和缺口，不宣称已完成 ABC 预处理。
- ABC 仅做文件元数据清点，不查看最终评估结果或失败样本。后续最终评估继续遵循项目隔离规则。
- datagen 的既有入口契约漂移不因拷贝而消失。FLUX 精选文件依据当前 GGUF 加载路径确定，部署后还需本地加载验证；其输出仍称“合成真实感图像”。

## 代码与工作树

本次开始清点时，主仓在 `codex/diffusion-training-throughput`，HEAD 为 `b55d830b8064fe7f44b7b11f9060396ba8bc940e`，工作区干净。不是默认假定的 `main`。

| 原路径 | 清点时 Git 状态 | 迁移处理 |
|---|---|---|
| `/mnt/d/python` | `b55d830b` | 保留当前工作区、Git 历史、分支与 tag |
| `/mnt/d/python_curve_eval` | `d3dda308`，`codex/postprocess-validation` | 保留源码及独立的近期训练 / 评估资产 |
| `/mnt/d/python_legacy_post_main` | `73b63b73`，detached | 保留代码状态供既有对照重现 |
| `/root/.codex/worktrees/data-generation/python` | `c6ac62ad`，detached，有未提交工作 | 已提交为 `2e4ed187`，分支 `codex/data-generation-migration`；不复制整个 Codex home |

数据生成工作树的未提交路径为 `checkpoints/image_generation/codex_scene_factory/CODE_LAYOUT.md`、同目录 `BLENDER_BUILD.md`、`packages/cad_scene_factory/src/cad_scene_factory/blender/build.py` 和 `packages/cad_scene_factory/tests/blender/test_build.py`。只打包已提交代码会漏掉这些内容。正式迁移时重读状态，保留后续修改。

主仓有四个 Git gitlink，但 `.gitmodules` 缺失，`git submodule status` 目前报错。因此携带实际 `thirdparty` 工作目录，不能依赖目标机 `submodule update` 自动恢复。工作树 `.git` 和 `.git/worktrees` 使用绝对路径，按原位置恢复；如果将来改位置，需显式修复工作树引用。

## 环境与运行状态

实际训练环境为 `/mnt/d/anaconda3/envs/torch`：Python 3.10.13、PyTorch 2.4.1+cu124、PyTorch Lightning 2.5.0.post0、PythonOCC 7.8.1.1，实际导入 NumPy **2.0.1**；Transformers 5.5.0、Diffusers 0.37.1。清单保存 256 条 installed distribution 记录和 220 个 conda 包记录。环境残留两份 NumPy dist-info，包元数据查询曾返回 2.1.2；本表以实际 `numpy.__version__` 为准，重复元数据也原样记录，不将其冒充干净锁文件。

`environment.yml` / `pyproject.toml` 与此环境存在差异，直接重新解算依赖不能保证同样行为。`conda-pack` 的默认预检也因已有包覆盖而失败。原先考虑保留同一 prefix；**用户后续明确取消复制旧环境**，现已按 RTX 5090 新建环境，PythonOCC API 保持 7.8.1 系列；不修改源环境。若需要 editable install，先验证依赖并使用 `--no-deps`，避免隐式改变 NumPy 等包。

源系统为 Ubuntu 22.04 / glibc 2.35，8 张 RTX 4090，NVIDIA 驱动 570.133.07。目标机后续已核验为 8×RTX 5090、驱动 610.43.02，详细新环境见部署记录；不同 GPU / 软件版本的实验不能承诺逐位一致。

scene_factory 的现有命令还使用 `/mnt/d/anaconda3/bin/python`；目标安装新 Anaconda base 保持解释器路径，同时新建 `generate` 环境恢复 FLUX 能力。源环境的原始清点情况为：它有 gguf / sentencepiece，训练用 `torch` 环境缺少这两项；但 `generate` 目前缺 Hydra / OmegaConf，不能宣称现成总入口已经可运行。目标侧需明确解释器、补齐必要入口依赖并验证。Blender 4.2.0 与 5.2.2、DINOv2 模型文件、FLUX 所需组件及中文报告字体分别列入资产清单。报告字体路径为 `/root/.cache/hola-browser-libs/fonts/NotoSansCJKsc-Regular.otf`。

清点时已通过 checkpoint 元数据读取若干实际 step，见 `weights.json`。当前灰模 diffusion 仍在训练，观察到已保存 step 18,213；这不是最终迁移 step。用户后续要求该运行权重暂缓迁移，之后再提醒补同步；需要续训时应复制稳定快照并核验完整 optimizer / trainer 状态，不把初始化加载当作续训。

旧 LFD 特征生成工具的硬编码路径 `/root/repo/python/thirdparty/light-field-distance` 与 `load_data/run.so` 在源机已经缺失；统一评估对已有 LFD pickle 的汇总不等于能够重生成旧特征。该缺口单列，不作为已复现能力。

## Codex 新对话上下文

本机 Codex CLI 为 0.160.1。配置中的默认模型为 `gpt-6-astra`、推理等级 `ultra`、personality 为 `pragmatic`；全局 `AGENTS.md` 为空，`memories/` 为空，自动记忆表有 0 条生成记录。未发现项目级 `.codex/config.toml`。

因此本项目的新对话主要依赖仓库各级 `AGENTS.md`、`checkpoints` 研究记录、历史 `experiments` 文档及源码 / 实际资产。自定义技能仅需保留 `/root/.codex/skills/spec-research-data-code`（3 个文件，共 4,873 B），计划放在目标项目 `.agents/skills` 中限定作用范围；模型等项目默认值放在项目 `.codex/config.toml`。官方文档分别支持 [项目指令发现](https://learn.chatgpt.com/docs/agent-configuration/agents-md)、[项目配置](https://learn.chatgpt.com/docs/config-file/config-basic) 和 [仓库技能目录](https://learn.chatgpt.com/docs/build-skills)。

不需要整包复制 `~/.codex`：其中包含跨项目会话、活跃 SQLite 数据库、锁、登录信息和插件运行状态。当前请求是让新对话获得近似项目体验，未要求迁移所有旧聊天。已删除的旧实验技能也不恢复。目标侧仍需核对已连接插件和实际技能发现；登录或服务凭证按需要单独配置，不写入仓库。

应用支持通过 SSH 打开远端项目，迁移完成后将新主机的 `/mnt/d/python` 添加为项目即可进入新的项目对话；桌面端连接状态须独立配置，复制项目文件不会自动创建连接。[官方远端连接说明](https://learn.chatgpt.com/docs/remote-connections)

## 传输与验收

执行顺序和完成判据见 [PLAN](PLAN.md)。保留 `/mnt/d/data`、模型、环境和工作树的原绝对路径，尽量沿用现有命令；原始文件、split、旧结果不删除或重命名。局域网内采用可续传的 SSH / rsync，不经公网中转。具体耗时需在登录成功后测量链路和磁盘吞吐，不能仅凭“同一局域网”假定带宽。

迁移验收包含文件与软链接闭包检查、稳定权重校验、环境导入与 native 库验证、开发数据小批量形状 / 推理接口、OCC 后处理和生成工具可用性。浏览器 profile、锁和认证文件不属于实验资产；W&B 全局调试日志链接与科学结果链接分开处理。实现检查不产生新的研究结论。
