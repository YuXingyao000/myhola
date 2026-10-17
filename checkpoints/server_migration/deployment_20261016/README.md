# RTX 5090 服务器部署记录

本轮跨时区执行，当前日期为 2026-10-17（北京时间；日志保留 UTC）。目录名保留开始部署时的 20261016。目标为 `root@172.31.224.142:12222`，项目路径保持 `/mnt/d/python`。前轮清点目录 `20261017` 使用当时的北京时间，保留原日期以免改写记录。

## 本轮明确调整

- 用户已经更改登录密码，本轮 SSH 认证成功；agent 不修改密码。
- 不迁移源机 Anaconda / torch / generate 环境。目标为 RTX 5090，重新安装适配的 CUDA / PyTorch 依赖，PythonOCC 保持 7.8.1 API 系列（源包为 pythonocc-core 7.8.1.1，OCCT 7.8.1）。
- 现有未提交代码先检查、测试并提交，再同步相同 commit 与工作区；不同研究工作树继续保留各自分支，不进行无关合并。
- ABC / FLUX 的可修入口问题直接修复；不隐式启动大规模 ABC 数据构建，不改 split，不使用最终评估数据调方案。
- 当前 diffusion 的训练状态由研究会话管理，整个 `checkpoints/diffusion_layerwise_svr` 运行目录暂缓迁移，部署交付时提醒后续补同步；不是停止训练或声称已迁最新权重。
- 配置 Anaconda、Node/npm、Codex 和项目范围的上下文 / 技能。

## 目标机实测

| 项目 | 登录后观测 |
|---|---|
| 系统 | Ubuntu 22.04.5 LTS，glibc 2.35 |
| GPU | 8 × NVIDIA GeForce RTX 5090，每卡 32,607 MiB |
| 驱动 | 610.43.02 |
| 内存 | 约503 GiB |
| `/mnt/d` | ext4，总15,873,100,804,096 B，可用15,073,065,787,392 B |
| inode | 可用488,292,335 |

## 执行状态

- 数据生成工作树已保存到 `codex/data-generation-migration`，提交 `2e4ed187da45ecede44b53bad2b686c475b337b7`；相关6项 Blender 集成测试和29项纯Python测试通过。
- 新环境基础安装与 8 卡 GPU / OCC / NCCL 验证已完成；35 组所选资产传输及剩余项目级验收现已完成。详细脚本 / 验证分别放在 `environment/` 与 `transfer/`；日志位于源机 `/mnt/d/migration-transfer-logs/` 和目标 `/mnt/d/migration/logs/`，不提交日志或权重。
- Node 采用官方 Node.js 24 LTS 二进制，固定 v24.22.0，下载核对官方 SHA-256；Codex 固定与源机一致的 0.160.1。脚本见 [setup_codex.sh](setup_codex.sh)。
- 项目 `.codex/config.toml` 保留当前模型 / 推理默认值，并允许完整读取嵌套指南；自定义数据研究技能放在 `.agents/skills`，随仓库迁移。

## 2026-10-17 最终验收结果

一次性完成 hook 已实际收尾，无需续传或安排第二次等待 hook。

- **资产**：队列于 07:22:27 UTC 完成，35/35 组均 `exit_code=0`，逐组最终 rsync 统计完整且末尾无错误，目标 262 个显式根路径全部存在。实际传输内容 1.619 TB，rsync 发送 819.217 GB（十进制；不是磁盘占用）。未重复已通过的 124 个链接和 104 份 training 数据哈希检查，也未宣称全库独立哈希通过。见 [最终资产验收](transfer/ACCEPTANCE.md)。
- **项目运行**：`training` 16.99 秒、`flux` 53.74 秒，均成功退出。完整 diffusion/AE 前后向、可信 AE 加载、缓存解码、OCC/post 导入以及 FLUX Q8 两步推理和 256px 图片导出均通过；只使用 training 数据、没有 optimizer 更新或正式训练。见 [最终项目报告](project_validation/README.md)。
- **代码**：写入最终记录前，两端主目录均为 `b540fd64`、`codex/diffusion-training-throughput`；另三个历史工作树 HEAD/分支亦一致，八个工作区均干净，八次 `git fsck --connectivity-only` 均为 0。首轮 tree/gitlink 问题已消除；未重跑旧 `sync_code.py` 或复制 `.git`。本次只提交迁移记录和紧凑报告，使用项目专用 sender 最终增量同步。提交后核对结果保存到源机 `/mnt/d/migration-transfer-logs/code/post-commit-verification.json`，同步状态以 `.git/hola-sync/status.json` 的 commit 与当前 HEAD 一致为准。见 [代码验收](code_validation/README.md)。
- **环境与 Codex**：沿用已经验收的新 Anaconda、torch/generate、PyTorch 2.8/cu128、PythonOCC 7.8.1.1/OCCT 7.8.1、8 张 5090/NCCL、Node/Codex 及真实账号请求结果。本次 SSH 成功，没有读取认证文件、其他项目聊天或整包复制 Codex home；无需重新安装。

**保留的边界与缺口**：

1. `checkpoints/diffusion_layerwise_svr/` 及其权重仍按用户要求暂缓迁移；源机后续训练由研究会话控制，本次未启动、停止或改预算，不能声称目标已有最新 diffusion 权重。
2. ABC 仅迁入所选原始 STEP/图像资产与显式提取入口，尚无完整 HoLa `data.npz` 几何、非空 ABC latent cache 及对应训练包；未查看最终评估内容，未改变 split。
3. 源端 visibility 已有 3 个 CAD 缺口保留。数据生成的打包、旋转映射、mask、逐图来源记录与配置转发等既有缺口未在本轮补齐，详见 [入口边界](data_entrypoints/README.md)。FLUX 小规模运行通过不能替代整条生产管线验收。
4. 目标 torchvision 的 JPEG SONAME 补丁为环境必要修复；重装同版本 torchvision 后需按 [环境记录](environment/README.md) 重施。Codex 网络仍依赖已配置的局域网代理。

以下交接保留为完成前的历史，不代表仍待传输或验收。

## 2026-10-17 交接：本地完成 hook（历史）

用户要求长传输期间不持续调用模型监督。当前已改为本地一次性 hook，每60秒只读状态文件；等待期间不调用模型。35组传完、队列异常退出或同一资产重复失败时，启动一次**独立的项目 Codex 后台验收**，使用下方交接文件接管。显式暂停队列不会触发自动恢复。此能力不保证原桌面聊天自动弹出消息。

- hook：`transfer/completion_hook.py`；任务说明：`transfer/acceptance_prompt.md`。
- 运行状态：源机 `/mnt/d/migration-transfer-logs/acceptance-hook/status.json`；最后答复将保存为同目录 `result.md`，详细日志在该目录内，不纳入Git。
- 等待中的hook用进程锁避免重复，并在调用Codex前记录一次性标记；进程重启不会重复启动已经触发的验收。
- 7项合成状态检查通过，确认传输中不触发、完成/重复失败/异常退出会触发、显式暂停不触发；检查没有调用模型。
- 全部资产尚未完成，最近一次人工验收是30/35组完成、约398.53GB已完成组内容，ABC STEP继续传输；实时数据以外部状态JSON为准。详细检查见 [资产验收](transfer/ACCEPTANCE.md)。

**已完成运行验证**：8卡基础计算/NCCL、原AE权重、真实DeepCAD小批次diffusion/AE梯度和解码、scene工具入口、Blender两版本及8卡OptiX。OCC与torchvision的JPEG动态库冲突已在目标两环境最小修复并通过多种导入顺序/IGES/STEP验证。FLUX完整Q8权重加载与两步计算已通过，验收图片导出脚本的BF16转换已修。

**当时留给完成hook（现已完成）**：重跑约34秒的项目training验证和约196秒的FLUX验证，取得整体成功报告；完成代码增量同步及Git/worktree最终核对，更新状态后提交与同步迁移记录。首轮代码复制恰逢另一个用户会话提交，曾短暂缺tree；代码同步脚本也已补上空gitlink目录保留，不能跳过最终fsck。具体结果见 [项目验证交接](project_validation/README.md) 和 [代码验收](code_validation/README.md)。

Codex真实请求与项目指南加载已通过，目标也已配置源机使用的局域网代理；见 [Codex配置](codex/README.md)。只迁项目指南、技能和必要登录设置，没有整包复制其他项目会话。

**diffusion提醒**：人工检查时源机旧运行已到100000 step并停止；另一个用户研究会话随后提交 `7bb75c7c`，将总预算改为10000000。本迁移不干预该训练，仍按用户要求暂缓其运行目录与权重，后续单独补同步。

参考：[Codex CLI](https://learn.chatgpt.com/docs/codex/cli)、[无界面服务器认证](https://learn.chatgpt.com/docs/auth)、[Node.js 官方发行目录](https://nodejs.org/dist/v24.22.0/)。

## 命令行与字体补充

按用户后续要求，目标 root 已启用 Zsh、Oh My Zsh、命令建议/语法高亮和 Powerlevel10k；保留 Conda，实际 SSH 登录通过。容器已安装标题/UI用 Jost、正文/内容用 Archivo、中文用思源黑体。具体终端审美待用户指定，详见 [配置记录](shell/README.md)。此补充不改变数据传输完成hook。
