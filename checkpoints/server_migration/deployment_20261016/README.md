# RTX 5090 服务器部署记录

本轮跨时区执行，当前日期为 2026-10-17（北京时间；日志保留 UTC）。目录名保留开始部署时的 20261016。目标为 `root@172.31.224.142:12222`，项目路径保持 `/mnt/d/python`。前轮清点目录 `20261017` 使用当时的北京时间，保留原日期以免改写记录。

## 本轮明确调整

- 用户已经更改登录密码，本轮 SSH 认证成功；agent 不修改密码。
- 不迁移源机 Anaconda / torch / generate 环境。目标为 RTX 5090，重新安装适配的 CUDA / PyTorch 依赖，PythonOCC 保持 7.8.1 API 系列（源包为 pythonocc-core 7.8.1.1，OCCT 7.8.1）。
- 现有未提交代码先检查、测试并提交，再同步相同 commit 与工作区；不同研究工作树继续保留各自分支，不进行无关合并。
- ABC / FLUX 的可修入口问题直接修复；不隐式启动大规模 ABC 数据构建，不改 split，不使用最终评估数据调方案。
- 当前 diffusion 继续在源机训练，整个活跃 `checkpoints/diffusion_layerwise_svr` 运行目录暂缓迁移，部署交付时提醒后续补同步；不是停止训练或声称已迁最新权重。
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
- 新环境基础安装与 8 卡 GPU / OCC / NCCL 验证已完成，必要资产传输及项目级验收进行中。详细脚本 / 验证分别放在 `environment/` 与 `transfer/`；日志位于源机 `/mnt/d/migration-transfer-logs/` 和目标 `/mnt/d/migration/logs/`，不提交日志或权重。
- Node 采用官方 Node.js 24 LTS 二进制，固定 v24.22.0，下载核对官方 SHA-256；Codex 固定与源机一致的 0.160.1。脚本见 [setup_codex.sh](setup_codex.sh)。
- 项目 `.codex/config.toml` 保留当前模型 / 推理默认值，并允许完整读取嵌套指南；自定义数据研究技能放在 `.agents/skills`，随仓库迁移。

尚未完成的步骤不能据此视为成功；后续会填入实际传输状态、目标环境验证、代码 commit 对应关系及剩余事项。

参考：[Codex CLI](https://learn.chatgpt.com/docs/codex/cli)、[无界面服务器认证](https://learn.chatgpt.com/docs/auth)、[Node.js 官方发行目录](https://nodejs.org/dist/v24.22.0/)。
