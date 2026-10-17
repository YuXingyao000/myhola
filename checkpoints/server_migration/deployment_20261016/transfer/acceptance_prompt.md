你是用户明确授权的服务器迁移完成 hook。用户要求等待期间只运行本地脚本，不消耗模型 token；现在脚本检测到传输队列完成、报错或意外退出，启动本次唯一的项目验收。请实际收尾，避免持续监督或无意义轮询。

先读本项目 AGENTS.md、checkpoints/server_migration/deployment_20261016/README.md 及其中 environment、project_validation、code_validation、codex、transfer 的验收/交接记录。最新数据状态在 /mnt/d/migration-transfer-logs/status.json，小摘要在 verification_summary.json。不要读取其他项目聊天或复制整个 Codex home。

目标是 root@172.31.224.142:12222，已认证的 SSH master 为 /tmp/hola-migration-ssh-%C：
`ssh -S /tmp/hola-migration-ssh-%C -o BatchMode=yes -p 12222 root@172.31.224.142 ...`
此任务仅继续已授权迁移、兼容修复和验收；不得改密码、删除源数据、重装源环境、停止/启动正式训练、发送外部消息、查看 ABC 最终评估内容或更改 split。当前 diffusion 运行及权重仍按用户要求暂缓迁移。源机曾达到100000 step，另一个用户研究会话随后将预算改到10000000；以当前实际状态为准，不干预。

执行：
1. 确认35组资产真实状态及退出码，不能用rsync增量列表99%当整库完成。仅补失败/遗漏项，不重传已完成的大库。先前124个科学链接与104份开发数据哈希已通过，报告里有边界，除新错误不重复全盘检查。
2. 检查代码同步结果 /mnt/d/migration-transfer-logs/code/ 与 source Git 最新状态。不同会话可能继续修改，保留未知工作，不代提交其半成品。只提交本迁移已经审阅的自身修复/文档/紧凑报告。现在已安装项目专用 Git 自动同步，见 code_sync/README.md；禁止再运行旧 sync_code.py 或复制整个 .git，以免覆盖两端机器专属配置、hook 或目标工作。主目录用 code_sync/sender.py enqueue /mnt/d/python 触发安全增量推送，读取 .git/hola-sync/status.json 确认；两侧分支不一致、工作区有修改或历史分叉时先报告并保留。其他3个历史工作树只核对现状，不自动更新。最终核对四个HEAD/分支/工作区与 git fsck --connectivity-only。
3. 新 Anaconda、torch/generate、PyTorch2.8cu128、PythonOCC7.8.1.1/OCCT7.8.1、8张5090/NCCL已验证。项目实际diffusion/AE前后向已通过；Torchvision自带JPEG与OCC的SONAME冲突由environment记录给出最小修复及备份。FLUX完整Q8加载与2步推理已通过，但验收脚本BF16转NumPy处曾失败，脚本已修正。按交接仅重跑未完成的项目验证，勿重新安装或无故重复全部基础检查。不做正式研究。
4. Codex0.160.1、Node24.22.0/npm11.19.1、账号真实请求、项目AGENTS上下文已通过；目标登录shell和Codex launcher使用源机已有局域网代理，配置在 /etc/profile.d/hola-network.sh。无需再读取或打印auth.json。项目技能在 .agents/skills，项目配置在 .codex/config.toml。
5. 把最终结果/剩余缺口写入 checkpoints/server_migration/deployment_20261016/README.md；提交本次完成的迁移记录并最终增量同步，核对两端。最终答复用中文，区分已完成与缺口，提醒暂缓的diffusion权重。ABC只有原始资产和显式提取入口，尚无完整HoLa训练包，不能称其已完整预处理。

若队列只是失败或意外退出，先做必要修复和断点续传，保存状态后重新安排本地一次性完成hook，立即结束本次模型调用，不能持续轮询几个小时。若SSH master失效、账号或依赖条件阻塞，记录具体原因和可执行恢复方法，不伪称成功，不无限重试模型。脚本会将你的最终答复保存到 /mnt/d/migration-transfer-logs/acceptance-hook/result.md；这是一项独立项目后台任务，不承诺原桌面聊天会自动弹消息。
