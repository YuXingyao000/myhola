# 项目代码自动同步（2026-10-17）

用户在源服务器讨论、修改并提交代码后，希望自动同步到 RTX 5090 服务器继续实验。已安装源机专属 Git hook，以普通后台进程执行，不调用 Codex 模型、不消耗等待 token，不依赖本次 SSH master。

## 已启用范围与行为

- 单向：本源机 `/mnt/d/python` → `172.31.224.142:12222` 的 `/mnt/d/python`。
- 本仓库 `.git/hooks/post-commit` 自动触发；`post-merge`、`post-rewrite` 补充覆盖合并及改写后的检查。源机器 ID 和规范化工作区路径双重限定。没有设置全局 hook，也不影响其他仓库；共享 Git 的另外三个实验工作树会跳过。
- 两端须在同名分支；只推送当前 HEAD 对应的提交和新增 Git 对象。目标通过 `receive.denyCurrentBranch=updateInstead` 更新工作区；不 force push，不自动切换分支、创建分支或删除分支。
- 目标有未提交修改、未跟踪文件、进行中的 Git 操作、历史分叉或新跟踪路径与忽略数据冲突时拒绝更新并保留原内容。已同步的提交若 amend/rebase，通常会触发分叉保护，需要人工协调，日常用后续提交修正。
- 不同步未提交修改、忽略的数据集、权重和运行输出，不同步源机 Python/CUDA 环境。Git 已跟踪的所有内容仍属于提交范围，不能靠此机制过滤误提交的资产或凭证。
- 连续提交的后台任务串行执行，取得执行权后读取最新 HEAD；成功同步过的同一 HEAD 会合并为一次处理。每次 Git push 最长120秒，网络类错误最多3次，间隔20秒、60秒；Git 安全拒绝立即停止。最终失败留在状态文件，修复后手动触发或下次提交再次尝试。不是永久联网守护服务。
- 状态表示**上一次同步结果**，不持续监控目标后续修改。目标也要开发时，应先暂停自动同步并协调分支；不要用双向无条件覆盖。
- 自动同步会更新目标工作区；需要训练期间源码完全固定时，应使用独立、固定提交的运行工作树。

## 安装位置与认证

实现源码为 [sender.py](sender.py)、[receiver.py](receiver.py)。运行副本位置：

- 源端 `.git/hola-sync/sender.py`、`config.json`（包含启用开关、源机器ID、工作区、目标URL和密钥路径）。
- 目标 `/mnt/d/migration/code-sync/receiver.py`；接收 hook 为目标 `.git/hooks/pre-receive` 和 `push-to-checkout`。原接收配置保存为 `/mnt/d/migration/code-sync/config-before.json`。
- 源端专用密钥 `/root/.ssh/hola_project_sync`，只在源机保存私钥。目标 authorized_keys 用 `restrict` 与固定 forced command 限定该密钥只能执行本项目的 Git receive-pack；普通 shell 命令验证已被拒绝。密码、私钥不进入 Git。
- hook 和机器配置不随普通提交自动安装到其他机器。修改受审阅的实现后，需要明确更新对应运行副本；避免 Git 提交自动改写安全接收器。

旧 `sync_code.py` 在本配置存在时主动拒绝运行，避免覆盖机器专属 Git 设置和 hook。迁移完成验收提示已同步更新，不再整目录拷贝 `.git` 或强行同步历史工作树。

## 状态与操作

查看上次结果：

```sh
python3 /mnt/d/python/.git/hola-sync/sender.py status /mnt/d/python
```

成功状态是 `synced`，`commit` 必须与当前 `git rev-parse HEAD` 相同。`blocked` 代表需要处理分支或目标工作区；`failed` 代表有限重试后仍未连通/完成。详情在 `.git/hola-sync/status.json`，历史日志在 `.git/hola-sync/worker.log`；hook 失败不会撤销已完成的本地提交。

手动后台重试：

```sh
python3 /mnt/d/python/.git/hola-sync/sender.py enqueue /mnt/d/python
```

暂停时将 `.git/hola-sync/config.json` 的 `enabled` 改为 `false`；恢复改为 `true` 并运行上述重试命令。已开始的一次推送不会因随后暂停被撤销。不要删除工作区文件或强推来解除保护。

## 验证

`python3 checkpoints/server_migration/deployment_20261016/code_sync/test_sync.py` 使用一次性真实 Git 仓库，8项测试覆盖：正常快进与忽略数据保留、目标脏修改、未跟踪文件、忽略文件碰撞、分支不同、提交历史分叉、受限密钥命令入口，以及连续3次真实提交自动同步与关联工作树不触发。安装后以正式项目提交触发 hook，再用独立读取两端 HEAD、工作区状态及文件 SHA256 验证实际落地。

协议依据：[Git hooks / push-to-checkout](https://git-scm.com/docs/githooks#_push_to_checkout)、[Git receive.denyCurrentBranch](https://git-scm.com/docs/git-config#Documentation/git-config.txt-receivedenyCurrentBranch)。
