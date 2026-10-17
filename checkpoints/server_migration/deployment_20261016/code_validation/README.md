# 代码与 Git 首轮复制验收

2026-10-17 首轮复制已结束，独立只读检查已执行。**首轮当时尚不能宣布 Git 工作区迁移完成**：发现并发提交造成的对象缺口，以及辅助工作树空 gitlink 目录遗漏。按用户最新要求停止持续监督，待数据传输终态的一次性 hook 接管最终验收。

## 已核验内容

- 四个工作树共 9,532 个 tracked 常规文件／软链接逐项核对内容 SHA256 或链接目标；9,528 项一致。四项差异来自复制期间的研究配置提交和部署验证脚本修改，见 JSON。
- 主工作树额外 thirdparty 的 1,218 个文件、1,580,739,905 字节全部一致；已初始化的四个 gitlink 均能解析 `.git`，HEAD 与索引一致且各自状态干净。
- 四个工作树的分支／detached 状态、`.git` 指向和公共仓库绝对路径一致；辅助三个工作树 HEAD 一致。
- 对照原 `20261017/code.json` 和 `worktree_code.json`：其余三个工作树的非空 thirdparty 内容均已 tracked，没有额外非 Git 源码遗漏。数据生成工作树原未提交文件已在 `2e4ed187` 提交并复制。

## 已发现问题与恢复要求

1. 复制过程中，另一研究会话将主分支从 `d231d509` 更新到 `7bb75c7c`（提高 SVR 总训练预算）。目标引用已更新，但缺少其 tree `26e0720f7d62e447b0ef84c5f02e9a192df03b31`；`git fsck --connectivity-only` 返回 2，主工作树 status 报 `bad tree object HEAD`。当时建议的整目录 `.git` 补拷已被后续安全 Git 增量推送方案取代，禁止再执行；不得回退该研究提交。
2. `sync_code.py` 排除了 gitlink 路径，导致源上原有的空目录没有复制。目标 curve_eval、legacy、data-generation 工作树分别显示 4、12、4 个 gitlink 被删除。应让同步脚本保留这些空目录，再检查状态；它们在源上原本未初始化，无需擅自下载依赖。
3. 首轮复制后又有迁移指南、部署脚本与验证记录变更。最终记录提交完成后需要再次增量同步。新机安装产生的 editable metadata／验证文件应按具体路径记录，不一概当作迁移错误。

最终验收应重新确认四个 HEAD／分支／tracked 文件状态，以及 `git fsck` 无对象缺失。既有 dangling 对象不是本轮清理对象；此次未修改 Git 状态、提交、删数据或修复目标文件。

紧凑证据见 [validation.json](validation.json)。原始逐文件快照暂存在源机 `/tmp/hola-code-validation-source.json` 与 `/tmp/hola-code-validation-target.json`，不是长期验收结果；再次接管时应重新取证。


## 最终只读验收：2026-10-17

[final_validation.json](final_validation.json) 为本次完成 hook 在写入最终记录前的双端快照。四个工作树的 HEAD、分支（含 detached 状态）一致，工作区全部干净；两端逐工作树 `git fsck --connectivity-only` 共 8 次均返回 0，无 missing/broken 输出。既有 dangling 对象保留（源端 604、目标 602 行），不影响连通性，未执行 prune。

| 工作树 | 两端 HEAD | 分支 |
|---|---|---|
| `/mnt/d/python` | `b540fd644c0971ee3dd0f6cc1bb879740b597d35` | `codex/diffusion-training-throughput` |
| `/mnt/d/python_curve_eval` | `d3dda308cd7dfb77646c1953eb7e1241d1f4c072` | `codex/postprocess-validation` |
| `/mnt/d/python_legacy_post_main` | `73b63b736af2d1002ffa01d5dfb397a9b76d3923` | detached |
| `/root/.codex/worktrees/data-generation/python` | `2e4ed187da45ecede44b53bad2b686c475b337b7` | `codex/data-generation-migration` |

本次还读取 `/mnt/d/migration-transfer-logs/code/` 的首轮及后续结果；04:21 的历史记录早于当前源提交，以实际双端快照为准。此前缺 tree 和 20 个空 gitlink 目录问题均已消除，不需要再次修复。未自动更新三个历史工作树，未复制 `.git`，未修改机器专属 hook/config。

最终迁移记录提交使用 [code_sync](../code_sync/README.md) 的 `sender.py enqueue /mnt/d/python` 安全增量同步；提交后的双端复核保存在 `/mnt/d/migration-transfer-logs/code/post-commit-verification.json`，避免将包含自身提交哈希的报告递归提交。该报告与 `.git/hola-sync/status.json` 必须共同确认当前 HEAD 落地，不能只凭提交成功宣称同步成功。
