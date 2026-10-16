# 2026-10-16 Git 整理记录

本次按用户要求，将数据生成／逐层条件 diffusion 与 VAE／后处理两条开发线合入本地 `main`，保留最新文件结构，归档阶段分支。未修改远端；fetch 因 `ssh.github.com` DNS 解析失败而未完成，未执行 push。

## 历史与合并结果

本地旧 `main=73b63b73` 属于原上游历史，共 1551 个提交；当前开发来自 `27017eb5` 的清理后历史。两者没有共同祖先。已先把旧 main 保存为 `archive/2026-10-16/main`，再将本地主分支指向新历史的整合结果，没有强行合并无关历史或恢复已删除的旧目录。

整理时缓存的 `origin/main=27017eb5` 是整合结果的祖先。远端实时状态仍须联网后重新 fetch 确认；不要以本地缓存为依据 force push。

| 内容 | 提交／处理 |
|---|---|
| 数据生成、模块化 diffusion、逐层图像条件 | `9b234962` 及其祖先 |
| VAE 归并、curve conditioning、point feedback、encoder edge value、后处理及报告 | `d3dda308` 及其祖先 |
| 原工作区未提交的布景工具扁平化、工作流、提示词、研究记录和依赖 | 单独保存为 `2f2ed73e` |
| 两线合并 | `ad81cf20`，保留双方提交历史 |
| 唯一合并冲突 | `CODE_LAYOUT.md`：保留包含 10 月 14—15 日实现状态的版本 |
| 合并后的说明 | 更新当前 `main` 与旧 diffusion baseline 的归档入口；历史验证 JSON 的原分支字段不改写 |

源码事实：VAE 注册指向归并后的 `models/vae.py`，Diffusion 指向 `diffusion/model.py`；训练入口同时保留严格 diffusion 加载、VAE resume 初始化清理及 `max_time`。没有为合并追加模型结构改动。

## 分支与工作区

本地分支由 18 个减为 2 个：`main` 与 `/mnt/d/python_curve_eval` 正在使用的 `codex/postprocess-validation`。其余 16 个本地分支已删除分支引用；整理前的全部 18 个分支尖端均有 `archive/2026-10-16/` 下的 tag，逐项映射见 [branches.json](branches.json)。未被当前历史包含的旧实验分支只做归档，不混入 main。

常用恢复入口：

```bash
# 查看整理前的引用；归档 tag 均为本地引用，尚未发布到远端。
git tag --list 'archive/2026-10-16/*'

# 按需从归档创建新分支，现有 main 不变。
git branch codex/recover-old-main archive/2026-10-16/main
git branch codex/reproduce-diffusion-baseline archive/2026-10-16/codex/diffusion-modular-baseline
```

`archive/2026-10-16/scene-factory-workspace` 额外保存 `2f2ed73e`。已有 `hola_brep_cross` tag、远端跟踪引用、两个旧 stash 和四个 worktree 均保留。

- `/mnt/d/python`：切到整合后的 `main`；本次开始时的改动已全部纳入。收尾期间又出现灰模训练配置及配套文档改动，按下文保留为未提交内容。
- `/mnt/d/python_curve_eval`：原分支、提交及文件不变。
- `/mnt/d/python_legacy_post_main`：继续 detached 在 `73b63b73`，旧评估基线不变。
- `/root/.codex/worktrees/data-generation/python`：保留原 detached 提交及 1 个修改、3 个未跟踪文件。其中独有旧 `build.py`／`test_build.py` 依赖已移除接口；本次不回灌现行扁平化代码。

本地附加备份位于 `.git/git-cleanup/2026-10-16/`：整理前引用、工作树清单、原工作区 binary diff、48 个未跟踪源码／笔记的压缩包及 SHA-256 清单、旧 data-generation worktree 的 diff／未跟踪文件备份、测试日志。此目录不纳入版本管理，也不构成异地备份。

原工作区全部 60 个改动／新增／删除路径在合并后逐项核对，内容与整理前一致。约 3.92 GB 的逐件评估轨迹、生成页面、浏览器缓存等保持本地；合并评估线的已有 `.gitignore` 后，不再出现在待提交列表。未删除运行产物，未提交浏览器配置、权重或数据集。

收尾时新增的 `configs/train_diffusion_layerwise_svr.yaml`、`checkpoints/diffusion/SVR_CONFIG_VALIDATION.json` 及根／源码 AGENTS、diffusion README／PLAN 中的灰模入口说明，均不属于本次开始时的快照。为保留进行中的工作，本次只暂存同一文档内的归档引用变更，灰模部分保持原样、未提交；下列合并验证也不替代这项新配置的验证。

## 验证范围

紧凑结果见 [validation.json](validation.json)。

- 合并接口相关的 14 个现有测试文件：201 passed、1 skipped、0 failed，23.08 秒。覆盖 diffusion、VAE/local/joint、encoder 权重迁移／resume、训练配置、真实 OCC 拟合与后处理策略。
- 唯一 skip：未指定 `HOLABREP_DIFFUSION_CACHE_SAMPLE` 的可选真实 latent cache 测试。
- 布景工作流控制：24/24；调用接口检查：13/13，使用替身及 CLI help，不调用模型或 Blender。
- 相对两线共同祖先的 58 个新增／改动 Python 文件通过 AST 解析；归档引用与工作区内容核对通过。
- `git -c core.whitespace=cr-at-eol diff --cached --check` 通过，保留原有 CSV 的 CRLF 字节。

本次是 Git 整合验证，没有训练、全数据集重评估、图片生成或研究效果结论。
