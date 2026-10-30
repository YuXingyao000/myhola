# Claude Code Guidelines

`AGENTS.md` 是本仓库的权威指南，本文件只是 Claude Code 的入口。改动某个文件前，先读覆盖该文件的 `AGENTS.md`；两者冲突时以 `AGENTS.md` 为准。

## 本 worktree

分支 `codex/room-generation`：数据集和图像生成（`packages/cad_scene_factory`，Codex 场景工厂工作流），以及无条件 diffusion 的对照实验。记录在 `checkpoints/image_generation/` 下。生成的图像都是合成的真实感图像，不能称为实拍数据。

这是 `/mnt/d/python` 的 worktree，不会自动同步到 5090；只有主 checkout 会同步。其他 worktree 的分工见主 checkout 的 `CLAUDE.md`。不要用裸 `git stash`，stash 栈在各 worktree 之间共享。

## 必读

- 全仓库：`AGENTS.md`（研究目标、数据契约、评估纪律）
- `src/brepnet/`、`src/brepnet/data/`、`src/brepnet/eval/`：各目录下的 `AGENTS.md`
- 研究记录：`checkpoints/<主题>/PLAN.md` 和 `README.md`/`NOTE.md`；架构快照写在 `checkpoints/<主题>/ARCHITECTURE.md`（ASCII 张量图，不用 Mermaid）
- `experiments/` 只是历史归档。`skills/brepnet-experiment-*` 已删除，不要依赖，也不要恢复。
- 外部论文的数字放进对比表之前，先核对它的口径：主 checkout 的 `checkpoints/baselines/README.md`

## 代码地图

- 入口：`python -m src.brepnet.train --config-name <config> [Hydra overrides]`；新实验用 Hydra override，不新增 shell wrapper。
- Diffusion：`src/brepnet/diffusion/` 包。`models/diffusion*.py` 和 `models/condition_encoders.py` 只是兼容的重新导出。
- VAE：本分支仍是单文件 `models/vae.py` + `models/blocks.py`（原 HoLa AE）。
- 数据生成：`packages/cad_scene_factory/`（有自己的 README 和 `prompts/`），以及 `src/brepnet/data/datagen/`、`configs/datagen/`。
- 后处理：`src/brepnet/post/`（`recover_v1` 等），评估：`src/brepnet/eval/`、`tools/vae_compare_*.py`。

## 评估口径（VAE / 端到端）

- 主合法标准：保存 STEP，以 `read.precision.mode=0` 重读，as-read 类型为 `TopAbs_SOLID` 且 `BRepCheck` 通过。容差 0.1 的兼容口径单独列出。success marker、文件存在、Compound 都不能替代主标准。
- 固定 DeepCAD validation 3495 件（原清单 3505 件），两种分母都报告；失败和超时留在分母里。用 validation 调规则，test 只在方案冻结后评估一次。
- raw 面/边 CD²、face-pair P/R/F1，以及 STEP 几何的"共同可测集合"分开报告。不能把 raw 几何的改善直接当作合法率的改善。

## 协作

- 讨论和笔记默认用中文，代码标识符用英文。区分源码事实、运行结果、文献结论、用户观察和研究假设。
- 模型代码直接呈现计算，不要逐层写 shape/dtype 校验或 `validate_*` 包装（见 `AGENTS.md` 的 2026-10-18 约定）。
- 保留未提交的工作（这个 worktree 里常有 Codex 会话的未提交改动）。不要 force push 或 reset，不要提交数据、权重和凭证。
