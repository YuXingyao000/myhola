# Img2Brep：Claude Code 指南

`AGENTS.md` 是本仓库的权威指南，本文件只是 Claude Code 的入口。改动某个文件前，先读覆盖该文件的 `AGENTS.md`；两者冲突时以 `AGENTS.md` 为准。

## 必读

- 全仓库：`AGENTS.md`（研究目标、仓库与外部目录、代码卫生、路径对照与 checkpoint 迁移、数据契约、评估纪律）
- `src/`：`src/AGENTS.md`；预处理 `src/preprocess/AGENTS.md`；评估 `src/eval/AGENTS.md`；diffusion 内部 `src/models/SKILL.md`
- 研究记录在仓库外：`/mnt/d/hola_research/<主题>/`（`PLAN.md`、`README.md`／`NOTE.md`、`ARCHITECTURE.md`，ASCII 张量图，不用 Mermaid）。外部论文数字进对比表前先读 `/mnt/d/hola_research/baselines/README.md`。

## 仓库形态（2026-10-30 整理后）

- 唯一长期分支 `main`；任务用短期分支 + PR。旧 worktree（`python_uv_train`、`python_curve_eval`、`python_room_generation`）的分支已合并或归档到 tag `archive/pre-cleanup-20261030/*`，不要在其上继续开发。
- 代码直接在 `src/` 下：`train.py`、`inference.py`、`evaluate.py`，以及 `common/ preprocess/ datagen/ datasets/ preference/ models/{vae,diffusion}/ postprocess/ eval/`。旧 `src/brepnet/...` 路径的对照表在 `AGENTS.md`。
- 运行目录默认 `${IMG2BREP_RUNS_ROOT:-/mnt/d/hola_runs}/<exp_name>/<UTC 时间戳>/`；整理前的 checkpoint 先用 `tools/migrate_checkpoint.py` 迁移再 resume。
- 当前状态：最好的 VAE 来源是 shared curve + UV best42471，validation 原生合法 3180/3495（90.99%）；近几轮（latent_expansion、shared_curve_uv、conditional_intersection）合法率都停在约 91%，失败多卡在裁剪面阶段（见 `/mnt/d/hola_research/uv_intersection/`、`conditional_intersection/`）。Diffusion 侧仍使用原版 HoLa AE 与 32 维 latent cache；换 VAE 需要重编码 cache 并重训 diffusion。

## 常用命令

```bash
python -m src.train --config-name <config> [Hydra overrides]
python -m src.evaluate --help
python -m pytest tests/test_<topic>.py          # 只跑相关测试
pre-commit run --all-files                       # 布局、死模块、import 分层、ruff
python tools/migrate_checkpoint.py OLD.ckpt NEW.ckpt --dry-run
```

## 评估口径（VAE / 端到端）

- 当前保留两种合法性口径：mode 0 重读后直接检查 as-read solid，以及历史 tolerance=0.1 后的 BRepCheck。baseline 的比较协议和最终主口径待讨论，分别报告两种口径及固定／已有目录分母，不用 success marker 替代合法性。
- 固定 DeepCAD validation 3495 件（`src/preprocess/splits/deepcad_v7_contract_validation.txt`；原清单 3505 件），两种分母都报告；失败和超时留在分母里。用 validation 调规则，test 只在方案冻结后评估一次。
- raw 面/边 CD²、face-pair P/R/F1，以及 STEP 几何的“共同可测集合”分开报告。不能把 raw 几何的改善直接当作合法率的改善。

## 协作

- 讨论和笔记默认用中文，代码标识符用英文。区分源码事实、运行结果、文献结论、用户观察和研究假设。
- 模型代码直接呈现计算，不要逐层写 shape/dtype 校验或 `validate_*` 包装（见 `AGENTS.md` 的 2026-10-18 约定）。
- 检查即规范：pre-commit 失败时修代码，不跳过 hook。一次性脚本与记录写到 `/mnt/d/hola_research`，不进代码仓库。
- 保留未提交的工作；不要 force push 或 reset；不要提交数据、权重和凭证。提交后核对 `.git/hola-sync/status.json` 再报告 5090 同步结果。
