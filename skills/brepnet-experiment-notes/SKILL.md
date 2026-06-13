---
name: brepnet-experiment-notes
description: Manage HoLa-BRep experiment folders under experiments/ with one dated folder per day, a mandatory NOTE.md, and per-experiment records that explain why, implementation, and results.
---

# BRepNet Experiment Notes

This skill is installed into multiple tools (Cursor / Claude Code / Codex); the canonical source is `skills/brepnet-experiment-notes/SKILL.md` (relative to repo root `/mnt/d/python`) — edit there and re-sync the copies (see README "Experiment Workflow & Agent Skills"). To plan a new experiment day, see the `brepnet-experiment-planning` skill (`skills/brepnet-experiment-planning/SKILL.md`).

Use this skill when creating, updating, summarizing, or reorganizing anything under `experiments/`.

The current project habit is note-first: commands and scripts are supporting artifacts, while `NOTE.md` is the durable experiment log.

## Core Rules

1. Use one folder per day: `experiments/YYYY-MM-DD/`.
2. Every date folder must have `NOTE.md`.
3. Record work as separate experiment entries, not only as a loose daily checklist.
4. Each experiment entry must state:
   - **Why**: what hypothesis, bug, ablation, or question motivated it.
   - **Implementation**: what script/config/data/checkpoint/command changed or ran.
   - **Result**: observed metrics, logs, failure mode, partial result, or current status.
5. One entry can be an iteration of a previous entry. Name the relationship explicitly.
6. If results are not available yet, write the status and where to find logs/checkpoints.
7. When adding or changing scripts/commands, update `NOTE.md` in the same folder in the same turn.
8. Always link, never just name. Whenever an entry mentions a paper, a previous day's note, a config, a script, a checkpoint, or a concrete data file, write it as a clickable markdown link, not as bare prose. If a link target genuinely does not exist yet, say so explicitly (e.g. `（暂无链接，论文待补）`).

## Links & References (must follow)

The biggest recurring gap is that entries name a paper / prior note / data file but never link to it. Every entry must make its references clickable.

Link conventions:

- Prior day's note: link relative to the current file, e.g. `[2026-06-01 笔记](../2026-06-01/NOTE.md)`. To point at a specific experiment, link the heading anchor, e.g. `[2026-06-01 #2](../2026-06-01/NOTE.md#2-ar-topology-transformer-smoke-run)`.
- Repo file (config / script / source): link the workspace-relative path, e.g. `[train_diffusion_learned_topology.yaml](../../configs/train_diffusion_learned_topology.yaml)` or for a file in the same day folder `[topology_ar_train.py](./topology_ar_train.py)`.
- Data file / checkpoint / output: link the absolute path, e.g. `[real_photo.npz](/mnt/d/data/deepcad_v7_cond/00000000/real_photo.npz)`, `[best.pt](/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt)`. Keep the human-readable name as the link text and the full path as the target.
- Paper / external reference: link to the canonical URL (arXiv abstract page preferred), e.g. `[HoLa-BRep (arXiv:2504.xxxxx)](https://arxiv.org/abs/2504.xxxxx)`. If only a local PDF exists, link the absolute PDF path. If you do not know the exact URL, do not invent one — write the title plus `（链接待补）`.
- W&B / dashboard run: link the run URL directly so it is one click from the note.

Rules:

1. Inside `具体实施` and `结果`, the first time you reference a paper / note / config / script / checkpoint / data file, make it a link. Later mentions in the same entry can be plain.
2. Each entry should end with a `**参考 / 关联**` bullet list collecting its key links (related prior entries, papers, primary data, output dir). This makes the trail navigable even when skimming.
3. Do not fabricate paths or URLs. A wrong link is worse than an honest `（待补）`.

## NOTE.md Shape

Prefer this structure:

```markdown
# YYYY-MM-DD 实验笔记

## 今日目标

一句话说明今天主要想推进或验证什么。

## 实验记录

### 1. 实验名

**为什么这么做**

- ...

**具体实施**

- 脚本/命令：
  - [xxx.py](./xxx.py)
  - [command.sh](./command.sh)
- 关键配置：
  - data root / condition root / latent root（链接到具体路径）
  - checkpoint：[best.pt](/mnt/d/data/.../best.pt)
  - Hydra overrides
- 依据/参考：[相关论文（arXiv:...）](https://arxiv.org/abs/...)、[昨天的实验](../YYYY-MM-DD/NOTE.md#...)

**结果**

- 指标、现象、失败日志、输出路径，或“还在跑 / 待补评估”。输出目录写成链接：[outputs/](./outputs/)。

**参考 / 关联**

- 关联实验：[YYYY-MM-DD #n](../YYYY-MM-DD/NOTE.md#...)
- 论文：[标题（arXiv:...）](https://arxiv.org/abs/...)
- 主要数据：[文件名](/mnt/d/data/...)
- 配置/脚本：[config.yaml](../../configs/config.yaml)

### 2. 实验名（上一条的迭代）

**为什么这么做**

- 承接 [#1](#1-实验名)，...

**具体实施**

- ...

**结果**

- ...

**参考 / 关联**

- ...

## 今日结论

- 保留真正影响后续决策的结论。

## 待办

- [ ] 下一步动作。
```

Keep the notes factual. Prefer concrete paths, exact config names, checkpoint paths, and metric values over broad summaries.

## Command And Script Rules

Commands may live in `command.sh`, `commands.sh`, or a dated run script when useful. Keep them easy to copy and trace from `NOTE.md`.

For command files:

1. Start with `cd /mnt/d/python`.
2. Use short section headers for groups of related runs.
3. Put a short comment above each command explaining what it does.
4. Keep commands self-contained with explicit paths and Hydra overrides.
5. Avoid wrapper indirection unless the script itself is the experiment artifact.

For Python experiment scripts:

1. Put narrow one-off experiment code in the date folder.
2. If it becomes reusable infrastructure, move it to `tools/` or `src/brepnet/...` and record that move in `NOTE.md`.
3. Save outputs under the same dated folder or an explicitly named external output directory.

## Current Data Contract

For current DeepCAD v7 diffusion experiments, prefer these paths unless the user says otherwise:

| What | Path |
|------|------|
| Raw B-Rep data | `/mnt/d/data/deepcad_v7` |
| Condition data | `/mnt/d/data/deepcad_v7_cond` |
| 24-view latent cache | `/mnt/d/data/ae_cache/1119_deepcad_aug1_11k_24` |
| AE checkpoint | `/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt` |
| Training list | `src/brepnet/data/list/deduplicated_deepcad_training_7_30.txt` |
| Validation list | `src/brepnet/data/list/deduplicated_deepcad_validation_7_30.txt` |
| Test list | `src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt` |

Current condition files use:

```text
deepcad_v7_cond/{model_id}/imgs.npz
  - svr_imgs
  - sketch_imgs

deepcad_v7_cond/{model_id}/real_photo.npz
  - flux
```

Do not call this `svr.npz` unless the repository actually changes back to that file contract.

## Common Experiment Entry Examples

### Training idea

```markdown
### 1. Learned topology bias with real-photo condition

**为什么这么做**

- 验证 frozen topology predictor 产生的 soft adjacency bias 是否能缩小 oracle topology 和 baseline 之间的差距。

**具体实施**

- 使用 [train_diffusion_learned_topology.yaml](../../configs/train_diffusion_learned_topology.yaml)。
- checkpoint: [best.pt](/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt)。
- 数据使用 [real_photo.npz](/mnt/d/data/deepcad_v7_cond/00000000/real_photo.npz) 的 `flux` key，`real_photo_ratio=1.0`。
- 思路依据 [HoLa-BRep（arXiv:2504.xxxxx）](https://arxiv.org/abs/2504.xxxxx) 的 topology bias 设计，承接 [2026-06-01 #3](../2026-06-01/NOTE.md#3-oracle-topology-baseline)。

**结果**

- 训练日志：[outputs/train.log](./outputs/train.log)
- 当前状态：跑到 epoch N，val/loss=...

**参考 / 关联**

- 关联实验：[2026-06-01 #3](../2026-06-01/NOTE.md#3-oracle-topology-baseline)
- 论文：[HoLa-BRep（arXiv:2504.xxxxx）](https://arxiv.org/abs/2504.xxxxx)
- 主要数据：[real_photo.npz](/mnt/d/data/deepcad_v7_cond/00000000/real_photo.npz)
- 配置：[train_diffusion_learned_topology.yaml](../../configs/train_diffusion_learned_topology.yaml)
```

### Failed run

```markdown
### 2. AR topology transformer smoke run

**为什么这么做**

- 测试 edge-token 序列建模是否能替代固定 face-slot adjacency 预测。

**具体实施**

- 运行 [topology_ar_train.py](./topology_ar_train.py)。
- batch size: 32，epochs: 30。

**结果**

- import 阶段失败：`ModuleNotFoundError: ...`。
- 修复：在脚本开头加入 repo root 到 `sys.path`。
- 下一步：重跑同一命令，日志写到 [outputs/ar_smoke.log](./outputs/ar_smoke.log)。

**参考 / 关联**

- 脚本：[topology_ar_train.py](./topology_ar_train.py)
- 论文：[标题（链接待补）]()
```

## Do Not

- Do not create an experiment folder without `NOTE.md`.
- Do not leave a new script or command unmentioned in `NOTE.md`.
- Do not write only “ran training” without why/implementation/result.
- Do not bury important outcomes only in terminal output or W&B.
- Do not create multiple nearly identical command variants without recording the experimental difference.
- Do not name a paper / prior note / config / checkpoint / data file in plain text when a link is possible — make it a clickable markdown link.
- Do not invent a paper URL or file path to satisfy the link rule; use `（链接待补）` instead.
