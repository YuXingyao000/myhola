---
name: brepnet-experiment-notes
description: Manage HoLa-BRep experiment folders under experiments/ with one dated folder per day, a mandatory NOTE.md, and per-experiment records that explain why, implementation, and results.
---

# BRepNet Experiment Notes

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
  - `experiments/YYYY-MM-DD/xxx.py`
  - `experiments/YYYY-MM-DD/command.sh`
- 关键配置：
  - data root / condition root / latent root
  - checkpoint
  - Hydra overrides

**结果**

- 指标、现象、失败日志、输出路径，或“还在跑 / 待补评估”。

### 2. 实验名（上一条的迭代）

**为什么这么做**

- ...

**具体实施**

- ...

**结果**

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

- 使用 `configs/train_diffusion_learned_topology.yaml`。
- checkpoint: `experiments/2026-06-01/outputs_svr_single/best.pt`。
- 数据使用 `deepcad_v7_cond/real_photo.npz["flux"]`，`real_photo_ratio=1.0`。

**结果**

- 训练日志：`...`
- 当前状态：跑到 epoch N，val/loss=...
```

### Failed run

```markdown
### 2. AR topology transformer smoke run

**为什么这么做**

- 测试 edge-token 序列建模是否能替代固定 face-slot adjacency 预测。

**具体实施**

- 运行 `experiments/YYYY-MM-DD/topology_ar_train.py`。
- batch size: 32，epochs: 30。

**结果**

- import 阶段失败：`ModuleNotFoundError: ...`。
- 修复：在脚本开头加入 repo root 到 `sys.path`。
- 下一步：重跑同一命令，日志写到 `...`。
```

## Do Not

- Do not create an experiment folder without `NOTE.md`.
- Do not leave a new script or command unmentioned in `NOTE.md`.
- Do not write only “ran training” without why/implementation/result.
- Do not bury important outcomes only in terminal output or W&B.
- Do not create multiple nearly identical command variants without recording the experimental difference.
