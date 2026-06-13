---
name: brepnet-experiment-planning
description: Plan and set up one day of HoLa-BRep experiments toward the long-term goal "Image-Conditioned Diffusion Model for B-Rep generation". Use when the user starts a new experiment day, asks "今天做什么实验 / 帮我规划今天的实验 / 跑实验", or wants the agent to design the next experiment, justify it with literature, write/modify scripts and commands, and prepare runs — then hand off to the reporting skill.
---

# BRepNet Experiment Planning (Daily Driver)

Use this skill to drive a single experiment day end-to-end: confirm goals, reflect against the long-term target, check literature, plan a concrete experiment, write or adapt scripts/commands, run only cheap validation, and hand off to the reporting skill.

This skill plans and sets up runs. It does NOT babysit long runs. Diffusion / VAE training can take >1 day; produce runnable scripts/commands, run only smoke checks, then stop and wait for the user to report results in a later turn.

Paths below are relative to the repo root (`/mnt/d/python`). This skill is installed into multiple tools (Cursor / Claude Code / Codex); the canonical source is `skills/brepnet-experiment-planning/SKILL.md` — edit there and re-sync the copies (see README "Experiment Workflow & Agent Skills").

Pair files:
- Long-term direction & recent progress: `experiments/ROADMAP.md` (read first, update at end).
- Reading list: `experiments/REFERENCES.md`.
- Writing the day's report: `skills/brepnet-experiment-notes/SKILL.md` (`brepnet-experiment-notes`).

## North Star (never silently drift)

**Image-Conditioned Diffusion Model for B-Rep model generation.** `image -> (topology mask) -> HoLa diffusion -> B-Rep`. The final judge is downstream utility (valid STEP rate + condition consistency), not any single intermediate metric. Every day's experiment must be traceable to this goal in one sentence.

## Daily Workflow (run phases in order)

### Phase 0 — Orient

1. Read `experiments/ROADMAP.md` fully: North Star, current phase, known best, open questions, next candidates, anti-goals.
2. Read the last ~1 week of `experiments/YYYY-MM-DD/NOTE.md` (most recent first). Extract: what was tried, what worked, what failed, what each entry's `待办` left open.
3. List the relevant output dirs / checkpoints / metrics files referenced by those notes so today's plan can link them.

Do not skip to planning. Today's plan must be grounded in what Phase 0 actually found, with links.

### Phase 1 — Confirm long-term goal

State the North Star in one sentence and name the current phase from `ROADMAP.md`. If the user's request seems to conflict with the North Star, say so explicitly before planning.

### Phase 2 — Confirm recent progress + mandatory daily reflection

Summarize, with links, the recent direction and recent results (from Phase 0). Then write a short **方向反思 (daily reflection)** — this is mandatory every day, even on "obvious" days:

1. 这个方向相对长期目标，现在卡在哪一步？(map to an open question in `ROADMAP.md`)
2. 最近几天是不是在对同一个指标 / 同一份数据反复微调？检查 anti-goals 列表。
   - Anti-local-loop trigger: if the last ~3 experiment days were all optimizing the **same single metric** on the **same component** with no meaningful breakthrough (and no new question answered), do NOT propose another tweak of the same kind. Instead, step back: re-derive whether that metric still serves the North Star, and propose either (a) moving to the next bottleneck in the pipeline, or (b) a downstream check that validates whether the local gains matter at all.
3. 基于反思，今天的方向是「继续推进上一条」还是「跳出局部、换下一个瓶颈」？给出明确选择和理由。

Keep this reflection honest and specific. It is the main guard against grinding one metric while forgetting the paper goal.

### Phase 3 — Literature check (required before writing any script/command)

No script or command for the day is written until literature is checked and a justification exists.

1. Read `experiments/REFERENCES.md`. If today's direction is already well-supported there, cite the specific entries.
2. If support is insufficient, use WebSearch to find **papers from the last 3 years** (or foundational theory: DDPM, VAE, LDM, classifier-free guidance, etc.) relevant to today's specific question. Prefer arXiv; do not invent URLs.
3. Write back any new, useful paper into `REFERENCES.md` with a one-line "为什么和我们相关", following its no-fabrication rule.
4. Produce a **为什么今天做这个实验** paragraph that connects: the open question (Phase 2) + the literature (Phase 3) + the expected signal. This paragraph is reused verbatim as the experiment entry's `为什么这么做` in the report.

### Phase 4 — Plan the concrete experiment

1. Define exactly one (or a small set of clearly-differentiated) experiment(s). For multiple variants, state the single variable each isolates (the 06-11 note's "不要同时改两个变量" discipline).
2. Specify the expected metric(s) and the decision rule: what result would make this a win / a kill / an iterate.
3. Reuse the modularized diffusion model — do not rewrite it. The model lives in `src/brepnet/models/`:
   - assembly: `src/brepnet/models/diffusion.py`
   - condition encoders: `src/brepnet/models/diffusion_condition.py`, `src/brepnet/models/condition_encoders.py`
   - denoiser / topo bias: `src/brepnet/models/diffusion_denoiser.py`
   - latent stats / padding: `src/brepnet/models/diffusion_latents.py`, `src/brepnet/models/diffusion_padding.py`
4. Prefer config assembly over code edits. A new diffusion variant is usually a new `configs/train_diffusion_*.yaml` composing existing groups:
   - `defaults:` pulls from `configs/{trainer,dataset,model,condition}/`.
   - condition options available: `none`, `single_img`, `multi_img`, `sketch`, `point_cloud`, `text` (see `configs/condition/`).
   - Entry point: `python -m src.brepnet.train --config-name <name> <hydra overrides> hydra.job.chdir=false`.
   - Only edit module code when a config field genuinely cannot express the change; if so, add the field to the owning config group and read it in the owning module (per repo AGENTS.md).
5. For one-off topology/CVAE-style probes that aren't part of the config-driven trainer, follow the existing pattern: a self-contained script in `experiments/YYYY-MM-DD/` that reuses `src/brepnet/models/*` building blocks (as 06-11 did with `DINOv2ImageEncoder`).

### Phase 5 — Implement + cheap validation only

1. Create today's folder `experiments/YYYY-MM-DD/` if missing.
2. Write the new config and/or script, or the new command line on an existing script. Use explicit paths and the Current Data Contract below.
3. Run only cheap validation — never launch full training:
   - `conda run -n torch python -m py_compile <new_script.py>` (syntax)
   - a tiny smoke test: 1 train + 1 val sample, batch size 1, output to `/tmp/...`, verifying forward / train step / eval path run. (06-11 used exactly this.)
   - for config changes: `python -c "from src.brepnet.models import build_model"` and a Hydra dry compose check.
4. Produce the full training command (8-GPU, real epochs/batch) ready to copy, but DO NOT start it. Make clear it is for the user to launch. Put eval commands (e.g. full val/test `sample@K`) alongside, to run after training.
5. Save the runnable commands into `experiments/YYYY-MM-DD/command.sh` (starts with `cd /mnt/d/python`), each command preceded by a short comment.

### Phase 6 — Hand off to reporting

Trigger the reporting skill `brepnet-experiment-notes` (`skills/brepnet-experiment-notes/SKILL.md`) to write/append `experiments/YYYY-MM-DD/NOTE.md` for today, reusing the Phase 3 justification as `为什么这么做`. Then update `experiments/ROADMAP.md`:
- add a timeline row if a conclusion was reached (or mark "still running" if only launched),
- update `当前最优` if a new best was confirmed,
- check off / add `下一步候选`.

If results are only available later, leave the report's `结果` as "已给出命令，待运行" and tell the user you'll finish the report when they say the run is done.

## Anti-Local-Loop Rules (the core guardrail)

1. The daily reflection (Phase 2) is mandatory and must reference `ROADMAP.md` open questions and anti-goals.
2. Never propose ≥3 consecutive days of tweaking the same single metric on the same component without a step-back that re-justifies it against the North Star or pivots to the next bottleneck / a downstream check.
3. `exact_adj_acc` and similar one-to-many metrics are sanity checks, not targets. Do not grind them.
4. When a local metric improves, ask whether it actually moves the downstream `image -> B-Rep` pipeline; if untested, propose the downstream check rather than more local tuning.
5. Prefer answering an open question over squeezing a saturated number.

## Current Data Contract

Reuse the contract from the reporting skill; for current DeepCAD v7 experiments prefer:

| What | Path |
|------|------|
| Raw B-Rep data | `/mnt/d/data/deepcad_v7` |
| Condition data | `/mnt/d/data/deepcad_v7_cond` |
| 24-view latent cache | `/mnt/d/data/ae_cache/1119_deepcad_aug1_11k_24` |
| AE checkpoint | `/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt` |
| Train / val / test lists | `src/brepnet/data/list/deduplicated_deepcad_{training,validation,testing}_7_30.txt` |

Condition files: `deepcad_v7_cond/{model_id}/imgs.npz` (`svr_imgs`, `sketch_imgs`) and `real_photo.npz` (`flux`). Do not call this `svr.npz`.

## Linking discipline

All references in the plan and in the report must be clickable markdown links (papers → arXiv URL, prior notes → `../YYYY-MM-DD/NOTE.md#anchor`, configs/scripts → workspace-relative, data/checkpoints/outputs → absolute path). Never fabricate a path or URL; use `（待补）`. This matches the `brepnet-experiment-notes` skill's Links & References rules.

## Do Not

- Do not write any script/command before the literature check and the "为什么今天做这个实验" justification exist.
- Do not skip the daily reflection, even on "obvious" continuation days.
- Do not launch full training; only run py_compile + tiny smoke + config sanity.
- Do not rewrite the diffusion model; compose configs / reuse `src/brepnet/models/*`.
- Do not grind a single metric / single sample for days while ignoring the North Star.
- Do not leave `ROADMAP.md` stale after reaching a conclusion.
- Do not end the day without handing off to the reporting skill.
