# Claude Code Guidelines — Evaluation

The canonical instructions for this directory are `AGENTS.md`; read it before changing evaluation code or reporting results. This file is a summary only. The repo-wide entry point is the root `CLAUDE.md`.

## Entry Points and Inputs

```bash
python -m src.evaluate \
    --pred-root /path/to/post_output \
    --gt-root /path/to/ground_truth \
    --split-list /path/to/frozen_model_ids.txt \
    --metrics condition,validity,complexity
```

- Post-processing: `python -m src.postprocess.construct_brep --data_root ... --out_root ...`; `recon_brep.step` may be a compound fallback, so `success.txt` is not a validity check.
- Modules: `protocol.py` (ID and directory resolution), `src/evaluate.py` + `parallel.py`/`io.py` (dispatch, parallelism, output), `metrics/{condition,visible_condition,validity,complexity,uniqueness,lfd}.py`.
- The CLI dispatch in `src/evaluate.py` is the source of truth: a metric module existing does not mean the metric is exposed. Native STEP validity, sampled CD and OCC diagnostics live in `step_metrics.py` and `occ_diagnostics.py`.

## Reporting Rules

- Freeze the intended model-ID list before evaluation and keep every stage status for every ID.
- At minimum report: requested, inference success, STEP produced, valid solid, geometry metrics computed, and each failure reason. Accuracy/success rates must state their denominator; report metric means on successful items alongside full-set success rates.
- `condition.prepare_eval_data` uses a degenerate origin-geometry fallback for missing/unreadable predictions; a numeric result does not imply prediction success.
- `compute_statistics` intersects the split with existing prediction directories and skips missing ones; `--only-valid` further filters by `success.txt`. Disclose both effects; never treat `num_total` as the intended total.
- Cross-method comparisons report both the common-success set and each method's failures on the fixed full set.
- Keep failures, timeouts, NaN/Inf, missing GT, and adapter errors distinct; do not silently delete them or report them as ordinary reconstruction error.

## Metric Semantics and Oracle Information

- The condition uses rotation 0 of the identity-first cube24 in `data/rotations.py`, not an optimal-rotation search. Do not interpret outputs with the old Euler64 or Blender numeric PNG order.
- CD is computed on merged point sets of the same primitive type; Hungarian matching is used for primitive-detection F1 and topology correspondence, not to average CD. Report normalization, sampling density, and F1 thresholds.
- Valid solid only means the current kernel/tolerance check passed; it does not prove shape correctness, manufacturability, or engineering constraints.
- Visibility comes from GT. Training supervision may use it; if it enters inference, candidate selection, or repair, mark the setting as oracle.
- `visible_condition` compares all predicted faces with visible GT faces; it is not restricted to camera-visible predicted surfaces and is not an image reprojection error.
- Real-photo samples without paired GT can support visible-evidence and validity analysis only; loader placeholder geometry is not GT.

## Shared-Boundary Output

- New NPZ outputs must set scalar `representation="shared_boundary_v1"` and provide `pred_edge_confidence[Q]`.
- Each active query has one predicted curve; `pred_edge_face_connectivity[Q,3]` references local predicted-face IDs, not legacy reciprocal half-edges.
- Raw exports keep duplicate pairs, same-face assignments, and all active queries; do not filter by GT edge count, connectivity, or matching results.
- Post-processing selects at most one edge per distinct face pair by confidence, geometric residual, and original edge ID; it records selection/removal reasons in `shared_boundary_post.json` without overwriting the raw NPZ. Native multi-edge/seam is not yet supported.

## Caches and Records

- Result caches prefer `eval_condition.npz` and also accept `eval.npz`; existing results are skipped by default. A change to predictions, GT, split, or metric protocol requires a new output directory or explicit recompute.
- New protocols and results go under `/mnt/d/hola_research/<topic>/`, run outputs under `/mnt/d/hola_runs/`. Freeze the split list, metric settings, post-processing settings, and per-item results.
- When changing metric implementations, validate on cases with known discriminating behavior (geometric differences, missing outputs, coordinate errors). Documentation changes do not trigger batch CAD evaluation.
