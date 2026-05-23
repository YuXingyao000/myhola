# Conditional Diffusion Debug Harness Plan

## 1. Objective

Build an isolated experiment harness for diagnosing why `0502_deepcad_flux_single_view_align_depth.ckpt` performs poorly on the 100-model conditional testing set.

The harness should answer these questions before any large retraining:

1. Does K-sampling contain occasional good CAD candidates?
2. Does the model materially depend on the image condition?
3. At which diffusion timestep does conditional denoising fail?
4. Are bad validity and Chamfer scores correlated with wrong face count?
5. Is the current training objective over-weighting global image-CAD alignment?
6. Would projection-based reranking recover better samples from existing candidates?
7. Is augmentation/view alignment safe for future training?

The first implementation should prioritize low-cost diagnostic experiments over new training.

## 2. Hard Constraints

All experiment code must stay inside:

```text
src/brepnet/experiments/condition_debug/
```

Existing project files must not be edited directly for this harness, including but not limited to:

```text
src/brepnet/diffusion_model.py
src/brepnet/train_diffusion.py
src/brepnet/dataset.py
src/brepnet/eval/eval_condition.py
src/brepnet/post/construct_brep.py
configs/brepnet/*.yaml
```

If an experiment requires behavior changes from an existing file, copy the file into this directory first and modify only the copied file. Example copied-file names:

```text
src/brepnet/experiments/condition_debug/diffusion_model_condition_debug.py
src/brepnet/experiments/condition_debug/train_diffusion_condition_debug.py
src/brepnet/experiments/condition_debug/dataset_condition_debug.py
```

Generated experiment outputs must be written outside the repository, under an explicit result root:

```text
/mnt/d/data/new_cond_results/exp_plan_0502_depth
```

No experiment script may overwrite an existing output directory unless a `--force` flag is explicitly provided.

## 3. Inputs

Primary checkpoint:

```text
/mnt/d/data/new_cond_ckpt/0502_deepcad_flux_single_view_align_depth.ckpt
```

Autoencoder checkpoint:

```text
/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt
```

Cached face latent root:

```text
/mnt/d/data/ae_cache/1119_deepcad_aug1_11k
```

Condition root:

```text
/mnt/d/data/deepcad_v6_cond
```

Test list:

```text
src/brepnet/data/list/deduplicated_deepcad_testing_7_30_sample_100.txt
```

GT root:

```text
<must be provided by caller>
```

The GT root must contain:

```text
<GT_ROOT>/<prefix>/normalized_shape.step
```

for each prefix in the 100-model test list.

## 4. Outputs

All output paths should live under:

```text
ROOT=/mnt/d/data/new_cond_results/exp_plan_0502_depth
```

Directory layout:

```text
$ROOT/
  E0_baseline/
  E0_baseline_post/
  E1_ksample/
    raw/
      <prefix>__s00/
      <prefix>__s01/
      ...
    post/
    oracle_summary.json
    oracle_summary.csv
  E2_condition_sensitivity/
    normal/
    shuffle/
    zero/
    summary.json
    summary.csv
  E3_denoise_probe/
    denoise_errors.json
    denoise_errors.csv
  E5_face_count/
    face_count_summary.json
    face_count_summary.csv
  logs/
```

Each experiment should write:

1. A command log.
2. A resolved config JSON.
3. A summary JSON.
4. A CSV table suitable for spreadsheet inspection.

## 5. Implementation Layout

All new harness code should be placed here:

```text
src/brepnet/experiments/condition_debug/
```

Planned files:

```text
README.md
HARNESS_ENGINEERING_PLAN.md
common.py
run_baseline.py
sample_condition.py
post_eval.py
denoise_gt_probe.py
summarize_eval.py
summarize_ksample_oracle.py
summarize_face_count.py
check_aug_alignment.py
run_0502_depth_00_baseline.sh
run_0502_depth_01_ksample.sh
run_0502_depth_02_condition_sensitivity.sh
run_0502_depth_03_denoise_probe.sh
run_0502_depth_05_face_count.sh
```

If local copies are required:

```text
diffusion_model_condition_debug.py
train_diffusion_condition_debug.py
dataset_condition_debug.py
```

## 6. Experiment Protocol

### E0: Baseline Single Sample

Goal:

Establish the exact current result for the 100-model testing set.

Protocol:

1. Run one sample per test item using the current checkpoint.
2. Post-process with `construct_brep`.
3. Evaluate with `eval_condition`.
4. Summarize validity, Chamfer, topology F-score, face count, and failure count.

Expected output:

```text
$ROOT/E0_baseline/
$ROOT/E0_baseline_post/
$ROOT/E0_baseline_summary.json
$ROOT/E0_baseline_summary.csv
```

Decision rule:

This is the reference for all later comparisons. Do not interpret later experiments without comparing against E0.

### E1: K-Sampling Oracle

Goal:

Check whether the conditional diffusion distribution contains good candidates even if single-sample testing misses them.

Protocol:

1. For each of the 100 prefixes, generate K candidates with the same condition.
2. Recommended first run: `K=8`.
3. Full run: `K=32`.
4. Save candidates without overwriting:

```text
<prefix>__s00
<prefix>__s01
...
```

5. Post-process all candidates.
6. Evaluate all candidates.
7. Aggregate per original prefix:

```text
single_face_cd
best_face_cd
best_valid_face_cd
any_valid
num_valid_candidates
best_candidate_suffix
```

Decision rule:

- If oracle best Chamfer or `any_valid` improves substantially over E0, prioritize projection reranking.
- If oracle does not improve, focus on condition strength, face-count control, or training objective.

### E2: Condition Sensitivity

Goal:

Determine whether the model meaningfully uses image condition.

Modes:

```text
normal
shuffle
zero
```

Protocol:

1. Use the same 100 prefixes.
2. Use fixed sampling seeds across modes when possible.
3. Generate outputs for:

```text
$ROOT/E2_condition_sensitivity/normal
$ROOT/E2_condition_sensitivity/shuffle
$ROOT/E2_condition_sensitivity/zero
```

4. Post-process and evaluate each mode.
5. Compare:

```text
face_cd
edge_cd
vertex_cd
valid_rate
num_recon_face
latent_distance_to_normal
```

Decision rule:

- If `normal`, `shuffle`, and `zero` are close, the image condition is weak.
- If `normal` is clearly better, condition is useful and sample selection/reranking becomes higher priority.

### E3: Noised-GT Denoise Probe

Goal:

Identify at what noise level conditional denoising stops being anchored to GT geometry.

Timesteps:

```text
50
100
200
500
800
```

Protocol:

1. Load GT face latent from cached `face_latents`.
2. Add scheduler noise at each timestep.
3. Predict epsilon or x0 using:

```text
normal condition
shuffle condition
zero condition
```

4. Write errors without running post-processing:

```text
epsilon_mse
x0_l1
x0_mse
per_timestep_mean
per_timestep_median
```

Decision rule:

- Low timestep good, high timestep bad: condition cannot control early global generation.
- Normal much better than shuffle/zero: condition contains useful signal.
- Normal close to shuffle/zero: condition path is too weak or too global.

### E4: Align-Loss Ablation

Goal:

Check whether global CLIP-style image-CAD alignment hurts local geometric correctness.

This experiment requires training or finetuning and should not be run until E1-E3 results justify it.

Candidate settings:

```text
lambda_align=0
lambda_align=0.05
lambda_align=0.1
lambda_align=1.0
```

Implementation rule:

Do not edit active `diffusion_model.py`. Copy it to:

```text
diffusion_model_condition_debug.py
```

and make `lambda_align` configurable only in the copy.

Decision rule:

If small or zero `lambda_align` improves Chamfer/validity on the 100-model set, global alignment should not be used at full weight.

### E5: Face-Count / Existence Diagnosis

Goal:

Determine whether invalidity and high Chamfer are driven by wrong predicted face count.

Protocol:

1. Read `eval.npz` from post-processed results.
2. Compute:

```text
num_recon_face
num_gt_face
face_count_error = num_recon_face - num_gt_face
abs_face_count_error
success.txt exists
face_cd
edge_cd
fe_fscore
ev_fscore
```

3. Group by:

```text
valid vs invalid
abs_face_count_error bins
over-generation
under-generation
```

Decision rule:

- If invalid results strongly correlate with face-count error, prioritize explicit face existence / count control.
- If face count is close but topology poor, prioritize adjacency/topology constraints.

### E6: Projection Reranking

Goal:

Use image evidence to select the best candidate from K-sampling.

This experiment depends on E1 producing multiple candidates.

First version scoring:

```text
score = 2.0 * silhouette_iou
      - 0.5 * edge_chamfer_2d
      - 0.1 * extra_projected_area
      + 0.2 * valid_bonus
```

Protocol:

1. Reuse E1 candidates.
2. Render/project candidate geometry to the input camera.
3. Compute mask/edge score against input image.
4. Select one candidate per prefix.
5. Compare:

```text
E0 single sample
E1 oracle best
E6 projection rerank
```

Decision rule:

- If rerank approaches oracle, sample selection is the main missing piece.
- If rerank fails while oracle succeeds, improve projection/mask/camera scoring.

### E7: Augmentation Alignment Check

Goal:

Verify future training with `dataset.is_aug=1` does not mismatch image view and latent pose.

Protocol:

1. Inspect mapping from cube view id to cached latent id.
2. For a few prefixes, save:

```text
selected cube_id
mapped id_aug
condition image filename/source
latent feature path
```

3. Confirm condition image changes when cube id changes.

Known risk:

Current `single_img` loading may overwrite `svr_imgs[v_id_aug]` with fixed `single_view.npz:flux`, which is unsafe if target latent pose changes.

Decision rule:

Do not run augmented single-image training until this check passes.

## 7. Metrics

Primary metrics:

```text
valid_rate
face_cd
edge_cd
vertex_cd
face_fscore
edge_fscore
vertex_fscore
fe_fscore
ev_fscore
num_recon_face / num_gt_face
abs_face_count_error
```

Secondary metrics:

```text
face_acc_cd
face_com_cd
edge_acc_cd
edge_com_cd
vertex_acc_cd
vertex_com_cd
num_valid_candidates
oracle_gap = single_face_cd - best_face_cd
condition_gap = shuffled_face_cd - normal_face_cd
zero_gap = zero_face_cd - normal_face_cd
```

Report both mean and median. Mean is sensitive to invalid fallback samples; median is often more stable.

## 8. Error Handling

### Missing Input Files

If a required checkpoint, list file, condition file, latent file, or GT STEP file is missing:

1. Print the exact missing path.
2. Mark the sample or run as failed.
3. Do not silently skip unless `--allow-missing` is explicitly provided.

### Existing Output Directory

If an output directory already exists:

1. Refuse to run by default.
2. Allow overwrite only with `--force`.
3. If `--force` is used, delete only the specific experiment output directory, never the shared root.

### Invalid STEP / Post-Processing Failure

If post-processing fails or `recon_brep.step` is invalid:

1. Keep the raw `data.npz`.
2. Mark validity as false.
3. Record failure reason when available.
4. Do not count the sample as missing.

### Empty or Corrupted `eval.npz`

If `eval.npz` is missing or unreadable:

1. Mark the candidate as `eval_missing`.
2. Exclude from numeric metric means unless explicitly computing missing-rate.
3. Include it in failure counts.

### NaN / Inf Metrics

If any metric is NaN or Inf:

1. Store the raw value in JSON as a string.
2. Exclude from mean/median.
3. Increment `nan_metric_count`.

### GPU / CUDA Failure

If sampling fails due to CUDA OOM:

1. Reduce batch size.
2. Keep K and seed schedule unchanged.
3. Restart only failed shard if possible.

### Ray Failure

If Ray post-processing fails or hangs:

1. Retry failed prefixes serially.
2. Log timeout prefix list.
3. Do not rerun successful prefixes unless `--force` is provided.

## 9. Reproducibility Rules

Each run must log:

```text
git status --short
command line
resolved arguments
checkpoint path
autoencoder checkpoint path
test list path
condition root
face_latents root
CUDA_VISIBLE_DEVICES
seed
K value
condition mode
timestamp
```

Sampling experiments must use deterministic seed assignment:

```text
seed = base_seed + sample_index
```

Recommended:

```text
base_seed = 20260509
```

## 10. Prohibited Actions

Do not:

1. Edit active project files outside `src/brepnet/experiments/condition_debug/`.
2. Overwrite existing experiment outputs without `--force`.
3. Delete shared result roots.
4. Change the 100-model test list during a comparison run.
5. Compare experiments using different GT roots.
6. Mix DINO and Depth checkpoints in the same summary table without marking the source.
7. Treat invalid STEP fallback Chamfer as equivalent to valid-shape Chamfer without reporting validity.
8. Start E4 training ablations before E1-E3 diagnostic outputs are reviewed.

## 11. Execution Order

Recommended order:

```text
1. E0 baseline
2. E1 K=8 smoke test
3. E1 K=32 full oracle
4. E2 condition sensitivity
5. E3 denoise probe
6. E5 face-count summary
7. Decide whether to implement E6 projection rerank
8. Decide whether E4 align-loss retraining is justified
9. Run E7 before any future is_aug=1 single-image training
```

## 12. Stop / Continue Criteria

Stop and inspect manually if:

```text
E0 cannot reproduce current poor result
more than 10% of samples are missing input files
post-processing fails for nearly all samples
normal/shuffle/zero runs accidentally use different prefix sets
GT root does not match the test list
```

Continue to reranking if:

```text
E1 oracle best is much better than E0 single sample
```

Continue to condition architecture/loss ablation if:

```text
E2 normal is close to shuffle/zero
or
E3 high-timestep normal denoising is close to shuffle/zero
```

Continue to face existence/count work if:

```text
E5 shows validity strongly correlates with abs_face_count_error
```

## 13. Deliverables

First deliverable batch:

```text
common.py
run_baseline.py
sample_condition.py
post_eval.py
denoise_gt_probe.py
summarize_eval.py
summarize_ksample_oracle.py
summarize_face_count.py
run_0502_depth_00_baseline.sh
run_0502_depth_01_ksample.sh
run_0502_depth_02_condition_sensitivity.sh
run_0502_depth_03_denoise_probe.sh
run_0502_depth_05_face_count.sh
```

Second deliverable batch, only after first diagnostics:

```text
projection_rerank.py
check_aug_alignment.py
diffusion_model_condition_debug.py
train_diffusion_condition_debug.py
```

## 14. Acceptance Criteria

The harness is acceptable when:

1. It can run E0 on the 100-model test list.
2. It can run E1 with `K=8` without overwriting candidates.
3. It can summarize E0 and E1 into JSON and CSV.
4. It can run E3 without post-processing.
5. It does not modify existing source files outside this directory.
6. It records enough metadata to reproduce each run.

