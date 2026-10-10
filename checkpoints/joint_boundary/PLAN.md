# Joint boundary decoder — implementation contract

## Task
Implement the first shared-boundary, bidirectional face/edge VAE decoder on a separate Git branch. Keep AR, diffusion training changes, native physical-edge re-extraction, and UV/loop prediction outside this implementation. Store new experiment records and runtime artifacts under `checkpoints/`, never add to `experiments/`.

## Assumptions
The existing DeepCAD contract represents at most one merged boundary per unordered pair of distinct faces, with directed half-edge samples. It excludes seams and does not retain all native physical edges. Preserve the old encoder inputs, face order, normalization, and posterior parameterization. A frozen encoder must also stay in evaluation mode. No new scientific improvement is assumed before training.

## Inputs
Existing `data.npz` face/edge samples, edge-face connectivity, negative pairs, fixed train/validation/test lists, and an optional explicitly supplied legacy AE checkpoint. Shared-boundary targets are added alongside, rather than substituted into, encoder inputs. Decoder inputs are face latents and per-object face masks only.

## Outputs
A registered VAE variant and Hydra training configs; tensor-native surfaces, shared curves, existence logits and two-sided face assignment logits; per-object raw NPZ exports with an explicit representation tag; matching/reconstruction diagnostics; a documented common run layout beneath `checkpoints/`.

## Processing Protocol
1. Group directed targets by unordered face pair, check reciprocal coverage and curve agreement up to reversal, and choose a deterministic representative. Closed curves may have different starting parameters: validate their geometric agreement and permit cyclic alignment. Reject unsupported/ambiguous targets with sample identifiers rather than silently inventing physical-edge labels.
2. Pad face features only for batching. Each object receives its own learned edge queries. Alternate supervised soft face assignment, face-to-edge updates, and optional edge-to-face updates.
3. Match all predicted edge queries to target boundaries with geometry plus ownership costs, including reversal, closed-curve cyclic alignment, and side-swap equivalences. Unmatched queries supervise no-boundary existence. GT connectivity is used for loss/matching, never as decoder input. Use 128 queries: the available training split contains 46,627 objects with at most 108 merged boundaries; 1,242 objects exceed 64.
4. Export predicted topology without GT edge counts or GT query selection. Preserve raw duplicates and invalid assignments for diagnostics; postprocessing must use a declared shared-boundary adapter rather than treating them as reciprocal half-edges.
5. Provide a frozen-encoder run and explicit joint-finetuning configuration. Loading legacy weights is a validated component migration, not permissive checkpoint guessing.

## Metrics
Face/curve reconstruction, existence/ownership losses and prediction counts; matched geometry and topology diagnostics; raw duplicate/invalid incidence counts; independent downstream geometry and solid validity under declared postprocessing settings. Shared-curve identity is a representation property, not an improvement metric.

## Decision Rule
Implementation is ready when real-data batching, forward/backward, finite gradients, frozen-encoder semantics, prediction-only export, checkpoint roundtrip, and CLI configuration pass appropriate checks. Research success requires later controlled training and held-out evaluation; no implementation smoke test establishes it.

## Edge Cases
Reject missing/corrupt/nonfinite geometry, invalid indices, inconsistent reciprocal boundaries, native multi-edge/seam inputs, empty face sets, and target counts above query capacity explicitly. Zero target boundaries must give a finite no-object loss. Matching and decoding must handle variable counts and preserve object isolation. Missing predictions remain failures in evaluation denominators.

## Validation / Sanity Checks
Use synthetic permutation/reversal/empty-target cases and actual DeepCAD samples. Check permutation equivariance, finite gradients through incidence and geometry, frozen encoder train/eval behavior, autonomous topology prediction, per-object exports, and strict new-checkpoint reload. Run OCC end-to-end when the installed environment supports it and report any unavailable checks.

## Implementation Constraints
Keep existing model/config entry points compatible. New behavior uses explicit configuration and separate modules. W&B is permitted and configurable; local checks need not create online runs. Do not launch a full research training job as a smoke test. No datasets, model weights, or generated geometry are committed.

## Deliverables
An experiment branch, reviewable commits, code and meaningful tests, records in this directory, organized output defaults, and runnable training/evaluation/resume commands with actual validation results.
