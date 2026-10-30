Task: inspect this candidate without modifying it.

The attached images are the host's current fixed review panel. Their ordered paths
and the actual technical report are supplied below. Inspect every full-resolution
image, including the empty scenes, against the original user requirements and any
provided references. Additional local image inspection is allowed when useful.

Check scale cues; functional relationships and visible density; major surfaces and
props; coherent wear; lighting and reflections; and the readability of the target's
silhouette, holes, grooves, and curved surfaces. Record observable image evidence,
and distinguish suspected causes from observations. Do not invent a realism score.

This is a read-only task: do not edit the template, assets, scene_plan.md, source
models, host code, reports, or images; do not rebuild or render. Return the review
as the structured final response. The host saves that response and compares input
and evidence hashes. This instruction is not an operating-system sandbox.

decision:
- ready: the current candidate is suitable for human review, with no unresolved
  issue requiring revision. This is not human approval or proof of realism.
- revise: specific issues can be addressed within the template. Suggest targeted
  changes; the host will schedule a separate repair turn.
- blocked: a required change is outside template scope or a required resource is
  unavailable. For example, report host CAD-normal defects rather than hiding them
  with material noise or altering the host importer/checker.

Use image basenames exactly as listed. inspected_images must list every image you
actually inspected. For each issue record image, region, observation, and
suggested_change. ready requires empty issues and remaining_issues; use summary
for the evidence supporting that decision and nonblocking scope limitations.
remaining_issues contains only defects or missing required evidence that must be
resolved before the current candidate is ready for human review. The host's fixed
checks_not_run list does not automatically require blocked: keep those unverified
checks in summary unless the current request explicitly requires them or their
absence prevents assessment of the current candidate. Pending human approval is
the expected next step and belongs in summary. Observed host defects requiring
correction still require blocked. Do not claim to have viewed images merely
because their paths appear in a report.

Current host evidence:
