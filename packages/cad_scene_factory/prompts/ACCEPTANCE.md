# Template Previews and Acceptance

This document separates facts that programs can check from quality that requires image inspection. It defines no automatic realism score and does not treat a small review set as proof that every CAD model will pass.

## Fixed Preview Set

The host pipeline uses 12 frozen training-set parts that pass the framing check, ordered by the original CAD's B-Rep face count. The current `review_models.json` retains the first workshop_002 review set and orders its parts from 7 to 28 faces; it is not a new sample spanning the full dataset's face-count distribution. STL triangle counts do not substitute for B-Rep face counts, and no additional shape-category quotas are required.

| Preset | Input | Output |
|---|---|---|
| `quick` | One part each from the lower, middle, and upper face counts of the fixed 12 | 3 images, 512 × 512, 64 samples |
| `review` | All 12 fixed parts | 12 images, 512 × 512, 256 samples, presented in face-count order |
| Additional environment checks in `review` | Environment root heights of -0.901, -0.4505, and 0, with no CAD present | 3 environment-only images, 512 × 512, 256 samples |

The host configuration supplies the camera, sample list, and seeds. Reuse the same configuration for every template; do not replace a part because it fails. If the review set changes later, record a new version and keep its results distinct from those of the previous set.

Use `quick` for feedback during construction. Completing `quick` does not complete `review`. The three environment heights check only those three compositions; they do not establish correctness over the entire continuous height range.

## Who Performs Each Stage

The initial Codex task writes `scene_plan.md` and creates the builders. The surrounding Python program then builds the template and runs the complete fixed `review`. It resumes the same Codex session for a dedicated read-only inspection of the current report and 15 images. Technical failures go directly to repair without a visual-review turn. If template changes are needed, the host requests a repair in that session and reruns the full review. The configured host repair budget defaults to two rounds after initial construction; this does not prescribe a number of internal edits or quick previews.

During read-only inspection, do not change the plan, builders, assets, images, or reports, and do not build or render. The host checks that review-input digests remain unchanged. This is an instruction plus a consistency check, not operating-system isolation. A changed input invalidates the review, even if the response says it passed.

## Technical Checks: Evidence Comes from Tool Results

The host uses `preview.py` to build and render the template; the agent may use its `quick` preset during construction or repair. Consult `--help` for arguments. Reports should preserve planned samples, hashes of the current source code and assets, actual settings, each sample's success or failure, image paths, and error reasons.

| Check | Evidence needed to pass |
|---|---|
| The template builds | Both current builders are actually loaded and return the agreed scene, support, and Material; the build succeeds |
| Assets are complete | Declared files exist and match their hashes; dependencies actually used satisfy packaging requirements |
| Fixed camera and geometric bounds | Camera and output settings match the contract; each part passes the existing framing, support-height, support-coverage, and protected-volume AABB checks |
| All planned images are generated | Every planned output exists, decodes, and has the correct dimensions; failures and missing outputs are recorded separately |
| Evidence matches the final version | Images and reports correspond to the final source code and assets; changed templates are checked again |
| The template is portable | After copying the package to another directory, it actually rebuilds using the packaged files, without dependencies on the original path or shared cache |

Mark checks that are not implemented or have not been run as unverified. Do not report a pass merely because the code looks reasonable. In particular, packed assets do not prove that relocation and rebuilding have been tested.

Use the number of planned inputs as the denominator for technical success counts. Keep import failures, geometry-check failures, timeouts, and missing images in that denominator. Do not present only successful outputs.

## Limits of the Technical Checks

- AABB checks are conservative and may reject slender or angled props that would not actually collide. Adjust the scene rather than bypassing the fixed checker.
- The top height and coverage of a support AABB do not by themselves prove that the actual support surface is flat, free of holes, or in contact with the part. Inspect the support construction and images as well.
- The protected volume is the convex hull of the camera position and the part's AABB. Background objects and the worktop may appear in the image. Passing this check establishes neither background realism nor the readability of every geometric detail of the part.
- Unstable poses and point contact are allowed. Do not add center-of-mass stability or rigid-body simulation requirements.
- These checks do not verify semantic fidelity from STEP to STL, compatibility with the entire dataset, or downstream reconstruction ability on real photographs.

## Visual Checks: Record Issues in Specific Images

During the dedicated review stage, actually open the current images supplied by the host, including all three empty-environment views. Use a contact sheet for navigation and inspect the individual images at their final resolution; a small thumbnail, file path, or successful render log does not prove that an image was inspected. Record the image, region, and observation rather than assigning an unsupported overall subjective score. Keep proposed causes and changes separate from what the image demonstrates.

| Item | Typical issues requiring revision |
|---|---|
| Physical scale | The part-to-tool or part-to-support proportions contradict the requested size; a recognizable prop has an implausible size; the plan omits or conflicts with the unit conversion. Cite the intended dimension and visible reference, without claiming exact metric dimensions can be recovered from a single image |
| Space and background | Exposed background edges, gaps in walls, implausible exterior views, or these issues appearing after the environment moves vertically |
| Functional layout | The requested work activity has no readable relationship between its visible props; unrelated objects are uniformly spaced like a display; substantial scene detail exists only outside the actual frame. Judge nearby groups, partial crops, and height layers within the protected-space constraints, without requiring a minimum object count |
| Major surfaces and props | Prominent complex props represented by crude primitives; large areas of stretched textures, seams, repetition, or incorrect scale; different materials all looking like the same plastic. Identify the dominant visible support/background surfaces and whether their structure and material variation fit the requested use history |
| State of use | A clear conflict with the user request; dirt or wear inconsistent with its location, material, or cause |
| Lighting and imaging | Clearly inconsistent light directions or reflections; uniform fill erasing the intended spatial cues; exposure, shadows, or reflections obscuring major holes, grooves, or silhouettes; obvious rendering artifacts. Inspect light direction, contact, dark/bright reflection regions, and target readability together |

Judge impact at the final 512 × 512 resolution. Small, distant objects do not each need added wear. A clean new target part and a visibly used environment can coexist. Strong metallic highlights, a tidy environment, or surfaces without rust are not failures in themselves.

In the review summary, briefly ground the judgment in the actual scale reference, functional arrangement, dominant surface state, and light/reflection behavior. Do not turn this into a per-object checklist. Distinguish template issues from shared host issues: target import, normals, tessellation, camera, source CAD, and checkers are outside the template agent's repair scope. A target-shading problem is not evidence that the environment needs more dirt or that the fixed material should be replaced.

## Structured Agent Review

Return the caller's structured object, without Markdown fences or additional prose. The fields are:

| Field | Meaning |
|---|---|
| `decision` | Exactly `ready`, `revise`, or `blocked` |
| `summary` | A concise account of the current technical evidence, visual findings, and nonblocking scope limitations |
| `issues` | A list of objects with `image`, `region`, `observation`, and `suggested_change`, all strings. Use the exact image basename supplied by the host; identify the relevant region in plain language |
| `remaining_issues` | A list of strings describing defects or missing required evidence that must be resolved before the current candidate is ready for human review; do not omit a known blocker |
| `inspected_images` | A list of the exact supplied basenames of images actually opened and inspected in this review task; do not include unavailable files or images remembered from an earlier version |

- **`ready`:** All 15 current images were inspected, configured technical checks passed, and no issue requires another template revision or host fix. Both `issues` and `remaining_issues` must be empty. This means ready for human review, not certified photographic realism. Describe the evidence and nonblocking scope limitations in `summary`.
- **`revise`:** Supported findings call for changes to the template's plan, assets, builders, layout, materials, or lighting. Include actionable issues. A diagnosed template error that prevented rendering can also require revision; preserve the missing outputs and do not claim a complete visual inspection.
- **`blocked`:** A required tool/resource or sufficient evidence is unavailable, or a defect requires work outside template scope. Identify what the host needs to resolve. Do not classify ordinary scene difficulty as a blocker or alter host code to avoid one.

The host's fixed `checks_not_run` list does not automatically prevent `ready`. Describe these unverified checks in `summary` as scope limitations unless the current request explicitly requires them or the missing evidence prevents assessment of the current candidate. Only then include the specific missing requirement and its impact in `remaining_issues`. Pending human approval is the expected next step, not a blocker to human review; record it in `summary`. These distinctions do not waive observed host defects that require correction, such as target-normal artifacts.

The host decides whether to request another repair using this response and its technical checks. Exhausting the configured repair budget does not turn `revise` or `blocked` into `ready`.

The inspected-image list is the agent's evidence record. Checking its coverage and the input digests does not independently prove that the visual judgment is correct.

## Status and Human Review

Keep technical and visual status separate rather than hiding unfinished work behind a single success label:

- Technical status: report passes and failures for checks that ran, plus any checks that remain unverified.
- Agent review decision: `ready`, `revise`, or `blocked`, backed by the current images and report.
- Human visual status: initially `pending`. The agent must not assign human `approved` status.
- After the host workflow ends, a human reviews the fixed previews and assigns `approved` or `revise`, with specific reasons. This is not an item-by-item question-and-answer dialog inside the agent loop.

Report “Technical checks for this iteration passed; visual review pending” only when every technical check required for the iteration has actually passed. If failures or unverified items remain, state their scope, retain the existing outputs, and do not claim complete acceptance.

The final delivery includes the template location, report and preview locations for the current version, planned/success/failure counts, and unresolved issues. Passing the review set does not remove the existing checks performed for each CAD during batch production.
