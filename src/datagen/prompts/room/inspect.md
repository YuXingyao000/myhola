Task: inspect the current candidate and references without changing either.

Open every current host-rendered image and every comparison reference image listed
in the supplied evidence. Inspect at full resolution where needed. You may read
the plan, builder, asset manifest, and reports to understand provenance and actual
changes. Do not edit files, run a build or render, acquire assets, or repair during
this turn. Do not claim to have viewed an image merely because its path is listed.

Compare the candidate against the user brief and the reference at room, activity,
and surface scales. Evaluate functional layout and proportions, major surface and
prop detail, believable material responses, source-consistent illumination,
shadows and reflections, and quality across the different camera directions.
Use runtime.reference_mode for the source relationship: adapt requires substantive
changes in layout, activity, and related lighting; changing cameras alone is not
an adaptation. Quality_only requires an independent space matching the brief,
not the reference's room type or layout. Evaluate what is visibly present, rather
than treating equipment names in the builder as proof of convincing equipment.
Distinguish an observable defect from an explanation; do not invent a numerical
realism score.

Return only the structured response matching the supplied schema:

- `decision`: `ready` when all required evidence is available and no significant
  unresolved quality or technical issue requires correction; `revise` when
  concrete changes inside the template can address the problems; `blocked` when
  a required resource, host capability, or evidence is missing and prevents a
  supported assessment or repair.
- `summary`: the supported overall judgment and any nonblocking scope limits.
  `ready` means ready for human review, with human approval still pending.
- `inspected_images`: the exact basenames of all candidate images actually opened,
  matching the host list. A ready review must cover the complete six-view set.
- `reference_images`: the exact basenames of the reference images actually opened,
  matching the supplied reference list.
- `reference_comparison`: four explicit observations under `layout`, `surfaces`,
  `lighting`, and `adaptation`. Describe retained quality and differences using
  image evidence. The legacy field `adaptation` remains in both modes: in adapt,
  identify substantive changes supported by the builder or plan; in quality_only,
  explain whether an independently assembled room fulfills the user's activity
  and room type. Do not require similarity to the reference's layout. Do not
  substitute generic praise or repeat the task brief.
- `issues`: concrete issues with `image`, `region`, `observation`, and
  `suggested_change`. Use an exact candidate image basename and a precise region.
  For a global defect choose a relevant overview image. Proposed changes should
  address the observation without sacrificing the other views.
- `remaining_issues`: defects or missing required evidence that must be resolved
  before this candidate is ready. Do not list pending human approval or optional
  future capabilities as blockers.

`ready` requires empty `issues` and `remaining_issues`. Do not lower the standard
because the files exist, a render completed, or repair budget is limited. If the
candidate remains visibly inferior to the reference through crude shapes, sparse
unmotivated staging, weak surfaces, or incoherent lighting, describe those actual
defects and request revision. Different composition or exposure alone is not a
failure. Describe the reference mode and asset reuse accurately; independently
assembling a room does not mean authoring all its assets from scratch.

Current host evidence follows.
