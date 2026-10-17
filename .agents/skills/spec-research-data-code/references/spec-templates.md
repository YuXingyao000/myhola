# Spec Templates

Use these templates after the skill triggers. Keep outputs short and contract-focused.

## Task Card

```md
Task:
[one sentence]

Known facts:
- ...
- ...

Hard constraints:
- ...
- ...

Do not:
- ...
- ...
```

## Assumptions Draft

```md
Assumptions

Data assumptions:
- ...

Task-boundary assumptions:
- ...

Implementation assumptions:
- ...
```

Rules:

- Include only assumptions that affect correctness, scope, or reproducibility.
- Mark uncertain items as `Needs confirmation`.
- Keep bullets short.

## Frozen Spec

```md
Task
[short paragraph]

Assumptions
- ...

Inputs
- type:
- required fields:
- dtype and shape:
- directory layout:

Outputs
- per-sample outputs:
- aggregate outputs:
- debug artifacts:
- file formats and naming:

Processing Protocol
1. ...
2. ...

Metrics
- name:
  meaning:
  computation rule:
  parameters and units:

Decision Rule
- ...

Edge Cases
- empty inputs:
- missing fields:
- invalid shapes:
- NaN / Inf:
- corrupted files:
- zero denominators:
- failed samples in batch mode:

Validation / Sanity Checks
- ...

Implementation Constraints
- language:
- allowed libraries:
- CPU or GPU expectations:
- parallelism policy:
- determinism requirement:

Deliverables
- ...
```

## Red-Team Review

```md
Spec risks:
- ...

Missing decisions:
- ...

Recommended fixes before coding:
- ...
```

Check for:

- hidden assumptions
- incompatible implementation choices
- missing edge-case behavior
- non-reproducible steps
- missing debug artifacts

## Compression Prompt

```md
Compress the frozen spec into a coding prompt.

Rules:
- keep only implementation-relevant content
- preserve contracts, edge-case behavior, outputs, and constraints
- remove motivation and repeated explanation
- keep the prompt reusable
```

## Reduced Form

Use this only for small one-off transformations with no ranking, export workflow, or reproducibility complexity.

```md
Task

Assumptions

Inputs

Outputs

Protocol

Edge Cases
```

## Repository Context Labels

Use these labels when repo context matters:

- `Observed in repository:`
- `Assumed for this task:`
- `Needs confirmation:`
