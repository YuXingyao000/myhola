---
name: spec-research-data-code
description: Turn vague research data-processing, dataset parsing, preprocessing, evaluation, metric, batch analysis, and experiment-aggregation requests into a compact execution spec before coding. Use when Codex needs to write or review scientific data workflow code and the inputs, metrics, output schema, or edge-case behavior are still underspecified.
---

# Spec Research Data Code

## Goal

Convert a rough research data task into a short execution contract before writing code. Use the contract to make assumptions explicit, define edge-case behavior up front, and keep the first implementation minimal and reproducible.

## Workflow

1. Extract a task card with `Task`, `Known facts`, `Hard constraints`, and `Do not`.
2. Draft assumptions in three groups: data assumptions, task-boundary assumptions, and implementation assumptions.
3. Mark uncertain items as `Needs confirmation`.
4. Freeze the execution spec before coding.
5. Red-team the spec for ambiguity, reproducibility risk, and missing debug artifacts.
6. Compress the frozen spec into a short coding prompt.
7. Write code only after the spec is stable.

## Frozen Spec

Write exactly these sections:

1. `Task`
2. `Assumptions`
3. `Inputs`
4. `Outputs`
5. `Processing Protocol`
6. `Metrics`
7. `Decision Rule`
8. `Edge Cases`
9. `Validation / Sanity Checks`
10. `Implementation Constraints`
11. `Deliverables`

## Decision Rules

- Treat repository context as evidence, not as the task contract.
- Separate observed facts, explicit assumptions, and uncertain inferences.
- Separate metric definitions from the rule that chooses the final result.
- Define behavior for empty inputs, missing fields, invalid shapes, NaN or Inf, corrupted files, zero denominators, and failed samples in batch mode.
- Prefer a minimal first version: single-process, low dependency count, deterministic behavior when practical, and visible debug artifacts.

## Output Standard

Use the full spec when the task includes batch processing, evaluation metrics, candidate ranking, file export, multiple input sources, or reproducibility constraints.

Use the reduced form only for small one-off transformations. The reduced form must still include `Task`, `Assumptions`, `Inputs`, `Outputs`, `Protocol`, and `Edge Cases`.

## Resources

Read [references/spec-templates.md](references/spec-templates.md) when you need reusable templates for the task card, frozen spec, red-team review, or compressed coding prompt.
