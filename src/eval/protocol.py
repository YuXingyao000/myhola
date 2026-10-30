"""Evaluation input/output protocol.

Evaluation is the last stage after post-processing.  The protocol here keeps
that boundary explicit: predictions live under `pred_root/sample_name`, ground
truth lives under `gt_root/sample_name`, and optional split lists select which
sample names are evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_CONDITION_RESULT = "eval_condition.npz"
LEGACY_CONDITION_RESULT = "eval.npz"
DEFAULT_ERROR_FILE = "eval_error.txt"


@dataclass(frozen=True)
class EvalSample:
    name: str
    pred_dir: Path
    gt_dir: Path | None = None


def read_split_list(split_list: str | Path | None) -> list[str]:
    if split_list is None or str(split_list) == "":
        return []
    path = Path(split_list)
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def collect_sample_names(
    pred_root: str | Path,
    split_list: str | Path | None = None,
    sample: str | None = None,
) -> list[str]:
    pred_root = Path(pred_root)
    if sample:
        return [sample]
    names = read_split_list(split_list)
    if not names:
        names = [path.name for path in pred_root.iterdir() if path.is_dir()]
    return sorted(names)


def iter_eval_samples(
    pred_root: str | Path,
    gt_root: str | Path | None = None,
    split_list: str | Path | None = None,
    sample: str | None = None,
) -> list[EvalSample]:
    pred_root = Path(pred_root)
    gt_root_path = Path(gt_root) if gt_root else None
    records: list[EvalSample] = []
    for name in collect_sample_names(pred_root, split_list=split_list, sample=sample):
        records.append(
            EvalSample(
                name=name,
                pred_dir=pred_root / name,
                gt_dir=(gt_root_path / name) if gt_root_path else None,
            )
        )
    return records


def first_existing_file(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def condition_result_path(sample: EvalSample, legacy: bool = False) -> Path:
    filename = LEGACY_CONDITION_RESULT if legacy else DEFAULT_CONDITION_RESULT
    return sample.pred_dir / filename


def error_path(sample: EvalSample) -> Path:
    return sample.pred_dir / DEFAULT_ERROR_FILE
