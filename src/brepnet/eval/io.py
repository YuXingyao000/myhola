"""Small IO helpers for evaluation results."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_npz_result(path: str | Path, results: dict[str, Any]) -> None:
    path = ensure_parent(path)
    np.savez_compressed(path, results=results, allow_pickle=True)


def write_text(path: str | Path, text: str) -> None:
    path = ensure_parent(path)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: str | Path, payload: Any) -> None:
    path = ensure_parent(path)
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def flatten_scalar_result(prefix: str, result: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"prefix": prefix}
    for key, value in result.items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            row[key] = value
    return row


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = ensure_parent(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    if "prefix" in fieldnames:
        fieldnames.remove("prefix")
        fieldnames.insert(0, "prefix")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
