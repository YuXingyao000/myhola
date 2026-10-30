"""Model ID list files: one 8-digit DeepCAD/ABC ID per line."""
from __future__ import annotations

import re
from pathlib import Path

MODEL_ID_PATTERN = re.compile(r"^\d{8}$")


def load_model_ids(list_path: Path) -> list[str]:
    """Load model IDs from a text file (one 8-digit ID per line).

    Skips empty lines and comments (lines starting with #).
    Deduplicates while preserving order.
    """
    if not list_path.is_file():
        raise FileNotFoundError(f"Model list does not exist: {list_path}")

    ids: list[str] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(list_path.read_text(encoding="utf-8").splitlines(), start=1):
        model_id = raw.strip()
        if not model_id or model_id.startswith("#"):
            continue
        if not MODEL_ID_PATTERN.match(model_id):
            raise ValueError(
                f"Invalid model id {model_id!r} in {list_path}:{line_number}; expected 8 digits."
            )
        if model_id in seen:
            continue
        seen.add(model_id)
        ids.append(model_id)
    return ids
