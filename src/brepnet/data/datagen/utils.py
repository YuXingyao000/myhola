"""Shared utilities for the data generation pipeline.

Extracted from run_flux.py, run_blender.py, and flux_scripts/generate_cube24_dynamic.py
to eliminate copy-paste duplication.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

MODEL_ID_PATTERN = re.compile(r"^\d{8}$")


def resolve_rank(explicit_rank: int | None) -> int | None:
    """Resolve GPU rank from explicit argument or environment variables.

    Checks RANK, LOCAL_RANK, SLURM_PROCID in that order.
    """
    import os

    if explicit_rank is not None:
        return explicit_rank

    for env_name in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        env_value = os.environ.get(env_name)
        if env_value is None or env_value == "":
            continue
        try:
            return int(env_value)
        except ValueError as exc:
            raise ValueError(
                f"Environment variable {env_name} must be an integer, got {env_value!r}."
            ) from exc
    return None


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


def find_model_ids(root: Path) -> list[str]:
    """Scan a directory for model ID subdirectories (8-digit names)."""
    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")
    return [
        path.name
        for path in sorted(root.iterdir())
        if path.is_dir() and MODEL_ID_PATTERN.match(path.name)
    ]


def split_for_gpus(model_ids: list[str], num_gpus: int, output_dir: Path) -> list[Path]:
    """Round-robin split model IDs into per-rank text files.

    Creates output_dir/rank_0.txt, rank_1.txt, etc.
    Returns list of file paths (one per rank).
    """
    if num_gpus <= 0:
        raise ValueError("num_gpus must be positive.")

    output_dir.mkdir(parents=True, exist_ok=True)

    rank_lists: list[list[str]] = [[] for _ in range(num_gpus)]
    for idx, model_id in enumerate(model_ids):
        rank_lists[idx % num_gpus].append(model_id)

    output_paths = []
    for rank, ids in enumerate(rank_lists):
        path = output_dir / f"rank_{rank}.txt"
        path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
        output_paths.append(path)
    return output_paths


def make_deterministic_seed(base_seed: int, *keys: str | int) -> int:
    """SHA256-based deterministic seed from a composite key.

    Useful for per-model / per-view reproducible randomness.
    """
    key = ":".join(str(k) for k in (base_seed, *keys)).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def resolve_model_list_arg(
    model_list: Path | None,
    machine_list_dir: Path | None,
    machine_index: int | None,
) -> Path | None:
    """Resolve a model list from CLI arguments (--model-list vs --machine-list-dir).

    Returns None if neither is specified (meaning: scan render root or use defaults).
    """
    if model_list is not None and machine_list_dir is not None:
        raise ValueError("Use either --model-list or --machine-list-dir, not both.")
    if model_list is not None:
        return model_list.resolve()
    if machine_list_dir is None:
        return None
    if machine_index is None:
        raise ValueError("--machine-list-dir requires --machine-index.")
    if machine_index < 0:
        raise ValueError("--machine-index must be non-negative.")
    return (machine_list_dir / f"machine_{machine_index}.txt").resolve()
