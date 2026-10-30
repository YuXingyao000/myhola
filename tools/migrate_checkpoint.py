"""Migrate a Lightning checkpoint across repository layout and module renames.

    python tools/migrate_checkpoint.py OLD.ckpt NEW.ckpt [--dry-run]

Every entry of MIGRATIONS is applied once, in order, and recorded in
checkpoint["hola_migrations"]. Tensor values are never changed:
  strings   ordered (old, new) substring replacements in every non-tensor
            field (hyper_parameters, callbacks, loops), e.g. dataset.name
            and paths that resume compares against the current config;
  prefixes  ordered (old, new) state_dict key prefix renames. Optimizer state
            is indexed by parameter order, not by name, so a pure rename keeps
            it valid; changing parameter registration order does not.
The output is a new file; the source checkpoint is never overwritten.
Add a new version for every rename and list it in AGENTS.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

LAYOUT_20261030 = [
    # Run outputs moved out of the repository.
    ("/mnt/d/python/checkpoints/", "/mnt/d/hola_runs/"),
    # Module paths (dotted), most specific first.
    ("src.brepnet.data.spatial_photo_dataset", "src.datasets.spatial_photo"),
    ("src.brepnet.data.photo_preference_dataset", "src.datasets.photo_preference"),
    ("src.brepnet.data.preference_dataset", "src.datasets.preference"),
    ("src.brepnet.data.photo_dataset", "src.datasets.photo"),
    ("src.brepnet.dataset", "src.datasets.brep"),
    ("src.brepnet.diffusion", "src.models.diffusion"),
    ("src.brepnet.models", "src.models"),
    ("src.brepnet.post", "src.postprocess"),
    ("src.brepnet.eval.run", "src.evaluate"),
    ("src.brepnet.eval", "src.eval"),
    ("src.brepnet.run_management", "src.common.run_management"),
    ("src.brepnet.prediction_export", "src.common.prediction_export"),
    ("src.brepnet.train", "src.train"),
    ("src.brepnet.inference", "src.inference"),
    # Repository-relative file paths.
    ("src/brepnet/data/list/", "src/preprocess/splits/"),
]

MIGRATIONS = [
    {"version": "2026-10-30-layout", "strings": LAYOUT_20261030, "prefixes": []},
]


def rewrite_strings(value, pairs, changes: list, path: str = ""):
    if isinstance(value, str):
        new = value
        for old, replacement in pairs:
            new = new.replace(old, replacement)
        if new != value:
            changes.append((path, value, new))
        return new
    if isinstance(value, dict):
        return {key: rewrite_strings(item, pairs, changes, f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, list):
        return [rewrite_strings(item, pairs, changes, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, tuple):
        return tuple(rewrite_strings(item, pairs, changes, f"{path}[{index}]") for index, item in enumerate(value))
    return value


def rename_prefixes(state_dict: dict, pairs) -> tuple[dict, int]:
    renamed, count = {}, 0
    for key, tensor in state_dict.items():
        new = key
        for old, replacement in pairs:
            if new.startswith(old):
                new = replacement + new[len(old):]
        count += new != key
        if new in renamed:
            raise ValueError(f"Prefix rename collides on {new!r}")
        renamed[new] = tensor
    return renamed, count


def migrate(checkpoint: dict) -> list[str]:
    applied = list(checkpoint.get("hola_migrations", []))
    report = []
    for migration in MIGRATIONS:
        if migration["version"] in applied:
            report.append(f"{migration['version']}: already applied")
            continue
        changes = []
        for field in list(checkpoint):
            if field not in ("state_dict", "optimizer_states"):
                checkpoint[field] = rewrite_strings(checkpoint[field], migration["strings"], changes, field)
        checkpoint["state_dict"], renamed = rename_prefixes(checkpoint["state_dict"], migration["prefixes"])
        applied.append(migration["version"])
        report.append(f"{migration['version']}: {len(changes)} strings, {renamed} state_dict keys")
        report.extend(f"  {path}: {old!r} -> {new!r}" for path, old, new in changes)
    checkpoint["hola_migrations"] = applied
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.target.exists():
        parser.error(f"Refusing to overwrite {args.target}")
    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False, mmap=True)
    print("\n".join(migrate(checkpoint)))
    if not args.dry_run:
        args.target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, args.target)
        print(f"wrote {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
