"""
Read model IDs from test_list/{train,test,val}.txt, deduplicate, split evenly
across N GPUs, and write rank_0.txt ... rank_{N-1}.txt plus all.txt.

The output is directly consumable by the Blender scripts via --model-list-dir.

Usage:
    python prepare_render_lists.py --list-dir ./test_list --output-dir ./render_lists --num-gpus 8
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

MODEL_ID_PATTERN = re.compile(r"^\d{8}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge split lists and partition into per-GPU rank files."
    )
    parser.add_argument(
        "--list-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "test_list",
        help="Directory containing {train,test,val}.txt (default: ./test_list)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "render_lists",
        help="Where to write rank_*.txt and all.txt (default: ./render_lists)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=8,
        help="Number of GPU ranks to split across (default: 8)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle model IDs before splitting (useful for load balancing).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --shuffle is set (default: 42)",
    )
    parser.add_argument(
        "--check-exists",
        action="store_true",
        help="Verify that DATA_ROOT/{model_id}/mesh.ply exists for every ID.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/mnt/d/data/deepcad_v6"),
        help="Model data root, only used with --check-exists (default: /mnt/d/data/deepcad_v6)",
    )
    return parser.parse_args()


def read_ids_from_file(path: Path) -> list[str]:
    """Return validated 8-digit model IDs from a text file."""
    if not path.is_file():
        raise FileNotFoundError(f"List file not found: {path}")
    ids: list[str] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        mid = raw.strip()
        if not mid:
            continue
        if not MODEL_ID_PATTERN.match(mid):
            raise ValueError(f"Invalid model ID {mid!r} at {path}:{lineno}")
        ids.append(mid)
    return ids


def merge_and_deduplicate(id_lists: list[list[str]]) -> list[str]:
    """Merge multiple ID lists, preserving first-seen order and warning on duplicates."""
    seen: set[str] = set()
    merged: list[str] = []
    dup_count = 0
    for ids in id_lists:
        for mid in ids:
            if mid in seen:
                dup_count += 1
                continue
            seen.add(mid)
            merged.append(mid)
    if dup_count:
        print(f"[warn] removed {dup_count} duplicate IDs across input files")
    return merged


def split_into_ranks(ids: list[str], num_ranks: int) -> list[list[str]]:
    """Round-robin partition so rank sizes differ by at most 1."""
    ranks: list[list[str]] = [[] for _ in range(num_ranks)]
    for i, mid in enumerate(ids):
        ranks[i % num_ranks].append(mid)
    return ranks


def main() -> None:
    args = parse_args()

    if args.num_gpus < 1:
        print("[error] --num-gpus must be >= 1", file=sys.stderr)
        sys.exit(1)

    list_dir: Path = args.list_dir.resolve()
    output_dir: Path = args.output_dir.resolve()

    list_files = sorted(list_dir.glob("*.txt"))
    if not list_files:
        print(f"[error] no .txt files found in {list_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"list_dir    = {list_dir}")
    print(f"output_dir  = {output_dir}")
    print(f"num_gpus    = {args.num_gpus}")
    print(f"list_files  = {[f.name for f in list_files]}")

    all_id_lists = []
    for lf in list_files:
        ids = read_ids_from_file(lf)
        print(f"  {lf.name}: {len(ids)} IDs")
        all_id_lists.append(ids)

    merged = merge_and_deduplicate(all_id_lists)
    print(f"total unique IDs: {len(merged)}")

    if args.check_exists:
        data_root: Path = args.data_root.resolve()
        missing = [mid for mid in merged if not (data_root / mid / "mesh.ply").is_file()]
        if missing:
            print(f"[warn] {len(missing)} model dirs missing mesh.ply under {data_root}:")
            for m in missing[:20]:
                print(f"  {m}")
            if len(missing) > 20:
                print(f"  ... and {len(missing) - 20} more")
            merged = [mid for mid in merged if mid not in set(missing)]
            print(f"remaining valid IDs: {len(merged)}")

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(merged)
        print(f"shuffled with seed={args.seed}")

    ranks = split_into_ranks(merged, args.num_gpus)

    output_dir.mkdir(parents=True, exist_ok=True)

    all_path = output_dir / "all.txt"
    all_path.write_text("\n".join(merged) + "\n", encoding="utf-8")
    print(f"wrote {all_path}  ({len(merged)} IDs)")

    for rank_idx, rank_ids in enumerate(ranks):
        rank_path = output_dir / f"rank_{rank_idx}.txt"
        rank_path.write_text("\n".join(rank_ids) + "\n", encoding="utf-8")
        print(f"wrote {rank_path}  ({len(rank_ids)} IDs)")

    print("done")


if __name__ == "__main__":
    main()
