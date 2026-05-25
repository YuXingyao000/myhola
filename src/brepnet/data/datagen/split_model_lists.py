from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

from .compat import PATHS, RUNTIME


MODEL_ID_PATTERN = re.compile(r"^\d{8}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge model list files and split model ids into machine_{i}.txt files."
    )
    parser.add_argument(
        "model_lists",
        nargs="+",
        type=Path,
        help="One or more txt files containing 8-digit model ids.",
    )
    parser.add_argument("--output-dir", type=Path, default=PATHS.machine_list_dir)
    parser.add_argument("--num-machines", type=int, required=True)
    parser.add_argument(
        "--strategy",
        choices=("round-robin", "contiguous"),
        default="round-robin",
        help="round-robin balances mixed-cost ids better; contiguous preserves larger chunks.",
    )
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=RUNTIME.list_seed)
    parser.add_argument(
        "--check-mesh",
        action="store_true",
        help="Drop ids whose model folder does not contain --mesh-name or mesh.ply.",
    )
    parser.add_argument("--model-root", type=Path, default=PATHS.model_root)
    parser.add_argument("--mesh-name", type=str, default=PATHS.mesh_name)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_machines <= 0:
        raise ValueError("--num-machines must be positive.")

    model_ids = merge_lists(args.model_lists)
    if args.check_mesh:
        before = len(model_ids)
        model_ids = [
            model_id
            for model_id in model_ids
            if has_mesh(args.model_root, model_id, args.mesh_name)
        ]
        print(f"mesh_check kept={len(model_ids)} dropped={before - len(model_ids)}")

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(model_ids)

    machine_lists = split_ids(model_ids, args.num_machines, args.strategy)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_path = args.output_dir / "all.txt"
    all_path.write_text("\n".join(model_ids) + ("\n" if model_ids else ""), encoding="utf-8")
    print(f"wrote {all_path} count={len(model_ids)}")

    for machine_idx, ids in enumerate(machine_lists):
        path = args.output_dir / f"machine_{machine_idx}.txt"
        path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
        print(f"wrote {path} count={len(ids)}")


def merge_lists(paths: list[Path]) -> list[str]:
    merged: list[str] = []
    seen = set()
    for path in paths:
        for model_id in read_model_ids(path):
            if model_id in seen:
                continue
            seen.add(model_id)
            merged.append(model_id)
    return merged


def read_model_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Model list does not exist: {path}")

    ids = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        model_id = raw.strip()
        if not model_id or model_id.startswith("#"):
            continue
        if not MODEL_ID_PATTERN.match(model_id):
            raise ValueError(f"Invalid model id {model_id!r} in {path}:{line_number}; expected 8 digits.")
        ids.append(model_id)
    return ids


def split_ids(model_ids: list[str], num_machines: int, strategy: str) -> list[list[str]]:
    if strategy == "round-robin":
        output: list[list[str]] = [[] for _ in range(num_machines)]
        for idx, model_id in enumerate(model_ids):
            output[idx % num_machines].append(model_id)
        return output

    chunk_size = (len(model_ids) + num_machines - 1) // num_machines
    return [
        model_ids[machine_idx * chunk_size : (machine_idx + 1) * chunk_size]
        for machine_idx in range(num_machines)
    ]


def has_mesh(model_root: Path, model_id: str, mesh_name: str) -> bool:
    model_dir = model_root / model_id
    return (model_dir / mesh_name).is_file() or (model_dir / "mesh.ply").is_file()


if __name__ == "__main__":
    main()
