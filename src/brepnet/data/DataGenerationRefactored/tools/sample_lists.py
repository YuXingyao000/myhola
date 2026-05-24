import argparse
import numpy as np
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample train/val/test id lists from deduplicated DeepCAD split files."
    )
    parser.add_argument(
        "--source_list_dir",
        type=str,
        default="/mnt/d/python/src/brepnet/data/DataGeneration/list",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/d/python/src/brepnet/data/DataGeneration/test_list",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for sampling.")
    parser.add_argument("--num_train", type=int, default=750)
    parser.add_argument("--num_val", type=int, default=125)
    parser.add_argument("--num_test", type=int, default=125)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_list_root = Path(args.source_list_dir)
    output_list_root = Path(args.output_dir)
    output_list_root.mkdir(exist_ok=True, parents=True)

    train_list_path = input_list_root / "deduplicated_deepcad_training_7_30.txt"
    valid_list_path = input_list_root / "deduplicated_deepcad_validation_7_30.txt"
    test_list_path = input_list_root / "deduplicated_deepcad_testing_7_30.txt"

    with open(train_list_path, "r") as f:
        train_list = [line.strip() for line in f if line.strip()]
    with open(valid_list_path, "r") as f:
        valid_list = [line.strip() for line in f if line.strip()]
    with open(test_list_path, "r") as f:
        test_list = [line.strip() for line in f if line.strip()]

    rng = np.random.default_rng(args.seed)

    if len(train_list) < args.num_train:
        raise ValueError(
            f"train_list contains only {len(train_list)} items, but {args.num_train} are required."
        )
    selected_train = rng.choice(train_list, args.num_train, replace=False)
    output_train_list_path = output_list_root / "train.txt"
    with open(output_train_list_path, "w") as f:
        for item in selected_train:
            f.write(item + "\n")

    if len(valid_list) < args.num_val:
        raise ValueError(
            f"valid_list contains only {len(valid_list)} items, but {args.num_val} are required."
        )
    selected_valid = rng.choice(valid_list, args.num_val, replace=False)
    output_valid_list_path = output_list_root / "val.txt"
    with open(output_valid_list_path, "w") as f:
        for item in selected_valid:
            f.write(item + "\n")

    if len(test_list) < args.num_test:
        raise ValueError(
            f"test_list contains only {len(test_list)} items, but {args.num_test} are required."
        )
    selected_test = rng.choice(test_list, args.num_test, replace=False)
    output_test_list_path = output_list_root / "test.txt"
    with open(output_test_list_path, "w") as f:
        for item in selected_test:
            f.write(item + "\n")

    print(
        f"wrote train/val/test -> {output_list_root} "
        f"({args.num_train}+{args.num_val}+{args.num_test} ids, seed={args.seed})"
    )
