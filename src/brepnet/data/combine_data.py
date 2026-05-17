import argparse
from pathlib import Path

import numpy as np
import ray


REQUIRED_IMGS_KEYS = ("svr_imgs", "sketch_imgs")
REQUIRED_SKETCH_KEYS = ("natural_img", "sketch_img")


def load_npz_arrays(npz_path: Path, required_keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(npz_path) as data:
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            raise KeyError(f"{npz_path} missing keys: {missing_keys}")
        return {key: data[key] for key in required_keys}


def save_combined_npz(output_path: Path, payload: dict[str, np.ndarray]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)


@ray.remote
def process_folder(folder: Path, outdir: Path, output_filename: str) -> None:
    imgs_path = folder / "imgs.npz"
    sketch_path = folder / "sketch_and_natural.npz"

    if not imgs_path.exists():
        print(f"skip {folder.name} (imgs.npz missing)")
        return
    if not sketch_path.exists():
        print(f"skip {folder.name} (sketch_and_natural.npz missing)")
        return

    try:
        imgs_data = load_npz_arrays(imgs_path, REQUIRED_IMGS_KEYS)
        sketch_data = load_npz_arrays(sketch_path, REQUIRED_SKETCH_KEYS)

        payload = {
            "svr_imgs": imgs_data["svr_imgs"],
            "sketch_imgs": imgs_data["sketch_imgs"],
            "natural_img": sketch_data["natural_img"],
            "sketch_img": sketch_data["sketch_img"],
        }

        source_output_path = folder / output_filename
        merged_output_path = outdir / folder.name / output_filename

        save_combined_npz(source_output_path, payload)
        save_combined_npz(merged_output_path, payload)
        print(f"saved {source_output_path} and {merged_output_path}")
    except Exception as exc:
        print(f"error processing {folder.name}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("/mnt/d/data/deepcad_v6_cond"),
        help="Root folder that contains per-shape condition folders.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Collector folder under /mnt/d/data for the merged npz files.",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="combined_imgs.npz",
        help="Filename for the merged npz written to both source and collector folders.",
    )

    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    ray.init()

    tasks = []
    for folder in sorted(args.data_root.iterdir()):
        if not folder.is_dir():
            continue
        tasks.append(process_folder.remote(folder, args.outdir, args.output_filename))
        # process_folder(folder, args.outdir, args.output_filename)

    ray.get(tasks)


if __name__ == "__main__":
    main()
