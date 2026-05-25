import argparse
import hashlib
import logging
import os
import re
import time
import traceback
from math import ceil
from pathlib import Path

import numpy as np
import ray
import torch
from PIL import Image

from .config import PHOTOREAL_POS_PROMPT, Img2BrepConfig
from .pipeline import Img2BrepPipeline

# -----------------------------
# Legacy Ray + sketch.png layout from the original Img2Brep generation package.
# -----------------------------
IMG_ROOT = Path("/mnt/d/data/abc_v2_natural_AA_Sketch2")
OUT_DIR = Path("/mnt/d/data/abc_v2_natural_AA_Sketch2_generated")
NUM_SERVER = 0
TOTAL_NUM_SERVER = 5
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

LOGGER = logging.getLogger("generation.generate")
MODEL_DIR_PATTERN = re.compile(r"^\d{8}$")


def collect_server_folders(img_root: Path, num_server: int, total_num_server: int) -> list[Path]:
    all_folders = sorted([p for p in img_root.iterdir() if p.is_dir()])
    num_folder_per_server = len(all_folders) // total_num_server
    folder_start = num_server * num_folder_per_server

    if num_server == total_num_server - 1:
        return all_folders[folder_start:]
    return all_folders[folder_start : folder_start + num_folder_per_server]


def chunk_list(xs: list[Path], n: int) -> list[list[Path]]:
    """Split xs into n nearly-equal chunks (n must be >= 1)."""
    if n <= 1:
        return [xs]
    k = ceil(len(xs) / n)
    return [xs[i * k : (i + 1) * k] for i in range(n)]


def load_img_from_npz(config, img_folder: Path) -> np.ndarray:
    data_file = img_folder / "data.npz"

    arr = np.load(data_file)["svr_imgs"]

    hashseed = hash(img_folder.stem) % (2**32)
    rng = np.random.default_rng(seed=hashseed + config.seed)
    idx = rng.integers(64, 128)
    img_data = arr[idx]
    return img_data


def load_img(img_folder: Path) -> np.ndarray:
    img_file = img_folder / "sketch.png"
    img_data = np.array(Image.open(img_file).convert("RGB"))
    return img_data


@ray.remote(num_gpus=1)
def generate(config_dict: dict, folder_chunk: list[str], chunk_id: int):
    """
    Pass strings/paths as strings to reduce serialization surprises.
    Build config inside worker to avoid pickling issues.
    """
    try:
        config = Img2BrepConfig()
        for k, v in config_dict.items():
            setattr(config, k, v)

        pipe = Img2BrepPipeline(config)

        out_root = Path(OUT_DIR.as_posix() + f"_{NUM_SERVER}")

        for folder_str in folder_chunk:
            img_folder = Path(folder_str)
            img_data = load_img(img_folder)

            # random_prompt = build_prompt(config.prompt, seed=hashseed)
            # config.prompt = random_prompt
            generated_img: Image.Image = pipe(img_data)

            output_path = out_root / img_folder.stem / "natural.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            generated_img.save(output_path)

        return (True, chunk_id, "")

    except Exception:
        return (False, chunk_id, traceback.format_exc())


# -----------------------------
# Blender render -> FLUX (DataGeneration pipeline step 04)
# Input:  render-root/{model_id}/{view}/{material_idx}.png
# Output: output-root/{model_id}/{view}/{material_idx}.png
# -----------------------------


def resolve_rank(explicit_rank: int | None) -> int | None:
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


def load_model_ids_from_list(list_path: Path) -> list[str]:
    if not list_path.is_file():
        raise FileNotFoundError(f"Model list does not exist: {list_path}")

    model_ids = []
    seen_ids = set()
    for line_number, raw_line in enumerate(list_path.read_text(encoding="utf-8").splitlines(), start=1):
        model_id = raw_line.strip()
        if not model_id or model_id.startswith("#"):
            continue
        if not MODEL_DIR_PATTERN.match(model_id):
            raise ValueError(
                f"Invalid model id {model_id!r} in {list_path} line {line_number}; expected 8 digits."
            )
        if model_id in seen_ids:
            continue
        seen_ids.add(model_id)
        model_ids.append(model_id)
    return model_ids


def find_model_ids(render_root: Path) -> list[str]:
    if not render_root.exists():
        raise FileNotFoundError(f"Render root does not exist: {render_root}")
    return [
        path.name
        for path in sorted(render_root.iterdir())
        if path.is_dir() and MODEL_DIR_PATTERN.match(path.name)
    ]


def get_model_ids_for_blender(args: argparse.Namespace) -> list[str]:
    if args.model_list is not None and args.model_list_dir is not None:
        raise ValueError("Use either --model-list or --model-list-dir, not both.")

    if args.model_list is not None:
        model_ids = load_model_ids_from_list(args.model_list)
        LOGGER.info("stage=load_model_list source=%s count=%d", args.model_list, len(model_ids))
        return model_ids

    if args.model_list_dir is not None:
        if args.rank is None:
            raise ValueError(
                "--model-list-dir was provided, but no rank was found. "
                "Pass --rank or set RANK/LOCAL_RANK/SLURM_PROCID."
            )
        rank_list = args.model_list_dir / f"rank_{args.rank}.txt"
        model_ids = load_model_ids_from_list(rank_list)
        LOGGER.info(
            "stage=load_rank_list rank=%d source=%s count=%d",
            args.rank,
            rank_list,
            len(model_ids),
        )
        return model_ids

    model_ids = find_model_ids(args.render_root)
    LOGGER.info("stage=scan_render_root count=%d", len(model_ids))
    return model_ids


def make_image_seed(base_seed: int, model_id: str, material_idx: int) -> int:
    key = f"{base_seed}:{model_id}:{material_idx}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def load_rgb_image(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as img:
        return np.array(img.convert("RGB"), dtype=np.uint8)


def main_blender_batch(args: argparse.Namespace) -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    args.render_root = args.render_root.resolve()
    args.output_root = args.output_root.resolve()
    if args.model_list is not None:
        args.model_list = args.model_list.resolve()
    if args.model_list_dir is not None:
        args.model_list_dir = args.model_list_dir.resolve()

    args.rank = resolve_rank(args.rank)

    transformer_path = args.transformer_path or Img2BrepConfig.transformer_path
    base_model_path = args.base_model_path or Img2BrepConfig.base_model_path
    args.transformer_path = Path(transformer_path).resolve()
    args.base_model_path = Path(base_model_path).resolve()

    LOGGER.info("stage=start render_root=%s", args.render_root)
    LOGGER.info("stage=start output_root=%s", args.output_root)
    if args.model_list is not None:
        LOGGER.info("stage=start model_list=%s", args.model_list)
    if args.model_list_dir is not None:
        LOGGER.info("stage=start model_list_dir=%s rank=%s", args.model_list_dir, args.rank)

    pos_prompt = args.prompt or PHOTOREAL_POS_PROMPT
    config = Img2BrepConfig(
        transformer_path=args.transformer_path,
        base_model_path=args.base_model_path,
        pos_prompt=pos_prompt,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        true_cfg_scale=args.true_cfg_scale,
        pre_encode_text=True,
    )
    LOGGER.info(
        "stage=config base_model_path=%s transformer_path=%s num_steps=%d guidance_scale=%.2f",
        config.base_model_path,
        config.transformer_path,
        config.num_steps,
        config.guidance_scale,
    )

    init_start = time.perf_counter()
    pipe = Img2BrepPipeline(config)
    LOGGER.info("stage=init_pipeline_done elapsed=%.2fs", time.perf_counter() - init_start)

    model_ids = get_model_ids_for_blender(args)
    if args.max_models is not None:
        if args.max_models <= 0:
            raise ValueError("--max-models must be a positive integer.")
        model_ids = model_ids[: args.max_models]
        LOGGER.info("stage=limit_models max_models=%d selected=%d", args.max_models, len(model_ids))

    total_start = time.perf_counter()
    rendered_images = 0
    skipped_images = 0
    model_failures = 0
    skipped_models = 0

    for model_id in model_ids:
        model_start = time.perf_counter()
        input_dir = args.render_root / model_id / str(args.view_index)
        if not input_dir.is_dir():
            LOGGER.error("model=%s stage=check_input error=missing input dir %s", model_id, input_dir)
            model_failures += 1
            continue

        if args.skip_existing:
            all_exist = all(
                (args.output_root / model_id / str(args.view_index) / f"{mi}.png").is_file()
                for mi in range(args.material_count)
            )
            if all_exist:
                skipped_models += 1
                skipped_images += args.material_count
                LOGGER.info(
                    "model=%s stage=skip_existing_model (all %d PNGs present)", model_id, args.material_count
                )
                continue

        for material_idx in range(args.material_count):
            input_path = input_dir / f"{material_idx}.png"
            output_path = args.output_root / model_id / str(args.view_index) / f"{material_idx}.png"

            if not input_path.is_file():
                LOGGER.error(
                    "model=%s material=%d stage=check_input error=missing image %s",
                    model_id,
                    material_idx,
                    input_path,
                )
                continue

            if args.skip_existing and output_path.is_file():
                skipped_images += 1
                LOGGER.info(
                    "model=%s material=%d stage=skip_existing output=%s",
                    model_id,
                    material_idx,
                    output_path,
                )
                continue

            try:
                seed = make_image_seed(args.seed, model_id, material_idx)
                input_img = load_rgb_image(input_path)
                output_img = pipe(input_img, seed=seed)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_img.save(output_path)
                rendered_images += 1
                LOGGER.info(
                    "model=%s material=%d stage=generate_done seed=%d output=%s",
                    model_id,
                    material_idx,
                    seed,
                    output_path,
                )
            except Exception as exc:
                LOGGER.error(
                    "model=%s material=%d stage=generate error=%s traceback=%s",
                    model_id,
                    material_idx,
                    exc,
                    traceback.format_exc().strip().replace("\n", " | "),
                )
                continue

        LOGGER.info(
            "model=%s stage=model_done elapsed=%.2fs",
            model_id,
            time.perf_counter() - model_start,
        )

    LOGGER.info(
        "stage=done models_total=%d skipped_models=%d model_failures=%d rendered_images=%d "
        "skipped_images=%d elapsed=%.2fs",
        len(model_ids),
        skipped_models,
        model_failures,
        rendered_images,
        skipped_images,
        time.perf_counter() - total_start,
    )


def main_legacy_sketch_ray() -> None:
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    assert num_gpus > 0, "No CUDA GPUs visible."
    ray.init(num_gpus=num_gpus)

    config = Img2BrepConfig()
    config_dict: dict = {"pre_encode_text": True}

    server_folders = collect_server_folders(IMG_ROOT, NUM_SERVER, TOTAL_NUM_SERVER)
    chunks = chunk_list(server_folders, num_gpus)
    chunks = [[p.as_posix() for p in chunk] for chunk in chunks]

    refs = [generate.remote(config_dict, chunks[i], i) for i in range(len(chunks))]
    outs = ray.get(refs)

    for ok, chunk_id, err in outs:
        if not ok:
            print(f"[ERROR] chunk {chunk_id} failed:\n{err}")
        else:
            print(f"[OK] chunk {chunk_id} done")


def build_top_level_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Img2Brep FLUX generation. Without --render-root: legacy Ray job over sketch.png folders. "
            "With --render-root: single-GPU batch over Blender outputs "
            "(render-root/{model_id}/{view}/{material}.png)."
        )
    )
    parser.add_argument(
        "--render-root",
        type=Path,
        default=None,
        help="Blender output root; if set, runs batch mode instead of legacy Ray.",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--model-list-dir", type=Path, default=None)
    parser.add_argument("--model-list", type=Path, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--view-index", type=int, default=0)
    parser.add_argument("--material-count", type=int, default=8)
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--true-cfg-scale", type=float, default=1.0)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--transformer-path", type=Path, default=None)
    parser.add_argument("--base-model-path", type=Path, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_top_level_parser()
    args = parser.parse_args(argv)

    if args.render_root is not None:
        if args.output_root is None:
            parser.error("--output-root is required when --render-root is set.")
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
        main_blender_batch(args)
        return

    blender_only = (
        args.output_root is not None
        or args.model_list_dir is not None
        or args.model_list is not None
        or args.skip_existing
        or args.transformer_path is not None
        or args.base_model_path is not None
        or args.prompt is not None
        or args.max_models is not None
    )
    if blender_only:
        parser.error("Blender batch options require --render-root.")

    main_legacy_sketch_ray()


if __name__ == "__main__":
    main()
