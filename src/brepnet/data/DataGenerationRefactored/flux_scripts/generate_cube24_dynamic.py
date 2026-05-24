r"""
Generate FLUX.1-Kontext images from Blender renders with dynamic industrial-material prompts.

This script is an adapter layer on top of the local `external/generation` package.
It disables prompt
pre-encoding and updates the positive prompt dynamically per model.

Supported input layouts:
    1. Flat multi-view single-material:
       render-root/{model_id}/{view_idx}.png
    2. Legacy nested layout:
       render-root/{model_id}/{view_idx}/{material_idx}.png

The script recursively scans all PNGs under each model folder and mirrors the
same relative path under output-root.

Typical usage:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 RANK=0 \
    python3 ./generate_flux_kontext_dynamic_material_from_blender.py \
        --render-root /path/to/output_cube24_single_material \
        --output-root /path/to/output_flux_cube24_dynamic \
        --model-list-dir /path/to/render_lists \
        --rank 0 \
        --skip-existing
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

_EXTERNAL_ROOT = Path(__file__).resolve().parents[1] / "external"
if not (_EXTERNAL_ROOT / "generation").is_dir():
    raise SystemExit(f"Expected generation package at {_EXTERNAL_ROOT / 'generation'}")
sys.path.insert(0, str(_EXTERNAL_ROOT))

from generation.config import PHOTOREAL_POS_PROMPT, Img2BrepConfig  # noqa: E402
from generation.pipeline import Img2BrepPipeline  # noqa: E402


LOGGER = logging.getLogger("generate_flux_kontext_dynamic_material_from_blender")
MODEL_DIR_PATTERN = re.compile(r"^\d{8}$")

DEFAULT_BASE_PROMPT = PHOTOREAL_POS_PROMPT

DEFAULT_NEG_PROMPT = (
    Img2BrepConfig().neg_prompt
    + ", wood grain, wicker, woven fabric, knitted texture, leather, plush, ceramic craft surfaces, "
    + "toy-like plastic, fantasy environment, decorative props, cluttered background, crowded scene"
)

INDUSTRIAL_MATERIAL_PROFILES = (
    "aluminum alloy",
)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate FLUX.1-Kontext images from Blender renders with dynamic industrial-material prompts. "
            "Recursively scans PNGs under each model folder and preserves relative paths under output-root."
        )
    )
    parser.add_argument("--render-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-list-dir", type=Path, default=None)
    parser.add_argument("--model-list", type=Path, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--true-cfg-scale", type=float, default=1.0)
    parser.add_argument("--transformer-path", type=Path, default=None)
    parser.add_argument("--base-model-path", type=Path, default=None)
    parser.add_argument("--prompt-base", type=str, default=DEFAULT_BASE_PROMPT)
    parser.add_argument("--neg-prompt", type=str, default=DEFAULT_NEG_PROMPT)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args(argv)


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


def get_model_ids(args: argparse.Namespace) -> list[str]:
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


def discover_render_pngs(model_input_root: Path) -> list[Path]:
    if not model_input_root.is_dir():
        raise FileNotFoundError(f"Model input directory does not exist: {model_input_root}")

    rel_paths = [
        path.relative_to(model_input_root)
        for path in model_input_root.rglob("*.png")
        if path.is_file()
    ]
    return sorted(rel_paths, key=lambda p: p.as_posix())


def choose_material_profile(base_seed: int, model_id: str) -> tuple[int, str]:
    key = f"{base_seed}:{model_id}:material_profile".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    profile_index = int.from_bytes(digest[:4], "big") % len(INDUSTRIAL_MATERIAL_PROFILES)
    return profile_index, INDUSTRIAL_MATERIAL_PROFILES[profile_index]


def build_dynamic_prompt(base_prompt: str, material_profile: str) -> str:
    if material_profile.lower() in base_prompt.lower():
        return base_prompt

    anchor = "real-world CAD part that has been used"
    if anchor in base_prompt:
        return base_prompt.replace(
            anchor,
            f"real-world CAD part made of {material_profile} that has been used",
            1,
        )
    return f"{base_prompt.rstrip('.')} The part should look like {material_profile}."


def make_image_seed(base_seed: int, model_id: str, relative_path: Path) -> int:
    key = f"{base_seed}:{model_id}:{relative_path.as_posix()}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def load_rgb_image(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as img:
        return np.array(img.convert("RGB"), dtype=np.uint8)


def main(argv: list[str] | None = None) -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    configure_logging()
    args = parse_args(argv)

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

    config = Img2BrepConfig(
        transformer_path=args.transformer_path,
        base_model_path=args.base_model_path,
        pos_prompt=args.prompt_base,
        neg_prompt=args.neg_prompt,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        true_cfg_scale=args.true_cfg_scale,
        pre_encode_text=False,
        seed=args.seed,
    )
    LOGGER.info(
        "stage=config base_model_path=%s transformer_path=%s num_steps=%d guidance_scale=%.2f true_cfg_scale=%.2f pre_encode_text=%s",
        config.base_model_path,
        config.transformer_path,
        config.num_steps,
        config.guidance_scale,
        config.true_cfg_scale,
        config.pre_encode_text,
    )

    init_start = time.perf_counter()
    pipe = Img2BrepPipeline(config)
    LOGGER.info("stage=init_pipeline_done elapsed=%.2fs", time.perf_counter() - init_start)

    model_ids = get_model_ids(args)
    if args.max_models is not None:
        if args.max_models <= 0:
            raise ValueError("--max-models must be a positive integer.")
        model_ids = model_ids[: args.max_models]
        LOGGER.info("stage=limit_models max_models=%d selected=%d", args.max_models, len(model_ids))

    total_start = time.perf_counter()
    rendered_images = 0
    skipped_images = 0
    skipped_models = 0
    model_failures = 0

    for model_id in model_ids:
        model_start = time.perf_counter()
        model_input_root = args.render_root / model_id

        try:
            relative_pngs = discover_render_pngs(model_input_root)
        except Exception as exc:
            LOGGER.error("model=%s stage=discover_inputs error=%s", model_id, exc)
            model_failures += 1
            continue

        if not relative_pngs:
            LOGGER.error("model=%s stage=discover_inputs error=no_pngs_found root=%s", model_id, model_input_root)
            model_failures += 1
            continue

        if args.skip_existing:
            all_exist = all((args.output_root / model_id / rel_path).is_file() for rel_path in relative_pngs)
            if all_exist:
                skipped_models += 1
                skipped_images += len(relative_pngs)
                LOGGER.info(
                    "model=%s stage=skip_existing_model image_count=%d",
                    model_id,
                    len(relative_pngs),
                )
                continue

        material_profile_idx, material_profile = choose_material_profile(args.seed, model_id)
        dynamic_prompt = build_dynamic_prompt(args.prompt_base, material_profile)
        pipe.config.pos_prompt = dynamic_prompt
        pipe.config.neg_prompt = args.neg_prompt
        LOGGER.info(
            "model=%s stage=prompt_selected material_profile_idx=%d material_profile=%s",
            model_id,
            material_profile_idx,
            material_profile,
        )
        LOGGER.info(
            "model=%s stage=prompt_text prompt=%s",
            model_id,
            dynamic_prompt,
        )

        for rel_path in relative_pngs:
            input_path = model_input_root / rel_path
            output_path = args.output_root / model_id / rel_path

            if args.skip_existing and output_path.is_file():
                skipped_images += 1
                LOGGER.info(
                    "model=%s file=%s stage=skip_existing output=%s",
                    model_id,
                    rel_path.as_posix(),
                    output_path,
                )
                continue

            try:
                seed = make_image_seed(args.seed, model_id, rel_path)
                input_img = load_rgb_image(input_path)
                output_img = pipe(input_img, seed=seed)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_img.save(output_path)
                rendered_images += 1
                LOGGER.info(
                    "model=%s file=%s stage=generate_done seed=%d output=%s",
                    model_id,
                    rel_path.as_posix(),
                    seed,
                    output_path,
                )
            except Exception as exc:
                LOGGER.error(
                    "model=%s file=%s stage=generate error=%s traceback=%s",
                    model_id,
                    rel_path.as_posix(),
                    exc,
                    traceback.format_exc().strip().replace("\n", " | "),
                )
                continue

        LOGGER.info(
            "model=%s stage=model_done elapsed=%.2fs image_count=%d",
            model_id,
            time.perf_counter() - model_start,
            len(relative_pngs),
        )

    LOGGER.info(
        "stage=done models_total=%d skipped_models=%d model_failures=%d rendered_images=%d skipped_images=%d elapsed=%.2fs",
        len(model_ids),
        skipped_models,
        model_failures,
        rendered_images,
        skipped_images,
        time.perf_counter() - total_start,
    )


if __name__ == "__main__":
    main()
