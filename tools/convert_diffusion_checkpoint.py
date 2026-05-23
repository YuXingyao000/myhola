"""Convert legacy diffusion checkpoint keys to the modular Diffusion layout.

The script keeps tensor values untouched and only rewrites state_dict keys.
It drops strategy, feature-mapper, distillation, and alignment-only weights
that no longer exist in the simplified model.

See tools/checkpoint_key_mapping.md for the explicit prefix mapping table.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import torch


DROP_PREFIXES = (
    "strategy.",
    "_feature_domain_mapper.",
    "cad_align_proj.",
    "img_align_proj.",
    "cad_align_head.",
    "img_align_head.",
    "img_adapters.",
    "cond_attn.",
    "learned_uncond_emb",
    "learned_svr_emb",
    "learned_mvr_emb",
    "learned_sketch_emb",
    "learned_pc_emb",
    "learned_txt_emb",
)

PREFIX_MAP = (
    ("p_embed.", "denoiser.input_projection."),
    ("net1.", "denoiser.backbone."),
    ("fc_out.", "denoiser.output_projection."),
    ("time_embed.", "denoiser.time_embedding."),
    ("cross_attn_pre_proj.", "denoiser.condition_fuser.query_projection."),
    ("cross_attn_add_cond.", "denoiser.condition_fuser.decoder."),
    ("cross_attn_post_proj.", "denoiser.condition_fuser.output_projection."),
    ("classifier.", "face_padder.validity_head."),
    ("padding.mask_head.", "face_padder.validity_head."),
    ("latent_codec.autoencoder.", "autoencoder."),
    ("ae_model.", "autoencoder."),
    ("condition_extractor.", "condition_encoder."),
    ("img_model.", "condition_encoder.image_encoder.img_model."),
    ("img_fc.", "condition_encoder.image_encoder.projection."),
    ("camera_embedding.", "condition_encoder.camera_embedding."),
    ("point_model.", "condition_encoder.point_encoder."),
    ("txt_model.", "condition_encoder.text_encoder.text_model."),
    ("txt_fc.", "condition_encoder.text_encoder.projection."),
)


def strip_lightning_prefix(key: str) -> str:
    if key.startswith("model."):
        return key[len("model."):]
    return key


def convert_key(key: str) -> tuple[str | None, str]:
    key = strip_lightning_prefix(key)
    if key.startswith(DROP_PREFIXES):
        return None, "dropped"

    for old_prefix, new_prefix in PREFIX_MAP:
        if key.startswith(old_prefix):
            return new_prefix + key[len(old_prefix):], old_prefix.rstrip(".")

    return key, "unchanged"


def convert_state_dict(
    state_dict: dict[str, Any],
    *,
    lightning_prefix: bool,
) -> tuple[dict[str, Any], Counter[str], list[tuple[str, str]]]:
    converted: dict[str, Any] = {}
    counts: Counter[str] = Counter()
    collisions: list[tuple[str, str]] = []

    for old_key, value in state_dict.items():
        new_key, bucket = convert_key(old_key)
        counts[bucket] += 1
        if new_key is None:
            continue

        if lightning_prefix:
            new_key = "model." + new_key

        if new_key in converted:
            collisions.append((old_key, new_key))
        converted[new_key] = value

    return converted, counts, collisions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Legacy checkpoint path")
    parser.add_argument("output", type=Path, help="Converted checkpoint path")
    parser.add_argument(
        "--no-lightning-prefix",
        action="store_true",
        help="Write raw module keys instead of Lightning 'model.' keys",
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    has_state_dict = isinstance(checkpoint, dict) and "state_dict" in checkpoint
    state_dict = checkpoint["state_dict"] if has_state_dict else checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected a state_dict-like checkpoint, got {type(state_dict)!r}")

    converted, counts, collisions = convert_state_dict(
        state_dict,
        lightning_prefix=not args.no_lightning_prefix,
    )
    if collisions:
        details = "\n".join(f"{old_key} -> {new_key}" for old_key, new_key in collisions[:20])
        raise RuntimeError(f"Key collisions during conversion:\n{details}")

    output_checkpoint = checkpoint
    if has_state_dict:
        output_checkpoint = dict(checkpoint)
        output_checkpoint["state_dict"] = converted
    else:
        output_checkpoint = converted

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, args.output)

    print(f"input: {args.input}")
    print(f"output: {args.output}")
    print(f"kept: {len(converted)}")
    for bucket, count in sorted(counts.items()):
        print(f"{bucket}: {count}")


if __name__ == "__main__":
    main()
