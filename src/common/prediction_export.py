"""Per-object exports, without silently replicating a whole VAE batch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def save_item_predictions(prefixes, items: list[dict], output_dir: Path) -> None:
    if len(prefixes) != len(items):
        raise ValueError(f"Prediction count {len(items)} differs from item count {len(prefixes)}.")
    for prefix, item in zip(prefixes, items):
        name = str(prefix)
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"Invalid per-object export identifier: {name!r}")
        arrays = {}
        for key, value in item.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            value = np.asarray(value)
            if value.dtype.hasobject:
                raise TypeError(f"Prediction field {key!r} cannot require pickle/object arrays.")
            arrays[key] = value
        item_dir = output_dir / name
        item_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(item_dir / "data.npz", **arrays)


def vae_prediction_items(prefixes, recon_data: dict, loss_dict: dict) -> list[dict]:
    if "items" in recon_data:
        items = recon_data["items"]
        if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
            raise TypeError("VAE per-object exports must be {'items': list[dict]}.")
        if len(items) != len(prefixes):
            raise ValueError("VAE returned a different number of objects than the input batch.")
        return items
    # Legacy forward(v_test=True) exports concatenated tensors and has no isolated
    # per-object decoder path. Keep its test batches at one rather than fabricate
    # offsets for cross-object predicted edges or write identical data per item.
    if len(prefixes) != 1:
        raise ValueError("Legacy VAE export requires batch_size=1 or explicit per-object 'items'.")
    item = {key: value for key, value in recon_data.items() if key in {
        "pred_face", "pred_edge", "gt_face", "gt_edge", "pred_face_adj", "gt_face_adj",
        "pred_edge_face_connectivity", "gt_edge_face_connectivity", "cached_latent_stats",
    }}
    for key in ("face_coords", "edge_coords"):
        if key in loss_dict:
            item[f"{key}_loss"] = loss_dict[key]
    return [item]
