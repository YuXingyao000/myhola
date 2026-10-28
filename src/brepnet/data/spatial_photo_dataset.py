"""空间条件 DDPM 的独立 GT 数据流；复用已溯源的原 VAE mean，不读偏好对。"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, default_collate

from .photo_dataset import photo_transform
from .preference_dataset import random_duplicate_padding, read_split, sha256_file


def spatial_data_provenance(config: dict) -> dict:
    """明确绑定原 AE、identity rotation、全 train/validation 清单及缓存根。"""
    path = Path(config["source_protocol"]).resolve()
    digest = sha256_file(path)
    if digest != config["source_protocol_sha256"]:
        raise ValueError("Spatial training source protocol SHA256 changed")
    protocol = json.loads(path.read_text())
    if (protocol["version"] != "photo_preference_generation_v1" or
            protocol["gt_cache"] != "original embedded AE / FP32 / mean32+std32 / rotation0" or
            protocol["condition"] != {"filename": "real_photo.npz", "key": "flux", "rotation_id": 0,
                                       "shape": [512, 512, 3], "dtype": "uint8"}):
        raise ValueError("Expected verified original-AE mean/std with identity-view flux images")
    if Path(config["cached_latent_root"]).resolve() != path.parent.parent / "gt_cache":
        raise ValueError("GT cache root differs from its recorded source")
    if Path(config["condition_data_root"]).resolve() != Path(protocol["condition_data_root"]).resolve():
        raise ValueError("Photo root differs from the recorded source")
    splits = {kind: read_split(config[field]) for kind, field in
              (("train", "train_dataset"), ("validation", "val_dataset"))}
    if splits != protocol["splits"] or set(splits["train"]) & set(splits["validation"]):
        raise ValueError("Use the full, disjoint source train/validation lists")
    return {"source_protocol_sha256": digest,
            "original_checkpoint_sha256": protocol["original_checkpoint_sha256"],
            "train_count": len(splits["train"]), "validation_count": len(splits["validation"]),
            "train_list_sha256": sha256_file(Path(config["train_dataset"])),
            "validation_list_sha256": sha256_file(Path(config["val_dataset"])),
            "condition_data_root": str(Path(config["condition_data_root"]).resolve()),
            "cached_latent_root": str(Path(config["cached_latent_root"]).resolve())}


class SpatialPhotoDataset(Dataset):
    """RGB 在线编码；GT 只进入加噪/epsilon target，不作为空间 query 输入。"""

    def __init__(self, training_mode: str, config: dict):
        if training_mode not in {"training", "validation"}:
            raise ValueError("Use a separately frozen protocol for downstream test evaluation")
        self.provenance = spatial_data_provenance(config)
        if config.get("provenance") is not None and dict(config["provenance"]) != self.provenance:
            raise ValueError("Spatial dataset changed after run preparation")
        self.ids = read_split(config["train_dataset" if training_mode == "training" else "val_dataset"])
        self.scale = int(config["epoch_scale_factor"]) if training_mode == "training" else 1
        if self.scale < 1:
            raise ValueError("epoch_scale_factor must be positive")
        self.images = Path(config["condition_data_root"])
        self.latents = Path(config["cached_latent_root"])
        protocol_path = Path(config["source_protocol"])
        protocol = json.loads(protocol_path.read_text())
        split = "train" if training_mode == "training" else "validation"
        self.sources = protocol_path.parent.parent / "candidates" / split
        self.raw_root = Path(protocol["raw_data_root"])
        self.transform = photo_transform()

    def __len__(self) -> int:
        return len(self.ids) * self.scale

    def __getitem__(self, index: int) -> dict:
        identity = self.ids[index % len(self.ids)]
        root = self.latents / f"{identity}_0"
        payload = (root / "features.npy").read_bytes()
        source = json.loads((root / "source.json").read_text())
        if (source["precision"] != "float32" or source["layout"] != "mean32,std32" or
                hashlib.sha256(payload).hexdigest() != source["output_sha256"]["features.npy"]):
            raise ValueError(f"GT latent provenance mismatch: {identity}")
        stats = np.load(io.BytesIO(payload), allow_pickle=False)
        if (stats.ndim != 2 or stats.shape[1] != 64 or not 1 <= len(stats) <= 30 or
                stats.dtype != np.float32 or not np.isfinite(stats).all() or (stats[:, 32:] < 0).any()):
            raise ValueError(f"Invalid original AE mean/std: {identity}")
        image_path = self.images / identity / "real_photo.npz"
        image_bytes = image_path.read_bytes()
        sources = json.loads((self.sources / identity / "sources.json").read_text())
        if (hashlib.sha256(image_bytes).hexdigest() != sources[str(image_path.resolve())] or
                source["gt_npz_sha256"] != sources[str(self.raw_root / identity / "data.npz")]):
            raise ValueError(f"Photo/GT inputs differ from their recorded pairing: {identity}")
        with np.load(io.BytesIO(image_bytes), allow_pickle=False) as data:
            image = data["flux"]
        if image.shape != (512, 512, 3) or image.dtype != np.uint8:
            raise ValueError(f"Expected flux uint8 [512,512,3]: {identity}")
        return {"v_prefix": identity, "gt_image": self.transform(image),
                "gt_latent": random_duplicate_padding(torch.from_numpy(stats[:, :32].copy()))}

    collate_fn = staticmethod(default_collate)
