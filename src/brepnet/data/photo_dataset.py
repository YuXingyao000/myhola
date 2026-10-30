"""照片 RGB baseline：只读已验证的冻结 DINO 特征与原版 AE mean。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, default_collate
from torchvision import transforms

from .preference_dataset import random_duplicate_padding, sha256_file


def photo_transform():
    """与原 SVR 的 RGB 缩放和归一化一致；不裁剪、不做随机增强。"""
    return transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def photo_cache_provenance(config: dict) -> dict:
    path = Path(config["cache_root"]).resolve() / "manifest.json"
    payload = path.read_bytes()
    manifest = json.loads(payload)
    if manifest.get("version") != "legacy_photo_rgb_v1" or manifest.get("complete") is not True:
        raise ValueError("Photo training requires a complete legacy_photo_rgb_v1 cache")
    if manifest.get("image_key") != "flux" or manifest.get("rotation_id") != 0:
        raise ValueError("Expected original flux photos paired with rotation 0")
    train = manifest["splits"]["training"]["ids"]
    val = manifest["splits"]["validation"]["ids"]
    if not train or not val or len(set(train)) != len(train) or len(set(val)) != len(val) or set(train) & set(val):
        raise ValueError("Photo cache splits must be nonempty, unique and disjoint")
    for info in manifest["splits"].values():
        for field in ("features", "means", "counts"):
            artifact = path.parent / info["files"][field]
            # 文件系统时间戳可能合并快速写入；每次启动按实际内容核验。
            if sha256_file(artifact) != info["sha256"][field]:
                raise ValueError(f"Photo cache content changed: {artifact}")
    return {
        "manifest": str(path), "manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "source_checkpoint_sha256": manifest["checkpoint_sha256"],
        "training_count": len(train), "validation_count": len(val),
    }


class PhotoFeatureDataset(Dataset):
    """冻结图像特征可缓存；可训练 img_fc 仍在每个训练 step 内执行。"""

    def __init__(self, training_mode: str, config: dict):
        if training_mode not in {"training", "validation"}:
            raise ValueError("This cache is for training/validation, not downstream test evaluation")
        provenance = photo_cache_provenance(config)
        if config.get("provenance") is not None and dict(config["provenance"]) != provenance:
            raise ValueError("Photo cache manifest changed since run preparation")
        root = Path(config["cache_root"])
        info = json.loads((root / "manifest.json").read_text())["splits"][training_mode]
        self.ids = info["ids"]
        self.features = np.load(root / info["files"]["features"], mmap_mode="r", allow_pickle=False)
        self.means = np.load(root / info["files"]["means"], mmap_mode="r", allow_pickle=False)
        self.counts = np.load(root / info["files"]["counts"], mmap_mode="r", allow_pickle=False)
        n = len(self.ids)
        if (self.features.shape != (n, 1024) or self.means.shape != (n, 30, 32)
                or self.counts.shape != (n,) or self.features.dtype != np.float32
                or self.means.dtype != np.float32 or self.counts.dtype != np.int16):
            raise ValueError("Photo cache shapes/dtypes differ from manifest contract")
        self.scale = int(config["epoch_scale_factor"]) if training_mode == "training" else 1
        if self.scale < 1:
            raise ValueError("epoch_scale_factor must be positive")

    def __len__(self):
        return len(self.ids) * self.scale

    def __getitem__(self, index: int) -> dict:
        index %= len(self.ids)
        n = int(self.counts[index])
        if not 1 <= n <= 30:
            raise ValueError(f"Invalid face count for {self.ids[index]}: {n}")
        mean = torch.from_numpy(self.means[index, :n].copy())
        feature = torch.from_numpy(self.features[index].copy())
        if not torch.isfinite(mean).all() or not torch.isfinite(feature).all():
            raise ValueError(f"Nonfinite cached input: {self.ids[index]}")
        return {"v_prefix": self.ids[index], "gt_latent": random_duplicate_padding(mean),
                "image_features": feature}

    collate_fn = staticmethod(default_collate)
