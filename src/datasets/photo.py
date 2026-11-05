"""照片 RGB baseline：只读已验证的冻结 DINO 特征与原版 AE mean。"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, default_collate
from torchvision import transforms

from src.datasets.preference import random_duplicate_padding, read_split, sha256_file


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


def photo_gt_data_provenance(config: dict) -> dict:
    """训练用完整冻结特征 cache，验证用独立全量 GT manifest；不读 pair。"""
    cache = photo_cache_provenance(config)
    path = Path(config["validation_manifest"]).resolve()
    payload = path.read_bytes()
    manifest = json.loads(payload)
    if manifest["version"] != "photo_gt_validation_v1":
        raise ValueError("Expected independent photo_gt_validation_v1 manifest")
    if (manifest["train_cache_manifest_sha256"] != cache["manifest_sha256"] or
            manifest["original_checkpoint_sha256"] != cache["source_checkpoint_sha256"]):
        raise ValueError("Full validation and photo training cache have different sources")
    if manifest["condition"] != {"filename": "real_photo.npz", "key": "flux", "rotation_id": 0,
                                 "shape": [512, 512, 3], "dtype": "uint8"}:
        raise ValueError("Expected rotation-0 flux photos paired with original AE latents")
    train, val = read_split(config["train_dataset"]), read_split(config["val_dataset"])
    for ids, field, key in ((train, "train_dataset", "train_split"),
                            (val, "val_dataset", "validation_split")):
        if (len(ids) != manifest[key]["count"] or
                sha256_file(Path(config[field])) != manifest[key]["sha256"]):
            raise ValueError("GT split differs from its audited full manifest")
    cache_manifest = json.loads(Path(cache["manifest"]).read_text())
    if train != cache_manifest["splits"]["training"]["ids"] or set(train) & set(val):
        raise ValueError("GT training cache and split must match and be disjoint from validation")
    records = manifest["records"]
    if [record["image_id"] for record in records] != val:
        raise ValueError("Validation manifest must cover the complete split in order")
    image_root = Path(config["condition_data_root"]).resolve()
    if any(Path(record["image_path"]).resolve() != image_root / record["image_id"] / "real_photo.npz"
           for record in records):
        raise ValueError("Validation photo root differs from its audited manifest")
    return {
        "version": "photo_gt_only_v1", "photo_cache": cache,
        "validation_manifest": str(path), "validation_manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "original_checkpoint_sha256": manifest["original_checkpoint_sha256"],
        "photo_checkpoint_sha256": manifest["photo_checkpoint_sha256"],
        "photo_global_step": manifest["photo_global_step"],
        "train_split_sha256": manifest["train_split"]["sha256"],
        "validation_split_sha256": manifest["validation_split"]["sha256"],
        "train_count": len(train), "validation_count": len(val),
        "condition_data_root": str(image_root),
    }


class PhotoGTOnlyDataset(Dataset):
    """纯 GT 流：训练读取 CLS cache；完整 validation 在线编码照片。"""

    def __init__(self, training_mode: str, config: dict):
        if training_mode not in {"training", "validation"}:
            raise ValueError("Use the explicit downstream protocol for test evaluation")
        self.provenance = photo_gt_data_provenance(config)
        if config.get("provenance") is not None and dict(config["provenance"]) != self.provenance:
            raise ValueError("Photo GT data changed after run preparation")
        self.training = training_mode == "training"
        if self.training:
            cache_config = {key: value for key, value in config.items() if key != "provenance"}
            self.cache = PhotoFeatureDataset("training", cache_config)
            self.ids = self.cache.ids
        else:
            self.records = json.loads(Path(config["validation_manifest"]).read_text())["records"]
            self.ids = [record["image_id"] for record in self.records]
            self.transform = photo_transform()

    def __len__(self) -> int:
        return len(self.cache) if self.training else len(self.records)

    def __getitem__(self, index: int) -> dict:
        if self.training:
            return self.cache[index]
        record = self.records[index]
        image_bytes = Path(record["image_path"]).read_bytes()
        latent_bytes = Path(record["latent_path"]).read_bytes()
        if (hashlib.sha256(image_bytes).hexdigest() != record["image_sha256"] or
                hashlib.sha256(latent_bytes).hexdigest() != record["latent_sha256"]):
            raise ValueError(f"Validation photo/GT changed: {record['image_id']}")
        with np.load(io.BytesIO(image_bytes), allow_pickle=False) as data:
            image = data["flux"]
        stats = np.load(io.BytesIO(latent_bytes), allow_pickle=False)
        if image.shape != (512, 512, 3) or image.dtype != np.uint8:
            raise ValueError(f"Invalid flux photo: {record['image_id']}")
        if (stats.shape != (record["num_faces"], 64) or not 1 <= len(stats) <= 30 or
                stats.dtype != np.float32 or not np.isfinite(stats).all() or (stats[:, 32:] < 0).any()):
            raise ValueError(f"Invalid original AE mean/std: {record['image_id']}")
        return {"v_prefix": record["image_id"], "gt_image": self.transform(image),
                "gt_latent": random_duplicate_padding(torch.from_numpy(stats[:, :32].copy()))}

    collate_fn = staticmethod(default_collate)
