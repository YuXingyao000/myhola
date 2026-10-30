"""照片DPO：复用baseline训练缓存，完整validation保持独立。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, default_collate

from .photo_dataset import PhotoFeatureDataset, photo_cache_provenance
from .preference_dataset import PreferenceDataset, random_duplicate_padding, read_split, sha256_file
from .preference_source import load_condition


def read_photo_pairs(manifest: Path, identities: list[str], split: str) -> list[dict]:
    """候选只能来自对应split/图片的原始latent，不接受外部路径。"""
    allowed, seen, pairs = set(identities), set(), []
    with manifest.open() as stream:
        for line in stream:
            row = json.loads(line)
            identity = row["image_id"]
            if identity not in allowed or identity in seen or row["split"] != split:
                raise ValueError(f"Duplicate or out-of-split photo preference: {identity}")
            seen.add(identity)
            if row["pair_type"] not in {"validity", "geometry"}:
                raise ValueError(f"Unknown photo preference type: {identity}")
            pair = {key: row[key] for key in ("image_id", "pair_type")}
            for kind in ("winner", "loser"):
                candidate = row[f"{kind}_id"]
                expected = Path("candidates") / split / identity / candidate / "latent.npy"
                path = Path(row[f"{kind}_path"])
                if candidate not in {f"c{i:02d}" for i in range(8)} or path != expected:
                    raise ValueError(f"Photo candidate path differs from its source: {identity}")
                pair[f"{kind}_path"] = manifest.parent / path
            if pair["winner_path"] == pair["loser_path"]:
                raise ValueError(f"Photo winner and loser must differ: {identity}")
            pairs.append(pair)
    if not pairs:
        raise ValueError(f"Empty photo preference manifest: {manifest}")
    return pairs


def photo_preference_provenance(config: dict, protocol: dict, fingerprint: str,
                               protocol_sha256: str, status: dict) -> dict:
    """把两份偏好清单、照片baseline缓存与完整split绑定到同一来源。"""
    manifest = Path(config["pairs_manifest"]).resolve()
    validation = Path(config["validation_pairs_manifest"]).resolve()
    if manifest.name != "pairs.jsonl" or validation != manifest.parent / "validation_pairs.jsonl":
        raise ValueError("Use the complete photo run's separate train/validation manifests")
    if (config.get("model_source") != "photo" or protocol.get("model_source") != "photo" or
            protocol["condition"] != {"filename": "real_photo.npz", "key": "flux", "rotation_id": 0,
                                       "shape": [512, 512, 3], "dtype": "uint8"}):
        raise ValueError("Photo DPO requires the recorded flux RGB condition")
    if Path(config["condition_data_root"]).resolve() != Path(protocol["condition_data_root"]):
        raise ValueError("Photo condition root differs from generation")
    if Path(config["cached_latent_root"]).resolve() != manifest.parent / "gt_cache":
        raise ValueError("Full validation GT must use the recorded generation cache")
    if set(protocol["splits"]["train"]) & set(protocol["splits"]["validation"]):
        raise ValueError("Photo preference train/validation overlap")
    digests = {}
    for split, field, path in (("train", "train_dataset", manifest),
                              ("validation", "val_dataset", validation)):
        identities = read_split(config[field])
        if identities != protocol["splits"][split]:
            raise ValueError(f"Photo {split} split differs from generation")
        digest = sha256_file(path)
        expected_key = "pairs_manifest_sha256" if split == "train" else "validation_pairs_manifest_sha256"
        if digest != config[expected_key]:
            raise ValueError(f"Photo {split} manifest SHA256 changed")
        pairs = read_photo_pairs(path, identities, split)
        if (len(pairs) != status[split]["pairs"] or len(identities) != status[split]["images"] or
                status[split]["candidates"] != len(identities) * 8):
            raise ValueError(f"Incomplete photo {split} generation")
        digests[expected_key] = digest
    cached = photo_cache_provenance(config)
    if cached != protocol["photo_training_provenance"]:
        raise ValueError("Use the same frozen image features and GT means as the photo baseline")
    if cached["source_checkpoint_sha256"] != protocol["original_checkpoint_sha256"]:
        raise ValueError("Photo cache DINO/AE differs from the pair source")
    cache_manifest = json.loads(Path(cached["manifest"]).read_text())
    if cache_manifest["splits"]["training"]["ids"] != protocol["splits"]["train"]:
        raise ValueError("Photo training feature order/split differs from generation")
    return {
        "protocol_version": protocol["version"], "pairs_manifest": str(manifest),
        "validation_pairs_manifest": str(validation), **digests,
        "generation_protocol": str(manifest.parent / "metadata/protocol.json"),
        "generation_protocol_sha256": protocol_sha256, "generation_fingerprint": fingerprint,
        "source_checkpoint_sha256": protocol["checkpoint_sha256"],
        "original_checkpoint_sha256": protocol["original_checkpoint_sha256"],
        "photo_global_step": protocol["photo_global_step"], "photo_cache": cached,
        "train_list_sha256": sha256_file(Path(config["train_dataset"])),
        "validation_list_sha256": sha256_file(Path(config["val_dataset"])),
        "validation_gt_cache": str(manifest.parent / "gt_cache"),
    }


class PhotoPreferenceDataset(Dataset):
    """训练GT与偏好对独立抽样；完整validation只监测GT去噪。"""

    def __init__(self, training_mode: str, config: dict):
        from .preference_dataset import preference_data_provenance

        if training_mode not in {"training", "validation"}:
            raise ValueError("Photo DPO uses explicit independent downstream evaluation")
        self.provenance = preference_data_provenance(config)
        if self.provenance["protocol_version"] != "photo_preference_generation_v1":
            raise ValueError("PhotoPreferenceDataset requires photo pairs")
        if config.get("provenance") is not None and dict(config["provenance"]) != self.provenance:
            raise ValueError("Photo DPO data changed after run preparation")
        self.config = config
        self.training = training_mode == "training"
        self.identities = read_split(config["train_dataset" if self.training else "val_dataset"])
        if self.training:
            cached_config = {"cache_root": config["cache_root"],
                             "epoch_scale_factor": config["epoch_scale_factor"],
                             "provenance": self.provenance["photo_cache"]}
            self.gt = PhotoFeatureDataset("training", cached_config)
            self.feature_indices = {identity: i for i, identity in enumerate(self.gt.ids)}
            self.pairs = read_photo_pairs(Path(config["pairs_manifest"]).resolve(), self.identities, "train")

    def __len__(self) -> int:
        return len(self.gt) if self.training else len(self.identities)

    def __getitem__(self, index: int) -> dict:
        if self.training:
            item = self.gt[index]
            pair = self.pairs[torch.randint(len(self.pairs), ()).item()]
            feature = torch.from_numpy(self.gt.features[self.feature_indices[pair["image_id"]]].copy())
            if not torch.isfinite(feature).all():
                raise ValueError(f"Nonfinite pair image feature: {pair['image_id']}")
            item.update({"pair_image_id": pair["image_id"], "pair_image_features": feature,
                         "pair_type": pair["pair_type"],
                         "winner_latent": PreferenceDataset._generated_latent(pair["winner_path"]),
                         "loser_latent": PreferenceDataset._generated_latent(pair["loser_path"])})
            return item
        identity = self.identities[index]
        path = Path(self.config["cached_latent_root"]) / f"{identity}_0" / "features.npy"
        stats = np.load(path, allow_pickle=False)
        if stats.ndim != 2 or stats.shape[1] != 64 or stats.dtype != np.float32 or not np.isfinite(stats).all():
            raise ValueError(f"Expected finite FP32 mean32/std32 validation cache: {path}")
        return {"v_prefix": identity,
                "gt_latent": random_duplicate_padding(torch.from_numpy(stats[:, :32].copy())),
                "gt_image": load_condition(self.config, identity)}

    collate_fn = staticmethod(default_collate)
