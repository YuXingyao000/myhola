"""原版 SVR 的独立 GT 数据流与离线偏好对；不重新编码生成 latent。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, default_collate
from torchvision import transforms

from src.brepnet.diffusion.image_loading import load_image_view


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_split(path: str | Path) -> list[str]:
    """保留清单顺序；重复对象不能悄悄改变训练权重。"""
    identities = Path(path).read_text().splitlines()
    if not identities or any(not item.strip() for item in identities):
        raise ValueError(f"Empty identity/split: {path}")
    identities = [item.strip() for item in identities]
    if len(set(identities)) != len(identities):
        raise ValueError(f"Duplicate identities in {path}")
    return identities


def preference_data_provenance(config: dict) -> dict:
    """只接受本轮生产协议；记录内容哈希，而不把文件路径当成数据版本。"""
    manifest = Path(config["pairs_manifest"]).resolve()
    status = json.loads((manifest.parent / "status.json").read_text())
    if status["status"] != "complete":
        raise ValueError("Preference generation must complete before training reads its pair pool.")
    protocol_path = manifest.parent / "metadata/protocol.json"
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    recorded_fingerprint = protocol.pop("fingerprint")
    actual_fingerprint = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode(),
    ).hexdigest()
    if recorded_fingerprint != actual_fingerprint:
        raise ValueError(f"Generation protocol fingerprint mismatch: {protocol_path}")
    if protocol["version"] != "svr_preference_generation_v1":
        raise ValueError(f"Unsupported generation protocol: {protocol['version']}")
    for mode, field in (("train", "train_dataset"), ("validation", "val_dataset")):
        if read_split(config[field]) != protocol["splits"][mode]:
            raise ValueError(f"{field} differs from the recorded generation split.")
    if Path(config["condition_data_root"]).resolve() != Path(protocol["condition_data_root"]):
        raise ValueError("Condition image root differs from the generation protocol.")
    if Path(config["cached_latent_root"]).resolve() != manifest.parent / "gt_cache":
        raise ValueError("Use this generation run's freshly encoded gt_cache.")
    return {
        "pairs_manifest": str(manifest),
        "pairs_manifest_sha256": sha256_file(manifest),
        "generation_protocol": str(protocol_path),
        "generation_protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "generation_fingerprint": recorded_fingerprint,
        "source_checkpoint_sha256": protocol["checkpoint_sha256"],
        "train_list_sha256": hashlib.sha256(Path(config["train_dataset"]).read_bytes()).hexdigest(),
        "validation_list_sha256": hashlib.sha256(Path(config["val_dataset"]).read_bytes()).hexdigest(),
    }


def random_duplicate_padding(mean: torch.Tensor, max_faces: int = 30) -> torch.Tensor:
    """每个真实面至少出现一次，其余随机复制，然后随机排列。"""
    num_faces = len(mean)
    if not 0 < num_faces <= max_faces:
        raise ValueError(f"Expected 1..{max_faces} faces; received {num_faces}")
    indices = torch.cat((torch.arange(num_faces), torch.randint(num_faces, (max_faces - num_faces,))))
    return mean[indices[torch.randperm(max_faces)]]


class PreferenceDataset(Dataset):
    """一个 GT 对象 + 独立均匀抽取的一个偏好对；验证只读取 GT。"""

    def __init__(self, training_mode: str, config: dict):
        self.provenance = preference_data_provenance(config)
        expected = config.get("provenance")
        if expected is not None and self.provenance != dict(expected):
            raise ValueError("Preference data changed after the training invocation was recorded.")
        split_keys = {"training": "train_dataset", "validation": "val_dataset", "testing": "test_dataset"}
        self.identities = read_split(config[split_keys[training_mode]])
        self.training_mode = training_mode
        self.latent_root = Path(config["cached_latent_root"])
        self.image_root = Path(config["condition_data_root"])
        self.epoch_scale = int(config["epoch_scale_factor"]) if training_mode == "training" else 1
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.pairs: list[dict] = []
        if training_mode == "training":
            manifest = Path(config["pairs_manifest"]).resolve()
            with manifest.open() as stream:
                for line in stream:
                    if line.strip():
                        row = json.loads(line)
                        self.pairs.append({
                            key: row[key] for key in ("image_id", "winner_path", "loser_path", "pair_type")
                        })
            pair_ids = [row["image_id"] for row in self.pairs]
            if not pair_ids:
                raise ValueError(f"No preference pairs in {manifest}")
            if len(set(pair_ids)) != len(pair_ids):
                raise ValueError("Preference protocol permits at most one pair per image.")
            if set(pair_ids) - set(self.identities):
                raise ValueError("Preference manifest contains identities outside the training split.")
            for row in self.pairs:
                if row["pair_type"] not in {"validity", "geometry"}:
                    raise ValueError(f"Unknown preference type: {row['pair_type']}")
                for field in ("winner_path", "loser_path"):
                    row[field] = manifest.parent / row[field]

    def __len__(self) -> int:
        return len(self.identities) * self.epoch_scale

    def _image(self, image_id: str) -> torch.Tensor:
        image = load_image_view(
            self.image_root / image_id / "imgs.npz", "svr_imgs", 0, read_single_view=True,
        )
        return self.transform(image)

    @staticmethod
    def _generated_latent(path: Path) -> torch.Tensor:
        values = np.load(path, allow_pickle=False)
        if values.shape != (30, 32) or values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError(f"Expected finite original float32 [30,32] latent: {path}")
        return torch.from_numpy(values)

    def __getitem__(self, index: int) -> dict:
        image_id = self.identities[index % len(self.identities)]
        path = self.latent_root / f"{image_id}_0" / "features.npy"
        stats = np.load(path, allow_pickle=False)
        if stats.ndim != 2 or stats.shape[1] != 64 or not np.isfinite(stats).all():
            raise ValueError(f"Expected finite mean32/std32 cache: {path}")
        item = {
            "v_prefix": image_id,
            "gt_latent": random_duplicate_padding(torch.from_numpy(stats[:, :32].copy()).float()),
            "gt_image": self._image(image_id),
        }
        if self.training_mode == "training":
            pair = self.pairs[torch.randint(len(self.pairs), ()).item()]
            item.update({
                "pair_image_id": pair["image_id"],
                "pair_image": self._image(pair["image_id"]),
                "winner_latent": self._generated_latent(pair["winner_path"]),
                "loser_latent": self._generated_latent(pair["loser_path"]),
                "pair_type": pair["pair_type"],
            })
        return item

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        return default_collate(batch)
