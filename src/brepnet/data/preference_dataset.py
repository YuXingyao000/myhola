"""原版 SVR 的独立 GT 数据流与离线偏好对；不重新编码生成 latent。"""

from __future__ import annotations

import hashlib
import io
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


def _read_protocol(protocol_path: Path) -> tuple[dict, str, str]:
    """Validate both the protocol content identity and its exact saved bytes."""
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    recorded_fingerprint = protocol.pop("fingerprint")
    actual_fingerprint = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode(),
    ).hexdigest()
    if recorded_fingerprint != actual_fingerprint:
        raise ValueError(f"Generation protocol fingerprint mismatch: {protocol_path}")
    return protocol, recorded_fingerprint, hashlib.sha256(protocol_bytes).hexdigest()


def _observation_provenance(
    manifest: Path, protocol: dict, status: dict, recorded_fingerprint: str,
) -> tuple[Path, dict]:
    """A reselected pool must still use the frozen source run's original assets."""
    if protocol.get("partial") is not False:
        raise ValueError("Partial observation preference runs cannot be used for training.")
    if status.get("protocol_fingerprint") != recorded_fingerprint:
        raise ValueError("Observation status protocol_fingerprint differs from the recorded protocol.")
    if status.get("partial") is not False:
        raise ValueError("Observation status must explicitly confirm partial=false before training.")
    if manifest.name != "pairs.jsonl":
        raise ValueError("Observation preference training requires the derived pairs.jsonl manifest.")
    source_run = Path(protocol["source_run"])
    if not source_run.is_absolute() or source_run != source_run.resolve():
        raise ValueError("Observation source_run must be an absolute canonical path.")
    source_status_bytes = (source_run / "status.json").read_bytes()
    source_status = json.loads(source_status_bytes)
    if source_status.get("status") != "complete":
        raise ValueError("Source preference generation must be complete before training.")
    source_status_sha256 = hashlib.sha256(source_status_bytes).hexdigest()
    if protocol.get("source_status_sha256") != source_status_sha256:
        raise ValueError("Source status SHA256 differs from the derived protocol.")
    source_path = source_run / "metadata/protocol.json"
    source, source_fingerprint, source_sha256 = _read_protocol(source_path)
    if source["version"] != "svr_preference_generation_v1":
        raise ValueError(f"Unsupported source generation protocol: {source['version']}")
    if source_fingerprint != protocol["source_protocol_fingerprint"]:
        raise ValueError("Source generation protocol fingerprint differs from the derived protocol.")
    if source_sha256 != protocol["source_protocol_sha256"]:
        raise ValueError("Source generation protocol SHA256 differs from the derived protocol.")
    for field in ("splits", "condition_data_root", "checkpoint_sha256"):
        if protocol[field] != source[field]:
            raise ValueError(f"Observation {field} differs from the source generation protocol.")
    if set(source["splits"]["train"]) & set(source["splits"]["validation"]):
        raise ValueError("Source training and validation splits overlap.")
    cache_root = source_run / "gt_cache"
    if Path(protocol["gt_cache_root"]) != cache_root or cache_root.resolve() != cache_root:
        raise ValueError("Observation gt_cache_root must reference the source run's original gt_cache.")
    digests = status.get("manifest_sha256", {})
    for name in ("pairs.jsonl", "validation_pairs.jsonl"):
        if digests.get(name) != sha256_file(manifest.parent / name):
            raise ValueError(f"Observation manifest SHA256 mismatch: {name}")
    return cache_root, {
        "protocol_version": protocol["version"],
        "source_run": str(source_run),
        "source_protocol": str(source_path),
        "source_protocol_sha256": source_sha256,
        "source_protocol_fingerprint": source_fingerprint,
        "source_status_sha256": source_status_sha256,
        "gt_cache_root": str(cache_root),
        "validation_pairs_manifest_sha256": digests["validation_pairs.jsonl"],
    }


def preference_data_provenance(config: dict) -> dict:
    """Validate generation/reselection provenance without relocating source assets."""
    manifest = Path(config["pairs_manifest"]).resolve()
    status = json.loads((manifest.parent / "status.json").read_text())
    if status["status"] != "complete":
        raise ValueError("Preference generation must complete before training reads its pair pool.")
    protocol_path = manifest.parent / "metadata/protocol.json"
    protocol, recorded_fingerprint, protocol_sha256 = _read_protocol(protocol_path)
    if protocol["version"] == "photo_preference_generation_v1":
        from .photo_preference_dataset import photo_preference_provenance
        return photo_preference_provenance(config, protocol, recorded_fingerprint, protocol_sha256, status)
    cache_root, extra = manifest.parent / "gt_cache", {}
    if protocol["version"] == "svr_observation_preference_v1":
        cache_root, extra = _observation_provenance(manifest, protocol, status, recorded_fingerprint)
    elif protocol["version"] != "svr_preference_generation_v1":
        raise ValueError(f"Unsupported generation protocol: {protocol['version']}")
    for mode, field in (("train", "train_dataset"), ("validation", "val_dataset")):
        if read_split(config[field]) != protocol["splits"][mode]:
            raise ValueError(f"{field} differs from the recorded generation split.")
    if Path(config["condition_data_root"]).resolve() != Path(protocol["condition_data_root"]):
        raise ValueError("Condition image root differs from the generation protocol.")
    if Path(config["cached_latent_root"]).resolve() != cache_root:
        raise ValueError("Use the recorded source generation run's freshly encoded gt_cache.")
    geometry = config.get("geometry") or {}
    if geometry.get("enabled", False):
        source = (_read_protocol(Path(extra["source_protocol"]))[0]
                  if "source_protocol" in extra else protocol)
        raw_root = Path(geometry["raw_data_root"]).resolve()
        if raw_root != Path(source["raw_data_root"]):
            raise ValueError("Geometry GT root differs from the source cache generation protocol")
        extra["geometry_supervision"] = {
            "version": "original_decoder_xyz_v1", "raw_data_root": str(raw_root),
            "cache_contract": "source.json binds raw NPZ and mean32/std32 features",
            "correspondence": "original face slots and ordered non-self half-edge owner pairs",
        }
    return {
        "pairs_manifest": str(manifest),
        "pairs_manifest_sha256": sha256_file(manifest),
        "generation_protocol": str(protocol_path),
        "generation_protocol_sha256": protocol_sha256,
        "generation_fingerprint": recorded_fingerprint,
        "source_checkpoint_sha256": protocol["checkpoint_sha256"],
        "train_list_sha256": hashlib.sha256(Path(config["train_dataset"]).read_bytes()).hexdigest(),
        "validation_list_sha256": hashlib.sha256(Path(config["val_dataset"]).read_bytes()).hexdigest(),
        **extra,
    }


def random_duplicate_padding(mean: torch.Tensor, max_faces: int = 30, *,
                             return_face_slots: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """每个真实面至少出现一次，其余随机复制，然后随机排列。"""
    num_faces = len(mean)
    if not 0 < num_faces <= max_faces:
        raise ValueError(f"Expected 1..{max_faces} faces; received {num_faces}")
    indices = torch.cat((torch.arange(num_faces), torch.randint(num_faces, (max_faces - num_faces,))))
    permutation = torch.randperm(max_faces)
    padded = mean[indices[permutation]]
    # Before permutation, the first N entries are the N original faces. Keep one
    # designated copy per face; duplicated slots still receive the original DDPM loss.
    return (padded, permutation.argsort()[:num_faces]) if return_face_slots else padded


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
        geometry = config.get("geometry") or {}
        self.geometry_enabled = bool(geometry.get("enabled", False)) and training_mode == "training"
        self.raw_root = Path(geometry["raw_data_root"]) if self.geometry_enabled else None
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
                    path = Path(row[field])
                    if self.provenance.get("protocol_version") == "svr_observation_preference_v1":
                        expected_parent = Path(self.provenance["source_run"]) / "candidates/train" / row["image_id"]
                        if (not path.is_absolute() or path != path.resolve() or
                                path.parent.parent != expected_parent or path.name != "latent.npy" or
                                path.parent.name not in {f"c{index:02d}" for index in range(8)}):
                            raise ValueError(
                                f"Observation {row['image_id']} {field} must reference its original source candidate latent."
                            )
                    row[field] = manifest.parent / path
                if (self.provenance.get("protocol_version") == "svr_observation_preference_v1" and
                        row["winner_path"] == row["loser_path"]):
                    raise ValueError(f"Observation {row['image_id']} winner and loser must be distinct candidates.")

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
        payload = path.read_bytes() if self.geometry_enabled else None
        stats = np.load(io.BytesIO(payload) if payload is not None else path, allow_pickle=False)
        if stats.ndim != 2 or stats.shape[1] != 64 or not np.isfinite(stats).all():
            raise ValueError(f"Expected finite mean32/std32 cache: {path}")
        padded = random_duplicate_padding(torch.from_numpy(stats[:, :32].copy()).float(),
                                          return_face_slots=self.geometry_enabled)
        item = {
            "v_prefix": image_id,
            "gt_latent": padded[0] if self.geometry_enabled else padded,
            "gt_image": self._image(image_id),
        }
        if self.geometry_enabled:
            item["gt_geometry"] = self._geometry(image_id, len(stats), padded[1],
                                                  hashlib.sha256(payload).hexdigest())
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

    def _geometry(self, image_id: str, num_faces: int, face_slots: torch.Tensor, cache_sha: str) -> dict:
        cache = self.latent_root / f"{image_id}_0"
        source = json.loads((cache / "source.json").read_text())
        payload = (self.raw_root / image_id / "data.npz").read_bytes()
        if (source.get("precision") != "float32" or source.get("layout") != "mean32,std32" or
                source.get("output_sha256", {}).get("features.npy") != cache_sha or
                source.get("gt_npz_sha256") != hashlib.sha256(payload).hexdigest()):
            raise ValueError(f"Geometry GT/cache source mismatch: {image_id}")
        with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
            faces = arrays["sample_points_faces"]
            edges = arrays["sample_points_lines"]
            connectivity = arrays["edge_face_connectivity"]
        if (faces.ndim != 4 or faces.shape[:3] != (num_faces, 16, 16) or faces.shape[-1] < 3
                or edges.ndim != 3 or edges.shape[1] != 16 or edges.shape[-1] < 3
                or not np.isfinite(faces).all() or not np.isfinite(edges).all()):
            raise ValueError(f"Invalid GT geometry shape/count/finite values: {image_id}")
        if (connectivity.ndim != 2 or connectivity.shape[1] != 3 or
                not np.issubdtype(connectivity.dtype, np.integer)):
            raise ValueError(f"Invalid GT connectivity: {image_id}")
        if len(connectivity) and (connectivity.min() < 0 or connectivity[:, 0].max() >= len(edges)
                                  or connectivity[:, 1:].max() >= num_faces):
            raise ValueError(f"GT connectivity index out of range: {image_id}")
        # Same non-self rule as encode_original_gt, retaining ordered half-edges.
        connectivity = connectivity[connectivity[:, 1] != connectivity[:, 2]]
        return {
            "face_slots": face_slots,
            "face_points": torch.from_numpy(faces[..., :3].copy()).float(),
            "edge_points": torch.from_numpy(edges[connectivity[:, 0], :, :3].copy()).float(),
            "edge_faces": torch.from_numpy(connectivity[:, 1:].copy()).long(),
        }

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        if "gt_geometry" not in batch[0]:
            return default_collate(batch)
        result = default_collate([{key: value for key, value in item.items() if key != "gt_geometry"}
                                  for item in batch])
        result["gt_geometry"] = [item["gt_geometry"] for item in batch]
        return result
