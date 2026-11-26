"""Native UV input alongside unchanged legacy VAE geometry and supervision."""
from pathlib import Path

import numpy as np
import torch

from src.common.io import sha256_file
from .brep import AutoEncoder_dataset3


class PcurveInputDataset(AutoEncoder_dataset3):
    strict_sample_loading = True

    def __init__(self, training_mode, config):
        super().__init__(training_mode, config)
        self.pcurve_root = Path(config["pcurve_input_root"])
        for cad_id in self.data_folders:
            path = self.pcurve_root / cad_id / "data.npz"
            if not path.is_file():
                raise FileNotFoundError(path)

    def __getitem__(self, index):
        legacy = super().__getitem__(index)
        cad_id = self.data_folders[index % len(self.data_folders)]
        path = self.pcurve_root / cad_id / "data.npz"
        with np.load(path, allow_pickle=False) as data:
            if str(data["schema"]) != "native_pcurve_v4":
                raise ValueError(f"{path}: expected native_pcurve_v4")
            if str(data["source_npz_sha256"]) != sha256_file(self.root / cad_id / "data.npz"):
                raise ValueError(f"{path}: source geometry hash mismatch")
            uv, owner = data["pcurve_uv"], data["owner"]
        faces = len(legacy[1])
        if (uv.shape != (len(owner), 16, 2) or not np.isfinite(uv).all()
                or owner.ndim != 1 or owner.dtype.kind not in "iu"
                or len(owner) == 0 or owner.min() < 0 or owner.max() >= faces
                or not np.all(np.bincount(owner, minlength=faces) > 0)):
            raise ValueError(f"{path}: invalid UV samples or missing face boundaries")
        # Rigid XYZ augmentation leaves this face's parameter chart unchanged.
        return legacy, torch.from_numpy(uv).float(), torch.from_numpy(owner).long()

    @staticmethod
    def collate_fn(items):
        batch = AutoEncoder_dataset3.collate_fn([item[0] for item in items])
        owners = []
        offset = 0
        for item, face_count in zip(items, batch["face_counts"].tolist()):
            owners.append(item[2] + offset)
            offset += face_count
        owner = torch.cat(owners)
        counts = torch.bincount(owner, minlength=offset)
        indices = torch.full((offset, int(counts.max())), -1, dtype=torch.long)
        order = torch.argsort(owner, stable=True)
        starts = counts.cumsum(0) - counts
        position = torch.arange(len(owner)) - starts.repeat_interleave(counts)
        indices[owner[order], position] = order
        batch["pcurve_uv"] = torch.cat([item[1] for item in items])
        batch["pcurve_indices"] = indices
        batch["pcurve_padding_mask"] = indices < 0
        return batch
