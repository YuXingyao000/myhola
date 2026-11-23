"""Native coedge data; rigid XYZ augmentation leaves UV and all indices intact."""

from pathlib import Path

import numpy as np
import torch

from src.preprocess.native_pcurve import SCHEMA
from src.preprocess.rotations import cube24_rotation_matrix


def local_geometry(points):
    xyz = points[..., :3]
    flat = xyz.flatten(1, -2)
    center = flat.mean(1)
    scale = torch.linalg.vector_norm(flat - center[:, None], dim=-1).amax(1, keepdim=True)
    expanded_center = center.reshape(len(points), *([1] * (points.ndim - 2)), 3)
    expanded_scale = scale.reshape(len(points), *([1] * (points.ndim - 2)), 1)
    local = (xyz - expanded_center) / (expanded_scale + 1e-6)
    normals = points[..., 3:]
    normals = normals / (torch.linalg.vector_norm(normals, dim=-1, keepdim=True) + 1e-6)
    return torch.cat([local, normals], -1), torch.cat([center, scale], -1)


class NativePcurveDataset(torch.utils.data.Dataset):
    def __init__(self, mode, cfg):
        split = {"training": "train_dataset", "validation": "val_dataset", "testing": "test_dataset"}[mode]
        self.ids = [s.strip() for s in Path(cfg[split]).read_text().splitlines() if s.strip()]
        self.root = Path(cfg["raw_data_root"])
        self.augment = mode == "training" and bool(cfg["is_aug"])
        self.scale = int(cfg.get("epoch_scale_factor", 1))
        missing = [s for s in self.ids if not (self.root / s / "data.npz").is_file()]
        if missing:
            raise FileNotFoundError(f"Native supervision missing for {len(missing)} CADs, first={missing[:5]}")

    def __len__(self):
        return len(self.ids) * self.scale

    def __getitem__(self, index):
        sample = self.ids[index % len(self.ids)]
        with np.load(self.root / sample / "data.npz") as z:
            if str(z["schema"]) != SCHEMA:
                raise ValueError(f"{sample}: incompatible native pcurve schema")
            item = {
                key: torch.from_numpy(z[key].copy())
                for key in ("owner", "mate", "next", "slot", "side", "vertex", "pcurve_uv")
            }
            face = torch.from_numpy(z["sample_points_faces"].copy())
            edge = torch.from_numpy(z["sample_points_lines"].copy())
        if self.augment:
            rot = torch.as_tensor(cube24_rotation_matrix(int(np.random.randint(24))), dtype=face.dtype)
            for points in (face, edge):
                points[..., :3] = points[..., :3] @ rot.T
                points[..., 3:] = points[..., 3:] @ rot.T
        face_norm, face_bbox = local_geometry(face)
        item.update(v_prefix=sample, face_points=face, edge_points=edge, face_norm=face_norm, face_bbox=face_bbox)
        return item

    @staticmethod
    def collate_fn(items):
        face_counts = torch.tensor([len(x["face_points"]) for x in items])
        edge_counts = torch.tensor([len(x["edge_points"]) for x in items])
        output = {
            key: torch.cat([x[key] for x in items])
            for key in ("face_points", "edge_points", "face_norm", "face_bbox", "pcurve_uv", "slot", "side")
        }
        face_offset = edge_offset = vertex_offset = 0
        owner, mate, nxt, vertices = [], [], [], []
        mask = torch.ones(int(face_counts.sum()), int(face_counts.sum()), dtype=torch.bool)
        for item, nface, nedge in zip(items, face_counts.tolist(), edge_counts.tolist()):
            owner.append(item["owner"] + face_offset)
            mate.append(item["mate"] + edge_offset)
            nxt.append(item["next"] + edge_offset)
            vertices.append(item["vertex"] + vertex_offset)
            mask[face_offset : face_offset + nface, face_offset : face_offset + nface] = False
            vertex_offset += int(item["vertex"].max()) + 1
            face_offset += nface
            edge_offset += nedge
        output.update(
            owner=torch.cat(owner),
            mate=torch.cat(mate),
            next=torch.cat(nxt),
            vertex=torch.cat(vertices),
            face_counts=face_counts,
            edge_counts=edge_counts,
            attn_mask=mask,
            v_prefix=[x["v_prefix"] for x in items],
        )
        output["edge_face_connectivity"] = torch.stack(
            [torch.arange(edge_offset), output["owner"], output["owner"][output["mate"]]], 1
        )
        return output
