"""Original VAE samples plus native IDs for each directed endpoint use."""
from functools import lru_cache
from pathlib import Path

import torch

from src.datasets.brep import AutoEncoder_dataset3
from src.preprocess.endpoint_vertices import load_endpoint_vertices


class VertexIdentityDataset(AutoEncoder_dataset3):
    strict_sample_loading = True

    def __init__(self, v_training_mode, v_conf):
        super().__init__(v_training_mode, v_conf)
        self.vertex_root = Path(v_conf['endpoint_vertex_root'])
        for sid in self.data_folders:
            path = self.vertex_root / sid / 'endpoint_vertices.npz'
            if not path.is_file():
                raise FileNotFoundError(f'Missing directed endpoint labels: {path}')

    @lru_cache(maxsize=256)
    def _labels(self, sid):
        import numpy as np
        source = self.root / sid / 'data.npz'
        with np.load(source, allow_pickle=False) as data:
            connectivity = data['edge_face_connectivity']
        connectivity = connectivity[connectivity[:, 1] != connectivity[:, 2]]
        return load_endpoint_vertices(
            self.vertex_root / sid / 'endpoint_vertices.npz',
            source_path=source, connectivity=connectivity,
        )

    def __getitem__(self, index):
        item = super().__getitem__(index)
        sid = self.data_folders[index % len(self.data_folders)]
        return item, torch.from_numpy(self._labels(sid).copy())

    @staticmethod
    def collate_fn(batch):
        samples, vertices = zip(*batch)
        result = AutoEncoder_dataset3.collate_fn(samples)
        packed, offset = [], 0
        for ids in vertices:
            valid = ids >= 0
            packed.append(torch.where(valid, ids + offset, ids))
            offset += int(ids[valid].max()) + 1 if valid.any() else 0
        result['endpoint_vertex_ids'] = torch.cat(packed)
        result['vertex_edge_counts'] = [len(ids) for ids in vertices]
        return result
