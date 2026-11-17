"""Native loop targets for Decoder supervision; original Encoder inputs retained."""
from pathlib import Path

import numpy as np
import torch

from .brep import AutoEncoder_joint_dataset


def load_endpoint_topology(path, source_sha256, connectivity):
    """Read the audited coedge cache, failing instead of inventing neighbors."""
    with np.load(path, allow_pickle=False) as data:
        if data['schema_version'].item() != 1 or data['source_sha256'].item() != source_sha256:
            raise ValueError(f'Coedge schema/source mismatch: {path}')
        ordered = connectivity[np.argsort(connectivity[:, 0])]
        count = len(ordered)
        if not np.array_equal(data['source_edge_ids'], ordered[:, 0]) or not np.array_equal(data['owner_faces'], ordered[:, 1]):
            raise ValueError(f'Coedge raw-edge/owner mismatch: {path}')
        # This experiment keeps original sampling and Encoder inputs unchanged.
        if np.any(data['orientation'] != 1):
            raise ValueError(f'Endpoint experiment requires loop-oriented original samples: {path}')
        previous, following = data['previous'].copy(), data['next'].copy()
        ids = np.arange(count)
        for indices in (previous, following):
            if indices.shape != (count,) or np.any(indices < 0) or np.any(indices >= count):
                raise ValueError(f'Invalid endpoint neighbor indices: {path}')
        if (not np.array_equal(previous[following], ids) or not np.array_equal(following[previous], ids)
                or np.any(ordered[previous, 1] != ordered[:, 1]) or np.any(ordered[following, 1] != ordered[:, 1])):
            raise ValueError(f'Coedge cycle crosses owners or is not inverse: {path}')
    return {'endpoint_previous': torch.from_numpy(previous), 'endpoint_next': torch.from_numpy(following)}


class BoundaryDataset(AutoEncoder_joint_dataset):
    def __init__(self, mode, cfg):
        super().__init__(mode, cfg)
        self.endpoint_root = Path(cfg['endpoint_topology_root']) if cfg.get('endpoint_topology_root') else None
        if self.endpoint_root:
            for sample_id in self.data_folders:
                path = self.endpoint_root / sample_id / 'coedge_topology.npz'
                if not path.is_file():
                    raise FileNotFoundError(path)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        if self.endpoint_root:
            sample_id = self.data_folders[idx % len(self.data_folders)]
            item[-1].update(load_endpoint_topology(
                self.endpoint_root / sample_id / 'coedge_topology.npz',
                self._source_sha256(sample_id), item[7].numpy(),
            ))
        return item

    @staticmethod
    def collate_fn(batch):
        result = AutoEncoder_joint_dataset.collate_fn(batch)
        if 'endpoint_next' in batch[0][-1]:
            previous, following, offset = [], [], 0
            for item in batch:
                previous.append(item[-1]['endpoint_previous'] + offset)
                following.append(item[-1]['endpoint_next'] + offset)
                offset += len(item[2])
            result['endpoint_targets'] = {'previous': torch.cat(previous), 'next': torch.cat(following)}
        return result
