"""Native loop labels appended to the unchanged HoLa geometry loader."""
from pathlib import Path

import numpy as np
import torch

from src.common.io import sha256_file
from .brep import AutoEncoder_dataset3


def load_corner_topology(path, source_sha256, connectivity):
    """Map the audited cache's sorted coedge axis to connectivity row indices."""
    order = np.argsort(connectivity[:, 0])
    ordered = connectivity[order]
    count = len(order)
    with np.load(path, allow_pickle=False) as data:
        if data['schema_version'].item() != 1 or data['source_sha256'].item() != source_sha256:
            raise ValueError(f'Coedge schema/source mismatch: {path}')
        if not np.array_equal(data['source_edge_ids'], ordered[:, 0]) or not np.array_equal(data['owner_faces'], ordered[:, 1]):
            raise ValueError(f'Coedge raw-edge/owner mismatch: {path}')
        if np.any(data['orientation'] != 1):
            raise ValueError(f'Corner exchange requires original samples oriented along the loop: {path}')
        previous, following = data['previous'].copy(), data['next'].copy()
        for indices in (previous, following):
            if indices.shape != (count,) or np.any(indices < 0) or np.any(indices >= count):
                raise ValueError(f'Invalid coedge neighbor indices: {path}')
        ids = np.arange(count)
        if (not np.array_equal(previous[following], ids) or not np.array_equal(following[previous], ids)
                or np.any(ordered[previous, 1] != ordered[:, 1]) or np.any(ordered[following, 1] != ordered[:, 1])):
            raise ValueError(f'Coedge cycle crosses owners or is not inverse: {path}')
    inverse = np.argsort(order)
    return {key: torch.from_numpy(order[value[inverse]])
            for key, value in (('corner_previous', previous), ('corner_next', following))}


class HolaCornerDataset(AutoEncoder_dataset3):
    strict_sample_loading = True

    def __init__(self, mode, cfg):
        super().__init__(mode, cfg)
        self.topology_root = Path(cfg['corner_topology_root'])
        self.source_hashes = {}
        for sample_id in self.data_folders:
            path = self.topology_root / sample_id / 'coedge_topology.npz'
            if not path.is_file():
                raise FileNotFoundError(path)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        sample_id = self.data_folders[idx % len(self.data_folders)]
        source = self.root / sample_id / 'data.npz'
        stat = source.stat()
        stamp = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        cached = self.source_hashes.get(sample_id)
        if cached is None or cached[0] != stamp:
            cached = (stamp, sha256_file(source))
            self.source_hashes[sample_id] = cached
        target = load_corner_topology(
            self.topology_root / sample_id / 'coedge_topology.npz', cached[1], item[7].numpy(),
        )
        return item, target

    @staticmethod
    def collate_fn(batch):
        items, targets = zip(*batch)
        offset, previous, following = 0, [], []
        for item, target in zip(items, targets):
            previous.append(target['corner_previous'] + offset)
            following.append(target['corner_next'] + offset)
            offset += len(item[7])
        result = AutoEncoder_dataset3.collate_fn(items)
        result['corner_previous'] = torch.cat(previous)
        result['corner_next'] = torch.cat(following)
        return result
