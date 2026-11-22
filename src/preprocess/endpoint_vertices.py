"""Directed endpoint identity sidecars; training does not import OCC."""
from pathlib import Path

import numpy as np

from src.common.io import sha256_file


SCHEMA_VERSION = 1


def load_endpoint_vertices(path, *, source_path, connectivity):
    """Labels follow connectivity rows, not sample_points_lines row numbers."""
    with np.load(Path(path), allow_pickle=False) as data:
        version = data['schema_version'].item()
        digest = data['source_sha256'].item()
        edge_ids = data['source_edge_ids']
        faces = data['face_pairs']
        vertices = data['endpoint_vertex_ids']
        valid = data['endpoint_valid']
    if version != SCHEMA_VERSION:
        raise ValueError(f'Unsupported directed vertex schema: {path}')
    if digest != sha256_file(source_path):
        raise ValueError(f'Directed vertex source digest mismatch: {path}')
    if not np.array_equal(edge_ids, connectivity[:, 0]) or not np.array_equal(faces, connectivity[:, 1:]):
        raise ValueError(f'Directed vertex connection order mismatch: {path}')
    if vertices.dtype.kind not in 'iu' or vertices.shape != (len(connectivity), 2) or (vertices < -1).any():
        raise ValueError(f'Invalid directed endpoint identities: {path}')
    if valid.dtype != np.bool_ or not np.array_equal(valid, vertices >= 0):
        raise ValueError(f'Invalid directed endpoint validity: {path}')
    return vertices
