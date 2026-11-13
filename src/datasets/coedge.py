"""Batch native-loop neighbors in the existing raw-curve index space."""
import torch


def collate_coedge_topology(targets, edge_counts):
    neighbors, orientations = [], []
    offset = 0
    for target, count in zip(targets, edge_counts):
        neighbors.append(target["coedge_neighbors"] + offset)
        orientations.append(target["coedge_orientation"])
        offset += count
    return {
        "coedge_neighbors": torch.cat(neighbors),
        "coedge_orientation": torch.cat(orientations),
    }
