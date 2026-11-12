"""Pack local raw-edge and native-vertex supervision without mixing objects."""
import torch


def collate_corner_targets(targets, edge_counts):
    pairs, pair_groups, group_objects = [], [], []
    edge_offset = group_offset = 0
    for object_id, (target, edge_count) in enumerate(zip(targets, edge_counts)):
        local_pairs = target["corner_pairs"].clone()
        local_pairs[..., 0] += edge_offset
        group_count = len(target["corner_group_vertices"])
        pairs.append(local_pairs)
        pair_groups.append(target["corner_pair_groups"] + group_offset)
        group_objects.append(torch.full((group_count,), object_id, dtype=torch.long))
        edge_offset += edge_count
        group_offset += group_count
    return {"pairs": torch.cat(pairs), "pair_groups": torch.cat(pair_groups),
            "group_objects": torch.cat(group_objects)}
