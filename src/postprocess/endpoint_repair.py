"""Targeted endpoint repair on the selected edge sample points.

Runs only inside recovery paths, after half-edge selection and before fitting.
It acts only where the relation is unambiguous:

* inside each face, two endpoints of different edges are paired when they are
  mutual nearest neighbours, closer than `max_gap`, and the second-nearest
  candidate of either endpoint is at least `ambiguity_margin` farther
  (tiny gaps below `small_gap` are accepted without the margin test);
* pairs are merged across faces into vertex clusters; a cluster is kept when
  its diameter is within `max_gap`, no edge contributes both ends to it, and
  it has at least two members;
* clustered endpoints are moved onto the cluster centroid and the interior
  of the edge follows with a chord-length linear blend; unclustered endpoints
  do not move;
* an open edge whose two ends are unpaired in both of its faces is dropped
  from the connectivity, as a superfluous boundary, when both faces keep at
  least two edges.

No GT is read. The returned diagnostics describe every decision.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class EndpointRepairConfig:
    max_gap: float = 0.2
    ambiguity_margin: float = 0.02
    small_gap: float = 0.01
    snap: bool = True
    drop_dangling: bool = True
    min_edges_per_face: int = 2


def _face_endpoints(edge_points, edge_ids, closed):
    ends = [(int(e), k) for e in edge_ids if not closed[e] for k in (0, -1)]
    xyz = np.array([edge_points[e][k] for e, k in ends]).reshape(-1, 3)
    return ends, xyz


def pair_face_endpoints(edge_points, edge_ids, closed, config=EndpointRepairConfig()):
    """Mutual-nearest, unambiguous endpoint pairs inside one face.

    Returns (pairs, dangling): pairs as ((edge, end), (edge, end), distance) with
    end in {0, -1}; dangling lists the endpoints that found no partner.
    """
    ends, xyz = _face_endpoints(edge_points, edge_ids, closed)
    n = len(ends)
    if n < 2:
        return [], list(ends)
    dist = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    same_edge = np.array([[a[0] == b[0] for b in ends] for a in ends])
    dist[same_edge] = np.inf
    order = np.argsort(dist, axis=1)

    def unambiguous(i, j):
        best = dist[i, j]
        if not np.isfinite(best) or best > config.max_gap:
            return False
        if best <= config.small_gap:
            return True
        second = dist[i, order[i, 1]] if n > 2 else np.inf
        return second - best >= config.ambiguity_margin

    pairs = []
    for i in range(n):
        j = int(order[i, 0])
        if j <= i and int(order[j, 0]) == i:
            continue  # handled from j's side
        if int(order[j, 0]) != i:
            continue
        if unambiguous(i, j) and unambiguous(j, i):
            pairs.append((ends[i], ends[j], float(dist[i, j])))
    paired = {end for a, b, _ in pairs for end in (a, b)}
    dangling = [end for end in ends if end not in paired]
    return pairs, dangling


class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_vertex_clusters(edge_points, face_edge_adj, closed, config=EndpointRepairConfig()):
    """Pair endpoints per face and merge the pairs into vertex clusters.

    Returns (clusters, per_face) where clusters is a list of dicts
    {"members": [(edge, end), ...], "target": xyz, "diameter": float} and
    per_face maps face index -> {"pairs": [...], "dangling": [...]}.
    """
    union = _UnionFind()
    per_face = {}
    for face_index, edge_ids in enumerate(face_edge_adj):
        pairs, dangling = pair_face_endpoints(edge_points, edge_ids, closed, config)
        per_face[face_index] = {"pairs": pairs, "dangling": dangling}
        for a, b, _ in pairs:
            union.union(a, b)
    groups = {}
    for end in list(union.parent):
        groups.setdefault(union.find(end), []).append(end)
    clusters = []
    rejected = {"single_member": 0, "edge_both_ends": 0, "diameter": 0}
    for members in groups.values():
        if len(members) < 2:
            rejected["single_member"] += 1
            continue
        edges = [e for e, _ in members]
        if len(set(edges)) != len(edges):
            rejected["edge_both_ends"] += 1
            continue
        xyz = np.array([edge_points[e][k] for e, k in members])
        diameter = float(np.max(np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)))
        if diameter > config.max_gap:
            rejected["diameter"] += 1
            continue
        clusters.append({"members": sorted(members), "target": xyz.mean(axis=0), "diameter": diameter})
    clusters.sort(key=lambda c: c["members"][0])
    return clusters, per_face, rejected


def dangling_edges(per_face, edge_face_connectivity, closed):
    """Open edges whose two ends are unpaired in both of their faces."""
    dangling_in = {}
    for face_index, info in per_face.items():
        for edge, _ in info["dangling"]:
            dangling_in.setdefault(edge, {}).setdefault(face_index, 0)
            dangling_in[edge][face_index] += 1
    result = []
    for edge, first, second in edge_face_connectivity:
        edge, first, second = int(edge), int(first), int(second)
        if closed[edge]:
            continue
        counts = dangling_in.get(edge, {})
        if counts.get(first, 0) == 2 and counts.get(second, 0) == 2:
            result.append(edge)
    return result


def snap_endpoints(edge_points, clusters, closed):
    """Move clustered endpoints to their cluster centroid with a chord-length blend."""
    repaired = np.array(edge_points, dtype=np.float64, copy=True)
    target_of = {}
    for cluster in clusters:
        for member in cluster["members"]:
            target_of[member] = cluster["target"]
    displacements = []
    for edge in range(len(repaired)):
        if closed[edge]:
            continue
        points = repaired[edge]
        d0 = target_of.get((edge, 0), points[0]) - points[0]
        d1 = target_of.get((edge, -1), points[-1]) - points[-1]
        if not np.any(d0) and not np.any(d1):
            continue
        segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
        total = segments.sum()
        t = np.concatenate([[0.0], np.cumsum(segments)]) / total if total > 0 else np.linspace(0, 1, len(points))
        repaired[edge] = points + (1 - t)[:, None] * d0 + t[:, None] * d1
        displacements.extend(float(np.linalg.norm(d)) for d, key in ((d0, (edge, 0)), (d1, (edge, -1)))
                             if key in target_of)
    return repaired.astype(edge_points.dtype), displacements


def repair_endpoints(shape, config=EndpointRepairConfig()):
    """Apply the repair in place on a prepared `Shape`; return diagnostics and clusters.

    Expects `shape.recon_edge_points`, `shape.edge_face_connectivity`,
    `shape.face_edge_adj` and `shape.openness` (True = closed edge).
    """
    closed = np.asarray(shape.openness, dtype=bool)
    diagnostics = {"config": asdict(config), "dropped_edges": [], "num_clusters": 0,
                   "cluster_sizes": {}, "num_snapped_endpoints": 0, "max_snap_displacement": 0.0}
    clusters, per_face, rejected = build_vertex_clusters(
        shape.recon_edge_points, shape.face_edge_adj, closed, config)
    diagnostics["num_pairs"] = sum(len(info["pairs"]) for info in per_face.values())
    diagnostics["num_dangling_endpoints"] = sum(len(info["dangling"]) for info in per_face.values())
    diagnostics["rejected_clusters"] = rejected

    if config.drop_dangling:
        drop = []
        for edge in dangling_edges(per_face, shape.edge_face_connectivity, closed):
            _, first, second = shape.edge_face_connectivity[shape.edge_face_connectivity[:, 0] == edge][0]
            if (len(shape.face_edge_adj[first]) - 1 >= config.min_edges_per_face
                    and len(shape.face_edge_adj[second]) - 1 >= config.min_edges_per_face):
                drop.append(edge)
        for edge in drop:
            _, first, second = shape.edge_face_connectivity[shape.edge_face_connectivity[:, 0] == edge][0]
            shape.face_edge_adj[first].remove(edge)
            shape.face_edge_adj[second].remove(edge)
        if drop:
            keep = ~np.isin(shape.edge_face_connectivity[:, 0], drop)
            shape.edge_face_connectivity = shape.edge_face_connectivity[keep]
            shape.remove_edge_idx_new.extend(int(e) for e in drop)
        diagnostics["dropped_edges"] = [int(e) for e in drop]

    diagnostics["num_clusters"] = len(clusters)
    sizes = {}
    for cluster in clusters:
        sizes[len(cluster["members"])] = sizes.get(len(cluster["members"]), 0) + 1
    diagnostics["cluster_sizes"] = {str(k): v for k, v in sorted(sizes.items())}
    if config.snap and clusters:
        shape.recon_edge_points, displacements = snap_endpoints(shape.recon_edge_points, clusters, closed)
        diagnostics["num_snapped_endpoints"] = len(displacements)
        diagnostics["max_snap_displacement"] = max(displacements, default=0.0)
    return diagnostics, clusters
