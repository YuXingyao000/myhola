"""Hard negative face pairs for the edge-existence classifier.

The classifier's negatives are ordered face pairs of one CAD that share no edge
(`zero_positions`). Two kinds are rare in that pool but produce most false edges:
pairs that share a vertex without sharing an edge, and pairs whose GT grids come
within `near_threshold` without sharing a vertex. This module labels them once from
the GT data.npz and the vertex sidecar, and composes a sample's negative set at
training time so every hard pair is present `repeat` times.

    python -m src.preprocess.hard_negatives --data-root /mnt/d/data/deepcad_v7 \
        --vertex-root /mnt/d/data/deepcad_v7_vertex_groups/v2_20261112 \
        --split src/preprocess/splits/deepcad_v7_contract_train.txt \
        --split src/preprocess/splits/deepcad_v7_contract_validation.txt \
        --output-root /mnt/d/data/deepcad_v7_hard_negatives/v1_20261126
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from src.common.io import sha256_file

SCHEMA_VERSION = 1


def face_vertex_sets(num_faces, face_pairs, endpoint_vertex_ids):
    """Native vertex ids touching each face, from the vertex sidecar's canonical edges."""
    sets = [set() for _ in range(num_faces)]
    for (a, b), (u, v) in zip(face_pairs, endpoint_vertex_ids):
        for vertex in (int(u), int(v)):
            if vertex >= 0:
                sets[int(a)].add(vertex)
                sets[int(b)].add(vertex)
    return sets


def grid_min_distances(face_grids):
    """[F, F] smallest distance between any two grid points of two faces."""
    points = np.asarray(face_grids, dtype=np.float32).reshape(len(face_grids), -1, 3)
    out = np.zeros((len(points), len(points)), dtype=np.float32)
    for i in range(len(points)):
        diff = points[i][None, :, None, :] - points[:, None, :, :]
        out[i] = np.sqrt((diff ** 2).sum(-1)).min(axis=(1, 2))
    return out


def hard_negative_pairs(face_adj, face_grids, face_pairs, endpoint_vertex_ids, near_threshold):
    """Ordered non-adjacent pairs that share a vertex, and ordered non-adjacent pairs that come close."""
    num_faces = face_adj.shape[0]
    sets = face_vertex_sets(num_faces, face_pairs, endpoint_vertex_ids)
    distance = grid_min_distances(face_grids)
    vertex, near, near_distance = [], [], []
    for i in range(num_faces):
        for j in range(num_faces):
            if i == j or face_adj[i, j]:
                continue
            if sets[i] & sets[j]:
                vertex.append((i, j))
            elif distance[i, j] < near_threshold:
                near.append((i, j))
                near_distance.append(distance[i, j])
    return {
        "vertex_pairs": np.asarray(vertex, dtype=np.int64).reshape(-1, 2),
        "near_pairs": np.asarray(near, dtype=np.int64).reshape(-1, 2),
        "near_distance": np.asarray(near_distance, dtype=np.float32),
        "near_threshold": np.float32(near_threshold),
        "schema_version": np.int64(SCHEMA_VERSION),
    }


def load_hard_negatives(path):
    """All hard ordered pairs of one CAD as [K, 2]."""
    with np.load(path) as data:
        if int(data["schema_version"]) != SCHEMA_VERSION:
            raise ValueError(f"{path}: hard-negative schema {int(data['schema_version'])} != {SCHEMA_VERSION}")
        return np.concatenate((data["vertex_pairs"], data["near_pairs"]), axis=0)


def compose_negatives(pool, hard, count, repeat, rng=np.random):
    """`count` ordered pairs: every hard pair `repeat` times, the rest drawn uniformly from the pool minus the hard pairs.

    Mirrors the plain sampler when there are no hard pairs. Pairs are never invented: hard pairs
    are a subset of the pool, and the uniform fill keeps the pool's remaining rows, diagonal included.
    """
    pool = np.asarray(pool, dtype=np.int64).reshape(-1, 2)
    hard = np.asarray(hard, dtype=np.int64).reshape(-1, 2)
    if len(hard) == 0:
        if len(pool) <= count:
            return pool
        return pool[rng.choice(len(pool), count, replace=False)]
    block = np.repeat(hard, repeat, axis=0)
    if len(block) >= count:
        return block if len(block) == count else block[rng.choice(len(block), count, replace=False)]
    width = int(max(pool.max(initial=0), hard.max(initial=0))) + 1
    keys = pool[:, 0] * width + pool[:, 1]
    rest = pool[~np.isin(keys, hard[:, 0] * width + hard[:, 1])]
    fill = rest[rng.choice(len(rest), min(count - len(block), len(rest)), replace=False)]
    return np.concatenate((block, fill), axis=0)


def label_one(args):
    data_root, vertex_root, output_root, near_threshold, sid = args
    try:
        with np.load(Path(data_root) / sid / "data.npz") as data:
            face_adj, grids = data["face_adj"].astype(bool), data["sample_points_faces"][..., :3]
        with np.load(Path(vertex_root) / sid / "vertex_groups.npz") as vertex:
            face_pairs, endpoints = vertex["face_pairs"], vertex["endpoint_vertex_ids"]
        labels = hard_negative_pairs(face_adj, grids, face_pairs, endpoints, near_threshold)
        target = Path(output_root) / sid
        target.mkdir()
        np.savez_compressed(target / "hard_negatives.npz", **labels)
        return {"id": sid, "status": "ok", "faces": int(face_adj.shape[0]),
                "vertex_pairs": int(len(labels["vertex_pairs"])), "near_pairs": int(len(labels["near_pairs"]))}
    except Exception as exc:  # one CAD must not stop the batch; the report keeps the reason
        return {"id": sid, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--vertex-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--near-threshold", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=16)
    args = parser.parse_args()
    if args.output_root.exists():
        parser.error("Output directory exists; choose a new label version")
    ids = list(dict.fromkeys(sid for split in args.split for sid in split.read_text().split()))
    if any(len(sid) != 8 or not sid.isdigit() for sid in ids):
        parser.error("Split must contain eight-digit CAD IDs")
    args.output_root.mkdir(parents=True)
    jobs = [(str(args.data_root), str(args.vertex_root), str(args.output_root), args.near_threshold, sid) for sid in ids]
    results = []
    with Pool(args.num_workers) as pool:
        for index, row in enumerate(pool.imap_unordered(label_one, jobs, chunksize=16), 1):
            results.append(row)
            if index % 2000 == 0 or index == len(jobs):
                print(f"{index}/{len(jobs)}", flush=True)
    ok = [row for row in results if row["status"] == "ok"]
    report = {"parameters": {key: str(value) for key, value in vars(args).items()},
              "split_sha256": {str(p): sha256_file(p) for p in args.split},
              "generator_sha256": sha256_file(Path(__file__)), "schema_version": SCHEMA_VERSION,
              "summary": {"ok": len(ok), "failed": len(results) - len(ok),
                          "with_vertex_pairs": sum(row["vertex_pairs"] > 0 for row in ok),
                          "with_near_pairs": sum(row["near_pairs"] > 0 for row in ok),
                          "vertex_pairs": sum(row["vertex_pairs"] for row in ok),
                          "near_pairs": sum(row["near_pairs"] for row in ok)},
              "samples": sorted(results, key=lambda row: row["id"])}
    (args.output_root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"]))
    return int(len(ok) != len(results))


if __name__ == "__main__":
    raise SystemExit(main())
