"""Native-loop topology projected onto the VAE's retained directed curves.

Only this NumPy module is imported by training. STEP traversal belongs to the
separate offline generator; spatial proximity never defines a neighbor here.
"""
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1


def quotient_coedges(owner, next_coedge, mate, connectivity):
    """Remove native seams, then contract consecutive face-pair chains.

    Crossing a removed seam continues after its *mate*. Simply skipping that
    occurrence would connect distinct boundary rings of a periodic face.
    Returned chains contain native occurrence IDs in their owner-loop order.
    """
    owner, next_coedge, mate, connectivity = [np.asarray(x, dtype=np.int64)
                                               for x in (owner, next_coedge, mate, connectivity)]
    count = len(owner)
    ids = np.arange(count)
    if (next_coedge.shape != (count,) or mate.shape != (count,)
            or np.any(next_coedge < 0) or np.any(next_coedge >= count)
            or np.any(mate < 0) or np.any(mate >= count)
            or not np.array_equal(np.sort(next_coedge), ids)
            or not np.array_equal(mate[mate], ids) or np.any(mate == ids)):
        raise ValueError("Native next/mate must be permutations with distinct involutive mates")
    if np.any(owner[next_coedge] != owner):
        raise ValueError("Native loop crosses owner faces")
    seam = owner == owner[mate]
    row_count = len(connectivity)
    if (connectivity.shape != (row_count, 3)
            or not np.array_equal(np.sort(connectivity[:, 0]), np.arange(row_count))):
        raise ValueError("Retained connectivity must cover each raw edge exactly once")
    pair_to_row = {tuple(row[1:]): int(row[0]) for row in connectivity}
    if len(pair_to_row) != row_count or any(a == b for a, b in pair_to_row):
        raise ValueError("Retained directed face pairs must be unique and non-seam")
    row_of_native = np.full(count, -1, dtype=np.int64)
    for h in np.flatnonzero(~seam):
        pair = int(owner[h]), int(owner[mate[h]])
        if pair not in pair_to_row:
            raise ValueError(f"Native boundary has no retained face pair: {pair}")
        row_of_native[h] = pair_to_row[pair]
    successor = np.full(count, -1, dtype=np.int64)
    for h in np.flatnonzero(~seam):
        q, seen = int(next_coedge[h]), set()
        while seam[q]:
            if q in seen:
                raise ValueError("Retained boundary enters a seam-only cycle")
            seen.add(q)
            q = int(next_coedge[mate[q]])
        successor[h] = q
    retained = np.flatnonzero(~seam)
    if not np.array_equal(np.sort(successor[retained]), retained):
        raise ValueError("Seam elimination did not produce disjoint boundary cycles")
    previous = np.full(row_count, -1, dtype=np.int64)
    following = np.full(row_count, -1, dtype=np.int64)
    chains = [None] * row_count
    visited = set()
    for start in retained:
        if int(start) in visited:
            continue
        cycle, h = [], int(start)
        while h not in visited:
            visited.add(h)
            cycle.append(h)
            h = int(successor[h])
        if h != start:
            raise ValueError("Boundary cycle entered another cycle")
        cuts = [k for k, h in enumerate(cycle) if row_of_native[h] != row_of_native[cycle[k - 1]]]
        if not cuts:
            groups = [cycle]
        else:
            cycle = cycle[cuts[0]:] + cycle[:cuts[0]]
            groups = [[cycle[0]]]
            for h in cycle[1:]:
                if row_of_native[h] != row_of_native[groups[-1][-1]]:
                    groups.append([])
                groups[-1].append(h)
        rows = [int(row_of_native[group[0]]) for group in groups]
        for k, (row, group) in enumerate(zip(rows, groups)):
            if chains[row] is not None:
                raise ValueError(f"Retained raw edge {row} occupies multiple disjoint loop positions")
            chains[row] = np.asarray(group, dtype=np.int64)
            previous[row] = rows[k - 1]
            following[row] = rows[(k + 1) % len(rows)]
    if any(chain is None for chain in chains):
        raise ValueError("Some retained curves have no native boundary chain")
    row_mate = np.empty(row_count, dtype=np.int64)
    for (a, b), row in pair_to_row.items():
        if (b, a) not in pair_to_row:
            raise ValueError("Retained face pair lacks its reverse")
        row_mate[row] = pair_to_row[(b, a)]
        native_other = row_of_native[mate[chains[row]]]
        if np.any(native_other != row_mate[row]):
            raise ValueError("Contracted native mates disagree with retained pairs")
    return {"previous": previous, "next": following, "mate": row_mate,
            "native_chains": chains, "seam_removed_count": int(seam.sum())}


def curve_orientation(points, segment_start, segment_end, minimum_cosine=0.5):
    """Check sampling direction against a known chain, including closed curves.

    Nearest segment projections select a tangent on this already identified
    chain. They do not decide curve membership, adjacency, or vertex identity.
    """
    points, segment_start, segment_end = [np.asarray(x, dtype=np.float64)
                                         for x in (points, segment_start, segment_end)]
    tangent = segment_end - segment_start
    lengths2 = np.sum(tangent * tangent, axis=-1)
    valid = lengths2 > 1e-24
    segment_start, tangent, lengths2 = segment_start[valid], tangent[valid], lengths2[valid]
    chord = np.diff(points, axis=0)
    chord_length = np.linalg.norm(chord, axis=-1)
    valid = chord_length > 1e-12
    midpoint = ((points[:-1] + points[1:]) * 0.5)[valid]
    chord, chord_length = chord[valid], chord_length[valid]
    if not len(chord) or not len(tangent):
        raise ValueError("Cannot orient a zero-length retained/native curve")
    delta = midpoint[:, None] - segment_start[None]
    parameter = np.clip(np.sum(delta * tangent[None], axis=-1) / lengths2[None], 0, 1)
    residual = delta - parameter[..., None] * tangent[None]
    distance2 = np.sum(residual * residual, axis=-1)
    nearest = distance2.argmin(axis=1)
    cosine = np.sum(chord * tangent[nearest], axis=-1) / (chord_length * np.sqrt(lengths2[nearest]))
    score = float(np.average(cosine, weights=chord_length))
    error = float(np.sqrt(distance2[np.arange(len(chord)), nearest].max()))
    if not np.isfinite(score) or abs(score) < minimum_cosine:
        raise ValueError(f"Ambiguous retained/native curve direction: cosine={score}")
    return (1 if score > 0 else -1), score, error


def validate_coedge_topology(arrays, *, source_sha256, connectivity, step_sha256=None):
    """Validate cache provenance and index semantics at the data boundary."""
    if int(arrays["schema_version"]) != SCHEMA_VERSION:
        raise ValueError("Unsupported coedge topology schema")
    if arrays["source_sha256"].item() != source_sha256:
        raise ValueError("Coedge source digest mismatch")
    if step_sha256 is not None and arrays["step_sha256"].item() != step_sha256:
        raise ValueError("Coedge STEP digest mismatch")
    connectivity = np.asarray(connectivity)
    count = len(connectivity)
    ordered = connectivity[np.argsort(connectivity[:, 0])]
    ids = np.arange(count)
    if not np.array_equal(ordered[:, 0], ids):
        raise ValueError("Retained connectivity does not cover raw edge IDs")
    if (not np.array_equal(arrays["source_edge_ids"], ids)
            or not np.array_equal(arrays["owner_faces"], ordered[:, 1])):
        raise ValueError("Coedge raw edge/owner order mismatch")
    for key in ("previous", "next", "mate"):
        value = arrays[key]
        if (value.dtype.kind not in "iu" or value.shape != (count,)
                or np.any(value < 0) or np.any(value >= count)):
            raise ValueError(f"Invalid coedge {key} indices")
    previous, following, mate = (arrays[key] for key in ("previous", "next", "mate"))
    owners = ordered[:, 1]
    if (not np.array_equal(following[previous], ids)
            or not np.array_equal(previous[following], ids)
            or not np.array_equal(mate[mate], ids) or np.any(mate == ids)
            or np.any(owners[previous] != owners) or np.any(owners[following] != owners)
            or not np.array_equal(ordered[mate, 1:], ordered[:, 1:][:, ::-1])):
        raise ValueError("Coedge cycle/mate/owner relationships are inconsistent")
    orientation = arrays["orientation"]
    if orientation.shape != (count,) or not np.isin(orientation, (-1, 1)).all():
        raise ValueError("Coedge orientations must be +1 or -1")
    lengths = arrays["native_chain_lengths"]
    if lengths.shape != (count,) or lengths.dtype.kind not in "iu" or np.any(lengths < 1):
        raise ValueError("Invalid native chain lengths")
    return {"coedge_neighbors": np.column_stack((previous, following, mate)).astype(np.int64),
            "coedge_orientation": orientation.astype(np.float32)[:, None]}


def load_coedge_topology(path, *, source_sha256, connectivity):
    with np.load(Path(path), allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    return validate_coedge_topology(arrays, source_sha256=source_sha256, connectivity=connectivity)
