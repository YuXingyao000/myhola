"""Deterministic UV-chart serialization; physical coedge uses stay distinct."""
import numpy as np

EDGE, CLOSE, END, BOS, PAD = range(5)


def canonical_loops(owner, nxt, uv, num_faces):
    if not np.array_equal(np.sort(nxt), np.arange(len(nxt))) or np.any(owner[nxt] != owner):
        raise ValueError("GT next must be a permutation within each owner")
    remaining = set(range(len(owner)))
    by_face = [[] for _ in range(num_faces)]
    while remaining:
        first = min(remaining)
        loop, h = [], first
        while h in remaining:
            remaining.remove(h)
            loop.append(h)
            h = int(nxt[h])
        keys = [tuple(uv[i].ravel()) for i in loop]
        smallest = min(keys)
        starts = [i for i, key in enumerate(keys) if key == smallest]
        start = min(starts, key=lambda i: tuple(keys[i:] + keys[:i]))
        loop = loop[start:] + loop[:start]
        polygon = uv[loop].reshape(-1, 2)
        area = 0.5 * np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1)
                            - polygon[:, 1] * np.roll(polygon[:, 0], -1))
        by_face[int(owner[first])].append((loop, abs(area), tuple(polygon.ravel())))
    if any(not loops for loops in by_face):
        raise ValueError("GT face without a boundary loop")
    return [[x[0] for x in sorted(loops, key=lambda x: (-x[1], x[2]))] for loops in by_face]


def relations_from_events(events):
    """Decode owner and next; an unfinished face is never silently closed."""
    owner, nxt, finished = [], [], []
    for face, sequence in enumerate(events):
        loop, closed, done = [], 0, False
        for event in sequence:
            if event == EDGE:
                loop.append(len(owner))
                owner.append(face)
                nxt.append(-1)
            elif event == CLOSE and loop:
                for a, b in zip(loop, loop[1:] + loop[:1]):
                    nxt[a] = b
                loop = []
                closed += 1
            elif event == END and not loop and closed:
                done = True
                break
            else:
                break
        finished.append(done)
    return np.asarray(owner, dtype=np.int64), np.asarray(nxt, dtype=np.int64), np.asarray(finished)
