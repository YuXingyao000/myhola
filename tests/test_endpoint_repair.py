"""Targeted endpoint repair: pairing rules, clusters, snapping, dangling edges, recovery policy."""
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.postprocess import construct_brep as post
from src.postprocess.decoded_prediction import interpolation_face_points
from src.postprocess.endpoint_repair import (
    EndpointRepairConfig,
    build_vertex_clusters,
    pair_face_endpoints,
    repair_endpoints,
)
from src.postprocess.geometry_optimization import optimize


def segment(start, end, n=16):
    return np.linspace(start, end, n, dtype=np.float32)


def square(corners, n=16):
    return [segment(corners[i], corners[(i + 1) % 4], n) for i in range(4)]


def plane(origin, u, v, n=16):
    s = np.linspace(0, 1, n)
    return np.array([[np.asarray(origin) + a * np.asarray(u) + b * np.asarray(v) for b in s] for a in s],
                    dtype=np.float32)


def make_shape(edges, connectivity, faces=None):
    edges = np.stack(edges).astype(np.float32)
    connectivity = np.asarray(connectivity, dtype=np.int64)
    num_faces = int(connectivity[:, 1:].max()) + 1
    adjacency = [[] for _ in range(num_faces)]
    for edge, first, second in connectivity:
        adjacency[first].append(int(edge))
        adjacency[second].append(int(edge))
    closed = np.linalg.norm(edges[:, 0] - edges[:, -1], axis=1) < 1e-6
    return types.SimpleNamespace(recon_edge_points=edges, edge_face_connectivity=connectivity,
                                 face_edge_adj=adjacency, openness=closed, remove_edge_idx_new=[],
                                 recon_face_points=faces, pair1=None, is_end_point=None)


class EndpointRepairTests(unittest.TestCase):
    def corner_shape(self):
        # Three edges meeting at the origin; e0 starts 0.1 short of the corner.
        e0 = segment((0.1, 0, 0), (1, 0, 0))
        e1 = segment((0, 0, 0), (0, 1, 0))
        e2 = segment((0, 0, 0), (0, 0, 1))
        faces = np.stack([plane((0, 0, 0), (1, 0, 0), (0, 1, 0)),
                          plane((0, 0, 0), (0, 1, 0), (0, 0, 1)),
                          plane((0, 0, 0), (1, 0, 0), (0, 0, 1))])
        return make_shape([e0, e1, e2], [[0, 0, 2], [1, 0, 1], [2, 1, 2]], faces)

    def test_unambiguous_corner_is_clustered_and_snapped(self):
        shape = self.corner_shape()
        original = shape.recon_edge_points.copy()
        diagnostics, clusters = repair_endpoints(shape)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]["members"], [(0, 0), (1, 0), (2, 0)])
        self.assertAlmostEqual(clusters[0]["diameter"], 0.1, places=6)
        target = clusters[0]["target"]
        np.testing.assert_allclose(target, [0.1 / 3, 0, 0], atol=1e-6)
        for edge in range(3):
            np.testing.assert_allclose(shape.recon_edge_points[edge][0], target, atol=1e-6)
            np.testing.assert_allclose(shape.recon_edge_points[edge][-1], original[edge][-1], atol=1e-6)
        # Interior follows a chord-length linear blend of the endpoint displacements.
        mid = shape.recon_edge_points[0][8]
        expected = original[0][8] + (1 - 8 / 15) * (target - original[0][0])
        np.testing.assert_allclose(mid, expected, atol=1e-5)
        self.assertEqual(diagnostics["num_snapped_endpoints"], 3)
        self.assertEqual(diagnostics["dropped_edges"], [])
        self.assertEqual(diagnostics["num_dangling_endpoints"], 6)

    def test_ambiguous_endpoint_gets_no_pair(self):
        e0 = segment((0, 0, 0), (-1, 0, 0))
        e1 = segment((0.1, 0, 0), (0.1, 1, 0))
        e2 = segment((0, 0.105, 0), (1, 0.105, 0))
        pairs, dangling = pair_face_endpoints(np.stack([e0, e1, e2]), [0, 1, 2], np.zeros(3, bool))
        self.assertEqual(pairs, [])
        self.assertEqual(len(dangling), 6)
        # Widening the margin requirement away makes the closer candidate win.
        pairs, _ = pair_face_endpoints(np.stack([e0, e1, e2]), [0, 1, 2], np.zeros(3, bool),
                                       EndpointRepairConfig(ambiguity_margin=0.0))
        self.assertEqual([(a, b) for a, b, _ in pairs], [((0, 0), (1, 0))])

    def test_dangling_edge_in_both_faces_is_dropped(self):
        outer = square([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
        other = square([(3, 0, 0), (4, 0, 0), (4, 1, 0), (3, 1, 0)])
        stray = segment((0.4, 0.4, 0), (0.6, 0.6, 0))
        edges = outer + other + [stray]
        connectivity = [[i, 0, 2 + i] for i in range(4)] + [[4 + i, 1, 6 + i] for i in range(4)] + [[8, 0, 1]]
        shape = make_shape(edges, connectivity)
        diagnostics, clusters = repair_endpoints(shape)
        self.assertEqual(diagnostics["dropped_edges"], [8])
        self.assertNotIn(8, shape.face_edge_adj[0])
        self.assertNotIn(8, shape.face_edge_adj[1])
        self.assertFalse((shape.edge_face_connectivity[:, 0] == 8).any())
        self.assertEqual(shape.remove_edge_idx_new, [8])
        self.assertEqual(len(clusters), 8)
        self.assertTrue(all(c["diameter"] == 0 for c in clusters))

    def test_closed_edges_have_no_endpoints(self):
        angles = np.linspace(0, 2 * np.pi, 16)
        circle = np.stack([np.cos(angles), np.sin(angles), np.zeros(16)], axis=1).astype(np.float32)
        line = segment((1, 0, 0), (2, 0, 0))
        shape = make_shape([circle, line], [[0, 0, 1], [1, 0, 2]])
        self.assertTrue(shape.openness[0])
        diagnostics, clusters = repair_endpoints(shape)
        self.assertEqual(clusters, [])
        self.assertEqual(diagnostics["dropped_edges"], [])

    def test_optimizer_accepts_vertex_clusters(self):
        shape = self.corner_shape()
        clusters, _, _ = build_vertex_clusters(shape.recon_edge_points, shape.face_edge_adj, shape.openness)
        interpolation = [interpolation_face_points(face, is_use_cuda=False) for face in shape.recon_face_points]
        faces, edges = optimize(interpolation, shape.recon_edge_points, shape.recon_face_points,
                                shape.edge_face_connectivity, None, None, shape.face_edge_adj,
                                v_islog=False, v_max_iter=5, use_cuda=False, vertex_clusters=clusters)
        self.assertEqual(edges.shape, shape.recon_edge_points.shape)
        self.assertEqual(faces.shape, shape.recon_face_points.shape)
        self.assertTrue(np.isfinite(edges).all() and np.isfinite(faces).all())

    def test_recover_v3_appends_repair_paths_after_seam_paths(self):
        original_construct = post.construct_brep_for_sample
        seen = []

        def legacy(data_root, out_root, sample_id, **kwargs):
            Path(out_root, sample_id).mkdir(parents=True, exist_ok=True)

        def complete(*args, report, split_face=True, endpoint_repair=None, **kwargs):
            seen.append((report["label"], split_face, endpoint_repair is not None))
            report["status"] = "incomplete_faces"
            return False

        with tempfile.TemporaryDirectory() as directory, patch.object(post, "construct_brep_for_sample", legacy), \
                patch.object(post, "_run_complete_post_path", complete):
            original_construct("unused", directory, "sample", post_policy="recover_v3")
            diagnostics = json.loads((Path(directory) / "sample/post_diagnostics.json").read_text())
        self.assertEqual(seen[-4:], [("seam_roundtrip_optimized", False, False), ("seam_roundtrip_fit_only", False, False),
                                     ("repair_roundtrip_fit_only", False, True), ("repair_roundtrip_optimized", False, True)])
        self.assertEqual(len(seen), 7)
        self.assertEqual(diagnostics["status"], "no_accepted_solid")
        self.assertTrue(diagnostics["paths"][-1]["endpoint_repair_enabled"])


if __name__ == "__main__":
    unittest.main()
