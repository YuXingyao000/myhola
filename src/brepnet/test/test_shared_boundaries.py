"""Target contract checks, independent of OCC and model training."""
import unittest

import numpy as np

from src.brepnet.data.shared_boundaries import build_shared_boundary_targets, shared_prediction_diagnostics


def make_boundary_arrays():
    u, v = np.meshgrid(np.linspace(-1, 1, 16), np.linspace(-1, 1, 16))
    faces = np.zeros((2, 16, 16, 6), dtype=np.float32)
    faces[:, :, :, 0] = u
    faces[:, :, :, 1] = v
    faces[1, :, :, 2] = .1
    faces[:, :, :, 5] = 1
    edges = np.zeros((2, 16, 6), dtype=np.float32)
    edges[0, :, 0] = np.linspace(-1, 1, 16)
    edges[0, :, 3] = 1
    edges[1] = edges[0, ::-1]
    return faces, edges, np.array([[0, 0, 1], [1, 1, 0]], dtype=np.int64)


class SharedBoundaryTargetTests(unittest.TestCase):
    def build(self, faces, edges, conn, **kwargs):
        args = dict(sample_id="sample-A", reciprocal_tolerance=.01, closed_tolerance=1e-5, allow_closed_phase=True)
        args.update(kwargs)
        return build_shared_boundary_targets(faces, edges, conn, **args)

    def test_reciprocal_rows_preserve_encoder_data_and_local_ids(self):
        faces, edges, conn = make_boundary_arrays()
        original = edges.copy()
        target = self.build(faces, edges, conn[::-1])
        np.testing.assert_array_equal(target["curves"], edges[:1, :, :3])
        np.testing.assert_array_equal(target["face_pairs"], [[0, 1]])
        np.testing.assert_array_equal(target["source_edge_ids"], [0])
        np.testing.assert_array_equal(edges, original)
        self.assertFalse(target["is_closed"][0])

    def test_closed_curve_start_phase_is_explicit(self):
        faces, edges, conn = make_boundary_arrays()
        angles = np.linspace(0, 2 * np.pi, 16)
        circle = .3 * np.stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)), axis=-1)
        edges[0, :, :3] = circle
        other = np.roll(circle[:-1], 4, axis=0)[::-1]
        edges[1, :, :3] = np.concatenate((other, other[:1]), axis=0)
        target = self.build(faces, edges, conn)
        self.assertTrue(target["is_closed"][0])
        self.assertTrue(target["closed_phase_used"][0])
        np.testing.assert_array_equal(target["curves"][0], edges[0, :, :3])
        with self.assertRaisesRegex(ValueError, "sample-A.*reciprocal residual"):
            self.build(faces, edges, conn, allow_closed_phase=False)

    def test_wrong_geometry_not_accepted_as_closed_phase(self):
        faces, edges, conn = make_boundary_arrays()
        edges[1, :, 1] += .2
        with self.assertRaisesRegex(ValueError, "sample-A.*face pair.*exceeds"):
            self.build(faces, edges, conn)

    def test_empty_boundaries_keep_shapes(self):
        faces, edges, conn = make_boundary_arrays()
        target = self.build(faces, edges[:0], conn[:0])
        self.assertEqual(target["curves"].shape, (0, 16, 3))
        self.assertEqual(target["face_pairs"].shape, (0, 2))
        self.assertEqual(target["is_closed"].shape, (0,))

    def test_unsupported_topology_fails_with_sample_id(self):
        faces, edges, conn = make_boundary_arrays()
        cases = [
            (edges[:1], conn[:1], "exactly two reciprocal"),
            (edges, np.array([[0, 0, 1], [1, 0, 1]]), "exactly two reciprocal"),
            (edges, np.array([[0, 0, 0], [1, 0, 0]]), "self-face/seam"),
            (edges, np.array([[0, 0, 3], [1, 3, 0]]), "face index"),
            (edges, np.array([[0, 0, 1], [0, 1, 0]]), "occurs more than once"),
        ]
        for bad_edges, bad_conn, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, "sample-A.*" + message):
                self.build(faces, bad_edges, bad_conn)

    def test_raw_diagnostics_do_not_hide_duplicate_or_self_queries(self):
        curves = np.zeros((4, 16, 3), dtype=np.float32)
        conn = np.array([[0, 0, 1], [1, 1, 0], [2, 0, 0], [3, 0, 9]])
        result = shared_prediction_diagnostics(2, curves, conn, np.ones(4))
        self.assertEqual(result["num_predicted_queries"], 4)
        self.assertEqual(result["num_duplicate_pair_assignments"], 1)
        self.assertEqual(result["num_self_assignments"], 1)
        self.assertEqual(result["num_invalid_face_assignments"], 1)


if __name__ == "__main__":
    unittest.main()
