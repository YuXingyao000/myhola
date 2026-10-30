"""Shared-query post selection is explicit and never uses target topology."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.preprocess.shared_boundaries import SHARED_BOUNDARY_REPRESENTATION
from src.postprocess.construct_brep import _prepare_shape_topology, load_prediction_shape
from src.postprocess.shape import Shape
from tests.test_shared_boundaries import make_boundary_arrays


class SharedBoundaryPostTests(unittest.TestCase):
    def test_more_than_two_queries_select_by_confidence_after_geometry_gate(self):
        shape = Shape.__new__(Shape)
        shape.representation = SHARED_BOUNDARY_REPRESENTATION
        shape.recon_face_points = np.zeros((2, 16, 16, 3))
        shape.recon_edge_points = np.zeros((5, 16, 3))
        shape.edge_face_connectivity = np.array([[0, 0, 1], [1, 1, 0], [2, 0, 1], [3, 0, 0], [4, 0, 8]])
        shape.edge_confidence = np.array([.99, .8, .7, .99, .99])
        distances = [.4, .2, .1]
        shape._edge_to_face_distance = lambda edge, *_: distances[edge]
        shape.select_shared_boundaries(edge2face_threshold=.3)
        np.testing.assert_array_equal(shape.selected_source_edge_ids, [1])
        np.testing.assert_array_equal(shape.edge_face_connectivity, [[0, 0, 1]])
        self.assertEqual(shape.post_adaptation_diagnostics["num_duplicate_pair_assignments"], 2)
        self.assertEqual(shape.post_adaptation_diagnostics["num_geometry_rejected"], 1)
        self.assertEqual(shape.post_adaptation_diagnostics["num_removed_queries"], 4)

    def test_tagged_npz_stays_raw_until_explicit_post_adaptation(self):
        output = Path(tempfile.gettempdir()) / "hola_tests"
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="post-", dir=output) as directory:
            directory = Path(directory)
            faces, edges, conn = make_boundary_arrays()
            raw_path = directory / "data.npz"
            np.savez(raw_path, representation=np.array(SHARED_BOUNDARY_REPRESENTATION), pred_face=faces[..., :3],
                     pred_edge=edges[..., :3], pred_edge_face_connectivity=conn, pred_edge_confidence=np.array([.7, .8]))
            raw_bytes = raw_path.read_bytes()
            shape = load_prediction_shape(raw_path)
            self.assertEqual(len(shape.recon_edge_points), 2)
            self.assertEqual(shape.representation, SHARED_BOUNDARY_REPRESENTATION)
            shape.select_shared_boundaries()
            self.assertEqual(len(shape.recon_edge_points), 1)
            self.assertEqual(raw_bytes, raw_path.read_bytes())

    def test_zero_queries_report_clear_post_failure(self):
        output = Path(tempfile.gettempdir()) / "hola_tests"
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="post-empty-", dir=output) as directory:
            faces, edges, conn = make_boundary_arrays()
            raw_path = Path(directory) / "data.npz"
            np.savez(raw_path, representation=np.array(SHARED_BOUNDARY_REPRESENTATION), pred_face=faces[..., :3],
                     pred_edge=edges[:0, :, :3], pred_edge_face_connectivity=conn[:0], pred_edge_confidence=np.empty(0))
            shape = load_prediction_shape(raw_path)
            with self.assertRaisesRegex(ValueError, "no usable predicted boundaries"):
                _prepare_shape_topology(shape, Path(directory) / "debug", False, False)
            self.assertTrue((Path(directory) / "shared_boundary_post.json").is_file())


if __name__ == "__main__":
    unittest.main()
