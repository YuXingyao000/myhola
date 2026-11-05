"""Raw data fidelity, frozen endpoint relations and matched boundary policies."""
from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("OCC")
from src.postprocess import construct_brep as post
from src.postprocess import continuous_occ as adapter
from src.postprocess.continuous_geometry import Config


def planar_shape():
    t = np.linspace(0, 1, 16)
    u, v = np.meshgrid(t, t, indexing="ij")
    faces = np.stack((np.stack((u, v, np.zeros_like(u)), -1),
                      np.stack((u, np.zeros_like(u), v), -1)))
    edges = np.stack((t, np.zeros_like(t), np.zeros_like(t)), -1)[None]
    return SimpleNamespace(recon_face_points=faces, recon_edge_points=edges,
                           edge_face_connectivity=np.array([[0, 0, 1]]),
                           face_edge_adj=[[0], [0]], pair1=None, replace_edge_idx=[])


def test_cascade_keeps_raw_observations_and_legacy_initial_poles(tmp_path):
    shape = planar_shape()
    raw_faces, raw_edges = shape.recon_face_points.copy(), shape.recon_edge_points.copy()
    offset = np.array([.03, .02, .01])
    shape.recon_face_points += offset
    shape.recon_edge_points += offset
    shape.recon_geom_faces = [post.create_surface(points) for points in shape.recon_face_points]
    shape.recon_geom_curves = [post.create_edge(points) for points in shape.recon_edge_points]
    initial = [adapter.from_occ_surface(face).poles.copy() for face in shape.recon_geom_faces]
    path = tmp_path / "continuous.json"
    config = Config(iterations=2, freeze_surface_iterations=0, surface_move_limit=.005)
    adapter.refine_occ_geometry(shape, asdict(config), path, raw_face_points=raw_faces,
                                raw_edge_points=raw_edges, endpoint_pairs=np.empty((0, 4), dtype=int))
    with np.load(path.with_suffix(".npz")) as saved:
        np.testing.assert_array_equal(saved["pre_legacy_face_points"], raw_faces)
        np.testing.assert_array_equal(saved["post_legacy_edge_points"], raw_edges + offset)
        np.testing.assert_array_equal(saved["curve_0_observation_xyz"], raw_edges[0])
        assert saved["endpoint_pairs"].shape == (0, 4)  # Do not recompute nearest pairs.
        for i, original in enumerate(initial):
            np.testing.assert_array_equal(saved[f"surface_{i}_reference_poles"], original)
            np.testing.assert_array_equal(saved[f"surface_{i}_observation_xyz"], raw_faces[i].reshape(-1, 3))
            assert abs(saved[f"surface_{i}_poles"] - original).max() <= .005 + config.constraint_tolerance
            np.testing.assert_allclose(adapter.from_occ_surface(shape.recon_geom_faces[i]).poles,
                                       saved[f"surface_{i}_poles"], rtol=0, atol=1e-14)
    report = json.loads(path.read_text())
    assert report["data_reference"] == "pre_legacy_raw_xyz"
    assert report["raw_initial_stage"] == "post_legacy_fit_before_continuous"
    assert report["cascade"]["curve"]["legacy_sample_displacement_from_raw"]["rms"] == pytest.approx(np.linalg.norm(offset))
    assert report["final"]["objective"] <= report["raw_initial"]["objective"] + config.descent_tolerance


@pytest.mark.parametrize("optimizer,optimized,preserve,expected", [
    ("legacy", True, False, ["legacy", "fit_replace", "trim_replace"]),
    ("legacy", True, True, ["legacy", "fit_preserve", "trim_preserve"]),
    ("continuous", True, False, ["fit_preserve", "continuous", "trim_preserve"]),
    ("continuous", True, True, ["fit_preserve", "continuous", "trim_preserve"]),
    ("legacy_continuous", True, True, ["legacy", "fit_preserve", "continuous", "trim_preserve"]),
    ("legacy_continuous", True, False, ["legacy", "fit_preserve", "continuous", "trim_preserve"]),
    ("legacy", False, True, ["fit_replace", "trim_replace"]),
    ("continuous", False, True, ["fit_replace", "trim_replace"]),
    ("legacy_continuous", False, True, ["fit_replace", "trim_replace"]),
])
def test_complete_path_order_and_boundary_policy(monkeypatch, tmp_path, optimizer, optimized, preserve, expected):
    shape = planar_shape()
    original_faces, original_edges = shape.recon_face_points.copy(), shape.recon_edge_points.copy()
    raw_pairs = adapter.soft_endpoint_pairs(shape)
    events = []
    monkeypatch.setattr(post, "load_prediction_shape", lambda path: shape)
    monkeypatch.setattr(post, "_prepare_shape_topology", lambda *args: None)

    def legacy(current, *args):
        events.append("legacy")
        # Deliberately mutate in place: captured observations must own copies.
        current.recon_face_points += .02
        current.recon_edge_points += .03
        current.face_edge_adj = []  # Detect any accidental re-pairing afterwards.

    def fit(current, *, fitting_policy, replace_natural_edges):
        events.append("fit_replace" if replace_natural_edges else "fit_preserve")
        if optimizer in {"legacy", "legacy_continuous"} and optimized:
            np.testing.assert_allclose(current.recon_edge_points, original_edges + .03)
        return ([], [], [], [])

    def continuous(current, config, path, **reference):
        events.append("continuous")
        if optimizer == "legacy_continuous":
            np.testing.assert_array_equal(reference["raw_face_points"], original_faces)
            np.testing.assert_array_equal(reference["raw_edge_points"], original_edges)
            np.testing.assert_array_equal(reference["endpoint_pairs"], raw_pairs)
        else:
            assert reference == {}
        return ([], [], [], [])

    def trim(faces, curves, connectivity, report, flush, deadline, *, allow_natural_boundary):
        events.append("trim_replace" if allow_natural_boundary else "trim_preserve")
        report["complete_faces"] = False
        return []

    monkeypatch.setattr(post, "_optimize_shape_geometry", legacy)
    monkeypatch.setattr(post, "_fit_occ_geometry", fit)
    monkeypatch.setattr(adapter, "refine_occ_geometry", continuous)
    monkeypatch.setattr(post, "_build_complete_trimmed_faces", trim)
    result = post._run_complete_post_path(
        tmp_path, tmp_path / "output", "sample", optimized=optimized, fitting_policy="legacy",
        max_iterations=200, use_cuda=False, report={}, flush=lambda: None, deadline=None,
        geometry_optimizer=optimizer, preserve_optimized_boundaries=preserve)
    assert not result
    assert events == expected
