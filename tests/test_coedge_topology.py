"""Topology contraction and provenance tests, independent of training weights."""
import numpy as np
import pytest

from src.preprocess.coedge_topology import (
    SCHEMA_VERSION, curve_orientation, load_coedge_topology, quotient_coedges,
)


def cylinder_quotient():
    # Side wire: top circle, seam forward, bottom circle, seam reversed.
    # Two caps each contain one circle; mates distinguish both seam uses.
    owner = np.array([0, 0, 0, 0, 1, 2])
    following = np.array([1, 2, 3, 0, 4, 5])
    mate = np.array([4, 3, 5, 1, 0, 2])
    connectivity = np.array([[0, 0, 1], [1, 1, 0], [2, 0, 2], [3, 2, 0]])
    return quotient_coedges(owner, following, mate, connectivity), connectivity


def test_periodic_seam_splicing_preserves_separate_boundary_rings():
    result, _ = cylinder_quotient()
    assert result["seam_removed_count"] == 2
    np.testing.assert_array_equal(result["previous"], np.arange(4))
    np.testing.assert_array_equal(result["next"], np.arange(4))
    np.testing.assert_array_equal(result["mate"], [1, 0, 3, 2])


def test_consecutive_native_segments_contract_to_one_retained_curve():
    result = quotient_coedges([0, 0, 0, 1, 1, 1], [1, 2, 0, 4, 5, 3],
                              [5, 4, 3, 2, 1, 0], [[0, 0, 1], [1, 1, 0]])
    assert [len(chain) for chain in result["native_chains"]] == [3, 3]
    np.testing.assert_array_equal(result["next"], [0, 1])


def test_disjoint_same_pair_positions_are_not_silently_merged():
    with pytest.raises(ValueError, match="multiple disjoint"):
        quotient_coedges([0, 0, 1, 1], [0, 1, 2, 3], [2, 3, 0, 1], [[0, 0, 1], [1, 1, 0]])


def test_seam_only_surface_has_no_retained_slots():
    result = quotient_coedges([0, 0], [1, 0], [1, 0], np.empty((0, 3), dtype=np.int64))
    assert len(result["next"]) == 0
    assert result["seam_removed_count"] == 2


def test_missing_native_pair_fails_instead_of_self_loop_fallback():
    with pytest.raises(ValueError, match="no retained face pair"):
        quotient_coedges([0, 1], [0, 1], [1, 0], np.empty((0, 3), dtype=np.int64))


@pytest.mark.parametrize("reverse, expected", [(False, 1), (True, -1)])
def test_closed_curve_orientation_is_independent_of_parameter_cut(reverse, expected):
    t = np.linspace(0, 2 * np.pi, 257)
    native = np.column_stack((np.cos(t), np.sin(t), np.zeros_like(t)))
    q = np.linspace(0.71, 2 * np.pi + 0.71, 16)
    points = np.column_stack((np.cos(q), np.sin(q), np.zeros_like(q)))
    if reverse:
        points = points[::-1]
    sign, score, _ = curve_orientation(points, native[:-1], native[1:])
    assert sign == expected
    assert abs(score) > 0.99


def test_ambiguous_direction_is_rejected():
    with pytest.raises(ValueError, match="Ambiguous"):
        curve_orientation([[0, 0, 0], [0, 1, 0]], [[-1, 0, 0]], [[1, 0, 0]])


def write_cache(path, **changes):
    result, connectivity = cylinder_quotient()
    chains = result.pop("native_chains")
    arrays = {**result, "schema_version": SCHEMA_VERSION, "source_sha256": "source",
              "step_sha256": "step", "source_edge_ids": np.arange(4),
              "owner_faces": connectivity[:, 1], "orientation": np.array([1, 1, -1, 1]),
              "native_chain_lengths": np.array([len(chain) for chain in chains])}
    arrays.update(changes)
    np.savez(path, **arrays)
    return connectivity


def test_loader_returns_raw_edge_order_and_explicit_roles(tmp_path):
    path = tmp_path / "coedge.npz"
    connectivity = write_cache(path)
    data = load_coedge_topology(path, source_sha256="source", connectivity=connectivity[::-1])
    assert data["coedge_neighbors"].shape == (4, 3)
    np.testing.assert_array_equal(data["coedge_neighbors"][:, 2], [1, 0, 3, 2])
    np.testing.assert_array_equal(data["coedge_orientation"][:, 0], [1, 1, -1, 1])


@pytest.mark.parametrize("changes", [
    {"source_sha256": "other"}, {"next": np.array([1, 0, 2, 3])},
    {"orientation": np.array([1, 0, 1, 1])}, {"mate": np.array([0, 1, 2, 3])},
])
def test_corrupted_cache_is_rejected(tmp_path, changes):
    path = tmp_path / "coedge.npz"
    connectivity = write_cache(path, **changes)
    with pytest.raises(ValueError):
        load_coedge_topology(path, source_sha256="source", connectivity=connectivity)


def test_real_occ_cylinder_seam_quotient():
    pytest.importorskip("OCC")
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from src.preprocess.extract_coedge_topology import native_coedges
    _, occurrences, owner, following, mate = native_coedges(BRepPrimAPI_MakeCylinder(1.0, 2.0).Shape())
    pairs = sorted({(int(owner[h]), int(owner[mate[h]])) for h in range(len(owner))
                    if owner[h] != owner[mate[h]]})
    connectivity = np.asarray([(i, a, b) for i, (a, b) in enumerate(pairs)])
    result = quotient_coedges(owner, following, mate, connectivity)
    assert len(occurrences) == 6
    assert result["seam_removed_count"] == 2
    np.testing.assert_array_equal(result["next"], np.arange(4))


def test_real_occ_box_keeps_four_coedges_per_face():
    pytest.importorskip("OCC")
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    from src.preprocess.extract_coedge_topology import native_coedges
    _, occurrences, owner, following, mate = native_coedges(BRepPrimAPI_MakeBox(1., 2., 3.).Shape())
    pairs = sorted({(int(owner[h]), int(owner[mate[h]])) for h in range(len(owner))})
    connectivity = np.asarray([(i, a, b) for i, (a, b) in enumerate(pairs)])
    result = quotient_coedges(owner, following, mate, connectivity)
    assert len(occurrences) == 24
    assert result["seam_removed_count"] == 0
    assert all(len(chain) == 1 for chain in result["native_chains"])
    cursor = np.arange(24)
    for _ in range(4):
        cursor = result["next"][cursor]
    np.testing.assert_array_equal(cursor, np.arange(24))
    assert not np.any(result["next"] == np.arange(24))
