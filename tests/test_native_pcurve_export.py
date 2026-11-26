"""Native UV export retains seam uses and the original face ordering."""
import json

import numpy as np
import pytest

from src.common.io import sha256_file
from src.preprocess.native_pcurve import SCHEMA, extract_native


def make_source(tmp_path, kind):
    pytest.importorskip("OCC")
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods
    from OCC.Extend.DataExchange import read_step_file, write_step_file

    source = tmp_path / kind
    source.mkdir()
    shape = (BRepPrimAPI_MakeBox(2., 1., 3.) if kind == "box"
             else BRepPrimAPI_MakeCylinder(1., 2.)).Shape()
    step = source / "normalized_shape.step"
    write_step_file(shape, str(step))
    restored = read_step_file(str(step), verbosity=False)
    explorer = TopExp_Explorer(restored, TopAbs_FACE)
    grids = []
    while explorer.More():
        surface = BRepAdaptor_Surface(topods.Face(explorer.Current()))
        us = np.linspace(surface.FirstUParameter(), surface.LastUParameter(), 16)
        vs = np.linspace(surface.FirstVParameter(), surface.LastVParameter(), 16)
        grids.append([[surface.Value(float(u), float(v)).Coord() for u in us] for v in vs])
        explorer.Next()
    # Deliberately use a different NPZ face order from the STEP explorer.
    np.savez(source / "data.npz", sample_points_faces=np.asarray(grids)[::-1])
    labels = source / "vertex_groups.npz"
    np.savez(labels, source_sha256=sha256_file(source / "data.npz"),
             step_sha256=sha256_file(step),
             parameters_json=json.dumps({"face_tolerance": 1e-6}))
    return source, labels


@pytest.mark.parametrize("kind, faces, coedges, vertices", [("box", 6, 24, 8), ("cylinder", 3, 6, 2)])
def test_native_export_topology_and_seam_charts(tmp_path, kind, faces, coedges, vertices):
    source, labels = make_source(tmp_path, kind)
    data, info = extract_native(source, labels)
    assert str(data["schema"]) == SCHEMA
    assert (info["faces"], info["coedges"], info["vertices"]) == (faces, coedges, vertices)
    owner, mate, following = data["owner"], data["mate"], data["next"]
    np.testing.assert_array_equal(mate[mate], np.arange(coedges))
    np.testing.assert_array_equal(np.sort(following), np.arange(coedges))
    np.testing.assert_array_equal(owner[following], owner)
    np.testing.assert_array_equal(data["vertex"][following[mate]], data["vertex"])
    np.testing.assert_allclose(data["sample_points_lines"][..., :3],
                               data["sample_points_lines"][mate, ::-1, :3], atol=1e-6)
    assert owner[0] == faces - 1
    assert info["face_error"] < 1e-6
    assert info["trace_error"] < 1e-6
    assert info["uv_gap"] < 1e-6
    seams = np.flatnonzero(owner == owner[mate])
    if kind == "box":
        assert len(seams) == 0
    else:
        assert len(seams) == 2
        u = data["pcurve_uv"][seams, :, 0]
        np.testing.assert_allclose(np.sort(u.mean(axis=1)), [0, 1], atol=1e-6)


def test_native_export_rejects_changed_source(tmp_path):
    source, labels = make_source(tmp_path, "box")
    with np.load(source / "data.npz") as data:
        grids = data["sample_points_faces"].copy()
    np.savez(source / "data.npz", sample_points_faces=grids + 0.1)
    with pytest.raises(ValueError, match="provenance_mismatch"):
        extract_native(source, labels)
