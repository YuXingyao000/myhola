"""UV label contract and small OCCT construction tests; no test-set data."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.preprocess.edge_uv import (
    atomic_save_npz, empty_edge_uv_sidecar, load_edge_uv_sidecar,
    read_source_geometry, sha256_file, validate_edge_uv_sidecar,
)
from src.preprocess.extract_edge_uv import (
    _chart, _load_lists, _points, _project_path, _shapes,
    continuous_chart_path, extract_edge_uv, main, match_face_grids,
)


def _payload(tmp_path: Path, rows: np.ndarray | None = None) -> tuple[Path, dict[str, np.ndarray]]:
    if rows is None:
        rows = np.array([[7, 1, 0], [2, 0, 1]], dtype=np.int64)
    source = tmp_path / "source.npz"
    np.savez(source, original=np.arange(3))
    labels = empty_edge_uv_sidecar(rows, 2, source_sha256=sha256_file(source), reason="fixture")
    return source, labels


def _valid(labels: dict[str, np.ndarray]) -> None:
    labels["face_uv_bounds"][:] = [0, 1, 0, 1]
    labels["face_step_indices"][:] = np.arange(len(labels["face_step_indices"]))
    labels["face_match_error"][:] = 0
    labels["face_periods"][:] = 0
    labels["edge_uv"][:] = 0.5
    labels["edge_uv_valid"][:] = True
    labels["edge_uv_method"][:] = 1
    labels["edge_uv_residual"][:] = 0
    labels["status"] = np.array("ok")


def _validate(labels: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return validate_edge_uv_sidecar(labels, connectivity=labels["edge_face_connectivity"], num_faces=len(labels["face_uv_bounds"]), source_sha256=str(labels["source_sha256"].item()))


def test_sidecar_roundtrip_tracks_rows_and_source_bytes(tmp_path: Path) -> None:
    source, labels = _payload(tmp_path)
    _valid(labels)
    output = tmp_path / "edge_uv.npz"
    atomic_save_npz(output, labels)
    loaded = load_edge_uv_sidecar(output, source_path=source, connectivity=labels["edge_face_connectivity"], num_faces=2)
    np.testing.assert_array_equal(loaded["edge_face_connectivity"][:, 0], [7, 2])
    with pytest.raises(ValueError, match="row order"):
        load_edge_uv_sidecar(output, source_path=source, connectivity=labels["edge_face_connectivity"][::-1], num_faces=2)
    np.savez(source, original=np.arange(4))
    with pytest.raises(ValueError, match="source_sha256"):
        load_edge_uv_sidecar(output, source_path=source, connectivity=labels["edge_face_connectivity"], num_faces=2)


@pytest.mark.parametrize("rows", [np.empty((0, 3), dtype=np.int64), np.array([[7, 1, 0]])])
def test_invalid_labels_allow_unknown_bounds_and_zero_rows(tmp_path: Path, rows: np.ndarray) -> None:
    _, labels = _payload(tmp_path, rows)
    assert _validate(labels)["status"].item() == "invalid"


@pytest.mark.parametrize("field,value,error", [
    ("edge_uv", float("nan"), "nonfinite"),
    ("edge_uv", 1.1, "outside"),
    ("edge_uv_residual", 0.01, "residual tolerance"),
    ("edge_uv_method", 0, "method"),
    ("face_step_indices", 5, "partial bijection"),
    ("face_match_error", 0.01, "verified face matching"),
])
def test_rejects_invalid_supervision(tmp_path: Path, field: str, value: float, error: str) -> None:
    _, labels = _payload(tmp_path)
    _valid(labels)
    labels[field][:] = value
    with pytest.raises(ValueError, match=error):
        _validate(labels)


def test_face_matching_uses_all_ordered_samples() -> None:
    rng = np.random.default_rng(4)
    faces = rng.normal(size=(3, 16, 16, 3))
    indices, error = match_face_grids(faces, faces[[2, 0, 1]], 1e-5)
    np.testing.assert_array_equal(indices, [1, 2, 0])
    assert error.max() == 0
    altered = faces.copy()
    altered[1, 8, 8, 0] += 0.01  # Corners and averages are insufficient.
    with pytest.raises(ValueError, match="not_unique"):
        match_face_grids(faces, altered, 1e-5)
    with pytest.raises(ValueError, match="not_unique"):
        match_face_grids(np.repeat(faces[:1], 2, axis=0), np.repeat(faces[:1], 2, axis=0), 1e-5)


def test_periodic_projection_preserves_order_and_closed_endpoints() -> None:
    period = 2 * np.pi
    bounds = np.array([0, period, -1, 1])
    periods = np.array([period, 0])
    reference = np.column_stack([np.linspace(0, period, 16), np.zeros(16)])
    wrapped = reference.copy()
    wrapped[-1, 0] = 0
    np.testing.assert_allclose(continuous_chart_path(wrapped, bounds, periods, native=False), reference, atol=1e-12)
    np.testing.assert_allclose(continuous_chart_path(wrapped[::-1], bounds, periods, native=False), reference[::-1], atol=1e-12)
    seam = np.zeros((16, 2))
    with pytest.raises(ValueError, match="ambiguous"):
        continuous_chart_path(seam, bounds, periods, native=False)
    np.testing.assert_array_equal(continuous_chart_path(seam, bounds, periods, native=True), seam)
    half_turn = seam.copy()
    half_turn[8:, 0] = np.pi
    with pytest.raises(ValueError, match="ambiguous"):
        continuous_chart_path(half_turn, bounds, periods, native=False)


def _box_source(root: Path, *, nonuniform: bool = False) -> tuple[Path, Path, np.ndarray]:
    pytest.importorskip("OCC")
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
    from OCC.Core.TopoDS import topods
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.gp import gp_Trsf, gp_Vec
    from OCC.Extend.DataExchange import write_step_file

    shape = BRepPrimAPI_MakeBox(0.4, 0.6, 0.8).Shape()
    transform = gp_Trsf()
    transform.SetTranslation(gp_Vec(1.7, -2.1, 0.3))
    shape.Location(TopLoc_Location(transform))
    faces = _shapes(shape, TopAbs_FACE, topods.Face)
    grids = np.stack([_chart(BRepAdaptor_Surface(face))[2] for face in faces])
    face_edges = [_shapes(face, TopAbs_EDGE, topods.Edge) for face in faces]
    i, j, edge = next((i, j, a) for i in range(len(faces)) for j in range(i + 1, len(faces)) for a in face_edges[i] if any(a.IsSame(b) for b in face_edges[j]))
    curve = BRepAdaptor_Curve(edge)
    time = np.linspace(0, 1, 16)
    if nonuniform:
        time = time**1.2
    time = curve.FirstParameter() + time * (curve.LastParameter() - curve.FirstParameter())
    xyz = _points([curve.Value(float(t)) for t in time]).astype(np.float32)
    edges = np.repeat(xyz[None], 5, axis=0)
    edges[1] = xyz[::-1]
    # NPZ deliberately reverses face order and uses noncontiguous edge IDs.
    rows = np.array([[3, len(faces) - 1 - i, len(faces) - 1 - j], [1, len(faces) - 1 - j, len(faces) - 1 - i]], dtype=np.int64)
    root.mkdir(parents=True, exist_ok=True)
    source, step = root / "data.npz", root / "normalized_shape.step"
    np.savez(source, sample_points_faces=grids[::-1].astype(np.float32), sample_points_lines=edges, edge_face_connectivity=rows)
    write_step_file(shape, str(step))
    return source, step, rows


def test_occ_native_pcurve_face_permutation_reverse_rows_and_location(tmp_path: Path) -> None:
    source, step, rows = _box_source(tmp_path)
    labels = extract_edge_uv(source, step)
    assert labels["status"].item() == "ok", labels["reason"]
    assert (labels["edge_uv_method"] == 1).all()
    np.testing.assert_array_equal(labels["edge_face_connectivity"], rows)
    np.testing.assert_array_equal(labels["face_step_indices"], np.arange(6)[::-1])
    np.testing.assert_allclose(labels["edge_uv"][0, 0], labels["edge_uv"][1, 1, ::-1], atol=1e-6)
    assert labels["edge_uv_residual"].max() < 1e-6


def test_occ_nonuniform_legacy_edge_uses_explicit_projection(tmp_path: Path) -> None:
    source, step, _ = _box_source(tmp_path, nonuniform=True)
    labels = extract_edge_uv(source, step)
    assert labels["status"].item() == "ok", labels["reason"]
    assert (labels["edge_uv_method"] == 2).all()
    assert labels["edge_uv_residual"].max() < 1e-6
    assert "projected_native_path_unavailable" in labels["reason"].item()


def test_occ_projection_applies_nonidentity_surface_location(tmp_path: Path) -> None:
    pytest.importorskip("OCC")
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopoDS import topods
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.gp import gp_Trsf, gp_Vec

    shape = BRepPrimAPI_MakeBox(1.0, 2.0, 3.0).Shape()
    transform = gp_Trsf()
    transform.SetTranslation(gp_Vec(12.0, -7.0, 4.0))
    shape.Location(TopLoc_Location(transform))
    face = _shapes(shape, TopAbs_FACE, topods.Face)[0]
    surface = BRepAdaptor_Surface(face)
    bounds, periods, _ = _chart(surface)
    uv = bounds[[0, 2]] + np.linspace(0.1, 0.9, 16)[:, None] * (bounds[[1, 3]] - bounds[[0, 2]])
    xyz = _points([surface.Value(float(u), float(v)) for u, v in uv])
    projected, residual = _project_path(xyz, face, surface, bounds, periods)
    np.testing.assert_allclose(projected, uv, atol=1e-10)
    assert residual.max() < 1e-10


def test_correspondence_failure_keeps_geometry_and_corrupt_source_raises(tmp_path: Path) -> None:
    source, step, rows = _box_source(tmp_path)
    geometry = read_source_geometry(source)
    geometry["sample_points_faces"][0, 8, 8, 0] += 0.01
    np.savez(source, **geometry)
    labels = extract_edge_uv(source, step)
    assert labels["status"].item() == "invalid"
    assert not labels["edge_uv_valid"].any()
    np.testing.assert_array_equal(labels["edge_face_connectivity"], rows)
    assert "face_match_not_unique" in labels["reason"].item()
    source.write_bytes(b"broken npz")
    with pytest.raises(ValueError, match="unreadable source"):
        extract_edge_uv(source, step)


def test_missing_or_unreadable_step_is_source_failure(tmp_path: Path) -> None:
    source, step, _ = _box_source(tmp_path)
    step.unlink()
    with pytest.raises(FileNotFoundError, match="required source file"):
        extract_edge_uv(source, step)
    step.write_text("corrupt STEP source")
    with pytest.raises(ValueError, match="unreadable STEP source"):
        extract_edge_uv(source, step)


@pytest.mark.parametrize("existing_sidecar", [False, True])
def test_cli_missing_step_records_failure_and_preserves_existing_sidecar(tmp_path: Path, existing_sidecar: bool) -> None:
    data = tmp_path / "input"
    _, step, _ = _box_source(data / "one")
    split = tmp_path / "validation.txt"
    split.write_text("one\n", encoding="utf-8")
    output = tmp_path / "labels"
    args = ["--data-root", str(data), "--output-root", str(output), "--split", str(split)]
    sidecar = output / "one" / "edge_uv.npz"
    if existing_sidecar:
        main(args)
        original_digest = sha256_file(sidecar)
    step.unlink()
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 1
    record = json.loads((output / "manifest.jsonl").read_text())
    assert record["status"] == "source_failed"
    assert "FileNotFoundError" in record["reason"]
    assert "normalized_shape.step" in record["reason"]
    assert json.loads((output / "summary.json").read_text())["status_counts"] == {"source_failed": 1}
    if existing_sidecar:
        assert sha256_file(sidecar) == original_digest
    else:
        assert not sidecar.exists()


def test_cli_resumes_limit_and_rejects_mixed_contract(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = tmp_path / "input"
    _box_source(data / "one")
    _box_source(data / "two", nonuniform=True)
    split = tmp_path / "validation.txt"
    split.write_text("one\ntwo\n", encoding="utf-8")
    output = tmp_path / "labels"
    args = ["--data-root", str(data), "--output-root", str(output), "--split", str(split)]
    main([*args, "--limit", "1"])
    first_digest = sha256_file(output / "one" / "edge_uv.npz")
    assert json.loads((output / "summary.json").read_text())["pending"] == 1
    main(args)
    assert first_digest == sha256_file(output / "one" / "edge_uv.npz")
    summary = json.loads((output / "summary.json").read_text())
    assert summary["recorded"] == 2 and summary["pending"] == 0
    records = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    assert all(record["generation_seconds"] > 0 for record in records)
    assert "resumed" in capsys.readouterr().out
    with pytest.raises(ValueError, match="contract differs"):
        main([*args, "--match-tolerance", "0.0001"])
    source = data / "one" / "data.npz"
    original = read_source_geometry(source)
    np.savez(source, **original, changed=np.array(1))
    with pytest.raises(ValueError, match="source_sha256"):
        main(args)


def test_split_validation_and_cross_split_membership(tmp_path: Path) -> None:
    first, second = tmp_path / "a.txt", tmp_path / "b.txt"
    first.write_text("one\ntwo\n")
    second.write_text("two\nthree\n")
    ids, _, memberships = _load_lists([first, second])
    assert ids == ["one", "two", "three"]
    assert len(memberships["two"]) == 2
    first.write_text("../escape\n")
    with pytest.raises(ValueError, match="unsafe"):
        _load_lists([first])
    first.write_text("one\none\n")
    with pytest.raises(ValueError, match="duplicate"):
        _load_lists([first])
