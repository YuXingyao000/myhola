"""UV labels must follow connection rows through filtering and packing."""
from pathlib import Path

import numpy as np
import pytest
import torch

from src.brepnet.data.edge_uv import atomic_save_npz, empty_edge_uv_sidecar, sha256_file
from src.brepnet.dataset import AutoEncoder_dataset3, AutoEncoder_joint_dataset
from src.brepnet.test.test_shared_boundaries import make_boundary_arrays


@pytest.fixture
def uv_dataset(tmp_path):
    data_root, uv_root = tmp_path / "data", tmp_path / "uv"
    ids = tmp_path / "ids.txt"
    ids.write_text("one\ntwo\n")
    for name in ("one", "two"):
        faces, edges, rows = make_boundary_arrays()
        # Deliberately make edge_id != connection row, while preserving geometry.
        edges = edges[::-1].copy()
        rows[:, 0] = rows[::-1, 0].copy()
        directory = data_root / name
        directory.mkdir(parents=True)
        adjacency = np.zeros((len(faces), len(faces)), dtype=bool)
        adjacency[rows[:, 1], rows[:, 2]] = True
        np.savez(directory / "data.npz", sample_points_faces=faces, sample_points_lines=edges,
                 edge_face_connectivity=rows, face_adj=adjacency, zero_positions=np.argwhere(~adjacency))
        labels = empty_edge_uv_sidecar(rows, len(faces), source_sha256=sha256_file(directory / "data.npz"))
        labels["face_uv_bounds"][:] = [0, 1, 0, 1]
        labels["face_step_indices"][:] = np.arange(len(faces))
        labels["face_match_error"][:] = 0
        labels["face_periods"][:] = 0
        labels["edge_uv"][0] = .2
        labels["edge_uv"][1] = .8
        labels["edge_uv_valid"][:] = True
        labels["edge_uv_residual"][:] = 0
        labels["edge_uv_method"][:] = 1
        labels["status"] = np.array("ok")
        atomic_save_npz(uv_root / name / "edge_uv.npz", labels)
    return dict(raw_data_root=str(data_root), edge_uv_root=str(uv_root), epoch_scale_factor=1,
                train_dataset=str(ids), val_dataset=str(ids), test_dataset=str(ids),
                condition_name="none", condition_data_root=None, cached_condition=False,
                is_aug=0, is_overfit=False, num_points=10000,
                boundary_reciprocal_tolerance=.01, boundary_closed_tolerance=1e-5,
                boundary_allow_closed_phase=True)


def test_joint_batch_preserves_row_labels_and_reusable_items(uv_dataset):
    dataset = AutoEncoder_joint_dataset("validation", uv_dataset)
    items = [dataset[0], dataset[1]]
    batch = dataset.collate_fn(items)
    torch.testing.assert_close(batch["edge_face_connectivity"],
                               torch.tensor([[1, 0, 1], [0, 1, 0], [3, 2, 3], [2, 3, 2]]))
    expected = torch.tensor([.2, .8, .2, .8])
    torch.testing.assert_close(batch["edge_uv_targets"]["uv"][:, 0, 0, 0], expected)
    torch.testing.assert_close(dataset.collate_fn(items)["edge_face_connectivity"],
                               batch["edge_face_connectivity"])
    assert batch["edge_uv_targets"]["valid"].dtype == torch.bool


def test_rotation_keeps_uv_and_rotates_xyz(uv_dataset):
    dataset = AutoEncoder_joint_dataset("testing", {**uv_dataset, "is_aug": 1})
    identity, rotated = dataset[0], dataset[2]
    assert rotated[0] == "one_1"
    assert not torch.equal(identity[1], rotated[1])
    torch.testing.assert_close(identity[11]["uv"], rotated[11]["uv"])
    torch.testing.assert_close(identity[11]["valid"], rotated[11]["valid"])


def test_sidecar_missing_or_wrong_source_is_not_silently_ignored(uv_dataset):
    path = Path(uv_dataset["edge_uv_root"]) / "one/edge_uv.npz"
    with np.load(path) as source:
        labels = dict(source)
    labels["source_sha256"] = np.array("0" * 64)
    atomic_save_npz(path, labels)
    with pytest.raises(ValueError, match="source_sha256 mismatch"):
        AutoEncoder_joint_dataset("validation", uv_dataset)[0]
    path.unlink()
    with pytest.raises(FileNotFoundError, match="missing UV sidecar"):
        AutoEncoder_joint_dataset("validation", uv_dataset)


def test_disabling_sidecar_preserves_legacy_tuple_and_batch(uv_dataset):
    dataset = AutoEncoder_joint_dataset("validation", {**uv_dataset, "edge_uv_root": None})
    item = dataset[0]
    assert len(item) == 12  # eleven legacy fields plus shared-boundary targets
    assert "edge_uv_targets" not in dataset.collate_fn([item])


def test_self_face_filter_is_applied_to_uv_rows(uv_dataset):
    # Legacy base loader filters seams; the strict joint loader rejects them.
    data_path = Path(uv_dataset["raw_data_root"]) / "one/data.npz"
    uv_path = Path(uv_dataset["edge_uv_root"]) / "one/edge_uv.npz"
    with np.load(data_path) as source:
        arrays = dict(source)
    arrays["edge_face_connectivity"][0, 2] = 0
    np.savez(data_path, **arrays)
    with np.load(uv_path) as source:
        labels = dict(source)
    labels["edge_face_connectivity"] = arrays["edge_face_connectivity"]
    labels["source_sha256"] = np.array(sha256_file(data_path))
    atomic_save_npz(uv_path, labels)
    item = AutoEncoder_dataset3("validation", uv_dataset)[0]
    assert item[7].shape[0] == 1
    torch.testing.assert_close(item[11]["uv"], torch.full((1, 2, 16, 2), .8))


def test_all_invalid_labels_keep_geometry_sample(uv_dataset):
    root = Path(uv_dataset["raw_data_root"])
    with np.load(root / "one/data.npz") as source:
        rows, count = source["edge_face_connectivity"], len(source["sample_points_faces"])
    labels = empty_edge_uv_sidecar(rows, count, source_sha256=sha256_file(root / "one/data.npz"),
                                  reason="projection_ambiguous")
    atomic_save_npz(Path(uv_dataset["edge_uv_root"]) / "one/edge_uv.npz", labels)
    dataset = AutoEncoder_joint_dataset("validation", uv_dataset)
    batch = dataset.collate_fn([dataset[0]])
    assert len(dataset) == 2
    assert batch["edge_points"].shape[0] == 2
    assert not batch["edge_uv_targets"]["valid"].any()
