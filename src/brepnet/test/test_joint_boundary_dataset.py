"""Exercise actual loading, config migration, rotations and batch offsets."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.brepnet.dataset import AutoEncoder_dataset3, AutoEncoder_joint_dataset
from src.brepnet.test.test_shared_boundaries import make_boundary_arrays


class JointBoundaryDatasetTests(unittest.TestCase):
    def setUp(self):
        output = Path(__file__).resolve().parents[3] / "checkpoints/joint_boundary/test_tmp"
        output.mkdir(parents=True, exist_ok=True)
        self.workspace = tempfile.TemporaryDirectory(prefix="dataset-", dir=output)
        self.root = Path(self.workspace.name)
        self.addCleanup(self.workspace.cleanup)
        self.listfile = self.root / "ids.txt"
        self.listfile.write_text("one\ntwo\n")
        for sample_id in ("one", "two"):
            self.write_sample(sample_id)
        self.config = dict(
            raw_data_root=str(self.root), epoch_scale_factor=1,
            train_dataset=str(self.listfile), val_dataset=str(self.listfile), test_dataset=str(self.listfile),
            condition_name="none", condition_data_root=None, cached_condition=False,
            is_aug=0, is_overfit=False, num_points=10000,
            boundary_reciprocal_tolerance=.01, boundary_closed_tolerance=1e-5, boundary_allow_closed_phase=True,
        )

    def write_sample(self, sample_id, empty=False, seam=False):
        faces, edges, conn = make_boundary_arrays()
        if empty:
            edges, conn = edges[:0], conn[:0]
        if seam:
            conn[:, 1:] = 0
        adjacency = np.zeros((2, 2), dtype=bool)
        adjacency[conn[:, 1], conn[:, 2]] = True
        directory = self.root / sample_id
        directory.mkdir(exist_ok=True)
        np.savez(directory / "data.npz", sample_points_faces=faces, sample_points_lines=edges,
                 edge_face_connectivity=conn, face_adj=adjacency, zero_positions=np.argwhere(~adjacency))

    def test_config_canonical_and_legacy_fields(self):
        modern = AutoEncoder_dataset3("validation", self.config)
        old = dict(self.config)
        for current, previous in (("raw_data_root", "data_root"), ("epoch_scale_factor", "scale_factor"),
                                  ("condition_name", "condition_names"), ("condition_data_root", "condition_root")):
            old[previous] = old.pop(current)
        old["condition_names"] = []
        legacy = AutoEncoder_dataset3("validation", old)
        torch.testing.assert_close(modern[0][1], legacy[0][1])
        self.assertEqual(modern.condition, [])
        with self.assertRaisesRegex(ValueError, "Conflicting VAE dataset config"):
            AutoEncoder_dataset3("validation", {**self.config, "data_root": "different"})

    def test_collate_keeps_local_targets_and_reusable_encoder_items(self):
        dataset = AutoEncoder_joint_dataset("validation", self.config)
        items = [dataset[0], dataset[1]]
        original = items[1][7].clone()
        batch = dataset.collate_fn(items)
        torch.testing.assert_close(batch["face_counts"], torch.tensor([2, 2]))
        torch.testing.assert_close(batch["edge_face_connectivity"][2:], torch.tensor([[2, 2, 3], [3, 3, 2]]))
        for target in batch["boundary_targets"]:
            torch.testing.assert_close(target["face_pairs"], torch.tensor([[0, 1]]))
        torch.testing.assert_close(items[1][7], original)
        torch.testing.assert_close(dataset.collate_fn(items)["edge_face_connectivity"], batch["edge_face_connectivity"])

    def test_targets_rotate_with_old_encoder_geometry(self):
        dataset = AutoEncoder_joint_dataset("testing", {**self.config, "is_aug": 1})
        item = dataset[2]  # first model, rotation 1
        self.assertEqual(item[0], "one_1")
        torch.testing.assert_close(item[-1]["curves"][0], item[2][0, :, :3])
        self.assertFalse(torch.equal(item[1], dataset[0][1]))

    def test_zero_boundary_sample_and_seam_failure(self):
        self.write_sample("one", empty=True)
        dataset = AutoEncoder_joint_dataset("validation", self.config)
        item = dataset[0]
        batch = dataset.collate_fn([item, dataset[1]])
        self.assertEqual(batch["boundary_targets"][0]["curves"].shape, (0, 16, 3))
        self.assertEqual(item[4].shape, (0, 16, 6))
        self.write_sample("two", empty=True)
        empty_batch = dataset.collate_fn([dataset[0], dataset[1]])
        self.assertEqual(empty_batch["edge_face_connectivity"].shape, (0, 3))
        self.assertEqual(empty_batch["edge_norm"].shape, (0, 16, 6))
        self.assertTrue(all(len(target["curves"]) == 0 for target in empty_batch["boundary_targets"]))
        self.write_sample("one", seam=True)
        with self.assertRaisesRegex(ValueError, "one.*self-face/seam"):
            dataset[0]

    def test_missing_sample_is_not_silently_filtered(self):
        (self.root / "two/data.npz").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "two.*missing"):
            AutoEncoder_joint_dataset("validation", self.config)


if __name__ == "__main__":
    unittest.main()
