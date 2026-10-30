"""Exercise current-chart labels through the real joint VAE dataset boundary."""
import hashlib
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.preprocess.edge_uv import sha256_file
from src.preprocess.uv_curve_targets import (
    REPRESENTATION, SCHEMA_VERSION, SURFACE_SIGNATURE,
    UVProjectionConfig, parameters_json,
)
from src.datasets.brep import AutoEncoder_joint_dataset
from tests import test_joint_boundary_dataset as dataset_fixture


class UVCurveTargetDatasetTests(unittest.TestCase):
    write_sample = dataset_fixture.JointBoundaryDatasetTests.write_sample

    def setUp(self):
        dataset_fixture.JointBoundaryDatasetTests.setUp(self)
        self.uv_root = self.root / "uv_targets"
        self.uv_root.mkdir()
        self.config["uv_curve_target_root"] = str(self.uv_root)
        params = parameters_json(UVProjectionConfig())
        self.config["uv_curve_target_parameters_sha256"] = hashlib.sha256(params.encode()).hexdigest()
        plain = AutoEncoder_joint_dataset("validation", {**self.config, "uv_curve_target_root": None})
        for index, sample_id in enumerate(("one", "two")):
            target = plain[index][-1]
            count = len(target["curves"])
            uv = np.linspace(0, 1, count * 2 * 16 * 2, dtype=np.float32).reshape(count, 2, 16, 2)
            self.write_targets(sample_id, {
                "schema_version": np.array(SCHEMA_VERSION),
                "representation": np.array(REPRESENTATION),
                "surface_signature": np.array(SURFACE_SIGNATURE),
                "source_sha256": np.array(sha256_file(self.root / sample_id / "data.npz")),
                "parameters_json": np.array(params),
                "parameters_sha256": np.array(hashlib.sha256(params.encode()).hexdigest()),
                "face_pairs": target["face_pairs"].numpy(),
                "source_edge_ids": target["source_edge_ids"].numpy(),
                "uv": uv, "valid": np.ones(uv.shape[:-1], dtype=bool),
                "residual": np.zeros(uv.shape[:-1], dtype=np.float32),
                "reason_flags": np.zeros(uv.shape[:-1], dtype=np.uint8),
                "is_closed": target["is_closed"].numpy(),
                "projection_metadata_json": np.array('{"fixture":"dataset contract only"}'),
            })

    def write_targets(self, sample_id, arrays):
        folder = self.uv_root / sample_id
        folder.mkdir(exist_ok=True)
        np.savez(folder / "uv_curve_targets.npz", **arrays)

    def read_targets(self, sample_id):
        with np.load(self.uv_root / sample_id / "uv_curve_targets.npz") as data:
            return {key: data[key].copy() for key in data.files}

    def test_opt_in_rotation_and_batch_locality(self):
        disabled = AutoEncoder_joint_dataset("validation", {**self.config, "uv_curve_target_root": None})
        self.assertNotIn("edge_uv", disabled[0][-1])
        dataset = AutoEncoder_joint_dataset("testing", {**self.config, "is_aug": 1})
        initial, rotated = dataset[0], dataset[2]
        self.assertEqual(rotated[0], "one_1")
        self.assertFalse(torch.equal(initial[1], rotated[1]))
        torch.testing.assert_close(initial[-1]["edge_uv"], rotated[-1]["edge_uv"])
        torch.testing.assert_close(initial[-1]["edge_uv_valid"], rotated[-1]["edge_uv_valid"])
        batch = dataset.collate_fn([rotated, dataset[3]])
        for target in batch["boundary_targets"]:
            torch.testing.assert_close(target["face_pairs"], torch.tensor([[0, 1]]))
            self.assertEqual(target["edge_uv"].shape, (1, 2, 16, 2))
            self.assertEqual(target["edge_uv_valid"].dtype, torch.bool)

    def test_missing_cache_is_fatal(self):
        (self.uv_root / "two/uv_curve_targets.npz").unlink()
        dataset = AutoEncoder_joint_dataset("validation", self.config)
        with self.assertRaisesRegex(ValueError, "two.*uv_curve_targets.npz"):
            dataset[1]

    def test_projection_protocol_digest_is_required_and_bound(self):
        for digest in (None, "not-a-sha256"):
            with self.subTest(digest=digest):
                config = {**self.config, "uv_curve_target_parameters_sha256": digest}
                with self.assertRaisesRegex(ValueError, "uv_curve_target_parameters_sha256"):
                    AutoEncoder_joint_dataset("validation", config)
        labels = self.read_targets("one")
        changed = parameters_json(UVProjectionConfig(iterations=21))
        labels["parameters_json"] = np.array(changed)
        labels["parameters_sha256"] = np.array(hashlib.sha256(changed.encode()).hexdigest())
        self.write_targets("one", labels)
        dataset = AutoEncoder_joint_dataset("validation", self.config)
        with self.assertRaisesRegex(ValueError, "projection protocol mismatch"):
            dataset[0]

    def test_wrong_chart_owner_and_representative_are_rejected(self):
        original = self.read_targets("one")
        for key, changed in (
            ("surface_signature", np.array("old_occ_chart")),
            ("face_pairs", np.array([[1, 0]], dtype=np.int64)),
            ("source_edge_ids", original["source_edge_ids"] + 1),
        ):
            with self.subTest(key=key):
                self.write_targets("one", {**original, key: changed})
                dataset = AutoEncoder_joint_dataset("validation", self.config)
                with self.assertRaisesRegex(ValueError, key):
                    dataset[0]

    def test_source_hash_is_cached_but_rechecked_after_source_changes(self):
        dataset = AutoEncoder_joint_dataset("validation", self.config)
        with patch("src.datasets.brep.sha256_file", wraps=sha256_file) as digest:
            dataset[0]
            dataset[0]
            self.assertEqual(digest.call_count, 1)
            path = self.root / "one/data.npz"
            with np.load(path) as source:
                arrays = {key: source[key].copy() for key in source.files}
            # Valid source geometry is unchanged, but its source identity changed.
            np.savez(path, **arrays, fixture_revision=np.array(2))
            with self.assertRaisesRegex(ValueError, "source_sha256 mismatch"):
                dataset[0]
            self.assertEqual(digest.call_count, 2)


if __name__ == "__main__":
    unittest.main()
