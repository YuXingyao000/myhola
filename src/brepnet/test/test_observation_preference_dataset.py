"""Reselected preference pools preserve the original latent and split provenance."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import torch

from src.brepnet.data.preference_dataset import (
    PreferenceDataset,
    preference_data_provenance,
    sha256_file,
)


def write_protocol(path: Path, protocol: dict) -> dict:
    content = {key: value for key, value in protocol.items() if key != "fingerprint"}
    content["fingerprint"] = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode(),
    ).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content))
    return content


class ObservationPreferenceDatasetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="hola-observation-dataset-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / "source"
        self.derived = self.root / "derived"
        self.derived.mkdir()
        self.source_protocol_path = self.source / "metadata/protocol.json"
        self.protocol_path = self.derived / "metadata/protocol.json"
        self.source_protocol = write_protocol(self.source_protocol_path, {
            "version": "svr_preference_generation_v1",
            "splits": {"train": ["a", "b"], "validation": ["c"]},
            "condition_data_root": str(self.root / "images"),
            "checkpoint_sha256": "original-checkpoint-digest",
        })
        (self.source / "status.json").write_text('{"status":"complete"}')
        self.protocol = write_protocol(self.protocol_path, {
            "version": "svr_observation_preference_v1", "partial": False,
            "splits": self.source_protocol["splits"],
            "condition_data_root": self.source_protocol["condition_data_root"],
            "checkpoint_sha256": self.source_protocol["checkpoint_sha256"],
            "source_run": str(self.source),
            "source_protocol_fingerprint": self.source_protocol["fingerprint"],
            "source_protocol_sha256": sha256_file(self.source_protocol_path),
            "source_status_sha256": sha256_file(self.source / "status.json"),
            "gt_cache_root": str(self.source / "gt_cache"),
        })
        for name, identities in (("train.txt", ["a", "b"]), ("val.txt", ["c"])):
            (self.root / name).write_text("\n".join(identities) + "\n")
        for image_id in ("a", "b", "c"):
            cache = self.source / "gt_cache" / f"{image_id}_0"
            cache.mkdir(parents=True)
            stats = np.zeros((3, 64), np.float32)
            stats[:, :32] = np.arange(3)[:, None]
            stats[:, 32:] = 777
            np.save(cache / "features.npy", stats)
            image = self.root / "images" / image_id
            image.mkdir(parents=True)
            np.savez_compressed(image / "imgs.npz", svr_imgs=np.zeros((1, 16, 16, 3), np.uint8))
        self.winner = np.arange(30 * 32, dtype=np.float32).reshape(30, 32)
        self.pair = {"image_id": "a", "pair_type": "geometry"}
        for field, candidate, latent in (
            ("winner_path", "c00", self.winner),
            ("loser_path", "c01", -self.winner),
        ):
            path = self.source / "candidates/train/a" / candidate / "latent.npy"
            path.parent.mkdir(parents=True)
            np.save(path, latent)
            self.pair[field] = str(path)
        (self.derived / "pairs.jsonl").write_text(json.dumps(self.pair) + "\n")
        (self.derived / "validation_pairs.jsonl").write_text("")
        self.write_status()
        self.config = {
            "train_dataset": self.root / "train.txt", "val_dataset": self.root / "val.txt",
            "test_dataset": self.root / "val.txt", "cached_latent_root": self.source / "gt_cache",
            "condition_data_root": self.root / "images", "epoch_scale_factor": 1,
            "pairs_manifest": self.derived / "pairs.jsonl",
        }

    def write_status(self, **overrides):
        status = {
            "status": "complete",
            "partial": False,
            "protocol_fingerprint": json.loads(self.protocol_path.read_text())["fingerprint"],
            "manifest_sha256": {
                name: sha256_file(self.derived / name)
                for name in ("pairs.jsonl", "validation_pairs.jsonl")
            },
            **overrides,
        }
        (self.derived / "status.json").write_text(json.dumps(status))

    def write_derived_protocol(self, protocol):
        write_protocol(self.protocol_path, protocol)
        self.write_status()

    def test_source_cache_and_generated_latents_are_read_without_copy_or_reencoding(self):
        dataset = PreferenceDataset("training", self.config)
        item = dataset[1]
        self.assertEqual(item["v_prefix"], "b")
        self.assertEqual(item["pair_image_id"], "a")
        torch.testing.assert_close(item["winner_latent"], torch.from_numpy(self.winner), rtol=0, atol=0)
        self.assertEqual(set(item["gt_latent"][:, 0].tolist()), {0.0, 1.0, 2.0})
        self.assertFalse((self.derived / "gt_cache").exists())
        provenance = dataset.provenance
        self.assertEqual(provenance["source_protocol_sha256"], sha256_file(self.source_protocol_path))
        self.assertEqual(provenance["source_status_sha256"], sha256_file(self.source / "status.json"))
        self.assertEqual(provenance["generation_protocol_sha256"], sha256_file(self.protocol_path))
        self.assertEqual(provenance["gt_cache_root"], str(self.source / "gt_cache"))
        self.assertEqual(provenance["validation_pairs_manifest_sha256"], sha256_file(self.derived / "validation_pairs.jsonl"))

    def test_validation_checks_manifests_but_does_not_read_candidate_latents(self):
        for field in ("winner_path", "loser_path"):
            Path(self.pair[field]).unlink()
        item = PreferenceDataset("validation", self.config)[0]
        self.assertEqual(item["v_prefix"], "c")
        self.assertNotIn("winner_latent", item)

    def test_partial_or_unfinished_derived_run_is_rejected(self):
        for partial in (True, None, "false"):
            self.write_derived_protocol({**self.protocol, "partial": partial})
            with self.subTest(partial=partial), self.assertRaisesRegex(ValueError, "Partial"):
                preference_data_provenance(self.config)
        self.write_derived_protocol(self.protocol)
        self.write_status(status="running")
        with self.assertRaisesRegex(ValueError, "complete"):
            preference_data_provenance(self.config)

    def test_derived_status_must_match_protocol_and_confirm_full_run(self):
        changes = (
            {"protocol_fingerprint": "different-run"},
            {"protocol_fingerprint": None},
            {"partial": True},
            {"partial": None},
        )
        for changed in changes:
            self.write_status(**changed)
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "Observation status"):
                preference_data_provenance(self.config)

    def test_source_status_digest_is_required_and_frozen(self):
        (self.source / "status.json").write_text('{"status":"complete", "new_counter":1}')
        with self.assertRaisesRegex(ValueError, "Source status SHA256"):
            preference_data_provenance(self.config)
        (self.source / "status.json").write_text('{"status":"complete"}')
        without_digest = {key: value for key, value in self.protocol.items() if key != "source_status_sha256"}
        self.write_derived_protocol(without_digest)
        with self.assertRaisesRegex(ValueError, "Source status SHA256"):
            preference_data_provenance(self.config)

    def test_source_status_and_protocol_version_are_checked(self):
        (self.source / "status.json").write_text('{"status":"running"}')
        with self.assertRaisesRegex(ValueError, "Source.*complete"):
            preference_data_provenance(self.config)
        (self.source / "status.json").write_text('{"status":"complete"}')
        write_protocol(self.source_protocol_path, {**self.source_protocol, "version": "unsupported"})
        with self.assertRaisesRegex(ValueError, "Unsupported source"):
            preference_data_provenance(self.config)

    def test_source_protocol_byte_changes_and_fingerprint_changes_are_rejected(self):
        original = self.source_protocol_path.read_bytes()
        self.source_protocol_path.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            preference_data_provenance(self.config)
        write_protocol(self.source_protocol_path, {**self.source_protocol, "checkpoint_sha256": "changed"})
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            preference_data_provenance(self.config)

    def test_invalid_source_or_derived_fingerprint_is_rejected(self):
        for path, protocol in ((self.source_protocol_path, self.source_protocol), (self.protocol_path, self.protocol)):
            original = path.read_bytes()
            path.write_text(json.dumps({**protocol, "fingerprint": "corrupted"}))
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "fingerprint"):
                preference_data_provenance(self.config)
            path.write_bytes(original)

    def test_reselection_cannot_change_source_split_checkpoint_or_condition_root(self):
        changes = {
            "splits": {"train": ["b", "a"], "validation": ["c"]},
            "checkpoint_sha256": "other-checkpoint",
            "condition_data_root": str(self.root / "different-images"),
        }
        for field, value in changes.items():
            self.write_derived_protocol({**self.protocol, field: value})
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                preference_data_provenance(self.config)

    def test_both_manifests_must_match_recorded_digests(self):
        for name in ("pairs.jsonl", "validation_pairs.jsonl"):
            path = self.derived / name
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                preference_data_provenance(self.config)
            path.write_bytes(original)
        self.write_status(manifest_sha256={})
        with self.assertRaisesRegex(ValueError, "SHA256"):
            preference_data_provenance(self.config)

    def test_validation_manifest_cannot_be_selected_as_training_manifest(self):
        config = {**self.config, "pairs_manifest": self.derived / "validation_pairs.jsonl"}
        with self.assertRaisesRegex(ValueError, "requires.*pairs.jsonl"):
            preference_data_provenance(config)

    def test_source_cache_cannot_be_silently_relocated(self):
        moved_cache = self.derived / "gt_cache"
        shutil.copytree(self.source / "gt_cache", moved_cache)
        config = {**self.config, "cached_latent_root": moved_cache}
        with self.assertRaisesRegex(ValueError, "source generation run"):
            preference_data_provenance(config)
        self.write_derived_protocol({**self.protocol, "gt_cache_root": str(moved_cache)})
        with self.assertRaisesRegex(ValueError, "original gt_cache"):
            preference_data_provenance(config)
        self.write_derived_protocol(self.protocol)
        shutil.rmtree(self.source / "gt_cache")
        (self.source / "gt_cache").symlink_to(moved_cache, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "original gt_cache"):
            preference_data_provenance(self.config)

    def test_pair_paths_must_stay_with_same_source_training_cad(self):
        paths = (
            "candidates/train/a/c00/latent.npy",
            str(self.derived / "candidates/train/a/c00/latent.npy"),
            str(self.source / "candidates/validation/a/c00/latent.npy"),
            str(self.source / "candidates/train/b/c00/latent.npy"),
            str(self.source / "candidates/train/a/c08/latent.npy"),
            str(self.source / "candidates/train/a/candidate_0/latent.npy"),
        )
        for path in paths:
            (self.derived / "pairs.jsonl").write_text(json.dumps({**self.pair, "winner_path": path}) + "\n")
            self.write_status()
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "original source candidate"):
                PreferenceDataset("training", self.config)

    def test_winner_and_loser_cannot_be_the_same_candidate(self):
        pair = {**self.pair, "loser_path": self.pair["winner_path"]}
        (self.derived / "pairs.jsonl").write_text(json.dumps(pair) + "\n")
        self.write_status()
        with self.assertRaisesRegex(ValueError, "distinct candidates"):
            PreferenceDataset("training", self.config)

    def test_existing_invocation_provenance_still_detects_reselection_changes(self):
        config = deepcopy(self.config)
        config["provenance"] = preference_data_provenance(config)
        (self.derived / "pairs.jsonl").write_text(json.dumps({**self.pair, "audit": "new selection"}) + "\n")
        self.write_status()
        with self.assertRaisesRegex(ValueError, "changed after"):
            PreferenceDataset("training", config)


if __name__ == "__main__":
    unittest.main()
