"""Preference manifests retain latent references, empty images, and run identity."""

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.preference.generate import _check_files, finish_image
from src.preference.io import prepare_run
from src.preference.selection import SelectionConfig
from src.common.io import sha256_file


def record(candidate_id: str, validity: str, face_cd: float = 1.) -> dict:
    metrics = {"face_cd_sq": face_cd, "edge_cd_sq": 1., "face_f1": .9}
    return {"image_id": "part", "candidate_id": candidate_id, "validity": validity,
            "status": "ok", "raw": metrics,
            "post": [{"repeat": repeat, **metrics} for repeat in range(3)]}


class PreferenceGenerateTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="hola-preference-generate-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.config = {"seed": 42, "selection": asdict(SelectionConfig())}
        check_output = subprocess.check_output

        def cpu_inventory(command, **kwargs):
            if command[0] == "nvidia-smi":
                return "CPU-only protocol test\n"
            return check_output(command, **kwargs)

        inventory = patch("src.preference.io.subprocess.check_output", cpu_inventory)
        inventory.start()
        self.addCleanup(inventory.stop)

    def test_selected_pair_references_original_latents_and_retains_evidence(self):
        run = self.root / "run"
        for candidate_id, value in (("c00", .125), ("c01", .75)):
            directory = run / "candidates/train/part" / candidate_id
            directory.mkdir(parents=True)
            np.save(directory / "latent.npy", np.full((30, 32), value, dtype=np.float32))
        pair, report = finish_image(
            run, "train", "part", [record("c00", "valid"), record("c01", "invalid")], self.config)
        self.assertEqual(pair["pair_type"], "validity")
        self.assertEqual(pair["split"], "train")
        self.assertEqual(pair["rule_version"], SelectionConfig().version)
        np.testing.assert_array_equal(np.load(run / pair["winner_path"]), .125)
        np.testing.assert_array_equal(np.load(run / pair["loser_path"]), .75)
        self.assertEqual(set(pair["metrics_evidence"]), {"c00", "c01"})
        saved = json.loads((run / "metadata/selection/train/part.json").read_text())
        self.assertEqual(saved, {"image_id": "part", "split": "train", "pair": pair, "report": report})

    def test_unpaired_image_remains_recorded_with_all_candidates_and_reasons(self):
        run = self.root / "run"
        records = [record(f"c{i:02}", "valid") for i in range(8)]
        pair, report = finish_image(run, "validation", "part", records, self.config)
        self.assertIsNone(pair)
        self.assertEqual(report["candidate_count"], 8)
        self.assertEqual(report["ordered_pair_reasons"], {"post_improvement_unstable_or_small": 56})
        saved = json.loads((run / "metadata/selection/validation/part.json").read_text())
        self.assertIsNone(saved["pair"])
        self.assertEqual(saved["split"], "validation")

    def test_generated_artifact_mutation_is_rejected(self):
        path = self.root / "latent.npy"
        np.save(path, np.zeros((30, 32), dtype=np.float32))
        hashes = {"latent.npy": sha256_file(path)}
        _check_files(self.root, hashes)
        np.save(path, np.ones((30, 32), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            _check_files(self.root, hashes)

    def protocol_config(self):
        checkpoint = self.root / "checkpoint.ckpt"
        checkpoint.write_bytes(b"fixed checkpoint fixture")
        dino = self.root / "dino"
        dino.mkdir()
        (dino / "hubconf.py").write_text("# DINO source identity fixture\n")
        train, validation = self.root / "train.txt", self.root / "validation.txt"
        train.write_text("train_a\ntrain_b\n")
        validation.write_text("val_a\n")
        return {**self.config, "train_list": str(train), "validation_list": str(validation),
                "checkpoint": str(checkpoint), "dino_repo": str(dino),
                "raw_data_root": str(self.root / "cad"), "condition_data_root": str(self.root / "images"),
                "candidates": 8, "gpu_cpu_threads": 2,
                "post": {"timeout_seconds": 180, "max_optimize_iter": 200}}

    def test_resume_rejects_changed_input_checkpoint_budget_and_split(self):
        config = self.protocol_config()
        run = self.root / "run"
        original = prepare_run(config, run, resume=False)
        self.assertEqual(prepare_run(config, run, resume=True), original)
        with self.assertRaises(FileExistsError):
            prepare_run(config, run, resume=False)
        with self.assertRaisesRegex(ValueError, "Resume protocol"):
            prepare_run({**config, "candidates": 16}, run, resume=True)
        with self.assertRaisesRegex(ValueError, "Resume protocol"):
            prepare_run({**config, "gpu_cpu_threads": 4}, run, resume=True)
        with self.assertRaisesRegex(ValueError, "Resume protocol"):
            prepare_run({**config, "post": {**config["post"], "timeout_seconds": 360}}, run, resume=True)
        Path(config["checkpoint"]).write_bytes(b"different checkpoint")
        with self.assertRaisesRegex(ValueError, "Resume protocol"):
            prepare_run(config, run, resume=True)
        Path(config["checkpoint"]).write_bytes(b"fixed checkpoint fixture")
        Path(config["train_list"]).write_text("train_b\ntrain_a\n")
        with self.assertRaisesRegex(ValueError, "Resume protocol"):
            prepare_run(config, run, resume=True)

    def test_split_overlap_cannot_be_hidden_by_a_limit(self):
        config = self.protocol_config()
        Path(config["train_list"]).write_text("train_a\nval_a\n")
        with self.assertRaisesRegex(ValueError, "overlap"):
            prepare_run({**config, "train_limit": 1}, self.root / "run", resume=False)


if __name__ == "__main__":
    unittest.main()
