"""独立暂存工具的 CPU 文件合同检查；只使用临时人工数据。"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[3] / "checkpoints/diffusion/stage_svr_inputs.py"
SPEC = importlib.util.spec_from_file_location("stage_svr_inputs", SCRIPT)
staging = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(staging)


class DiffusionStagingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="svr-staging-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.train = self.root / "train.txt"
        self.val = self.root / "val.txt"
        self.train.write_bytes(b"two\r\none \r\ntwo\r\n")
        self.val.write_bytes(b"three\none")
        self.conditions = self.root / "source_conditions"
        self.latents = self.root / "source_latents"
        self.dst = self.root / "staged"
        self.sources = []
        for model_id in ("one", "two", "three"):
            image = self.conditions / model_id / "imgs.npz"
            latent = self.latents / f"{model_id}_0" / "features.npy"
            image.parent.mkdir(parents=True)
            latent.parent.mkdir(parents=True)
            image.write_bytes(b"image\x00\xff" + model_id.encode())
            latent.write_bytes(b"latent\x00\xfe" + model_id.encode())
            self.sources.extend((image, latent))
            # 这些文件不得复制或读取。
            (image.parent / "real_photo.npz").write_bytes(b"unused-photo")
            other_view = self.latents / f"{model_id}_1" / "features.npy"
            other_view.parent.mkdir()
            other_view.write_bytes(b"unused-view")
        self.arguments = dict(
            train_split=self.train, val_split=self.val, condition_root=self.conditions,
            latent_root=self.latents, dst_root=self.dst, workers=2,
        )

    def test_bytes_split_order_duplicates_and_manifest_are_preserved(self):
        manifest = staging.stage_inputs(**self.arguments)
        self.assertEqual((self.dst / "splits/train.txt").read_bytes(), self.train.read_bytes())
        self.assertEqual((self.dst / "splits/validation.txt").read_bytes(), self.val.read_bytes())
        self.assertFalse((self.dst / "INCOMPLETE").exists())
        self.assertEqual(manifest, json.loads((self.dst / "manifest.json").read_text()))
        self.assertEqual(manifest["id_counts"], dict(
            train_entries=3, validation_entries=2, train_unique=2, validation_unique=2,
            train_validation_overlap=1, unique_total=3,
        ))
        self.assertEqual(manifest["file_count"], 8)
        self.assertEqual(manifest["protocol"]["rotation_id"], 0)
        for record in manifest["files"]:
            source = Path(record["source"])
            destination = self.dst / record["destination"]
            self.assertTrue(source.is_absolute())
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            self.assertEqual(record["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual(record["size_bytes"], source.stat().st_size)
            self.assertEqual(record["mtime_ns"], source.stat().st_mtime_ns)
        copied = sorted(path.relative_to(self.dst).as_posix() for path in self.dst.rglob("*") if path.is_file())
        self.assertEqual(copied, sorted([record["destination"] for record in manifest["files"]] + ["manifest.json"]))

    def test_dry_run_only_reads_split_payloads_and_creates_no_destination(self):
        original_open = Path.open

        def reject_input_read(path, *args, **kwargs):
            if path in self.sources:
                self.fail(f"dry-run read input bytes: {path}")
            return original_open(path, *args, **kwargs)

        with mock.patch.object(Path, "open", reject_input_read):
            manifest = staging.stage_inputs(**self.arguments, dry_run=True)
        self.assertFalse(self.dst.exists())
        self.assertTrue(manifest["dry_run"])
        self.assertEqual(manifest["file_count"], 8)
        self.assertEqual(manifest["total_bytes"], sum(path.stat().st_size for path in self.sources + [self.train, self.val]))
        self.assertTrue(all("sha256" not in item for item in manifest["files"] if not item["destination"].startswith("splits/")))

    def test_missing_source_leaves_incomplete_destination(self):
        self.sources[0].unlink()
        with self.assertRaises(FileNotFoundError):
            staging.stage_inputs(**self.arguments)
        self.assertTrue((self.dst / "INCOMPLETE").is_file())
        self.assertFalse((self.dst / "manifest.json").exists())

    def test_existing_destination_is_never_overwritten(self):
        self.dst.mkdir()
        sentinel = self.dst / "keep"
        sentinel.write_bytes(b"existing")
        for dry_run in (False, True):
            with self.assertRaises(FileExistsError):
                staging.stage_inputs(**self.arguments, dry_run=dry_run)
        self.assertEqual(sentinel.read_bytes(), b"existing")
        self.assertFalse((self.dst / "INCOMPLETE").exists())

    def test_path_traversal_and_empty_ids_fail_without_silent_filtering(self):
        for index, model_id in enumerate(("../escape", "/absolute", "a/b", "a\\b", "..", "", "x\x00y")):
            with self.subTest(model_id=model_id):
                self.train.write_bytes((model_id + "\n").encode())
                destination = self.root / f"invalid-{index}"
                with self.assertRaises(ValueError):
                    staging.stage_inputs(**dict(self.arguments, dst_root=destination))
                self.assertTrue((destination / "INCOMPLETE").exists())
                self.assertFalse((destination / "manifest.json").exists())

    def test_source_change_during_copy_keeps_incomplete_marker(self):
        source = self.sources[0]
        original_stat = staging.checked_stat
        observations = 0

        def mutate_after_first_stat(path):
            nonlocal observations
            if path == source:
                observations += 1
                if observations == 2:
                    old = path.stat()
                    os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns + 1_000_000))
            return original_stat(path)

        with mock.patch.object(staging, "checked_stat", mutate_after_first_stat):
            with self.assertRaisesRegex(RuntimeError, "Source changed during copy"):
                staging.stage_inputs(**self.arguments)
        self.assertTrue((self.dst / "INCOMPLETE").exists())
        self.assertFalse((self.dst / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
