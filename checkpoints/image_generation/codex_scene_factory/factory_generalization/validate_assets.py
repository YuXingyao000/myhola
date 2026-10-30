"""用本地目录验证素材获取、缓存与两种模板清单，不调用网络、Blender 或 LLM。"""

import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "packages" / "cad_scene_factory"))
import assets


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


class AssetManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="factory-assets-check-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        (self.source / "textures").mkdir(parents=True)
        (self.source / "model.gltf").write_bytes(b"local-model-fixture")
        (self.source / "textures" / "metal.png").write_bytes(b"local-texture-fixture")
        self.catalog = self.root / "catalog.json"
        write_json(self.catalog, [{
            "provider": "local", "asset_id": "fixture", "name": "Fixture tool",
            "kind": "model", "source_dir": "source", "source_url": "local-fixture",
            "license": "CC0-1.0", "license_url": assets.CC0_URL, "author": "Test fixture",
        }])
        self.template = self.root / "template"
        self.cache = self.root / "cache"
        self.template.mkdir()
        # 本地获取必须走真实复制路径；只禁止意外发起网络请求。
        network = patch.object(assets, "_open_url", side_effect=AssertionError("Network disabled"))
        network.start()
        self.addCleanup(network.stop)

    def fetch(self, template=None):
        return assets.fetch_asset("local", "fixture", template or self.template,
                                  self.cache, catalog=self.catalog)

    def manifest(self, template=None):
        return json.loads(((template or self.template) / "assets.json").read_text())

    def test_new_manifest_copies_real_files_and_records_provenance(self):
        result = self.fetch()
        self.assertFalse(result["cache_hit"])
        self.assertIsInstance(self.manifest(), list)
        self.assertEqual(len(self.manifest()), 2)
        for record in result["files"]:
            path = self.template / record["path"]
            self.assertTrue(path.is_file())
            self.assertFalse(path.is_symlink())
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"])
            self.assertEqual(record["source_url"], "local-fixture")
            self.assertEqual(record["license"], "CC0-1.0")
            self.assertEqual(record["author"], "Test fixture")
        self.assertTrue((self.template / "assets/local/fixture/original/textures/metal.png").is_file())

    def test_cache_hit_copies_without_source_or_catalog(self):
        self.fetch()
        shutil.rmtree(self.source)
        self.catalog.unlink()
        destination = self.root / "second_template"
        result = self.fetch(destination)
        self.assertTrue(result["cache_hit"])
        self.assertEqual(self.manifest(), self.manifest(destination))
        for record in result["files"]:
            self.assertEqual((self.template / record["path"]).read_bytes(),
                             (destination / record["path"]).read_bytes())

    def test_flat_manifest_preserves_other_assets_and_custom_fields(self):
        self.fetch()
        existing = self.manifest()
        existing[0]["usage"] = {"role": "lathe accessory", "scale": 0.5}
        extra = {"path": "assets/old.bin", "sha256": hashlib.sha256(b"old").hexdigest(),
                 "license": "CC0-1.0", "custom": [1, 2]}
        (self.template / extra["path"]).write_bytes(b"old")
        existing.append(extra)
        write_json(self.template / "assets.json", existing)
        self.fetch()
        merged = self.manifest()
        self.assertIsInstance(merged, list)
        self.assertEqual(len(merged), 3)
        self.assertEqual({item["path"]: item for item in merged},
                         {item["path"]: item for item in existing})

    def test_room_manifest_preserves_source_and_per_file_custom_fields(self):
        self.fetch()
        records = self.manifest()
        records[0]["usage"] = {"role": "machine housing", "orientation": [0, 0, 1]}
        room = {
            "reference_id": "shed_quality_example",
            "source": {"license": "CC0-1.0", "notes": ["quality reference only"]},
            "custom_scene_metadata": {"revision": 2},
            "files": records,
        }
        write_json(self.template / "assets.json", room)
        self.fetch()
        self.assertEqual(self.manifest(), room)

    def test_room_manifest_accepts_new_files_without_flattening(self):
        inherited = self.template / "assets" / "reference.txt"
        inherited.parent.mkdir()
        inherited.write_bytes(b"inherited source")
        record = {"path": "assets/reference.txt", "sha256": hashlib.sha256(inherited.read_bytes()).hexdigest(),
                  "custom": "preserve inherited asset"}
        room = {"reference_id": "fixture_room", "source": {"url": "reference-fixture"},
                "files": [record]}
        write_json(self.template / "assets.json", room)
        self.fetch()
        merged = self.manifest()
        self.assertEqual(merged["source"], room["source"])
        self.assertEqual(merged["reference_id"], room["reference_id"])
        self.assertEqual(len(merged["files"]), 3)
        self.assertIn(record, merged["files"])

    def test_corrupt_cache_is_rejected_before_copy(self):
        self.fetch()
        cached = self.cache / "local/fixture/original/files/model.gltf"
        cached.write_bytes(b"corrupted cache")
        destination = self.root / "rejected_template"
        with self.assertRaisesRegex(ValueError, "Cache SHA-256 mismatch"):
            self.fetch(destination)
        self.assertFalse(destination.exists())

    def test_edited_template_is_not_overwritten(self):
        self.fetch()
        model = self.template / "assets/local/fixture/original/model.gltf"
        model.write_bytes(b"manually edited model")
        manifest_before = (self.template / "assets.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "Template file has different content"):
            self.fetch()
        self.assertEqual(model.read_bytes(), b"manually edited model")
        self.assertEqual((self.template / "assets.json").read_bytes(), manifest_before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
