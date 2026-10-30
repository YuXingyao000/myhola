"""CPU-only contracts for the DeepCAD extraction entrypoint; no CAD corpus or model weights."""
import sys

import pytest

from src.brepnet.data import extract_brep


def extraction_args(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    ids = tmp_path / "ids.txt"
    ids.write_text("00000002\n00000001\n00000002\n")
    return ["--data-root", str(source), "--output-root", str(tmp_path / "output"), "--model-list", str(ids)]


def test_extraction_dry_run_preserves_source_and_does_not_import_occ(tmp_path):
    argv = extraction_args(tmp_path)
    excluded = tmp_path / "excluded.txt"
    excluded.write_text("00000001\n")
    args = extract_brep.parse_args(argv + ["--exclude-list", str(excluded)])
    ids, manifest = extract_brep.prepare_plan(args)
    assert ids == ["00000002"]
    assert manifest["source_count"] == 2
    assert manifest["excluded_count"] == 1
    extract_brep.main(argv + ["--dry-run"])
    assert not (tmp_path / "output").exists()
    assert "src.brepnet.data.extract_brep_from_step_half_structure" not in sys.modules


def test_extraction_preserves_existing_outputs(tmp_path):
    argv = extraction_args(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    existing = output / "existing.txt"
    existing.write_text("preserve")
    with pytest.raises(FileExistsError, match="preserved"):
        extract_brep.prepare_plan(extract_brep.parse_args(argv))
    assert existing.read_text() == "preserve"


def test_extraction_rejects_overlapping_paths(tmp_path):
    args = extract_brep.parse_args(extraction_args(tmp_path))
    args.output_root = args.data_root / "derived"
    with pytest.raises(ValueError, match="overlap"):
        extract_brep.prepare_plan(args)
