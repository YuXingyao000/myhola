"""Photo generation binds the right input/model and preserves failed requests."""

from dataclasses import asdict
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.preference.generate import (
    _completed_records, input_failure_records, main, prepare_image, run_phase,
)
from src.preference.io import prepare_run
from src.preference.selection import SelectionConfig
from src.preference.source import (
    image_path, is_photo, load_condition, retain_native_validity, source_protocol,
)
from tools.vae_compare_post import read_json, sha256_file


def config_fixture(root):
    original, checkpoint = root / "original.ckpt", root / "photo.ckpt"
    original.write_bytes(b"frozen original checkpoint")
    torch.save({"global_step": 310000, "state_dict": {}, "hyper_parameters": {
        "model": {"name": "PhotoBaselineDiffusion"},
        "dataset": {"provenance": {"source_checkpoint_sha256": sha256_file(original)}},
    }}, checkpoint)
    (root / "train.txt").write_text("train_a\n")
    (root / "validation.txt").write_text("val_a\nval_b\n")
    (root / "dino").mkdir()
    (root / "dino/hubconf.py").write_text("# frozen source\n")
    return {"model_source": "photo", "checkpoint": str(checkpoint),
            "original_checkpoint": str(original), "expected_global_step": 310000,
            "dino_repo": str(root / "dino"), "train_list": str(root / "train.txt"),
            "validation_list": str(root / "validation.txt"), "condition_data_root": str(root / "images"),
            "raw_data_root": str(root / "cad"), "gpu_cpu_threads": 2, "seed": 42, "candidates": 8,
            "selection": asdict(SelectionConfig(version="photo_preference_v1")),
            "post": {"timeout_seconds": 180, "max_optimize_iter": 200},
            "stop_if_no_validation_pairs": False}


def test_photo_reads_flux_without_any_grey_render(tmp_path):
    cfg = config_fixture(tmp_path)
    path = image_path(cfg, "part")
    path.parent.mkdir(parents=True)
    photo = np.full((512, 512, 3), (32, 64, 192), dtype=np.uint8)
    np.savez(path, flux=photo, flux_masked=np.zeros_like(photo))
    result = load_condition(cfg, "part")
    expected = (torch.tensor([32, 64, 192])/255 - torch.tensor([.485, .456, .406])) / torch.tensor([.229, .224, .225])
    assert result.shape == (3, 224, 224)
    torch.testing.assert_close(result[:, 0, 0], expected)
    np.savez(path, flux=np.zeros((24, 512, 512, 3), np.uint8))
    with pytest.raises(ValueError, match="flux"):
        load_condition(cfg, "part")
    np.savez(path, flux_masked=photo)
    with pytest.raises(KeyError):
        load_condition(cfg, "part")
    with pytest.raises(ValueError, match="model_source"):
        is_photo({"model_source": "guess"})


def test_protocol_binds_photo_original_model_and_full_validation(tmp_path):
    cfg = config_fixture(tmp_path)
    run = tmp_path / "run"
    protocol = prepare_run(cfg, run, False)
    assert protocol["version"] == "photo_preference_generation_v1"
    assert protocol["splits"]["validation"] == ["val_a", "val_b"]
    assert protocol["splits"]["train"] == ["train_a"]
    assert protocol["condition"]["key"] == "flux" and protocol["photo_global_step"] == 310000
    assert prepare_run(cfg, run, True) == protocol
    with pytest.raises(ValueError, match="Resume protocol"):
        prepare_run({**cfg, "stop_if_no_validation_pairs": True}, run, True)
    with pytest.raises(ValueError, match="global_step"):
        source_protocol({**cfg, "expected_global_step": 4553})
    Path(cfg["original_checkpoint"]).write_bytes(b"different AE")
    with pytest.raises(ValueError, match="different original"):
        source_protocol(cfg)


def test_missing_image_keeps_eight_unknowns_and_resumes_without_retry(tmp_path):
    cfg = config_fixture(tmp_path)
    run = tmp_path / "run"
    with patch('src.models.diffusion.legacy.encode_original_gt', side_effect=AssertionError('No encoding on missing input')):
        assert prepare_image(None, cfg, run, "validation", "val_a") is None
        records = input_failure_records(run, "validation", "val_a", cfg)
    assert len(records) == 8 and all(row["validity"] == "unknown" for row in records)
    assert _completed_records(run, "validation", "val_a", cfg) == records
    with patch('src.preference.generate.mp.get_context', side_effect=AssertionError('No workers')):
        summary = run_phase("validation", ["val_a"], cfg, run)
    assert summary["candidates"] == 8 and summary["pairs"] == 0
    assert (run / "validation_pairs.jsonl").is_file()
    assert not (run / "pairs.jsonl").exists()
    photo = image_path(cfg, "val_a")
    photo.parent.mkdir(parents=True)
    photo.write_bytes(b"newly appeared input")
    with pytest.raises(ValueError, match="Previously missing"):
        _completed_records(run, "validation", "val_a", cfg)


def test_native_result_survives_later_error_without_inventing_validity():
    unknown = {"status": "timeout", "validity": "unknown"}
    assert retain_native_validity(unknown) is unknown
    native = {**unknown, "native": {"native_valid_solid": True}, "post": []}
    result = retain_native_validity(native)
    assert result["status"] == "timeout" and result["validity"] == "valid" and result["post"] == []
    assert native["validity"] == "unknown"
    failed = {**unknown, "native": {"native_valid_solid": False}}
    assert retain_native_validity(failed)["validity"] == "invalid"


def test_empty_validation_pairs_still_runs_training_split(tmp_path):
    cfg = {**config_fixture(tmp_path), "gpus": [0], "cpu_workers": 1, "queue_size": 1}
    path = tmp_path / "config.yaml"
    path.write_text(json.dumps(cfg))
    run = tmp_path / "run"
    phase_results = [{"images": 2, "candidates": 16, "pairs": 0},
                     {"images": 1, "candidates": 8, "pairs": 1}]
    argv = ['generation', '--config', str(path), '--run-dir', str(run)]
    with patch('sys.argv', argv), patch('src.preference.generate.run_phase',
                                       side_effect=phase_results) as phases:
        main()
    assert [call.args[0] for call in phases.call_args_list] == ['validation', 'train']
    status = read_json(run / 'status.json')
    assert status['status'] == 'complete' and status['validation']['pairs'] == 0
    assert status['train']['pairs'] == 1
