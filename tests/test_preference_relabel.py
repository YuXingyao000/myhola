"""CPU orchestration tests for saved-pool relabeling, failures, and resumption."""

from dataclasses import asdict
import json
import multiprocessing as mp
import os
import sys
import time
from types import SimpleNamespace

import pytest

from src.preference import relabel
from src.preference.observation_selection import SelectionConfig
from src.common.io import atomic_json, fingerprint, read_json, sha256_file


def _config():
    return {"fingerprint": "derived", "candidates": 8, "seed": 42,
            "selection": asdict(SelectionConfig()), "image_timeout_seconds": 1.}


def _metrics(split, image_id, config, *, complete=True):
    rows = []
    for index in range(config["candidates"]):
        observation = {"status": "ok", "silhouette_iou": .8,
                       "visible_face_cd_sq": .01, "silhouette_boundary_distance": .02}
        if not complete and index >= 2:
            observation = {"status": "running" if index == 2 else "pending"}
        rows.append({"image_id": image_id, "candidate_id": f"c{index:02d}",
                     "validity": "valid" if index == 0 else "invalid",
                     "raw": {"face_cd_sq": .01, "edge_cd_sq": .02, "face_f1": .8},
                     "latent_path": f"/saved/{image_id}/c{index:02d}/latent.npy",
                     "dataset_protocol_fingerprint": config["fingerprint"],
                     "source_fingerprint": f"immutable-{index}", "observation": observation})
    return {"protocol_fingerprint": config["fingerprint"], "attempt_id": config.get("attempt_id"),
            "image_id": image_id, "split": split, "source_fingerprint": "immutable-image",
            "status": "complete" if complete else "running", "candidates": rows}


def _success_child(source, output, split, image_id, config):
    atomic_json(output / "cases" / split / image_id / "metrics.json",
                _metrics(split, image_id, config))


def _timeout_child(source, output, split, image_id, config):
    atomic_json(output / "cases" / split / image_id / "metrics.json",
                _metrics(split, image_id, config, complete=False))
    time.sleep(10)


def _partial_crash_child(source, output, split, image_id, config):
    atomic_json(output / "cases" / split / image_id / "metrics.json",
                _metrics(split, image_id, config, complete=False))
    os._exit(17)


def _crash_before_verification_child(source, output, split, image_id, config):
    os._exit(17)


def _integrity_error_child(source, output, split, image_id, config):
    atomic_json(output / "cases" / split / image_id / "error.json",
                {"error": "SourceIntegrityError: recorded hash changed"})
    os._exit(1)


def _coordinator(tasks, results, source, output, config):
    while True:
        task = tasks.get()
        if task is None:
            return
        split, image_id, old = task
        results.put({"result": relabel._bounded(source, output, split, image_id, config, old)})


def _dead_coordinator(tasks, results, source, output, config):
    os._exit(19)


def _failed_coordinator(tasks, results, source, output, config):
    tasks.get()
    results.put({"error": "SourceIntegrityError: changed input"})


def test_bounded_success_records_deterministic_source_paths_and_denominator(tmp_path, monkeypatch):
    monkeypatch.setattr(relabel, "_child", _success_child)
    output = tmp_path / "derived"
    result = relabel._bounded(tmp_path / "source", output, "train", "cad", _config(), None)
    row = read_json(output / "cases/train/cad/selection.json")
    assert result["counts"]["images"] == 1
    assert result["counts"]["candidates"] == 8
    assert result["counts"]["observation_ok"] == 8
    assert row["pair"]["winner_path"] == "/saved/cad/c00/latent.npy"
    assert row["pair"]["pair_type"] == "validity"
    first = row["pair"]
    relabel._bounded(tmp_path / "source", output, "train", "cad", _config(), first)
    resumed = read_json(output / "cases/train/cad/selection.json")
    assert resumed["pair"] == first
    assert resumed["change"] == "retained"


@pytest.mark.parametrize("child,status", [(_timeout_child, "timeout"), (_partial_crash_child, "crash")])
def test_bounded_process_failure_preserves_finished_candidates_and_all_denominators(
    tmp_path, monkeypatch, child, status,
):
    monkeypatch.setattr(relabel, "_child", child)
    config = _config()
    config["image_timeout_seconds"] = .15
    started = time.monotonic()
    result = relabel._bounded(tmp_path / "source", tmp_path / "out", "validation", "cad", config, None)
    assert time.monotonic() - started < 5
    assert result["counts"]["candidates"] == 8
    assert result["counts"]["observation_ok"] == 2
    assert result["counts"][f"observation_{status}"] == 6
    assert result["counts"]["pairs"] == 1
    row = read_json(tmp_path / "out/cases/validation/cad/metrics.json")
    assert row["status"] == status
    for candidate in row["candidates"][2:]:
        checkpoint = (tmp_path / "out/candidates/validation/cad" /
                      candidate["candidate_id"] / "observation.json")
        assert read_json(checkpoint)["observation"]["status"] == status


def test_resume_cannot_accept_stale_metrics_when_child_dies_before_source_verification(tmp_path, monkeypatch):
    config = _config()
    output = tmp_path / "out"
    stale = _metrics("train", "cad", {**config, "attempt_id": "old-attempt"})
    atomic_json(output / "cases/train/cad/metrics.json", stale)
    monkeypatch.setattr(relabel, "_child", _crash_before_verification_child)
    with pytest.raises(RuntimeError):
        relabel._bounded(tmp_path / "source", output, "train", "cad", config, None)
    assert not (output / "cases/train/cad/selection.json").exists()


def test_recorded_integrity_error_is_fatal_even_with_stale_complete_metrics(tmp_path, monkeypatch):
    config = _config()
    output = tmp_path / "out"
    atomic_json(output / "cases/train/cad/metrics.json", _metrics("train", "cad", config))
    monkeypatch.setattr(relabel, "_child", _integrity_error_child)
    with pytest.raises(RuntimeError, match="recorded hash changed"):
        relabel._bounded(tmp_path / "source", output, "train", "cad", config, None)


@pytest.mark.parametrize("corruption", ["protocol", "image", "split", "candidate_ids"])
def test_worker_result_identity_and_protocol_must_match_job(tmp_path, monkeypatch, corruption):
    def malformed_child(source, output, split, image_id, config):
        row = _metrics(split, image_id, config)
        if corruption == "protocol":
            row["protocol_fingerprint"] = "other"
        elif corruption == "image":
            row["image_id"] = "other"
        elif corruption == "split":
            row["split"] = "validation"
        else:
            row["candidates"][1]["candidate_id"] = "c00"
        atomic_json(output / "cases" / split / image_id / "metrics.json", row)

    monkeypatch.setattr(relabel, "_child", malformed_child)
    with pytest.raises((ValueError, RuntimeError)):
        relabel._bounded(tmp_path / "source", tmp_path / "out", "train", "cad", _config(), None)


def _protocol(source, *, partial=False):
    atomic_json(source / "metadata/protocol.json", {"fixture": "source-protocol"})
    atomic_json(source / "status.json", {"status": "complete"})
    for name in relabel.MANIFESTS.values():
        (source / name).write_text("")
    return {**_config(), "source_run": str(source), "partial": partial,
            "splits": {"train": ["train-b", "train-a"], "validation": ["val-a"]},
            "source_protocol_sha256": sha256_file(source / "metadata/protocol.json"),
            "source_status_sha256": sha256_file(source / "status.json"),
            "source_manifest_sha256": {name: sha256_file(source / name) for name in relabel.MANIFESTS.values()},
            "source_sha256": {}}


def _fork_coordinators(monkeypatch, target):
    original_context = mp.get_context
    monkeypatch.setattr(relabel.mp, "get_context", lambda method: original_context("fork"))
    monkeypatch.setattr(relabel, "_worker_loop", target)


def test_queue_finalization_and_resume_rewrite_manifest_in_frozen_id_order(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "out"
    output.mkdir()
    protocol = _protocol(source)
    _fork_coordinators(monkeypatch, _coordinator)
    monkeypatch.setattr(relabel, "_child", _success_child)
    initial_children = {child.pid for child in mp.active_children()}
    status = relabel.run_jobs(source, output, protocol, 2, {})
    assert status["completed"] == status["total"] == 3
    assert status["train"]["candidates"] == 16
    assert status["validation"]["candidates"] == 8
    assert {child.pid for child in mp.active_children()} == initial_children
    relabel.finalize(output, protocol, status)
    first = (output / "pairs.jsonl").read_bytes()
    rows = [json.loads(line) for line in first.splitlines()]
    assert [row["image_id"] for row in rows] == ["train-b", "train-a"]
    assert read_json(output / "summary.json")["splits"]["train"]["intended_candidates"] == 16
    assert read_json(output / "status.json")["manifest_sha256"]["pairs.jsonl"] == sha256_file(output / "pairs.jsonl")
    # Resume may finish workers in another order, but never appends duplicate pairs.
    status = relabel.run_jobs(source, output, protocol, 1, {})
    relabel.finalize(output, protocol, status)
    assert (output / "pairs.jsonl").read_bytes() == first
    (source / "pairs.jsonl").write_text("changed during run\n")
    with pytest.raises(ValueError, match="Frozen source changed"):
        relabel.finalize(output, protocol, status)


@pytest.mark.parametrize("target", [_dead_coordinator, _failed_coordinator])
def test_coordinator_failure_is_prompt_and_does_not_leave_running_status_or_children(tmp_path, monkeypatch, target):
    source, output = tmp_path / "source", tmp_path / "out"
    output.mkdir()
    _fork_coordinators(monkeypatch, target)
    initial_children = {child.pid for child in mp.active_children()}
    started = time.monotonic()
    with pytest.raises(RuntimeError):
        relabel.run_jobs(source, output, _protocol(source), 2, {})
    assert time.monotonic() - started < 8
    assert read_json(output / "status.json")["status"] == "failed"
    assert {child.pid for child in mp.active_children()} == initial_children


def test_finalize_keeps_abstentions_in_denominator_and_rejects_stale_selection(tmp_path):
    protocol = _protocol(tmp_path / "source", partial=True)
    output = tmp_path / "out"
    output.mkdir()
    for split, ids in protocol["splits"].items():
        for image_id in ids:
            atomic_json(output / "cases" / split / image_id / "selection.json", {
                "image_id": image_id, "split": split, "protocol_fingerprint": protocol["fingerprint"],
                "pair": None,
            })
    status = {"completed": 3, "total": 3, "partial": True, "status": "finalizing",
              "train": {"images": 2, "candidates": 16, "pairs": 0},
              "validation": {"images": 1, "candidates": 8, "pairs": 0}}
    relabel.finalize(output, protocol, status)
    summary = read_json(output / "summary.json")
    assert summary["partial"] is True
    assert summary["splits"]["train"]["intended_images"] == 2
    assert summary["splits"]["train"]["pair_coverage"] == 0.
    assert (output / "pairs.jsonl").read_text() == ""
    incomplete = {**status, "train": {"images": 1, "candidates": 8, "pairs": 0}}
    with pytest.raises(ValueError, match="Incomplete image denominator"):
        relabel.finalize(output, protocol, incomplete)
    row_path = output / "cases/train/train-b/selection.json"
    row = read_json(row_path)
    row["protocol_fingerprint"] = "stale"
    atomic_json(row_path, row)
    with pytest.raises(ValueError, match="stale selection"):
        relabel.finalize(output, protocol, status)


def test_full_partial_protocol_and_changed_source_reject_resume(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "derived"
    old = {"version": "svr_preference_generation_v1", "candidates": 8,
           "splits": {"train": ["train-b", "train-a"], "validation": ["val-a"]},
           "checkpoint_sha256": "frozen-checkpoint", "condition_data_root": str(tmp_path / "images"),
           "raw_data_root": str(tmp_path / "raw")}
    old["fingerprint"] = fingerprint(old)
    atomic_json(source / "metadata/protocol.json", old)
    atomic_json(source / "status.json", {"status": "complete"})
    for name in relabel.MANIFESTS.values():
        (source / name).write_text("")
    monkeypatch.setitem(sys.modules, "OCC", SimpleNamespace(VERSION="fixture"))
    args = relabel.parser().parse_args(["--source-run", str(source), "--output-dir", str(output)])
    protocol = relabel.build_protocol(args)
    assert protocol["partial"] is False
    assert protocol["splits"] == old["splits"]
    args.limit = 1
    partial = relabel.build_protocol(args)
    assert partial["partial"] is True
    assert partial["splits"]["train"] == ["train-b"]
    args.limit = None
    args.splits = ["validation"]
    validation_only = relabel.build_protocol(args)
    assert validation_only["partial"] is True
    assert validation_only["splits"]["train"] == []
    atomic_json(output / "metadata/protocol.json", protocol)
    args.resume = True
    args.splits = ["validation", "train"]
    (source / "pairs.jsonl").write_text('{"edited": true}\n')
    changed = relabel.build_protocol(args)
    with pytest.raises(ValueError, match="unchanged source"):
        relabel.prepare_run(output, changed, args)


@pytest.mark.parametrize("splits", [
    {"train": ["a", "a"], "validation": ["b"]},
    {"train": ["a"], "validation": ["a"]},
    {"train": [], "validation": ["a"]},
    {"train": ["../escape"], "validation": ["b"]},
])
def test_source_split_integrity(splits):
    with pytest.raises(ValueError):
        relabel.validate_splits(splits)


@pytest.mark.parametrize("old,new,expected", [
    (None, None, "neither"), (None, ("a", "b"), "added"), (("a", "b"), None, "abstained"),
    (("a", "b"), ("a", "b"), "retained"), (("a", "b"), ("b", "a"), "flipped"),
    (("a", "b"), ("a", "c"), "replaced"),
])
def test_old_new_pair_transition_statistics(old, new, expected):
    convert = lambda pair: {"winner_id": pair[0], "loser_id": pair[1]} if pair else None
    assert relabel.pair_change(convert(old), convert(new)) == expected
