"""Method-aware scheduling preserves the historical geometric measurements."""
import json
from pathlib import Path

import pytest

from tools import legacy_failure_provenance as provenance
from tools import legacy_failure_saved_step as saved_step
from tools import legacy_failure_upstream as upstream
from tools.vae_compare_post import atomic_json


def make_case(tmp_path, method, sample_id="00000001", historical=False):
    paths = {}
    for name, filename in (("raw_npz", "data.npz"), ("post_step", "recon_brep.step"),
                           ("fit_step", "recon_brep.step"), ("gt_npz", "data.npz"),
                           ("gt_step", "model.step")):
        path = tmp_path / "sources" / name / sample_id / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"source")
        paths[name] = str(path)
    case = {"id": sample_id, "method": method, "paths": paths}
    for stage in ("raw", "post", "fit"):
        case[stage + "_record"] = {"method": method, "native_valid_solid": True}
    if historical:
        case.pop("method")
        for stage in ("raw", "post", "fit"):
            case[stage + "_legacy"] = case.pop(stage + "_record")
        for name in ("raw_npz", "post_step", "fit_step"):
            key, extension = name.split("_")
            paths[key + "_legacy_" + extension] = paths.pop(name)
    return case


RUNNERS = [(provenance, "--output-root", "provenance"),
           (upstream, "--root", "upstream"),
           (saved_step, "--root", "saved_step")]


@pytest.mark.parametrize("module,root_argument,metadata_name", RUNNERS)
@pytest.mark.parametrize("method,historical", [("legacy", True), ("intersection", False)])
def test_runners_route_jobs_and_timeout_status_by_method(
        tmp_path, monkeypatch, module, root_argument, metadata_name, method, historical):
    case = make_case(tmp_path, method, historical=historical)
    manifest = tmp_path / "manifest.json"
    atomic_json(manifest, {"cases": [case]})
    root = tmp_path / "diagnostic"
    jobs_seen = []

    def runner(jobs, metadata, workers, timeout, process_target):
        jobs_seen.extend(jobs)
        for job in jobs:
            assert job["method"] == method
            if module is provenance:
                assert job["original"]["native_valid_solid"] is True
            atomic_json(metadata / "samples" / method / (job["sample_id"] + ".json"),
                        {"status": "timeout", "error": "diagnostic timeout"})

    monkeypatch.setattr(module, "run_isolated_jobs", runner)
    monkeypatch.setattr(module, "configure_cpu", lambda: None)
    if module is provenance:
        monkeypatch.setattr(module, "preload_cpu_dependencies", lambda: None)
    module.main(["--manifest", str(manifest), root_argument, str(root), "--workers", "1"])
    assert len(jobs_seen) == 1
    protocol = json.loads((root / "metadata" / metadata_name / "protocol.json").read_text())
    assert protocol["method"] == method
    if module is upstream:
        assert protocol["contract_helper_sha256"]
    else:
        assert protocol["source_sha256"]["tools/failure_census_contract.py"]
    result_path = root / ("traces" if module is provenance else metadata_name) / (case["id"] + ".json")
    result = json.loads(result_path.read_text())
    assert result["method"] == method
    assert result["status"] == "timeout"
    assert not (root / "metadata" / metadata_name / "samples" / ("legacy" if method == "intersection" else "intersection")).exists()


@pytest.mark.parametrize("module,root_argument,metadata_name", RUNNERS)
def test_mixed_methods_rejected_before_only_ids_filter(tmp_path, monkeypatch, module, root_argument, metadata_name):
    cases = [make_case(tmp_path, "legacy"), make_case(tmp_path, "intersection", "00000002")]
    manifest = tmp_path / "manifest.json"
    atomic_json(manifest, {"cases": cases})
    monkeypatch.setattr(module, "configure_cpu", lambda: None)
    with pytest.raises(ValueError, match="method"):
        module.main(["--manifest", str(manifest), root_argument, str(tmp_path / "out"),
                     "--only-ids", "00000002"])


@pytest.mark.parametrize("name", ["raw_npz", "post_step", "fit_step", "gt_npz", "gt_step"])
def test_provenance_cannot_overwrite_any_source_directory(tmp_path, name):
    case = make_case(tmp_path, "intersection")
    source = Path(case["paths"][name]).parent
    root = tmp_path / "diagnostic"
    (root / "replay").mkdir(parents=True)
    (root / "replay" / case["id"]).symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="overlap source"):
        provenance.validate_replay_paths(case, root)


def test_fit_only_uses_fit_source_and_original_success_record(tmp_path, monkeypatch):
    case = make_case(tmp_path, "intersection")
    case["post_record"]["native_valid_solid"] = False
    manifest = tmp_path / "manifest.json"
    atomic_json(manifest, {"cases": [case]})
    jobs_seen = []
    monkeypatch.setattr(provenance, "configure_cpu", lambda: None)
    monkeypatch.setattr(provenance, "preload_cpu_dependencies", lambda: None)
    monkeypatch.setattr(provenance, "run_isolated_jobs", lambda jobs, *args, **kwargs: jobs_seen.extend(jobs))
    provenance.main(["--manifest", str(manifest), "--output-root", str(tmp_path / "fit_diagnostic"),
                     "--post-mode", "fit_only", "--only-ids", case["id"]])
    assert jobs_seen[0]["original_step"] == case["paths"]["fit_step"]
    assert jobs_seen[0]["original"]["native_valid_solid"] is True
    assert jobs_seen[0]["mode"] == "fit_only"
