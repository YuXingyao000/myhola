"""Maintenance checks that run without PyTorch, CUDA or PythonOCC."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools import audit_deepcad_v7_dataset as audit
from tools import check_repo
from tools import filter_deepcad_v7_lists as filters


def test_layout_rejects_gitlinks_even_without_gitmodules(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "update-index", "--add", "--cacheinfo", "160000", "1" * 40, "thirdparty/pointnet2"],
        cwd=tmp_path, check=True,
    )
    monkeypatch.setattr(check_repo, "ROOT", tmp_path)
    assert any("Git submodules" in error for error in check_repo.check_layout())


@pytest.mark.parametrize("tool", ["audit", "filter"])
def test_audit_and_filter_require_explicit_asset_and_output_paths(tool, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["maintenance-tool"])
    with pytest.raises(SystemExit) as error:
        if tool == "audit":
            audit.build_arg_parser().parse_args([])
        else:
            filters.parse_args()
    assert error.value.code == 2


@pytest.mark.parametrize("tool", ["audit", "filter", "filter_in_place"])
def test_generated_reports_and_lists_cannot_write_into_repository(tool, tmp_path, monkeypatch):
    repo = Path(check_repo.__file__).resolve().parents[1]
    args = ["tool", "--data-root", str(tmp_path), "--condition-root", str(tmp_path),
            "--latent-root", str(tmp_path)]
    if tool == "audit":
        args += ["--out-dir", str(repo / "experiments" / "forbidden")]
        entry = audit.main
    else:
        args += ["--output-dir", str(tmp_path / "lists"), "--report-dir",
                 str(tmp_path / "reports") if tool == "filter_in_place" else str(repo / "experiments")]
        if tool == "filter_in_place":
            args += ["--in-place"]
        entry = filters.parse_args
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit) as error:
        entry()
    assert error.value.code == 2


def test_filter_records_missing_assets_and_preserves_source_split(tmp_path, monkeypatch):
    source = tmp_path / "input.txt"
    source.write_text("00000001\n00000002\n00000001\n")
    data, condition, latent = (tmp_path / name for name in ("data", "conditions", "latents"))
    for path in (data / "00000001/data.npz", condition / "00000001/imgs.npz",
                 latent / "00000001_0/features.npy"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    output, report = tmp_path / "filtered", tmp_path / "report"
    monkeypatch.setattr(sys, "argv", [
        "filter", "--list", str(source), "--data-root", str(data), "--condition-root", str(condition),
        "--latent-root", str(latent), "--output-dir", str(output), "--report-dir", str(report),
        "--latent-views", "identity", "--no-require-real-photo",
    ])
    filters.main()
    assert source.read_text() == "00000001\n00000002\n00000001\n"
    assert (output / "input.txt").read_text() == "00000001\n"
    assert "00000002" in (report / "removed.tsv").read_text()
    assert json.loads((report / "summary.json").read_text())
