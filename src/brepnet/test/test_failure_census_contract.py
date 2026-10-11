from pathlib import Path

import pytest

from tools.failure_census_contract import case_method, case_path, case_record, manifest_method


def test_legacy_manifest_is_supported_without_relabelling():
    case = {"id":"00000001", "raw_legacy":{"method":"legacy","model_id":"00000001"},
            "paths":{"raw_legacy_npz":"/old/00000001/data.npz"}}
    assert case_method(case) == "legacy"
    assert case_record(case,"raw") is case["raw_legacy"]
    assert case_path(case,"raw_npz") == Path("/old/00000001/data.npz")
    assert manifest_method([case]) == "legacy"
    case["method"] = "intersection"
    with pytest.raises(ValueError, match="record"):
        case_record(case,"raw")
    with pytest.raises(ValueError, match="path"):
        case_path(case,"raw_npz")


def test_generic_case_identity_and_method_are_checked():
    case = {"id":"00000001", "method":"intersection",
            "post_record":{"method":"intersection","sample_id":"00000001"},
            "paths":{"post_step":"/new/00000001/recon_brep.step"}}
    assert case_record(case,"post") is case["post_record"]
    assert case_path(case,"post_step").name == "recon_brep.step"
    case["post_record"]["sample_id"] = "00000002"
    with pytest.raises(ValueError,match="identity"):
        case_record(case,"post")
    case["post_record"]["sample_id"] = "00000001"
    case["post_record"]["method"] = "legacy"
    with pytest.raises(ValueError,match="method"):
        case_record(case,"post")


def test_present_invalid_generic_fields_do_not_fall_back():
    case = {"id":"00000001","raw_record":None,"raw_legacy":{},
            "paths":{"raw_npz":"","raw_legacy_npz":"old.npz"}}
    with pytest.raises(ValueError):
        case_record(case,"raw")
    with pytest.raises(ValueError):
        case_path(case,"raw_npz")
    with pytest.raises(ValueError):
        case_method({"method":None})


def test_empty_or_mixed_method_manifest_rejected():
    for cases in ([], [{"method":"legacy"},{"method":"intersection"}]):
        with pytest.raises(ValueError,match="exactly one method"):
            manifest_method(cases)
