"""Focused diagnostics tests; no training or changes to the post algorithm."""
import math
import random

import numpy as np

from tools.legacy_failure_trace import PipelineTrace, comparison, json_value, observations


def test_comparison_does_not_confuse_brepcheck_with_valid_solid():
    original = {"native_valid_solid": False, "shape_type": 3, "native_brepcheck_valid": True}
    assert comparison(original, original)["same_recorded_outcome"]
    replay = {**original, "native_valid_solid": True, "shape_type": 2}
    assert set(comparison(original, replay)["differences"]) == {"native_valid_solid", "shape_type"}


def test_json_supports_list_connectivity_and_nonfinite_measurements():
    assert json_value([[np.int64(2), 3, 4]]) == [[2, 3, 4]]
    assert json_value({"distance": np.float32(math.nan)}) == {"distance": None}


def test_observations_keep_main_and_fallback_separate():
    trace = {"trim_runs": [{"context": "main", "num_faces": 2, "successful_faces": 1,
                            "failed_face_indices": [1]},
                           {"context": "fallback", "num_faces": 2, "successful_faces": 2,
                            "failed_face_indices": []}], "fallback_used": True}
    text = "\n".join(observations(trace))
    assert "main 裁剪：1/2" in text and "fallback 裁剪：2/2" in text
    assert "不能单独判定" in text


def test_diagnostic_error_is_visible_without_raising():
    trace = {}
    observer = PipelineTrace(trace, lambda: None)
    result = observer.safe(lambda: 1 / 0)
    assert "diagnostic_error" in result and len(trace["instrumentation_errors"]) == 1


def test_occ_wire_observer_preserves_rng_and_restores_original():
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    from OCC.Core.gp import gp_Pnt
    from src.brepnet.post import construct_brep as cb

    points = [gp_Pnt(0, 0, 0), gp_Pnt(1, 0, 0), gp_Pnt(1, 1, 0), gp_Pnt(0, 1, 0)]
    edges = [BRepBuilderAPI_MakeEdge(points[i], points[(i + 1) % 4]).Edge() for i in range(4)]
    original = cb.create_wire_from_unordered_edges
    random.seed(918)
    expected = original(edges.copy(), 0.002)
    expected_state = random.getstate()
    trace = {}
    random.seed(918)
    with PipelineTrace(trace, lambda: None) as observer:
        observer.current_trim = {"attempts": []}
        observed = cb.create_wire_from_unordered_edges(edges.copy(), 0.002)
        assert random.getstate() == expected_state
        assert len(expected) == len(observed) == 1
        assert observer.current_trim["attempts"][0]["wire_count"] == 1
    assert cb.create_wire_from_unordered_edges is original
    assert not trace.get("instrumentation_errors")
