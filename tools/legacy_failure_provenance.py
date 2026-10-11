"""Read-only stage provenance for the frozen legacy failure census.

Original post functions are called once. A code-object-filtered Python trace
observes existing branches; additional geometric projections run only after
STEP output is complete, using copied fitted surfaces/curves. Labels describe
observations, never a unique causal explanation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import inspect
import os
from pathlib import Path
import random
import sys
import time
import traceback

from tools.legacy_failure_trace import PipelineTrace, SOURCE_FILES, comparison, json_value, observations
from tools.vae_compare_post import (atomic_json, configure_cpu, fingerprint, native_validity,
    preload_cpu_dependencies, read_json, read_step_at_precision, run_isolated_jobs,
    sha256_file, stable_seed, validate_raw_prediction)

VERSION = "legacy_failure_provenance_v1"
SAMPLING = {
    "face_input": "16 points: original 16x16 grid [::4,::4]",
    "curve_input": "16 original curve points",
    "curve_owner": "9 uniform native-parameter samples including endpoints",
    "distance": "Euclidean normalized XYZ length; surface projection / finite-edge distance at fixed samples, not Hausdorff or proof of UV validity",
    "surface_domain": "fitted finite BSpline surface bounds; before wire trimming",
    "fit_internal_score": "sqrt(mean(point-to-shape Euclidean distance)) + max(distance); historical score, NOT RMSE",
}


def step_data_hash(path):
    """Exclude nondeterministic STEP HEADER; do not normalize DATA bytes."""
    data = Path(path).read_bytes()
    start = data.index(b"DATA;") + len(b"DATA;")
    end = data.index(b"ENDSEC;", start)
    return hashlib.sha256(data[start:end]).hexdigest()


def line_of(function, text, occurrence=0):
    lines, start = inspect.getsourcelines(function)
    matches = [start + i for i, line in enumerate(lines) if line.strip() == text]
    if len(matches) <= occurrence:
        raise ValueError(f"Trace anchor absent: {function.__qualname__}: {text}")
    return matches[occurrence]


def existing_analyzer_status(analyzer, shape):
    """Read the ORIGINAL analyzer's results; never run ShapeFix or another analyzer."""
    from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_EDGE, TopAbs_WIRE, TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID
    from tools.legacy_failure_geometry import primitive_map, _status_names, SHAPE_NAMES
    bad, errors = [], []
    for kind, enum in (("vertices", TopAbs_VERTEX), ("edges", TopAbs_EDGE), ("wires", TopAbs_WIRE),
                       ("faces", TopAbs_FACE), ("shells", TopAbs_SHELL), ("solids", TopAbs_SOLID)):
        items = primitive_map(shape, enum)
        for index in range(1, items.Size() + 1):
            try:
                check = analyzer.Result(items.FindKey(index))
                if check is None:
                    continue
                status = [x for x in _status_names(check.Status()) if x != "BRepCheck_NoError"]
                contexts = []
                check.InitContextIterator()
                while check.MoreShapeInContext():
                    names = [x for x in _status_names(check.StatusOnShape()) if x != "BRepCheck_NoError"]
                    if names:
                        contexts.append({"shape_type": SHAPE_NAMES[int(check.ContextualShape().ShapeType())], "statuses": names})
                    check.NextShapeInContext()
                if status or contexts:
                    bad.append({"kind": kind, "id": index - 1, "statuses": status, "contexts": contexts})
            except Exception as exc:
                errors.append(f"{kind}/{index - 1}: {type(exc).__name__}: {exc}")
    return {"with_errors": bad, "inspection_errors": errors,
            "id_scope": "local to this transient trimmed candidate; not STEP face IDs"}


def distribution(values):
    import numpy as np
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return {"count": len(values), "finite_count": len(finite),
            "status": "ok" if len(finite) == len(values) and len(values) else "partial" if len(finite) else "unavailable",
            "mean": float(finite.mean()) if len(finite) else None,
            "max": float(finite.max()) if len(finite) else None,
            "rms": float(np.sqrt(np.mean(finite ** 2))) if len(finite) else None}


def surface_distances(points, surface):
    from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
    from OCC.Core.gp import gp_Pnt
    values, errors = [], []
    for index, point in enumerate(points):
        try:
            projection = GeomAPI_ProjectPointOnSurf(gp_Pnt(*map(float, point)), surface, *surface.Bounds(), 1e-7)
            if projection.NbPoints() == 0:
                raise ValueError("no projection")
            values.append(projection.LowerDistance())
        except Exception as exc:
            values.append(float("nan"))
            errors.append({"sample": index, "error": f"{type(exc).__name__}: {exc}"})
    return {**distribution(values), "errors": errors}


def curve_distances(points, curve, first, last):
    # A nearest point can be an endpoint with no stationary orthogonal curve
    # projection. DistShapeShape includes finite-edge endpoint solutions.
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeVertex
    from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
    from OCC.Core.gp import gp_Pnt
    edge = BRepBuilderAPI_MakeEdge(curve, first, last).Edge()
    values, errors = [], []
    for index, point in enumerate(points):
        try:
            vertex = BRepBuilderAPI_MakeVertex(gp_Pnt(*map(float, point))).Vertex()
            distance = BRepExtrema_DistShapeShape(vertex, edge)
            if not distance.IsDone() or distance.NbSolution() == 0:
                raise ValueError("no finite-edge distance solution")
            values.append(distance.Value())
        except Exception as exc:
            values.append(float("nan"))
            errors.append({"sample": index, "error": f"{type(exc).__name__}: {exc}"})
    return {**distribution(values), "errors": errors}


class ProvenanceTrace(PipelineTrace):
    def __init__(self, trace, flush):
        super().__init__(trace, flush)
        self.frames = {}
        self.targets = {}
        self.fitted = None
        self.previous_trace = None

    def __enter__(self):
        from src.brepnet.post import occ_topology as top, occ_fit as fit, geometry_optimization as opt
        from src.brepnet.post import construct_brep as cb
        functions = {"optimize": opt.optimize, "trim": top.create_trimmed_face_from_wire,
                     "solid": top.get_solid, "surface": fit.create_surface, "curve": fit.create_edge,
                     "fit": cb._fit_occ_geometry, "pipeline": cb.construct_brep_for_sample,
                     "save": cb._try_save_valid_solid}
        for kind, function in functions.items():
            self.targets[function.__code__] = kind
        self.anchors = {
            "opt_loss": line_of(opt.optimize, "optimizer.zero_grad()"),
            "wire_new": line_of(top.create_trimmed_face_from_wire, "wire_fixer = ShapeFix_Wire(wire, topo_face, connected_tolerance)"),
            "wire_fixed": line_of(top.create_trimmed_face_from_wire, "if wire_fixer.Wire().NbChildren() == 1 and not wire_fixer.Wire().Closed():"),
            "skip_single": line_of(top.create_trimmed_face_from_wire, "continue", 0),
            "skip_open": line_of(top.create_trimmed_face_from_wire, "continue", 1),
            "added": line_of(top.create_trimmed_face_from_wire, "fixed_wire_list.append(fixed_wire)"),
            "periodic": line_of(top.create_trimmed_face_from_wire, "wire_list = [final_wire]"),
            "no_wires": line_of(top.create_trimmed_face_from_wire, "return None", 0),
            "fix_exception": line_of(top.create_trimmed_face_from_wire, "return None", 1),
            "null_face": line_of(top.create_trimmed_face_from_wire, "return None", 2),
            "invalid_face": line_of(top.create_trimmed_face_from_wire, "return None", 3),
            "valid_face": line_of(top.create_trimmed_face_from_wire, "return face_occ"),
            "solid_compound": line_of(top.get_solid, "return None", 0),
            "solid_failed": line_of(top.get_solid, "return None", 2),
            "solid_exception": line_of(top.get_solid, "return None", 3),
            "save_no_solid": line_of(cb._try_save_valid_solid, "return False", 0),
            "save_rejected": line_of(cb._try_save_valid_solid, "os.remove(step_path)"),
            "save_passed": line_of(cb._try_save_valid_solid, "write_stl_file(solid, str(out_sample_dir / \"recon_brep.stl\"),"),
            "trim_gate": line_of(cb.construct_brep_for_sample, "if len(valid_trimmed_faces) < 0.8 * num_faces:"),
        }
        trim = top.create_trimmed_face_from_wire
        self.trim_stages = {line_of(trim, text, occurrence): label for text, occurrence, label in (
            ("face_fixer.Perform()", 0, "perform"), ("face_fixer.FixOrientation()", 0, "orientation_1"),
            ("face_fixer.FixMissingSeam()", 0, "missing_seam"),
            ("face_fixer.FixIntersectingWires()", 0, "intersecting_wires"),
            ("face_fixer.FixOrientation()", 1, "orientation_2"))}
        self.solid_stages = {line_of(top.get_solid, text): label for text, label in (
            ("fix_shell.Perform()", "shell_fix"), ("maker.Build()", "make_solid"),
            ("fix_solid.Perform()", "solid_fix"), ("set_tolerance(solid, connected_tolerance)", "final_check"))}
        self.fit_score_lines = {"surface": line_of(fit.create_surface, "if error < 1e-2:"),
                                "curve": line_of(fit.create_edge, "if error < 1e-2:")}
        super().__enter__()
        self.previous_trace = sys.gettrace()
        if self.previous_trace is not None:
            super().__exit__(None, None, None)
            raise RuntimeError("Refusing to replace an existing Python trace function")
        sys.settrace(self.dispatch)
        return self

    def __exit__(self, *exception):
        sys.settrace(self.previous_trace)
        self.frames.clear()
        return super().__exit__(*exception)

    def dispatch(self, frame, event, arg):
        kind = self.targets.get(frame.f_code)
        if kind is None:
            return None
        try:
            self.observe(frame, event, arg, kind)
        except Exception as exc:
            self.trace.setdefault("instrumentation_errors", []).append(
                f"{kind}:{event}:{frame.f_lineno}: {type(exc).__name__}: {exc}")
        return self.dispatch

    def observe(self, frame, event, arg, kind):
        values, number = frame.f_locals, frame.f_lineno
        if event == "call":
            record = {"exceptions": []}
            self.frames[id(frame)] = record
            if kind == "trim":
                record.update(wire_events=[], periodic_wire_substitution=False, face_fix_stages=[], outcome="unknown")
                if self.current_trim and self.current_trim["attempts"]:
                    self.current_trim["attempts"][-1]["trim_detail"] = record
            elif kind == "solid" and self.sewing_context is not None:
                record.update(stages=[], reject_reason=None)
                self.sewing_context["detail"] = record
            elif kind == "save":
                record.update(outcome="unknown", in_memory_solid_returned=None, legacy_saved_step_check=None)
                self.trace.setdefault("save_attempts", []).append(record)
            elif kind == "optimize":
                record.update(loss_records=[], loss_timing="before each AdamW update; final returned coordinates include the last update")
                self.trace["optimization_detail"] = record
            elif kind in ("surface", "curve"):
                entries = self.trace.setdefault("fit_attempts", {}).setdefault(kind, [])
                record.update(index=len(entries), scores=[])
                entries.append(record)
            return
        record = self.frames.get(id(frame), {})
        if event == "exception":
            exc_type, exc, _ = arg
            record.setdefault("exceptions", []).append({"line": number, "type": exc_type.__name__, "message": str(exc)})
        if event == "line":
            if kind == "optimize" and number == self.anchors["opt_loss"]:
                record["loss_records"].append({"iteration": int(values["iter"]), **{
                    key: float(values[key].detach().item()) for key in
                    ("loss", "adj_distance_loss", "corners_loss", "wire_connected_loss")}})
            elif kind == "trim":
                self.observe_trim(record, values, number)
            elif kind == "solid":
                if number in self.solid_stages:
                    record["stages"].append(self.solid_stages[number])
                if number == self.anchors["solid_compound"]:
                    record["reject_reason"] = "sewn_compound_early_return"
                elif number == self.anchors["solid_failed"]:
                    record["reject_reason"] = "final_not_valid_solid"
                elif number == self.anchors["solid_exception"]:
                    record["reject_reason"] = "caught_exception"
            elif kind == "save":
                if number == self.anchors["save_no_solid"]:
                    record.update(outcome="no_solid_returned", in_memory_solid_returned=False)
                elif number == self.anchors["save_rejected"]:
                    record.update(outcome="saved_step_rejected", in_memory_solid_returned=True, legacy_saved_step_check=False)
                elif number == self.anchors["save_passed"]:
                    record.update(outcome="saved_step_passed", in_memory_solid_returned=True, legacy_saved_step_check=True)
            elif kind in ("surface", "curve") and number == self.fit_score_lines[kind]:
                record["scores"].append({"precision": float(values["precision"]), "score": float(values["error"])})
            elif kind == "pipeline" and number == self.anchors["trim_gate"]:
                count, total = len(values["valid_trimmed_faces"]), values["num_faces"]
                self.trace["trim_gate"] = {"successful_faces": count, "total_faces": total,
                    "solid_attempt_allowed": count > .8 * total, "at_exact_eighty_percent": count == .8 * total}
        if event == "return":
            if kind == "optimize":
                record.update(final_iteration_count=int(values.get("iter", 0)),
                    returned_original=bool(values.get("is_optimization_diverged", False)),
                    init_loss=values.get("init_loss"), best_loss=values.get("best_loss"))
            elif kind in ("surface", "curve"):
                scores = record["scores"]
                record["selected"] = min(scores, key=lambda item: item["score"]) if scores else None
                record["rough_fallback"] = not bool(scores)
            elif kind == "fit" and arg is not None:
                self.capture_fitted(values["shape"], arg)
            self.frames.pop(id(frame), None)

    def observe_trim(self, record, values, number):
        if number == self.anchors["wire_new"]:
            wire = values["wire"]
            record["wire_events"].append({"wire_index": len(record["wire_events"]),
                "original_closed": bool(wire.Closed()), "repaired_closed": None, "added": False, "skip_reason": None})
        elif number == self.anchors["wire_fixed"]:
            wire = values["wire_fixer"].Wire()
            record["wire_events"][-1].update(repaired_closed=bool(wire.Closed()), repaired_children=int(wire.NbChildren()))
        elif number in (self.anchors["skip_single"], self.anchors["skip_open"]):
            record["wire_events"][-1]["skip_reason"] = "single_edge_open" if number == self.anchors["skip_single"] else "open_after_fix"
        elif number == self.anchors["added"]:
            record["wire_events"][-1]["added"] = True
        elif number == self.anchors["periodic"]:
            record["periodic_wire_substitution"] = True
        if number in self.trim_stages:
            record["face_fix_stages"].append(self.trim_stages[number])
        outcomes = {"no_wires": "no_admitted_wires", "fix_exception": "face_fix_exception",
                    "null_face": "null_face", "invalid_face": "invalid_face", "valid_face": "success"}
        for anchor, outcome in outcomes.items():
            if number == self.anchors[anchor]:
                record["outcome"] = outcome
        if number in (self.anchors["invalid_face"], self.anchors["valid_face"]):
            record["final_brepcheck"] = existing_analyzer_status(values["face_analyzer"], values["face_occ"])

    def capture_fitted(self, shape, fitted):
        from OCC.Core.Geom import Geom_Surface, Geom_Curve
        from OCC.Core.BRep import BRep_Tool
        surfaces = [Geom_Surface.DownCast(item.Copy()) for item in fitted[0]]
        curves = [(Geom_Curve.DownCast(item.Copy()), float(item.FirstParameter()), float(item.LastParameter())) for item in fitted[2]]
        effective = []
        for edge in fitted[3]:
            curve, first, last = BRep_Tool.Curve(edge)
            effective.append((Geom_Curve.DownCast(curve.Copy()), first, last))
        self.fitted = (surfaces, curves, effective)

    def fitted_residuals(self):
        import numpy as np
        if self.fitted is None:
            return {"status": "unknown", "reason": "fit snapshot unavailable", "sampling": SAMPLING}
        surfaces, curves, effective = self.fitted
        faces = self.geometry.get("optimized_faces", self.geometry["selected_faces"])
        edges = self.geometry.get("optimized_edges", self.geometry["selected_edges"])
        result = {"status": "ok", "sampling": SAMPLING, "faces": [], "curves": [], "curve_owner": []}
        for index, surface in enumerate(surfaces):
            result["faces"].append({"face_index": index, **surface_distances(faces[index, ::4, ::4].reshape(-1, 3), surface)})
        for index, (curve, first, last) in enumerate(curves):
            result["curves"].append({"edge_index": index, **curve_distances(edges[index], curve, first, last)})
        for edge, face1, face2 in self.geometry["selected_connectivity"]:
            row = {"edge_index": int(edge), "face_pair": [int(face1), int(face2)]}
            for name, series in (("independent_curve", curves), ("after_periodic_replacement", effective)):
                curve, first, last = series[edge]
                points = [curve.Value(float(u)).Coord() for u in np.linspace(first, last, 9)]
                row[name] = [{"face_index": int(face), **surface_distances(points, surfaces[face])} for face in (face1, face2)]
            result["curve_owner"].append(row)
        measured = result["faces"] + result["curves"] + [side for row in result["curve_owner"]
            for name in ("independent_curve", "after_periodic_replacement") for side in row[name]]
        result["status"] = "ok" if all(item["status"] == "ok" for item in measured) else "partial"
        return result


def mechanism_flags(trace):
    """Three-state stage observations; absent/partial instrumentation is unknown."""
    complete = trace.get("status") == "ok" and not trace.get("instrumentation_errors")
    runs = trace.get("trim_runs", [])
    attempts = [a for run in runs for a in run.get("attempts", [])]
    details = [a["trim_detail"] for a in attempts if "trim_detail" in a]
    def observed(value):
        return True if value else False if complete else None
    wires = [w for d in details for w in d.get("wire_events", [])]
    return {
        "wire_not_returned": observed(any(a.get("wire_returned") is False for a in attempts)),
        "wire_skipped_open_after_fix": observed(any(w.get("skip_reason") for w in wires)),
        "all_wires_rejected": observed(any(d.get("outcome") == "no_admitted_wires" for d in details)),
        "partial_wire_drop_with_face_returned": observed(any(d.get("outcome") == "success" and
            any(w.get("skip_reason") for w in d.get("wire_events", [])) for d in details)),
        "closed_wire_admitted_but_trim_failed": observed(any(d.get("outcome") not in ("success", "unknown") and
            any(w.get("added") and w.get("repaired_closed") for w in d.get("wire_events", [])) for d in details)),
        "face_context_brepcheck_error": observed(any(d.get("final_brepcheck", {}).get("with_errors") for d in details)),
        "face_fix_exception": observed(any(d.get("outcome") == "face_fix_exception" for d in details)),
        "periodic_wire_substitution": observed(any(d.get("periodic_wire_substitution") for d in details)),
        "sewing_compound_early_return": observed(any(s.get("detail", {}).get("reject_reason") == "sewn_compound_early_return"
            for s in trace.get("solid_attempts", []))),
        "sewing_free_edges": observed(any(s.get("sewing", {}).get("free_edges", 0) > 0 for s in trace.get("solid_attempts", []))),
        "sewing_multiple_edges": observed(any(s.get("sewing", {}).get("multiple_edges", 0) > 0 for s in trace.get("solid_attempts", []))),
        "no_free_edge_but_solid_failed": observed(any(s.get("sewing", {}).get("free_edges") == 0 and
            s.get("result", {}).get("present") is False for s in trace.get("solid_attempts", []))),
        "trim_gate_blocked": trace.get("trim_gate", {}).get("solid_attempt_allowed") is False if "trim_gate" in trace else None,
        "saved_step_rejected": observed(any(s.get("legacy_saved_step_check") is False for s in trace.get("save_attempts", []))),
        "saved_check_passed_native_failed": observed(any(s.get("legacy_saved_step_check") is True for s in trace.get("save_attempts", [])) and trace.get("replay", {}).get("native_valid_solid") is False),
        "uv_specific_failure": None,
    }


def execute_provenance(job, status_path):
    import numpy as np
    from src.brepnet.post.construct_brep import construct_brep_for_sample
    started = time.monotonic()
    sample_id = job["sample_id"]
    trace_path = Path(job["trace_path"])
    trace = {"sample_id": sample_id, "version": VERSION, "protocol_fingerprint": job["protocol_fingerprint"],
             "raw_sha256": job["raw_sha256"], "seed": stable_seed(sample_id, "post"),
             "status": "running", "stage": "startup", "fallback_used": False,
             "original": job["original"], "post_mode": job["mode"], "instrumentation_errors": []}
    def flush():
        trace["elapsed_seconds"] = time.monotonic() - started
        atomic_json(trace_path, json_value(trace))
        atomic_json(status_path, {key: trace[key] for key in
            ("sample_id", "protocol_fingerprint", "status", "stage", "elapsed_seconds")})
    observer = ProvenanceTrace(trace, flush)
    flush()
    try:
        trace["raw"] = validate_raw_prediction(Path(job["raw_root"]) / sample_id / "data.npz")
        random.seed(trace["seed"])
        np.random.seed(trace["seed"])
        with observer:
            construct_brep_for_sample(job["raw_root"], job["post_root"], sample_id,
                v_drop_num=0, is_ray=False, is_log=False, is_optimize_geom=job["mode"] == "optimized",
                v_max_optimize_iter=200, isdebug=False, use_cuda=False, from_scratch=True, is_save_data=False)
        output = Path(job["post_root"]) / sample_id
        np.savez_compressed(output / "trace_geometry.npz", **observer.geometry)
        step = output / "recon_brep.step"
        replay = {"step_exists": step.is_file(), "success_marker": (output / "success.txt").is_file()}
        if step.is_file():
            replay.update(native_validity(read_step_at_precision(step, mode=0)), step_sha256=sha256_file(step))
            trace["step_data_comparison"] = {"original": step_data_hash(job["original_step"]), "replay": step_data_hash(step)}
            trace["step_data_comparison"]["equal"] = trace["step_data_comparison"]["original"] == trace["step_data_comparison"]["replay"]
        trace.update(replay=replay, comparison=comparison(job["original"], replay), stage="fit_residuals")
        flush()
        trace["fit_residuals"] = observer.safe(observer.fitted_residuals)
        trace.update(status="ok", stage="done")
    except Exception as exc:
        traceback.print_exc()
        trace.update(status="post_error", error=f"{type(exc).__name__}: {exc}")
    finally:
        trace["mechanisms"] = mechanism_flags(trace)
        trace["unresolved"] = ["Closed wires and small 3D projection residuals do not establish correct UV trimming or identify a unique cause.",
            "Face/edge indices before sewing are decoder indices; STEP subshape IDs have a separate scope.",
            "Loss trajectory values precede updates; displacement alone does not mean optimization worsened geometry."]
        trace["observations"] = observations(trace)
        flush()


def provenance_process(job, status_path, log_path):
    os.setsid()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as handle:
        os.dup2(handle.fileno(), 1)
        os.dup2(handle.fileno(), 2)
        execute_provenance(job, status_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--post-mode", choices=("optimized", "fit_only"), default="optimized")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--only-ids", nargs="+")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 8 or not 0 < args.timeout <= 180:
        parser.error("workers must be 1..8; timeout must be >0 and <=180 seconds")
    root, manifest_path = args.output_root.resolve(), args.manifest.resolve()
    cases = read_json(manifest_path)["cases"]
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate IDs")
    if args.only_ids:
        if set(args.only_ids) - set(ids):
            raise ValueError("Unknown --only-ids")
        cases = [case for case in cases if case["id"] in set(args.only_ids)]
    protocol = {"version": VERSION, "manifest_sha256": sha256_file(manifest_path),
        "source_sha256": {name: sha256_file(Path(name)) for name in (*SOURCE_FILES, "tools/legacy_failure_provenance.py", "tools/legacy_failure_geometry.py")},
        "post_mode": args.post_mode, "workers": args.workers, "timeout_seconds": args.timeout, "cpu_threads": 1,
        "seed": "stable_seed(id, 'post')", "optimization_initial_budget": 200,
        "optimization_adaptive_limit": 600, "drop_num": 0, "sampling": SAMPLING, "output_root": str(root)}
    key = fingerprint(protocol)
    metadata = root / "metadata" / "provenance"
    previous = read_json(metadata / "protocol.json")
    if previous and (previous != protocol or not args.resume):
        raise ValueError("Existing output requires --resume and identical protocol")
    atomic_json(metadata / "protocol.json", protocol)
    jobs = []
    for case in cases:
        sid, raw = case["id"], Path(case["paths"]["raw_legacy_npz"])
        trace_path = root / "traces" / f"{sid}.json"
        if raw.parent.name != sid or (root / "replay" / sid).resolve() == Path(case["paths"]["post_legacy_step"]).parent.resolve():
            raise ValueError("Unsafe or inconsistent input/output paths")
        raw_hash = sha256_file(raw) if raw.is_file() else None
        old = read_json(trace_path)
        if old:
            if old.get("protocol_fingerprint") != key or old.get("raw_sha256") != raw_hash:
                raise ValueError(f"Changed protocol/input: {sid}")
            if args.resume and old.get("status") in ("ok", "post_error", "timeout", "crash"):
                continue
            raise ValueError(f"Refusing overwrite: {sid}")
        jobs.append({"sample_id": sid, "method": "legacy", "mode": args.post_mode, "protocol_fingerprint": key,
            "raw_sha256": raw_hash, "raw_root": str(raw.parents[1]), "post_root": str(root / "replay"),
            "trace_path": str(trace_path), "original": case["post_legacy"] if args.post_mode == "optimized" else case["fit_legacy"],
            "original_step": case["paths"]["post_legacy_step" if args.post_mode == "optimized" else "fit_legacy_step"]})
    configure_cpu()
    preload_cpu_dependencies()
    run_isolated_jobs(jobs, metadata, args.workers, args.timeout, process_target=provenance_process)
    traces = []
    for case in cases:
        path = root / "traces" / f"{case['id']}.json"
        trace = read_json(path)
        status = read_json(metadata / "samples" / "legacy" / f"{case['id']}.json")
        if status.get("status") in ("timeout", "crash"):
            trace.update(sample_id=case["id"], status=status["status"], error=status.get("error"))
            trace["mechanisms"] = mechanism_flags(trace)
            atomic_json(path, json_value(trace))
        traces.append(trace)
    summary = {"version": VERSION, "protocol_fingerprint": key, "intended_count": len(cases),
        "status_counts": dict(Counter(t.get("status", "missing") for t in traces)),
        "instrumentation_error_cases": [t["sample_id"] for t in traces if t.get("instrumentation_errors")],
        "step_data_equal_count": sum(t.get("step_data_comparison", {}).get("equal") is True for t in traces),
        "step_data_different_ids": [t["sample_id"] for t in traces if t.get("step_data_comparison", {}).get("equal") is False],
        "mechanisms": {name: {label: sum(t.get("mechanisms", {}).get(name) is value for t in traces)
            for label, value in (("true", True), ("false", False), ("unknown", None))} for name in mechanism_flags({})},
        "elapsed_seconds": distribution([t.get("elapsed_seconds", float("nan")) for t in traces]),
        "limitations": ["Overlapping stage observations, not unique root causes.", "Unknown measurements remain in the intended denominator."]}
    atomic_json(root / "provenance_summary.json", summary)
    print(__import__("json").dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
