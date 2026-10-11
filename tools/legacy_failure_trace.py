"""Observe the unchanged legacy OCC pipeline in isolated, bounded CPU jobs.

The wrappers call each original function once with unchanged arguments. They do
not add repairs, consume random numbers, or replace the original comparison.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import inspect
import json
import os
from pathlib import Path
import random
import time
import traceback
from unittest.mock import patch

from tools.vae_compare_post import (
    atomic_json, configure_cpu, fingerprint, native_validity,
    preload_cpu_dependencies, read_json, read_step_at_precision,
    run_isolated_jobs, sha256_file, stable_seed, validate_raw_prediction,
)


VERSION = "legacy_failure_trace_v1"
SOURCE_FILES = (
    "src/brepnet/post/construct_brep.py", "src/brepnet/post/shape.py",
    "src/brepnet/post/geometry_optimization.py", "src/brepnet/post/occ_fit.py",
    "src/brepnet/post/occ_topology.py", "src/brepnet/post/constants.py",
    "src/brepnet/post/distance.py", "src/brepnet/eval/metrics/validity.py",
    "tools/vae_compare_post.py", "tools/legacy_failure_trace.py",
)


def json_value(value):
    """Strict JSON conversion; nonfinite diagnostics remain visible as null."""
    import numpy as np
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def occ_shape_summary(shape):
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_SHELL, TopAbs_SOLID
    from shared.occ_utils import get_primitives
    if shape is None or shape.IsNull():
        return {"present": False}
    return {
        "present": True, "shape_type": int(shape.ShapeType()),
        "brepcheck_valid": bool(BRepCheck_Analyzer(shape).IsValid()),
        **{name: len(get_primitives(shape, kind, v_remove_half=True)) for name, kind in (
            ("num_faces", TopAbs_FACE), ("num_edges", TopAbs_EDGE),
            ("num_shells", TopAbs_SHELL), ("num_solids", TopAbs_SOLID))},
    }


def comparison(original, replay):
    keys = ("native_valid_solid", "shape_type", "native_brepcheck_valid",
            "num_faces", "num_edges", "num_vertices", "success_marker")
    differences = {key: {"original": original.get(key), "replay": replay.get(key)}
                   for key in keys if original.get(key) != replay.get(key)}
    return {"same_recorded_outcome": not differences, "differences": differences,
            "scope": "Validity, shape type and primitive counts; not full geometric identity."}


def observations(trace):
    """Stage observations, deliberately not a root-cause classifier."""
    result = []
    selection = trace.get("selection", {})
    if selection:
        result.append(f"半边选择：{selection['input_edge_count']} 个原始半边候选变为 "
                      f"{selection['output_edge_count']} 条边；完全移除 "
                      f"{len(selection['removed_face_pairs'])} 个无向面邻接对。")
    for run in trace.get("trim_runs", []):
        missing = run.get("failed_face_indices", [])
        result.append(f"{run['context']} 裁剪：{run.get('successful_faces', 0)}/{run['num_faces']} 个面成功；"
                      f"失败面索引 {missing}。")
    solids = trace.get("solid_attempts", [])
    result.append(f"尝试 {len(solids)} 次 solid 构造；返回非空 solid "
                  f"{sum(item.get('result', {}).get('present', False) for item in solids)} 次。")
    if trace.get("fallback_used"):
        result.append("调用 fallback：再次裁剪，失败面以未裁剪的拟合面替代，再缝合并写出结果。")
    if trace.get("comparison", {}).get("differences"):
        result.append("重放的已记录结果与原评估存在差异；原案例分类保留，详见 comparison。")
    result.append("这些记录描述失败出现的阶段；不能单独判定 decoder、离散表示或 OCC 算法的根因。")
    return result


class PipelineTrace:
    def __init__(self, trace, flush):
        self.trace, self.flush = trace, flush
        self.stack = ExitStack()
        self.current_trim = None
        self.sewing_context = None
        self.context = "main"
        self.geometry = {}

    def safe(self, function, *args, **kwargs):
        # Instrumentation errors must not alter the original algorithm's flow.
        try:
            return function(*args, **kwargs)
        except Exception as exc:
            self.trace.setdefault("instrumentation_errors", []).append(
                f"{getattr(function, '__name__', type(function).__name__)}: {type(exc).__name__}: {exc}")
            return {"diagnostic_error": f"{type(exc).__name__}: {exc}"}

    def __enter__(self):
        import numpy as np
        import torch
        from src.brepnet.post import construct_brep as cb
        from src.brepnet.post import occ_topology as topology
        from src.brepnet.post.shape import Shape, _SKIP_FACE_PAIR

        original_choose = Shape._select_edge_for_face_pair
        def choose(shape, edge_ids, face1, face2, threshold):
            result = original_choose(shape, edge_ids, face1, face2, threshold)
            self.trace.setdefault("pair_selection", []).append({
                "face_pair": [int(face1), int(face2)], "candidate_raw_edge_ids": list(map(int, edge_ids)),
                "threshold": float(threshold),
                "selected_raw_edge_id": None if result is None or result is _SKIP_FACE_PAIR else int(result),
                "outcome": "unsupported_multiplicity" if result is _SKIP_FACE_PAIR else
                           "all_candidates_geometry_rejected" if result is None else "selected",
            })
            return result
        self.stack.enter_context(patch.object(Shape, "_select_edge_for_face_pair", choose))

        original_prepare = cb._prepare_shape_topology
        def prepare(shape, *args, **kwargs):
            before = shape.edge_face_connectivity.copy()
            self.trace["stage"] = "prepare_topology"
            self.flush()
            result = original_prepare(shape, *args, **kwargs)
            after = shape.edge_face_connectivity.copy()
            before_pairs = {tuple(sorted(map(int, row[1:]))) for row in before}
            after_pairs = {tuple(sorted(map(int, row[1:]))) for row in after}
            selected = {tuple(item["face_pair"]): item["selected_raw_edge_id"]
                        for item in self.trace.get("pair_selection", []) if item["outcome"] == "selected"}
            self.trace["selection"] = {
                "input_edge_count": len(before), "output_edge_count": len(shape.recon_edge_points),
                "input_pair_count": len(before_pairs), "output_pair_count": len(after_pairs),
                "removed_face_pairs": sorted(before_pairs - after_pairs),
                "connectivity": after.tolist(),
                "selected_raw_edge_ids_by_output_edge": [selected.get(tuple(map(int, row[1:]))) for row in after],
                "face_edge_adjacency": shape.face_edge_adj,
                "corner_constraint_count": 0 if shape.pair1 is None else len(shape.pair1),
            }
            self.geometry.update(selected_faces=shape.recon_face_points.copy(),
                                 selected_edges=shape.recon_edge_points.copy(), selected_connectivity=after)
            self.flush()
            return result
        self.stack.enter_context(patch.object(cb, "_prepare_shape_topology", prepare))

        original_optimize = cb._optimize_shape_geometry
        def optimize(shape, *args, **kwargs):
            self.trace["stage"] = "optimize"
            self.flush()
            started = time.monotonic()
            steps, losses = 0, []
            original_step = torch.optim.AdamW.step
            def step(optimizer, *step_args, **step_kwargs):
                nonlocal steps
                frame = inspect.currentframe().f_back
                if frame.f_code.co_name == "optimize":
                    values = frame.f_locals
                    if steps % 50 == 0 or steps >= int(values.get("v_max_iter", 200)) - 2:
                        losses.append({"iteration": steps, **{
                            name: float(values[name].detach().item()) for name in
                            ("loss", "adj_distance_loss", "corners_loss", "wire_connected_loss")}})
                del frame
                steps += 1
                return original_step(optimizer, *step_args, **step_kwargs)
            with patch.object(torch.optim.AdamW, "step", step):
                result = original_optimize(shape, *args, **kwargs)
            displacement = {}
            for kind, points in (("faces", shape.recon_face_points), ("edges", shape.recon_edge_points)):
                distances = np.linalg.norm(points - self.geometry["selected_" + kind], axis=-1)
                displacement[kind] = {"mean_displacement": float(distances.mean()),
                                      "max_displacement": float(distances.max())}
                self.geometry["optimized_" + kind] = points.copy()
            self.trace["optimization"] = {"adam_steps": steps, "loss_checkpoints": losses,
                                           "displacement": displacement, "seconds": time.monotonic() - started}
            self.flush()
            return result
        self.stack.enter_context(patch.object(cb, "_optimize_shape_geometry", optimize))

        original_fit = cb._fit_occ_geometry
        def fit(shape):
            self.trace["stage"] = "fit"
            self.flush()
            started = time.monotonic()
            result = original_fit(shape)
            self.trace["fitting"] = {
                "seconds": time.monotonic() - started,
                "faces": [{"face_index": index, "u_periodic": bool(surface.IsUPeriodic()),
                           "v_periodic": bool(surface.IsVPeriodic())} for index, surface in enumerate(result[0])],
                "num_curves": len(result[2]),
                "periodic_replaced_edge_indices": list(map(int, getattr(shape, "replace_edge_idx", []))),
                "note": "Independent fit return only; no fitted residual or chosen fitting tolerance is exposed.",
            }
            self.flush()
            return result
        self.stack.enter_context(patch.object(cb, "_fit_occ_geometry", fit))

        original_build = cb._build_trimmed_faces
        def build(faces, curves, connectivity):
            run = {"index": len(self.trace.setdefault("trim_runs", [])), "context": self.context,
                   "num_faces": len(faces), "connectivity": json_value(connectivity), "attempts": []}
            self.trace["trim_runs"].append(run)
            self.current_trim = run
            self.trace["stage"] = "trim_" + self.context
            self.flush()
            try:
                result = original_build(faces, curves, connectivity)
                run.update(successful_faces=sum(face is not None for face in result),
                           failed_face_indices=[i for i, face in enumerate(result) if face is None])
                return result
            finally:
                self.current_trim = None
                self.flush()
        self.stack.enter_context(patch.object(cb, "_build_trimmed_faces", build))

        original_wire = cb.create_wire_from_unordered_edges
        def wire(edges, tolerance, *args, **kwargs):
            frame = inspect.currentframe().f_back
            face_index = frame.f_locals.get("i_face") if frame.f_code.co_name == "_build_trimmed_faces" else None
            edge_indices = frame.f_locals.get("face_edge_idx", [])
            del frame
            entry = {"face_index": face_index, "edge_indices": list(map(int, edge_indices)),
                     "tolerance": float(tolerance)}
            result = original_wire(edges, tolerance, *args, **kwargs)
            entry.update(wire_returned=result is not None, wire_count=0 if result is None else len(result),
                         wires=[] if result is None else [self.safe(lambda item: {
                             "closed_flag": bool(item.Closed()), "brepcheck_valid": occ_shape_summary(item)["brepcheck_valid"]},
                             item) for item in result])
            if self.current_trim is not None:
                self.current_trim["attempts"].append(entry)
            return result
        self.stack.enter_context(patch.object(cb, "create_wire_from_unordered_edges", wire))

        original_trim = cb.create_trimmed_face_from_wire
        def trim(face, edges, wires, tolerance):
            result = original_trim(face, edges, wires, tolerance)
            if self.current_trim is not None and self.current_trim["attempts"]:
                self.current_trim["attempts"][-1].update(
                    trim_returned=result is not None, trimmed_shape=self.safe(occ_shape_summary, result))
            return result
        self.stack.enter_context(patch.object(cb, "create_trimmed_face_from_wire", trim))

        original_sewing = topology.BRepBuilderAPI_Sewing
        observer = self
        class ObservedSewing:
            def __init__(self, *args, **kwargs):
                self.actual = original_sewing(*args, **kwargs)
            def __getattr__(self, name):
                return getattr(self.actual, name)
            def Perform(self, *args, **kwargs):
                result = self.actual.Perform(*args, **kwargs)
                if observer.sewing_context is not None:
                    observer.sewing_context["sewn_before_shape_fix"] = observer.safe(occ_shape_summary, self.actual.SewedShape())
                    observer.sewing_context["sewing"] = {
                        name: observer.safe(lambda method=method: int(getattr(self.actual, method)()))
                        for name, method in (("free_edges", "NbFreeEdges"),
                                             ("multiple_edges", "NbMultipleEdges"),
                                             ("contiguous_edges", "NbContigousEdges"))}
                return result
        self.stack.enter_context(patch.object(topology, "BRepBuilderAPI_Sewing", ObservedSewing))

        def wrap_sewing(original, key):
            def wrapped(faces, tolerance):
                entry = {"tolerance": float(tolerance), "input_face_count": len(faces)}
                self.trace.setdefault(key, []).append(entry)
                self.sewing_context = entry
                self.trace["stage"] = key
                try:
                    result = original(faces, tolerance)
                    entry["result"] = self.safe(occ_shape_summary, result)
                    return result
                finally:
                    self.sewing_context = None
                    self.flush()
            return wrapped
        self.stack.enter_context(patch.object(cb, "get_solid", wrap_sewing(cb.get_solid, "solid_attempts")))
        self.stack.enter_context(patch.object(cb, "get_compound", wrap_sewing(cb.get_compound, "fallback_sewing")))

        original_fallback = cb._write_compound_fallback
        def fallback(*args, **kwargs):
            self.trace["fallback_used"] = True
            self.context = "fallback"
            try:
                return original_fallback(*args, **kwargs)
            finally:
                self.context = "main"
        self.stack.enter_context(patch.object(cb, "_write_compound_fallback", fallback))
        return self

    def __exit__(self, *exception):
        return self.stack.__exit__(*exception)


def execute_trace(job, status_path):
    import numpy as np
    from src.brepnet.post.construct_brep import construct_brep_for_sample
    started = time.monotonic()
    sample_id = job["sample_id"]
    trace_path = Path(job["trace_path"])
    trace = {"sample_id": sample_id, "protocol_fingerprint": job["protocol_fingerprint"],
             "raw_sha256": job["raw_sha256"], "seed": stable_seed(sample_id, "post"),
             "status": "running", "stage": "startup", "fallback_used": False,
             "original": job["original"], "instrumentation_errors": []}
    def flush():
        trace["elapsed_seconds"] = time.monotonic() - started
        atomic_json(trace_path, json_value(trace))
        atomic_json(status_path, {key: trace[key] for key in
                                  ("sample_id", "protocol_fingerprint", "status", "stage", "elapsed_seconds")})
    flush()
    observer = PipelineTrace(trace, flush)
    try:
        trace["raw"] = validate_raw_prediction(Path(job["raw_root"]) / sample_id / "data.npz")
        random.seed(trace["seed"])
        np.random.seed(trace["seed"])
        with observer:
            construct_brep_for_sample(job["raw_root"], job["post_root"], sample_id,
                                      v_drop_num=0, is_ray=False, is_log=False,
                                      is_optimize_geom=True, v_max_optimize_iter=200,
                                      isdebug=False, use_cuda=False, from_scratch=True, is_save_data=False)
        output = Path(job["post_root"]) / sample_id
        np.savez_compressed(output / "trace_geometry.npz", **observer.geometry)
        step = output / "recon_brep.step"
        replay = {"step_exists": step.is_file(), "success_marker": (output / "success.txt").is_file()}
        if step.is_file():
            replay.update(native_validity(read_step_at_precision(step, mode=0)), step_sha256=sha256_file(step))
        trace.update(replay=replay, comparison=comparison(job["original"], replay), status="ok", stage="done")
    except Exception as exc:
        traceback.print_exc()
        trace.update(status="post_error", error=f"{type(exc).__name__}: {exc}")
    finally:
        trace["observations"] = observations(trace)
        flush()


def trace_process(job, status_path, log_path):
    os.setsid()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as handle:
        os.dup2(handle.fileno(), 1)
        os.dup2(handle.fileno(), 2)
        execute_trace(job, status_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--only-ids", nargs="+")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 8 or not 0 < args.timeout <= 180:
        parser.error("workers must be 1..8; timeout must be >0 and <=180 seconds")
    root = args.output_root.resolve()
    manifest_path = args.manifest.resolve()
    manifest = read_json(manifest_path)
    cases = manifest["cases"]
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate IDs in selection manifest")
    if args.only_ids:
        if set(args.only_ids) - set(ids):
            raise ValueError("--only-ids contains IDs outside selection")
        cases = [case for case in cases if case["id"] in set(args.only_ids)]
    protocol = {"version": VERSION, "manifest_sha256": sha256_file(manifest_path),
                "source_sha256": {name: sha256_file(Path(name)) for name in SOURCE_FILES},
                "seed": "stable_seed(id, 'post')", "workers": args.workers, "timeout_seconds": args.timeout,
                "cpu_threads": 1, "optimization_initial_budget": 200, "optimization_adaptive_limit": 600,
                "drop_num": 0, "original_outputs": "read only", "output_root": str(root)}
    protocol_key = fingerprint(protocol)
    metadata = root / "metadata" / "trace"
    protocol_path = metadata / "protocol.json"
    previous = read_json(protocol_path)
    if previous and (previous != protocol or not args.resume):
        raise ValueError("Existing trace root: --resume and identical protocol are required")
    atomic_json(protocol_path, protocol)
    jobs = []
    for case in cases:
        sample_id = case["id"]
        raw = Path(case["paths"]["raw_legacy_npz"])
        output = root / "replay" / sample_id
        if raw.parent.name != sample_id or root == raw.parents[3] or output == Path(case["paths"]["post_legacy_step"]).parent:
            raise ValueError("Inconsistent or unsafe raw/replay paths")
        trace_path = root / "traces" / f"{sample_id}.json"
        old = read_json(trace_path)
        raw_hash = sha256_file(raw)
        if old:
            if old.get("protocol_fingerprint") != protocol_key or old.get("raw_sha256") != raw_hash:
                raise ValueError(f"Existing trace has a different protocol/input: {sample_id}")
            if args.resume and old.get("status") in ("ok", "post_error", "timeout", "crash"):
                continue
            raise ValueError(f"Refusing to overwrite existing trace: {trace_path}")
        jobs.append({"sample_id": sample_id, "method": "legacy", "mode": "optimized",
                     "protocol_fingerprint": protocol_key, "raw_sha256": raw_hash,
                     "raw_root": str(raw.parents[1]), "post_root": str(root / "replay"),
                     "trace_path": str(trace_path), "original": case["post_legacy"]})
    configure_cpu()
    preload_cpu_dependencies()
    run_isolated_jobs(jobs, metadata, args.workers, args.timeout, process_target=trace_process)
    traces = []
    for case in cases:
        trace_path = root / "traces" / f"{case['id']}.json"
        trace = read_json(trace_path)
        status = read_json(metadata / "samples" / "legacy" / f"{case['id']}.json")
        if status.get("status") in ("timeout", "crash"):
            trace.update(status=status["status"], error=status.get("error"), stage=trace.get("stage", "startup"))
            trace["observations"] = observations(trace)
            atomic_json(trace_path, trace)
        traces.append(trace)
    summary = {"protocol_fingerprint": protocol_key, "selected_count": len(cases),
               "status_counts": dict(Counter(trace.get("status", "missing") for trace in traces)),
               "instrumentation_error_cases": [trace["sample_id"] for trace in traces if trace.get("instrumentation_errors")],
               "changed_outcome_ids": [trace["sample_id"] for trace in traces if trace.get("comparison", {}).get("differences")],
               "main_trim_failure_cases": [trace["sample_id"] for trace in traces if any(
                   run["context"] == "main" and run.get("failed_face_indices") for run in trace.get("trim_runs", []))],
               "fallback_cases": [trace["sample_id"] for trace in traces if trace.get("fallback_used")],
               "limitations": ["Selected failure cases are stratified; their frequencies are not population estimates.",
                               "A stage observation does not establish an architectural root cause.",
                               "OCC wrappers observe outputs; exceptions swallowed inside original helpers remain unclassified."]}
    atomic_json(root / "trace_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
