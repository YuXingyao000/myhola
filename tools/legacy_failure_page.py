"""Build an offline, synchronized CAD failure browser from frozen diagnostics.

The page is a view of recorded results, never a new validity evaluation. Large
geometry is loaded per case as local scripts, so opening index.html directly
does not depend on an HTTP server or cross-origin fetch permissions.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request


THREE_URL = "https://cdn.jsdelivr.net/npm/three@0.160.1/build/three.min.js"
THREE_SHA256 = "170c6789f43217c96b3170f4b42fafe135de7f7cd48497a4218f9757ee1d49fa"


def read_json(path: Path):
    return json.loads(path.read_text())


def compact_trace(trace: dict) -> dict:
    """Keep observable stages in the page; complete per-attempt logs stay linked."""
    keys = ("sample_id", "status", "stage", "observations", "selection", "optimization",
            "fitting", "fallback_used", "comparison", "replay", "instrumentation_errors", "error")
    result = {key: trace[key] for key in keys if key in trace}
    result["trim_runs"] = [
        {key: value for key, value in run.items() if key != "connectivity"}
        for run in trace.get("trim_runs", [])
    ]
    result["solid_attempts"] = trace.get("solid_attempts", [])
    result["fallback_sewing"] = trace.get("fallback_sewing", [])
    return result


def write_compact_reports(root: Path, cases: list, manifest_sha256: str) -> None:
    """Versionable per-case evidence and exact replay DATA comparison."""
    summary, audit = [], []
    raw_keys = ("gt_face_count", "face_mae", "face_cd_sq", "edge_cd_sq", "topology_fp", "topology_fn",
                "legacy_reciprocal_cd_sq_mean", "legacy_reciprocal_cd_sq_max")
    for case in cases:
        item = case["id"]
        diagnostic, trace = case["diagnostics"], case["trace"]
        post = diagnostic["panels"]["post"]
        summary.append({
            "id": item, "stratum": case["stratum"], "raw": {key: case["raw_legacy"][key] for key in raw_keys},
            "post": {key: post[key] for key in ("shape_type_name", "native_brepcheck_valid", "native_valid_solid", "counts", "shell_orientation")},
            "post_incidence": {key: value for key, value in post["incidence"].items() if key != "edges"},
            "post_brepcheck_subshapes": post["brepcheck_subshapes"],
            "main_trim": [{key: run[key] for key in ("successful_faces", "failed_face_indices", "num_faces")}
                          for run in trace["trim_runs"] if run["context"] == "main"],
            "solid_attempts": [{"tolerance": attempt["tolerance"], "returned_solid": attempt.get("result", {}).get("present", False),
                                "sewing": attempt.get("sewing")} for attempt in trace["solid_attempts"]],
            "fallback_used": trace["fallback_used"], "replay_matches_original": trace["comparison"]["same_recorded_outcome"],
            "extent": {key: value for key, value in (case.get("extent") or {}).items() if key != "sources"},
            "detail_sources": {name + "_sha256": hashlib.sha256((root / directory / f"{item}.json").read_bytes()).hexdigest()
                               for name, directory in (("diagnostics", "diagnostics"), ("trace", "traces"))},
        })
        files = [Path(case["paths"]["post_legacy_step"]), root / "replay" / item / "recon_brep.step"]
        if not all(path.is_file() for path in files):
            audit.append({"id": item, "status": "missing_step", "data_section_bytes_equal": None})
            continue
        contents = [path.read_bytes() for path in files]
        sections = []
        for content in contents:
            start = content.find(b"DATA;")
            end = content.find(b"ENDSEC;", start + 5)
            if start < 0 or end < 0:
                raise ValueError(f"STEP lacks a DATA section: {item}")
            sections.append(content[start + 5:end])
        audit.append({"id": item, "complete_bytes_equal": contents[0] == contents[1],
                      "data_section_bytes_equal": sections[0] == sections[1],
                      "original_data_sha256": hashlib.sha256(sections[0]).hexdigest(),
                      "replay_data_sha256": hashlib.sha256(sections[1]).hexdigest()})
    (root / "case_summary.json").write_text(json.dumps({"version": "legacy_failure_case_summary_v1",
        "manifest_sha256": manifest_sha256, "cases": summary}, ensure_ascii=False, indent=2) + "\n")
    (root / "trace_step_data_audit.json").write_text(json.dumps({
        "description": "Exact STEP DATA section byte comparison, excluding STEP HEADER and ENDSEC. This is stricter than validity/count comparison but not a semantic STEP equivalence test.",
        "count": len(audit), "equal_data_sections": sum(row.get("data_section_bytes_equal") is True for row in audit),
        "equal_complete_files": sum(row.get("complete_bytes_equal") is True for row in audit), "cases": audit}, indent=2) + "\n")


def build(root: Path, allow_partial: bool = False) -> dict:
    manifest_path = root / "selected_cases.json"
    manifest = read_json(manifest_path)
    cases = manifest["cases"]
    ids = [case["id"] for case in cases]
    if len(ids) != 50 or len(set(ids)) != 50:
        raise ValueError("The browser requires the frozen 50 unique cases")
    if any(case["post_legacy"]["native_valid_solid"] is not False for case in cases):
        raise ValueError("Selected case is not an original native-solid failure")
    page = root / "page"
    extent_path = root / "diagnostics_extent.json"
    extents = read_json(extent_path) if extent_path.is_file() else {}
    if extents and extents.get("manifest_sha256") != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
        raise ValueError("Extent analysis uses a different manifest")
    geometry_dir = page / "geometry"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    missing, counts, source_hashes = [], {}, {}
    for case in cases:
        item = case["id"]
        geometry_path = root / "visualizations" / f"{item}.json"
        diagnostic_path = root / "diagnostics" / f"{item}.json"
        trace_path = root / "traces" / f"{item}.json"
        image_path = root / "visualizations" / f"{item}.png"
        for source in (geometry_path, diagnostic_path, trace_path, image_path):
            if not source.is_file():
                missing.append(str(source.relative_to(root)))
        if geometry_path.is_file():
            geometry = read_json(geometry_path)
            if geometry.get("id") != item or set(geometry.get("panels", {})) != {"gt", "raw", "post"}:
                raise ValueError(f"Invalid geometry identity/panels: {geometry_path}")
            payload = json.dumps(geometry, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            (geometry_dir / f"{item}.js").write_text(f"window.LEGACY_GEOMETRY_CALLBACK({payload});\n")
            case["geometry_counts"] = {key: panel.get("counts", {}) for key, panel in geometry["panels"].items()}
            counts[item] = case["geometry_counts"]
            source_hashes[item] = hashlib.sha256(geometry_path.read_bytes()).hexdigest()
        case["diagnostics"] = read_json(diagnostic_path) if diagnostic_path.is_file() else None
        if case["diagnostics"] is not None and case["diagnostics"].get("id") != item:
            raise ValueError(f"Diagnostic identity mismatch: {diagnostic_path}")
        trace = read_json(trace_path) if trace_path.is_file() else None
        if trace is not None and trace.get("sample_id") != item:
            raise ValueError(f"Trace identity mismatch: {trace_path}")
        case["trace"] = compact_trace(trace) if trace is not None else None
        case["extent"] = extents.get("cases", {}).get(item)
    if missing and not allow_partial:
        raise ValueError(f"Missing {len(missing)} case assets; first: {missing[:5]}")
    if not missing:
        write_compact_reports(root, cases, hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    summary = read_json(root / "selection_summary.json")
    population = {
        "intended": summary["population"]["accepted_test"],
        "failures": summary["population"]["native_failed"],
        "successes": summary["population"]["native_valid"],
        # Independently validated by the frozen selection audit, not 50-case prevalence.
        "topology_exact_failures": sum(stratum["population_count"] for stratum in summary["strata"]
                                       if stratum["raw_pair_topology_exact"]),
    }
    payload = {"schema": "legacy_failure_page_v1", "population": population, "cases": cases,
               "selection_protocol": manifest["selection_protocol"]}
    (page / "data.js").write_text("window.LEGACY_FAILURE_DATA=" +
                                  json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + ";\n")
    vendor = page / "vendor" / "three-0.160.1.min.js"
    vendor.parent.mkdir(parents=True, exist_ok=True)
    if not vendor.is_file():
        with urllib.request.urlopen(THREE_URL, timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != THREE_SHA256:
            raise ValueError("Downloaded Three.js differs from pinned checksum")
        vendor.write_bytes(content)
    if hashlib.sha256(vendor.read_bytes()).hexdigest() != THREE_SHA256:
        raise ValueError("Local Three.js differs from pinned checksum")
    report = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(), "case_count": len(cases),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "geometry_source_sha256": source_hashes, "missing_assets": missing,
        "three_url": THREE_URL, "three_sha256": THREE_SHA256,
        "entry_point": str((page / "index.html").resolve()),
        "offline": True, "geometry_format": "triangulated CAD display, not a replacement for native STEP checks",
    }
    (root / "page_build.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true", help="Development preview only; missing assets remain explicit")
    args = parser.parse_args()
    result = build(args.root.resolve(), args.allow_partial)
    print(json.dumps({key: result[key] for key in ("case_count", "missing_assets", "entry_point")}))


if __name__ == "__main__":
    main()
