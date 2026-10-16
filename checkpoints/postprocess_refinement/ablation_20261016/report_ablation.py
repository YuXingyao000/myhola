"""Audit and report the frozen model x post-policy ablation; never run OCC.

Run from any directory with this file's absolute path. Missing/nonterminal rows
are errors, not implicit failures. Explicit terminal failures stay in all fixed
denominators. ``--self-check`` exercises rejection and denominator contracts on
synthetic inputs without reading either research split.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import json
import math
from pathlib import Path
import sys
import tempfile
import time

REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY))

from tools.vae_compare_post import (
    REFERENCE_CONTRACT_FIELDS, SAMPLING, TERMINAL, atomic_json, fingerprint,
    sha256_file, stable_seed, statistics,
)

ROOT = Path(__file__).resolve().parent
MODELS = ("legacy", "conditioned")
POLICIES = ("legacy", "recover_v1")
CELLS = tuple(f"{model}/{policy}" for model in MODELS for policy in POLICIES)
SPLITS = {"validation": ("val", 3495, 3505), "test": ("test", 2417, 2424)}
CHECKPOINTS = {
    "legacy": "8b714943d24c3f3f18f3a1e64222b1f3577323ddbf42e4b90d603186ab7e917a",
    "conditioned": "4c7910b8169206c74fcf38c358cf171359149e0429c8debbc227d391aa56613f",
}
COMPARISONS = {
    "model_at_legacy": ("legacy/legacy", "conditioned/legacy"),
    "model_at_recover_v1": ("legacy/recover_v1", "conditioned/recover_v1"),
    "post_at_legacy_model": ("legacy/legacy", "legacy/recover_v1"),
    "post_at_conditioned_model": ("conditioned/legacy", "conditioned/recover_v1"),
    "combined": ("legacy/legacy", "conditioned/recover_v1"),
}
GEOMETRY = ("face_cd", "edge_cd")
COSTS = ("wall_seconds", "post_elapsed_seconds", "geometry_elapsed_seconds")
TOLERANCES = tuple(f"native_{kind}_tolerance_max" for kind in ("face", "edge", "vertex"))
NATIVE = "read.precision.mode=0; as-read TopAbs_SOLID and BRepCheck before mutation"
COMPAT = "fresh read mode=1, precision=.1; SetTolerance(.1); TopAbs_SOLID and BRepCheck"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict:
    # Unlike the runner's read_json, an absent artifact must never become {}.
    with path.open() as stream:
        value = json.load(stream)
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def exact_map(rows: list[dict], ids: list[str], key: str, context: str) -> dict:
    require(len(ids) > 0 and len(ids) == len(set(ids)), "Invalid intended IDs")
    observed = [row.get(key) for row in rows]
    require(len(observed) == len(set(observed)), f"Duplicate IDs: {context}")
    require(set(observed) == set(ids), f"Missing/unexpected IDs: {context}")
    return dict(zip(observed, rows))


def finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_rows(rows: list[dict], ids: list[str], protocol: dict,
                  model: str, policy: str) -> dict:
    result = exact_map(rows, ids, "sample_id", f"{model}/{policy}")
    digest = fingerprint(protocol)
    for sample_id, row in result.items():
        context = f"{model}/{policy}/{sample_id}"
        require(row.get("status") in TERMINAL, f"Nonterminal record: {context}")
        require(row.get("method") == model and row.get("mode") == "optimized",
                f"Model/mode mismatch: {context}")
        require(row.get("post_policy", "legacy") == policy, f"Policy mismatch: {context}")
        require(row.get("protocol_fingerprint") == digest, f"Protocol mismatch: {context}")
        require(row.get("seed") == stable_seed(sample_id, "post"), f"Seed mismatch: {context}")
        for name in ("native_valid_solid", "compat_valid_solid", "step_exists", "step_readable"):
            require(isinstance(row.get(name, False), bool), f"Nonboolean {name}: {context}")
        if row.get("native_valid_solid"):
            require(row.get("step_readable") and row.get("step_exists") and
                    row.get("native_is_solid_type") and row.get("native_brepcheck_valid"),
                    f"Native verdict lacks supporting checks: {context}")
        if row.get("geometry_status") == "ok":
            require(all(finite(row.get(name)) and row[name] >= 0 for name in GEOMETRY),
                    f"Successful geometry has invalid metrics: {context}")
        for name in COSTS + TOLERANCES:
            require(row.get(name) is None or (finite(row[name]) and row[name] >= 0),
                    f"Invalid numeric field {name}: {context}")
    return result


def read_rows(directory: Path, ids: list[str], protocol: dict,
              model: str, policy: str) -> dict:
    require(directory.is_dir(), f"Missing record directory: {directory}")
    rows = []
    for path in sorted(directory.glob("*.json")):
        row = load_json(path)
        require(path.stem == row.get("sample_id"), f"Filename/ID mismatch: {path}")
        rows.append(row)
    return validate_rows(rows, ids, protocol, model, policy)


def validate_protocol(protocol: dict, split_hash: str, n: int, original: int,
                      policy: str, *, historical: bool = False) -> None:
    expected = {
        "version": "vae_compare_post_v1", "source": "posterior_mean", "mode": "optimized",
        "split_sha256": split_hash, "intended_count": n, "original_test_count": original,
        "timeout_seconds": 180, "max_optimize_iter": 200, "adaptive_max_optimize_iter": 600,
        "drop_num": 0, "use_cuda": False, "threads_per_process": 1, "sampling": SAMPLING,
        "native_validity": NATIVE, "compatibility_validity": COMPAT,
        "timeout_scope": "whole subprocess from launch, including native/compat validity and geometry metrics",
        "baseline_method": "legacy", "candidate_method": "intersection" if historical else "conditioned",
    }
    for name, value in expected.items():
        require(protocol.get(name) == value, f"Post protocol mismatch: {policy}/{name}")
    require(protocol.get("post_policy", "legacy") == policy, f"Post protocol policy: {policy}")
    require(bool(protocol.get("post_algorithm_files")), "Missing post source hashes")


def same_contract(left: dict, right: dict, fields=REFERENCE_CONTRACT_FIELDS) -> None:
    for name in fields:
        require(left.get(name) == right.get(name), f"Different comparison contract: {name}")


def validate_raw_pair_hashes(maps: dict, ids: list[str]) -> None:
    for model in MODELS:
        for sample_id in ids:
            values = [maps[f"{model}/{policy}"][sample_id].get("raw_sha256") for policy in POLICIES]
            require(values[0] == values[1], f"Cross-policy raw mismatch: {model}/{sample_id}")


def cell_summary(rows: list[dict], n: int, original: int, *, historical=False) -> dict:
    valid = [row for row in rows if row.get("native_valid_solid")]
    computable = [row for row in rows if row.get("geometry_status") == "ok"]
    counts = {name: sum(bool(row.get(name)) for row in rows) for name in
              ("native_valid_solid", "compat_valid_solid", "step_exists", "step_readable")}
    counts.update(terminal_records=len(rows), geometry_computable=len(computable),
                  raw_available=sum(row.get("raw_sha256") is not None for row in rows))
    result = {
        "counts": counts, "native_valid_count": len(valid),
        "rate_accepted": len(valid) / n, "rate_original": len(valid) / original,
        "coverage_over_accepted": {k: v / n for k, v in counts.items()},
        "coverage_over_original": {k: v / original for k, v in counts.items()},
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "terminal_stage_counts": dict(Counter(row.get("stage", "unknown") for row in rows)),
        "geometry_all_computable": {name: statistics([r[name] for r in computable]) for name in GEOMETRY},
        "geometry_native_valid": {name: statistics([r[name] for r in computable if r.get("native_valid_solid")])
                                  for name in GEOMETRY},
        "stored_tolerances_native_valid": {name: statistics([r.get(name) for r in valid]) for name in TOLERANCES},
        "valid_face_count_differs_from_raw": sum(r.get("num_faces") != r.get("raw_num_faces") for r in valid),
    }
    if historical:
        result["latency_note"] = "历史测时不作本轮速度比较；本报告不导入其耗时统计。"
    else:
        result["latency_seconds"] = {name: statistics([r.get(name) for r in rows]) for name in COSTS}
    return result


def paired_summary(left: dict, right: dict, ids: list[str], original: int) -> dict:
    groups = Counter()
    for sample_id in ids:
        a, b = bool(left[sample_id].get("native_valid_solid")), bool(right[sample_id].get("native_valid_solid"))
        groups[("both_valid" if b else "regressed") if a else ("rescued" if b else "both_failed")] += 1
    counts = {key: groups[key] for key in ("both_valid", "rescued", "regressed", "both_failed")}
    require(sum(counts.values()) == len(ids), "Paired count conservation failed")
    geometry = {}
    for cohort in ("common_computable", "common_valid"):
        common = [i for i in ids if all(r.get("geometry_status") == "ok" for r in (left[i], right[i]))
                  and (cohort != "common_valid" or all(r.get("native_valid_solid") for r in (left[i], right[i])))]
        geometry[cohort] = {"count": len(common), "coverage_over_accepted": len(common) / len(ids),
                            "coverage_over_original": len(common) / original,
                            "metrics": {name: paired_numbers(left, right, common, name) for name in GEOMETRY}}
    return {"counts": counts, "net_native_valid_gain": counts["rescued"] - counts["regressed"],
            "native_gain_percentage_points": 100 * (counts["rescued"] - counts["regressed"]) / len(ids),
            "geometry": geometry,
            "latency_seconds": {name: paired_numbers(left, right, ids, name) for name in COSTS},
            "stored_tolerances_common_native_valid": {
                name: paired_numbers(left, right, [i for i in ids if left[i].get("native_valid_solid")
                                                  and right[i].get("native_valid_solid")], name)
                for name in TOLERANCES}}


def paired_numbers(left: dict, right: dict, ids: list[str], name: str) -> dict:
    common = [i for i in ids if finite(left[i].get(name)) and finite(right[i].get(name))]
    return {"count": len(common), "baseline": statistics([left[i][name] for i in common]),
            "candidate": statistics([right[i][name] for i in common]),
            "candidate_minus_baseline": statistics([right[i][name] - left[i][name] for i in common])}


def aggregate(maps: dict, ids: list[str], original: int) -> dict:
    require(original >= len(ids) and len(ids) > 0, "Invalid denominator")
    require(set(maps) == set(CELLS), "Missing/unexpected ablation cells")
    for cell, rows in maps.items():
        require(set(rows) == set(ids), f"Incomplete cell: {cell}")
        require(all(row.get("status") in TERMINAL for row in rows.values()), f"Nonterminal cell: {cell}")
    validate_raw_pair_hashes(maps, ids)
    return {"intended_count": len(ids), "original_count": original,
            "excluded_before_accepted_count": original - len(ids),
            "cells": {cell: cell_summary(list(maps[cell].values()), len(ids), original) for cell in CELLS},
            "paired": {name: {"baseline": left, "candidate": right,
                               **paired_summary(maps[left], maps[right], ids, original)}
                       for name, (left, right) in COMPARISONS.items()}}


def scalar(archive, name: str) -> str:
    value = archive[name]
    require(value.ndim == 0, f"Expected scalar provenance: {name}")
    return str(value.item())


def validate_inference(protocol: dict, split_hash: str, n: int) -> None:
    expected = {"version": "frozen_model_post_ablation_inference_v1", "mode": "posterior_mean",
                "methods": list(MODELS), "split_sha256": split_hash, "intended_count": n,
                "identity_view": True, "encoder_gt_topology": True, "decoder_gt_topology": False,
                "decoder_face_deduplication": False, "edge_threshold": .5, "smoke_subset": False,
                "input_contract": "exact_same_verified_reference_posterior_mean"}
    expected.update({f"{model}_checkpoint_sha256": digest for model, digest in CHECKPOINTS.items()})
    for name, value in expected.items():
        require(protocol.get(name) == value, f"Inference protocol mismatch: {name}")
    for name, value in {"dtype": "float32", "autocast": False, "seed": 42,
                        "matmul_allow_tf32": False, "cudnn_allow_tf32": False}.items():
        require(protocol.get("numeric_settings", {}).get(name) == value, f"Inference numeric mismatch: {name}")
    audit = protocol.get("model_audit", {})
    require(audit.get("encoder_equal") is True and audit.get("encoder_equal_tensor_count") == 243,
            "Inference frozen-encoder audit mismatch")
    require(bool(protocol.get("source_sha256")), "Missing inference source hashes")


def audit_raw(root: Path, split: str, ids: list[str], maps: dict, inference: dict,
              reference_root: Path, reference_maps: dict, digest_file) -> dict:
    import numpy as np
    from tools.vae_compare_inference import latent_sha256

    manifest_path = root / split / "metadata/raw_manifest.json"
    manifest = load_json(manifest_path)
    require(manifest.get("intended_count") == len(ids), "Raw manifest denominator mismatch")
    manifest_rows = exact_map(manifest["records"], ids, "model_id", f"{split} raw manifest")
    reference_protocol_path = reference_root / "metadata/inference/posterior_mean/protocol.json"
    reference_protocol = load_json(reference_protocol_path)
    require(inference.get("reference_protocol_sha256") == fingerprint(reference_protocol) and
            inference.get("reference_protocol_file_sha256") == digest_file(reference_protocol_path),
            "Reference inference protocol changed")
    require(inference.get("reference_split_sha256") == reference_protocol.get("split_sha256"),
            "Reference inference split hash mismatch")
    require(Path(inference["reference_root"]).resolve() == reference_root.resolve(), "Wrong reference root")
    verified = missing = 0
    verification_rows = []
    for index, sample_id in enumerate(ids):
        row = manifest_rows[sample_id]
        require(set(row.get("raw_sha256", {})) == set(MODELS), f"Manifest model mismatch: {sample_id}")
        values = {}
        for model in MODELS:
            expected_hash = maps[f"{model}/legacy"][sample_id].get("raw_sha256")
            require(row["raw_sha256"][model] == expected_hash, f"Manifest raw mismatch: {model}/{sample_id}")
            raw = root / split / "raw_predictions/posterior_mean" / model / sample_id / "data.npz"
            if not raw.is_file():
                require(expected_hash is None and all(maps[f"{model}/{p}"][sample_id]["status"] == "missing_raw"
                                                     for p in POLICIES), f"Required raw missing: {raw}")
                missing += 1
                continue
            require(expected_hash == digest_file(raw), f"Raw changed after post: {model}/{sample_id}")
            for policy in POLICIES:
                reused = root / split / "post_variants" / policy / "raw_predictions/posterior_mean" / model / sample_id / "data.npz"
                require(reused.is_file() and digest_file(reused) == expected_hash,
                        f"Reused raw mismatch: {model}/{policy}/{sample_id}")
            with np.load(raw, allow_pickle=False) as archive:
                latents = archive["face_latents"]
                require(latents.dtype == np.float32 and latents.ndim == 2 and latents.shape[1] == 32
                        and len(latents) > 0 and np.isfinite(latents).all(), f"Invalid latent: {model}/{sample_id}")
                digest = latent_sha256(latents)
                expected = {"input_latent_sha256": digest, "protocol_sha256": fingerprint(inference),
                            "comparison_mode": "posterior_mean", "method": model,
                            "checkpoint_sha256": CHECKPOINTS[model],
                            "reference_protocol_sha256": fingerprint(reference_protocol),
                            "input_source_kind": "verified_common_posterior_mean",
                            "encoder_checkpoint_sha256": CHECKPOINTS["legacy"]}
                for name, value in expected.items():
                    require(scalar(archive, name) == value, f"Raw provenance mismatch: {model}/{sample_id}/{name}")
                source = Path(inference["data_root"]) / sample_id / "data.npz"
                reference = reference_root / "raw_predictions/posterior_mean/intersection" / sample_id / "data.npz"
                require(Path(scalar(archive, "input_source_path")).resolve() == source.resolve() and
                        scalar(archive, "input_source_sha256") == digest_file(source), f"Source CAD mismatch: {sample_id}")
                require(Path(scalar(archive, "reference_raw_path")).resolve() == reference.resolve() and
                        scalar(archive, "reference_raw_sha256") == digest_file(reference), f"Reference raw mismatch: {sample_id}")
            require(digest == row.get("latent_sha256"), f"Manifest latent mismatch: {sample_id}")
            values[model] = digest
        if len(values) == len(MODELS):
            require(len(set(values.values())) == 1, f"Cross-model latent mismatch: {sample_id}")
            verified += 1
        reference = reference_root / "raw_predictions/posterior_mean/intersection" / sample_id / "data.npz"
        reference_hash = digest_file(reference)
        require(all(reference_maps[p][sample_id].get("raw_sha256") == reference_hash for p in POLICIES),
                f"Historical post raw mismatch: {sample_id}")
        if values:
            with np.load(reference, allow_pickle=False) as archive:
                digest = latent_sha256(archive["face_latents"])
                require(scalar(archive, "input_latent_sha256") == digest and
                        all(value == digest for value in values.values()), f"Reference latent mismatch: {sample_id}")
        verification_rows.append({"sample_id": sample_id, "raw_sha256": row["raw_sha256"], "latent_sha256": values})
        if (index + 1) % 1000 == 0:
            print(f"{split}: input audit {index + 1}/{len(ids)}", flush=True)
    return {"cross_model_latents_verified": verified, "missing_raw_artifacts": missing,
            "all_cross_model_latents_verified": verified == len(ids),
            "manifest_sha256": digest_file(manifest_path), "verified_inputs_fingerprint": fingerprint(verification_rows)}


def read_split(root: Path, split: str, hash_cache: dict) -> tuple[dict, dict, list[str]]:
    prefix, n, original = SPLITS[split]
    split_path = REPOSITORY / f"checkpoints/joint_boundary/validation/{prefix}_contract_audit/accepted.txt"
    ids = split_path.read_text().split()
    require(len(ids) == n and len(set(ids)) == n, f"Frozen split count mismatch: {split}")
    split_hash = sha256_file(split_path)

    def digest_file(path: Path) -> str:
        resolved = path.resolve()
        if resolved not in hash_cache:
            hash_cache[resolved] = sha256_file(resolved)
        return hash_cache[resolved]

    inference_path = root / split / "metadata/inference/posterior_mean/protocol.json"
    inference = load_json(inference_path)
    validate_inference(inference, split_hash, n)
    maps, protocols, historical, history_protocols = {}, {}, {}, {}
    reference_root = root.parent / f"{split}_20261016"
    for policy in POLICIES:
        metadata = root / split / "post_variants" / policy / "metadata/post/optimized"
        protocol = load_json(metadata / "protocol.json")
        validate_protocol(protocol, split_hash, n, original, policy)
        require(Path(protocol["comparison_root"]).resolve() == metadata.parents[2].resolve(), "Wrong post output root")
        require(Path(protocol["data_root"]).resolve() == Path(inference["data_root"]).resolve(), "Post/inference CAD root mismatch")
        for model in MODELS:
            maps[f"{model}/{policy}"] = read_rows(metadata / "samples" / model, ids, protocol, model, policy)
        protocols[policy] = protocol
        history_meta = reference_root / "post_variants" / policy / "metadata/post/optimized"
        old = load_json(history_meta / "protocol.json")
        validate_protocol(old, split_hash, n, original, policy, historical=True)
        same_contract(protocol, old)
        historical[policy] = read_rows(history_meta / "samples/intersection", ids, old, "intersection", policy)
        history_protocols[policy] = old
    same_contract(protocols["legacy"], protocols["recover_v1"])
    require(protocols["legacy"]["runner_sha256"] == protocols["recover_v1"]["runner_sha256"], "Post runners differ")
    # Verify the frozen executable source, not report/diagram work in progress.
    for name, digest in {**inference["source_sha256"], **protocols["legacy"]["post_algorithm_files"],
                         "tools/vae_compare_post.py": protocols["legacy"]["runner_sha256"]}.items():
        require(digest_file(REPOSITORY / name) == digest, f"Frozen executable source changed: {name}")
    result = aggregate(maps, ids, original)
    result["input_audit"] = audit_raw(root, split, ids, maps, inference, reference_root, historical, digest_file)
    result["historical_refinement_only"] = {
        "note": "仅 refinement、未加入 curve conditioning 的旧权重；同输入均值与后处理合同；历史耗时不比较。",
        "root": str(reference_root),
        "policies": {p: cell_summary(list(historical[p].values()), n, original, historical=True) for p in POLICIES},
        "protocol_fingerprints": {p: fingerprint(history_protocols[p]) for p in POLICIES}}
    result["protocols"] = protocols
    result["inference_protocol"] = inference
    result["split_sha256"] = split_hash
    raw_metrics = root / split / "raw_metrics/posterior_mean/summary.json"
    if raw_metrics.is_file():
        metrics = load_json(raw_metrics)
        require(metrics.get("protocol_valid") is True and metrics.get("split_sha256") == split_hash and
                metrics.get("inference_protocol_sha256") == digest_file(inference_path), "Raw metric protocol mismatch")
        for name, value in {"mode": "posterior_mean", "intended_count": n, "original_count": original,
                            "baseline_method": "legacy", "candidate_method": "conditioned",
                            "input_contract": "identical_latents"}.items():
            require(metrics.get(name) == value, f"Raw metric contract mismatch: {name}")
        require(set(metrics.get("methods", {})) == set(MODELS) and
                all(sum(metrics["methods"][m]["status_counts"].values()) == n for m in MODELS),
                "Raw metric coverage mismatch")
        require(metrics.get("metric_source_sha256") == digest_file(REPOSITORY / "tools/vae_compare_metrics.py"),
                "Raw metric source changed")
        result["raw_metrics"] = {"source": str(raw_metrics), "source_sha256": digest_file(raw_metrics), "summary": metrics}
    else:
        result["raw_metrics"] = None
    return result, maps, ids


def write_csv(path: Path, rows: list[dict]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def export_tables(summary: dict, all_maps: dict, output: Path) -> None:
    cells, pairs, samples = [], [], []
    for split, data in summary["splits"].items():
        entries = [(cell, value, False) for cell, value in data["cells"].items()]
        entries.extend((f"historical_refinement_only/{p}", value, True)
                       for p, value in data["historical_refinement_only"]["policies"].items())
        for cell, value, historical in entries:
            row = {"split": split, "cell": cell, "historical_reference": historical,
                   "intended_count": data["intended_count"], "original_count": data["original_count"],
                   **{k: value[k] for k in ("native_valid_count", "rate_accepted", "rate_original")},
                   **value["counts"], "status_counts": json.dumps(value["status_counts"], sort_keys=True)}
            for group in ("geometry_all_computable", "geometry_native_valid", "stored_tolerances_native_valid", "latency_seconds"):
                for metric, stats in value.get(group, {}).items():
                    row.update({f"{group}.{metric}.{name}": number for name, number in stats.items()})
            cells.append(row)
        for name, value in data["paired"].items():
            row = {"split": split, "comparison": name, "baseline": value["baseline"], "candidate": value["candidate"],
                   **value["counts"], "net_native_valid_gain": value["net_native_valid_gain"]}
            for cohort, geometry in value["geometry"].items():
                row[f"{cohort}.count"] = geometry["count"]
                for metric, sides in geometry["metrics"].items():
                    for side in ("baseline", "candidate", "candidate_minus_baseline"):
                        row.update({f"{cohort}.{metric}.{side}.{k}": v for k, v in sides[side].items()})
            pairs.append(row)
        for cell, rows in all_maps[split].items():
            for sample_id, row in rows.items():
                names = ("status", "stage", "raw_sha256", "native_valid_solid", "compat_valid_solid", "step_exists",
                         "step_readable", "geometry_status", "num_faces", "raw_num_faces") + GEOMETRY + COSTS + TOLERANCES
                samples.append({"split": split, "cell": cell, "sample_id": sample_id, **{k: row.get(k) for k in names}})
    write_csv(output / "cells.csv", cells)
    write_csv(output / "paired.csv", pairs)
    write_csv(output / "samples.csv", samples)


def draw_results(summary: dict, output: Path, font_path: Path) -> None:
    from tools.postprocess_validation_report import configure_plot
    plt = configure_plot(font_path)
    plt.rcParams["svg.fonttype"] = "path"
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4), facecolor="#f6f8fc")
    colors = ("#68788e", "#16877c")
    for ax, (split, data) in zip(axes, summary["splits"].items()):
        n, original = data["intended_count"], data["original_count"]
        for j, policy in enumerate(POLICIES):
            values = [data["cells"][f"{model}/{policy}"] for model in MODELS]
            values.append(data["historical_refinement_only"]["policies"][policy])
            x = [i + (j - .5) * .34 for i in range(3)]
            bars = ax.bar(x, [100 * v["rate_accepted"] for v in values], width=.31, color=colors[j],
                          label="原后处理" if policy == "legacy" else "原流程优先＋失败再修复")
            bars[-1].set_hatch("///"); bars[-1].set_alpha(.65)
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.3,
                        f"{100*value['rate_accepted']:.2f}%\n{value['native_valid_count']}/{n}",
                        ha="center", va="bottom", fontsize=9.5)
        ax.axvline(1.5, color="#aeb8c4", linestyle="--", linewidth=1)
        ax.set(xticks=range(3), xticklabels=["原模型", "Refinement＋\nCurve conditioning", "历史旁列\n仅 Refinement"],
               ylim=(0, 108), yticks=range(0, 101, 20), ylabel="原生 STEP solid 合法率（%）",
               title=f"{split} · accepted {n} / 原始 {original}")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
    axes[0].legend(loc="lower left", fontsize=10)
    fig.suptitle("模型 × 后处理：固定输入均值的 2×2 对照", x=.06, ha="left", fontsize=19, weight="bold")
    fig.text(.06, .035, "柱高使用 accepted 分母；原始分母合法率另见结果表。失败、超时和缺失保留在分母；斜线为历史结果，其耗时不作本轮比较。", fontsize=10, color="#52647a")
    fig.tight_layout(rect=[0, .09, 1, .92])
    for extension in ("png", "svg"):
        fig.savefig(output / f"native_validity.{extension}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def self_check() -> dict:
    """Counterexamples exercise the audit contract, not snapshots of its code."""
    started = time.monotonic()
    ids = ["a", "b", "c", "d"]
    protocol = {"synthetic": True}
    maps = {}
    valid_sets = ({"a", "b"}, {"a", "b", "c"}, {"a", "c"}, {"a", "b", "c"})
    for cell, valid in zip(CELLS, valid_sets):
        model, policy = cell.split("/")
        rows = []
        for index, sample_id in enumerate(ids):
            ok = sample_id in valid
            rows.append({"sample_id": sample_id, "method": model, "mode": "optimized", "post_policy": policy,
                         "protocol_fingerprint": fingerprint(protocol), "seed": stable_seed(sample_id, "post"),
                         "raw_sha256": None if sample_id == "d" else f"{model}-{sample_id}",
                         "status": "missing_raw" if sample_id == "d" else ("ok" if ok else "timeout"),
                         "native_valid_solid": ok, "native_is_solid_type": ok, "native_brepcheck_valid": ok,
                         "step_exists": ok, "step_readable": ok, "geometry_status": "ok" if ok else "not_run",
                         "face_cd": index + .1 if ok else None, "edge_cd": index + .2 if ok else None,
                         "wall_seconds": 180 if not ok else 2 + index})
        maps[cell] = validate_rows(rows, ids, protocol, model, policy)
    checks = []

    def reject(name, callback):
        try:
            callback()
        except (ValueError, FileNotFoundError):
            checks.append(name)
        else:
            raise AssertionError(f"Expected rejection: {name}")

    rows = list(maps[CELLS[0]].values())
    reject("duplicate_record_rejected", lambda: validate_rows(rows + [rows[0]], ids, protocol, "legacy", "legacy"))
    reject("missing_record_rejected", lambda: validate_rows(rows[:-1], ids, protocol, "legacy", "legacy"))
    broken = copy.deepcopy(rows); broken[0]["status"] = "running"
    reject("nonterminal_record_rejected", lambda: validate_rows(broken, ids, protocol, "legacy", "legacy"))
    broken = copy.deepcopy(rows); broken[0]["protocol_fingerprint"] = "foreign"
    reject("foreign_protocol_rejected", lambda: validate_rows(broken, ids, protocol, "legacy", "legacy"))
    broken_maps = copy.deepcopy(maps); broken_maps["legacy/recover_v1"]["a"]["raw_sha256"] = "changed"
    reject("reused_raw_mismatch_rejected", lambda: aggregate(broken_maps, ids, 5))
    broken = copy.deepcopy(rows); broken[0]["face_cd"] = float("nan")
    reject("nonfinite_success_metric_rejected", lambda: validate_rows(broken, ids, protocol, "legacy", "legacy"))
    report = aggregate(maps, ids, 5)
    require(report["cells"]["legacy/legacy"]["rate_accepted"] == .5 and
            report["cells"]["legacy/legacy"]["rate_original"] == .4,
            "Failures were dropped from denominators")
    require(report["cells"]["legacy/legacy"]["status_counts"]["missing_raw"] == 1, "Missing raw lost")
    checks.append("timeout_and_missing_raw_retained_in_both_denominators")
    require(report["paired"]["model_at_legacy"]["counts"] ==
            {"both_valid": 1, "rescued": 1, "regressed": 1, "both_failed": 1}, "Incorrect paired accounting")
    checks.append("rescues_regressions_and_count_conservation")
    broken_maps = copy.deepcopy(maps)
    for rows_by_id in broken_maps.values():
        for row in rows_by_id.values():
            row["geometry_status"] = "not_run"
    empty = aggregate(broken_maps, ids, 5)["paired"]["combined"]["geometry"]["common_valid"]
    require(empty["count"] == 0 and empty["metrics"]["face_cd"]["baseline"]["mean"] is None,
            "Empty geometry cohort was not null")
    checks.append("empty_common_geometry_is_null")
    # A later measurement error cannot erase an already confirmed solid.
    broken_maps = copy.deepcopy(maps); broken_maps["legacy/legacy"]["a"]["status"] = "geometry_error"
    require(aggregate(broken_maps, ids, 5)["cells"]["legacy/legacy"]["native_valid_count"] == 2,
            "Measurement failure erased native verdict")
    checks.append("geometry_failure_preserves_confirmed_validity")
    with tempfile.TemporaryDirectory() as folder:
        reject("missing_required_artifact_rejected", lambda: load_json(Path(folder) / "missing.json"))
        report["historical_refinement_only"] = {"policies": {
            p: cell_summary(list(maps[f"legacy/{p}"].values()), 4, 5, historical=True) for p in POLICIES}}
        synthetic = {"splits": {split: copy.deepcopy(report) for split in SPLITS}}
        export_tables(synthetic, {split: maps for split in SPLITS}, Path(folder))
        require(all((Path(folder) / name).is_file() for name in ("cells.csv", "paired.csv", "samples.csv")), "CSV export missing")
        historical = synthetic["splits"]["test"]["historical_refinement_only"]["policies"]["legacy"]
        require("latency_seconds" not in historical, "Historical speed comparison leaked into report")
    checks.append("csv_exports_and_historical_cost_exclusion")
    checks.extend(check_raw_artifacts())
    return {"passed": True, "checks": checks, "check_count": len(checks),
            "elapsed_seconds": time.monotonic() - started, "synthetic_only": True,
            "production_model_post_runner_modified": False}


def check_raw_artifacts() -> list[str]:
    """Real NPZ files: changing content must fail even with plausible row JSON."""
    import numpy as np
    from tools.vae_compare_inference import latent_sha256

    with tempfile.TemporaryDirectory() as folder:
        base = Path(folder)
        root, reference = base / "ablation", base / "reference"
        source = base / "cad/a/data.npz"
        source.parent.mkdir(parents=True); source.write_bytes(b"synthetic canonical CAD")
        latents = np.arange(32, dtype=np.float32).reshape(1, 32)
        latent_hash = latent_sha256(latents)
        old_protocol = {"split_sha256": "synthetic-split"}
        old_protocol_path = reference / "metadata/inference/posterior_mean/protocol.json"
        atomic_json(old_protocol_path, old_protocol)
        old_raw = reference / "raw_predictions/posterior_mean/intersection/a/data.npz"
        old_raw.parent.mkdir(parents=True)
        np.savez(old_raw, face_latents=latents, input_latent_sha256=latent_hash)
        inference = {"data_root": str(source.parents[1]), "reference_root": str(reference),
                     "reference_protocol_sha256": fingerprint(old_protocol),
                     "reference_protocol_file_sha256": sha256_file(old_protocol_path),
                     "reference_split_sha256": "synthetic-split"}
        maps, raw_hashes, arrays = {}, {}, {}
        for model in MODELS:
            raw = root / "validation/raw_predictions/posterior_mean" / model / "a/data.npz"
            raw.parent.mkdir(parents=True)
            arrays[model] = {"face_latents": latents, "input_latent_sha256": latent_hash,
                             "protocol_sha256": fingerprint(inference), "comparison_mode": "posterior_mean",
                             "method": model, "checkpoint_sha256": CHECKPOINTS[model],
                             "reference_protocol_sha256": fingerprint(old_protocol),
                             "input_source_kind": "verified_common_posterior_mean",
                             "encoder_checkpoint_sha256": CHECKPOINTS["legacy"],
                             "input_source_path": str(source), "input_source_sha256": sha256_file(source),
                             "reference_raw_path": str(old_raw), "reference_raw_sha256": sha256_file(old_raw)}
            np.savez(raw, **arrays[model]); raw_hashes[model] = sha256_file(raw)
            for policy in POLICIES:
                reused = root / "validation/post_variants" / policy / "raw_predictions/posterior_mean" / model / "a/data.npz"
                reused.parent.mkdir(parents=True); reused.symlink_to(raw)
                maps[f"{model}/{policy}"] = {"a": {"raw_sha256": raw_hashes[model], "status": "ok"}}
        manifest = root / "validation/metadata/raw_manifest.json"
        atomic_json(manifest, {"intended_count": 1, "records": [
            {"model_id": "a", "raw_sha256": raw_hashes, "latent_sha256": latent_hash}]})
        old_maps = {p: {"a": {"raw_sha256": sha256_file(old_raw)}} for p in POLICIES}

        def run():
            return audit_raw(root, "validation", ["a"], maps, inference, reference, old_maps, sha256_file)

        require(run()["cross_model_latents_verified"] == 1, "Valid raw fixture failed")
        checks = ["actual_npz_latents_and_reference_hashes_verified"]
        reused = root / "validation/post_variants/recover_v1/raw_predictions/posterior_mean/legacy/a/data.npz"
        canonical = reused.resolve(); reused.unlink(); reused.write_bytes(b"different physical input")
        try:
            run()
        except ValueError as error:
            require("Reused raw mismatch" in str(error), "Unexpected physical-input rejection")
            checks.append("physical_reused_raw_mismatch_rejected")
        else:
            raise AssertionError("Different reused file accepted")
        reused.unlink(); reused.symlink_to(canonical)
        # Keep all file hashes consistent but lie about the bytes of one model's
        # latent; the recomputed content digest must still reject it.
        raw = root / "validation/raw_predictions/posterior_mean/conditioned/a/data.npz"
        arrays["conditioned"]["face_latents"] = latents + 1
        np.savez(raw, **arrays["conditioned"])
        raw_hashes["conditioned"] = sha256_file(raw)
        for policy in POLICIES:
            maps[f"conditioned/{policy}"]["a"]["raw_sha256"] = raw_hashes["conditioned"]
        atomic_json(manifest, {"intended_count": 1, "records": [
            {"model_id": "a", "raw_sha256": raw_hashes, "latent_sha256": latent_hash}]})
        try:
            run()
        except ValueError as error:
            require("input_latent_sha256" in str(error), "Unexpected latent-content rejection")
            checks.append("latent_bytes_mismatch_rejected_despite_matching_raw_hashes")
        else:
            raise AssertionError("Changed latent bytes accepted")
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--font", type=Path, default=Path("/root/.cache/hola-browser-libs/fonts/NotoSansCJKsc-Regular.otf"))
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--check-output", type=Path)
    args = parser.parse_args()
    if args.self_check:
        result = self_check()
        result.update(command=[sys.executable, *sys.argv], report_script_sha256=sha256_file(Path(__file__)))
        if args.check_output:
            atomic_json(args.check_output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    require(args.check_output is None, "--check-output requires --self-check")
    output = args.output or args.root / "report"
    summary = {"version": "frozen_model_post_ablation_report_v1", "splits": {},
               "notes": ["已训练权重的系统对照；额外微调与损失变化未被单独控制，不能归因于单个 block。",
                         "test 只汇总冻结方案；历史 DeepCAD test 曾参与开发，不称完全盲测。",
                         "CD²沿用原测量；不代表连续曲面最大误差或拓扑完全正确。",
                         "耗时为128 worker并发下的本轮记录；历史旁列耗时不比较。"],
               "report_script_sha256": sha256_file(Path(__file__)), "command": [sys.executable, *sys.argv]}
    all_maps, all_ids, cache = {}, {}, {}
    for split in SPLITS:
        summary["splits"][split], all_maps[split], all_ids[split] = read_split(args.root, split, cache)
    require(not set(all_ids["validation"]) & set(all_ids["test"]), "Validation/test overlap")
    # Never produce a partially validated summary from a missing second split.
    output.mkdir(parents=True, exist_ok=True)
    export_tables(summary, all_maps, output)
    draw_results(summary, output, args.font)
    atomic_json(output / "summary.json", summary)
    print(json.dumps({"output": str(output), "native_valid_counts": {
        split: {cell: value["native_valid_count"] for cell, value in data["cells"].items()}
        for split, data in summary["splits"].items()}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
