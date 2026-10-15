"""Fixed last-checkpoint supplement; never select checkpoints using these results."""
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
SPLIT = REPO / "checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt"
IDS = SPLIT.read_text().splitlines()
assert len(IDS) == len(set(IDS)) == 2417
SOURCES = {}
CASES = {
    "conditioned": ("candidate", "conditioned", "Baseline"),
    "control_1089": ("control", "encoder_gat_control", "Original GAT\nstep 1,089"),
    "candidate_1089": ("candidate", "encoder_edge_value", "Edge-value GAT\nstep 1,089"),
    "candidate_14520": ("endpoint_14520/candidate", "encoder_edge_value", "Edge-value GAT\nstep 14,520"),
}
METRICS = ("face_mae", "face_cd_sq", "edge_cd_sq", "edge_gt_to_pred_sq",
           "edge_pred_to_gt_sq", "legacy_reciprocal_cd_sq_mean")


def read_bytes(path):
    before = path.read_bytes()
    digest = hashlib.sha256(before).hexdigest()
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Changing source: {path}")
    SOURCES[str(path.resolve())] = digest
    return before


def read_table(path, method, key):
    rows = list(csv.DictReader(io.StringIO(read_bytes(path).decode())))
    selected = [row for row in rows if row["method"] == method]
    result = {row[key]: row for row in selected}
    if len(selected) != len(result) or set(result) != set(IDS):
        raise ValueError(f"Expected exactly 2417 unique IDs: {path}/{method}")
    return result


def number(value):
    if value in (None, ""):
        return None
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"Invalid metric {value}")
    return value


def boolean(value):
    if value == "True":
        return True
    if value == "False":
        return False
    if value == "":
        return None
    raise ValueError(f"Unexpected boolean {value!r}")


def stats(values):
    known = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if not len(known):
        return {"count": 0, "missing": len(values)}
    return {"count": len(known), "missing": len(values) - len(known),
            "mean": float(known.mean()), "median": float(np.median(known)),
            **{f"p{p}": float(np.percentile(known, p)) for p in (90, 95, 99)},
            "max": float(known.max())}


def main():
    destination = ROOT / "endpoint_diagnostics.json"
    if destination.exists():
        raise FileExistsError(destination)
    if read_bytes(SPLIT).decode().splitlines() != IDS:
        raise ValueError("Split changed")
    read_bytes(Path(__file__))
    read_bytes(ROOT / "PLAN.md")
    raw, post, output = {}, {}, {}
    for label, (relative, method, _) in CASES.items():
        directory = ROOT / relative
        raw[label] = read_table(directory / "raw_metrics/posterior_mean/per_object.csv", method, "model_id")
        summary = json.loads(read_bytes(directory / "raw_metrics/posterior_mean/summary.json"))
        if not summary["protocol_valid"] or summary["input_contract"] != "same_geometry_source":
            raise ValueError("Invalid input contract")
        for row in raw[label].values():
            if row["shared_input_status"] != "verified" or row["latent_digest_status"] != "verified":
                raise ValueError("Unverified source/latent")
            if row["input_source_sha256"] != row["actual_gt_source_sha256"]:
                raise ValueError("GT source mismatch")
        metrics = {key: stats([number(raw[label][sid][key]) for sid in IDS]) for key in METRICS}
        for key in METRICS:
            for statistic in ("mean", "median", "p90"):
                if not np.isclose(metrics[key][statistic], summary["methods"][method]["metrics"][key][statistic], rtol=1e-12, atol=1e-15):
                    raise ValueError(f"Raw summary mismatch: {label}/{key}/{statistic}")
        output[label] = {"raw": metrics, "adjacency_exact": sum(
            row["topology_status"] == "ok" and number(row["topology_fp"]) == 0 and number(row["topology_fn"]) == 0
            for row in raw[label].values()), "post": {}}
        post[label] = {}
        for mode in ("fit_only", "optimized"):
            post[label][mode] = read_table(directory / "post" / mode / "samples.csv", method, "sample_id")
            summary = json.loads(read_bytes(directory / "post" / mode / "summary.json"))
            if summary["methods"][method]["counts"]["terminal_records"] != len(IDS):
                raise ValueError("Incomplete post records")
            values = [boolean(post[label][mode][sid]["native_valid_solid"]) for sid in IDS]
            valid = sum(value is True for value in values)
            if valid != summary["methods"][method]["counts"]["native_valid_solid"]:
                raise ValueError("Solid count differs from upstream summary")
            output[label]["post"][mode] = {"valid": valid, "rate": valid / len(IDS),
                "invalid": sum(value is False for value in values),
                "unknown": sum(value is None for value in values),
                "status_counts": dict(Counter(row["status"] for row in post[label][mode].values()))}
    for sid in IDS:
        if len({raw[label][sid]["input_source_sha256"] for label in CASES}) != 1:
            raise ValueError(f"Canonical CAD mismatch: {sid}")
    comparisons = {}
    for reference in ("conditioned", "candidate_1089"):
        pair = {"raw": {}, "post": {}}
        for key in METRICS:
            values = [(number(raw["candidate_14520"][sid][key]), number(raw[reference][sid][key])) for sid in IDS]
            common = [(a, b) for a, b in values if a is not None and b is not None]
            deltas = [a - b for a, b in common]
            reference_mean = float(np.mean([b for _, b in common])) if common else None
            pair["raw"][key] = {"improved": sum(d < -1e-12 for d in deltas),
                "tied": sum(abs(d) <= 1e-12 for d in deltas), "worsened": sum(d > 1e-12 for d in deltas),
                "missing": len(IDS) - len(deltas), "mean_delta": float(np.mean(deltas)) if deltas else None,
                "relative_mean_change": float(np.mean(deltas)) / reference_mean if reference_mean else None}
        for mode in ("fit_only", "optimized"):
            transitions = {name: [] for name in ("both_valid", "rescued", "regressed", "both_failed", "unknown")}
            for sid in IDS:
                a = boolean(post[reference][mode][sid]["native_valid_solid"])
                b = boolean(post["candidate_14520"][mode][sid]["native_valid_solid"])
                name = "unknown" if a is None or b is None else (
                    "both_valid" if a and b else "rescued" if b else "regressed" if a else "both_failed")
                transitions[name].append(sid)
            pair["post"][mode] = {"counts": {k: len(v) for k, v in transitions.items()}, "ids": transitions}
        comparisons[reference] = pair
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    labels = [record[2] for record in CASES.values()]
    colors = ["#8c98a4", "#d49a36", "#3a76b7", "#26877b"]
    for ax, key, title in zip(axes[0], ("face_mae", "edge_cd_sq"),
                              ("Raw face MAE (lower is better)", "Raw edge squared CD (lower is better)")):
        values = [output[label]["raw"][key]["mean"] for label in CASES]
        bars = ax.bar(labels, values, color=colors)
        ax.bar_label(bars, labels=[f"{v:.4g}" for v in values], padding=3)
        ax.set_ylim(0, max(values) * 1.18)
        ax.set_title(title)
    for ax, mode in zip(axes[1], ("fit_only", "optimized")):
        values = [output[label]["post"][mode]["rate"] * 100 for label in CASES]
        bars = ax.bar(labels, values, color=colors)
        ax.bar_label(bars, labels=[f"{output[label]['post'][mode]['valid']}/2417\n{value:.2f}%" for label, value in zip(CASES, values)], padding=3)
        ax.set_ylim(0, 100)
        ax.set_title(f"Native valid solid: {mode} (higher is better)")
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Fixed 2,417 CAD objects / own posterior means\n1,089-step architecture comparison and separately predeclared 14,520-step endpoint", fontsize=13)
    visual = ROOT / "visualizations"
    visual.mkdir(exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(visual / f"four_checkpoint_summary.{suffix}", dpi=170)
    plt.close(fig)
    for name, digest in SOURCES.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source changed during aggregation: {name}")
    destination.write_text(json.dumps({"intended_count": len(IDS), "endpoint_selection": "last_saved_not_test_selected",
        "methods": output, "endpoint_comparisons": comparisons, "source_sha256": SOURCES,
        "limitations": ["Last-step candidate has no equal-step original-GAT control.",
                        "Raw CD uses squared sample distances, not CAD tolerance.",
                        "Both failed includes invalid solids; unknown is reported separately."]}, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
