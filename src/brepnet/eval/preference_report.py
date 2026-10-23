"""Fixed-list comparison of original SVR and the final DPO checkpoint.

Read completed inference/evaluation records only. Geometry failures remain
missing measurements; validity and coverage always use the full CAD list.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np


METHODS = ("baseline", "dpo_5000")
RAW_METRICS = ("face_cd_sq", "edge_cd_sq", "face_f1", "face_count_abs_error")
POST_METRICS = RAW_METRICS[:3]
INFERENCE_TERMINAL = {"ok", "generation_error", "interrupted"}
EVALUATION_TERMINAL = {
    "ok", "input_error", "no_step", "evaluation_error", "timeout", "crash", "interrupted",
}
VALIDITIES = ("valid", "invalid", "unknown")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)


def load_rows(run_dir: Path, expected_count: int = 100) -> tuple[list[str], dict[str, list[dict]]]:
    """Require every intended object to have reached an explicit terminal state."""
    ids = [line.strip() for line in (run_dir / "validation100.txt").read_text().splitlines() if line.strip()]
    if len(ids) != expected_count or len(set(ids)) != expected_count:
        raise ValueError(f"Expected {expected_count} distinct validation IDs")
    rows = {method: [] for method in METHODS}
    for method in METHODS:
        for image_id in ids:
            raw_dir = run_dir / "raw_predictions" / method / image_id
            inference_path = raw_dir / "inference.json"
            if not inference_path.is_file():
                raise ValueError(f"Missing inference terminal record: {method}/{image_id}")
            inference = read_json(inference_path)
            if inference.get("status") not in INFERENCE_TERMINAL:
                raise ValueError(f"Unfinished inference: {method}/{image_id}")
            evaluation_dir = run_dir / "evaluations" / method / image_id
            evaluation = {}
            if inference["status"] == "ok":
                if not all((raw_dir / name).is_file() for name in ("data.npz", "latent.npy")):
                    raise ValueError(f"Successful inference has missing outputs: {method}/{image_id}")
                path = evaluation_dir / "metrics.json"
                if not path.is_file():
                    raise ValueError(f"Missing evaluation terminal record: {method}/{image_id}")
                evaluation = read_json(path)
                if evaluation.get("status") not in EVALUATION_TERMINAL:
                    raise ValueError(f"Unfinished evaluation: {method}/{image_id}")
            validity = evaluation.get("validity", "unknown")
            if validity not in VALIDITIES:
                raise ValueError(f"Unknown validity label: {method}/{image_id}: {validity}")
            row = {
                "image_id": image_id, "method": method, "seed": inference.get("seed"),
                "inference_status": inference["status"],
                "evaluation_status": evaluation.get("status", "not_run"),
                "validity": validity, "inference_seconds": inference.get("seconds"),
                "evaluation_seconds": evaluation.get("wall_seconds"),
                "raw_exists": (raw_dir / "data.npz").is_file(),
                "step_exists": (evaluation_dir / "post/c00/recon_brep.step").is_file(),
                "step_readable": bool(evaluation.get("native", {}).get("step_readable")),
                "error": evaluation.get("error") or inference.get("error", ""),
            }
            raw = evaluation.get("raw", {})
            for metric in POST_METRICS:
                row[f"raw_{metric}"] = raw.get(metric) if finite_number(raw.get(metric)) else None
            counts = [raw.get(name) for name in ("pred_face_count", "gt_face_count")]
            row["raw_face_count_abs_error"] = abs(counts[0] - counts[1]) if all(map(finite_number, counts)) else None
            repeats = evaluation.get("post", [])
            complete = (
                validity == "valid" and len(repeats) == 3
                and {item.get("repeat") for item in repeats} == {0, 1, 2}
                and all(finite_number(item.get(metric)) for item in repeats for metric in POST_METRICS)
            )
            row.update(post_repeat_count=len(repeats), post_complete=complete)
            for metric in POST_METRICS:
                row[f"post_{metric}"] = float(np.mean([item[metric] for item in repeats])) if complete else None
            rows[method].append(row)
    for baseline, dpo in zip(rows[METHODS[0]], rows[METHODS[1]]):
        if baseline["seed"] is None or baseline["seed"] != dpo["seed"]:
            raise ValueError(f"Missing or unequal paired inference seeds: {baseline['image_id']}")
    return ids, rows


def statistics(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "median": None}
    return {"n": len(values), "mean": float(np.mean(values)), "median": float(np.median(values))}


def paired_statistics(left: dict[str, float], right: dict[str, float], *,
                      bootstrap_repeats: int = 10000, seed: int = 42) -> dict:
    """Resample CAD pairs, never independent methods or geometry repeats."""
    ids = sorted(left.keys() & right.keys())
    result = {"n": len(ids), "ids": ids, "baseline_mean": None, "dpo_5000_mean": None,
              "dpo_minus_baseline": None, "ci95": None,
              "bootstrap_repeats": bootstrap_repeats, "bootstrap_seed": seed}
    if not ids:
        return result
    baseline = np.array([left[image_id] for image_id in ids])
    dpo = np.array([right[image_id] for image_id in ids])
    differences = dpo - baseline
    rng = np.random.default_rng(seed)
    samples = rng.integers(len(ids), size=(bootstrap_repeats, len(ids)))
    interval = np.quantile(differences[samples].mean(axis=1), [.025, .975])
    result.update(baseline_mean=float(baseline.mean()), dpo_5000_mean=float(dpo.mean()),
                  dpo_minus_baseline=float(differences.mean()), ci95=interval.tolist())
    return result


def summarize(ids: list[str], rows: dict[str, list[dict]], *, bootstrap_repeats: int = 10000) -> dict:
    total = len(ids)
    summary = {
        "version": "svr_dpo_validation100_v1", "intended_count": total,
        "methods": {}, "metrics": {},
        "protocol": {
            "candidates_per_image": 1, "candidate_id": "c00", "selection": "none",
            "comparison": "original SVR vs final DPO step 5000; no checkpoint selection",
            "post_policy": "recover_v1", "post_repeats_required": 3,
            "post_aggregation": "mean of three repeats within each CAD, then mean across CADs",
            "missing_geometry": "null, never zero", "cd": "symmetric squared Chamfer; no square root",
            "bootstrap": "paired CAD resampling; percentile 95% interval; 10000 samples; seed 42",
            "validity_bootstrap": "valid=1; invalid/unknown=0; unknown counts reported separately",
        },
    }
    for method in METHODS:
        records = rows[method]
        if len(records) != total or {r["image_id"] for r in records} != set(ids):
            raise ValueError(f"{method} does not cover the complete frozen list")
        counts = {
            "inference_success": sum(r["inference_status"] == "ok" for r in records),
            "raw_outputs": sum(r["raw_exists"] for r in records),
            "step_outputs": sum(r["step_exists"] for r in records),
            "readable_steps": sum(r["step_readable"] for r in records),
            **{f"validity_{label}": sum(r["validity"] == label for r in records) for label in VALIDITIES},
            "post_complete": sum(r["post_complete"] for r in records),
            "partial_post": sum(0 < r["post_repeat_count"] < 3 for r in records),
            "valid_without_complete_post": sum(r["validity"] == "valid" and not r["post_complete"] for r in records),
            "inference_errors": sum(r["inference_status"] != "ok" for r in records),
            "evaluation_errors": sum(r["evaluation_status"] not in ("ok", "not_run") for r in records),
            "timeouts": sum(r["evaluation_status"] == "timeout" for r in records),
        }
        summary["methods"][method] = {
            "counts": counts, "rates_over_intended": {name: count / total for name, count in counts.items()},
            "inference_status_counts": dict(Counter(r["inference_status"] for r in records)),
            "evaluation_status_counts": dict(Counter(r["evaluation_status"] for r in records)),
        }
    for stage, metrics in (("raw", RAW_METRICS), ("post", POST_METRICS)):
        for metric in metrics:
            key = f"{stage}_{metric}"
            values = {method: {r["image_id"]: r[key] for r in rows[method] if r[key] is not None} for method in METHODS}
            summary["metrics"][key] = {
                "own": {method: statistics(list(values[method].values())) for method in METHODS},
                "paired_common": paired_statistics(values[METHODS[0]], values[METHODS[1]], bootstrap_repeats=bootstrap_repeats),
            }
    valid = {method: {r["image_id"]: float(r["validity"] == "valid") for r in rows[method]} for method in METHODS}
    summary["validity_paired"] = paired_statistics(valid[METHODS[0]], valid[METHODS[1]], bootstrap_repeats=bootstrap_repeats)
    maps = {method: {r["image_id"]: r["validity"] for r in rows[method]} for method in METHODS}
    transitions = {f"{left}_to_{right}": [] for left in VALIDITIES for right in VALIDITIES}
    for image_id in ids:
        transitions[f"{maps[METHODS[0]][image_id]}_to_{maps[METHODS[1]][image_id]}"].append(image_id)
    summary["validity_transitions"] = {name: {"n": len(items), "ids": items} for name, items in transitions.items()}
    return summary


def write_csv(path: Path, ids: list[str], rows: dict[str, list[dict]]) -> None:
    maps = {method: {row["image_id"]: row for row in records} for method, records in rows.items()}
    paired = []
    for image_id in ids:
        row = {"image_id": image_id}
        for method in METHODS:
            row.update({f"{method}_{key}": value for key, value in maps[method][image_id].items() if key not in ("image_id", "method")})
        row["validity_transition"] = f"{maps[METHODS[0]][image_id]['validity']}_to_{maps[METHODS[1]][image_id]['validity']}"
        for stage, metrics in (("raw", RAW_METRICS), ("post", POST_METRICS)):
            for metric in metrics:
                key = f"{stage}_{metric}"
                left, right = (maps[method][image_id][key] for method in METHODS)
                row[f"delta_{key}"] = right - left if left is not None and right is not None else None
        paired.append(row)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)


def plot_comparison(summary: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    colors = ("#4775a7", "#dc7856")
    fig = plt.figure(figsize=(17, 10), facecolor="#fafafa")
    grid = fig.add_gridspec(2, 3, height_ratios=(1.55, 1), hspace=.6, wspace=.65)
    ax_table = fig.add_subplot(grid[0, :])
    ax_table.axis("off")
    ax_table.set_title("Single-view reconstruction: original SVR vs DPO (5,000 steps)", loc="left", fontsize=19, pad=20)

    def number(value, scale=1):
        return "n/a" if value is None else f"{value * scale:.3f}"

    table_rows = []
    valid = summary["validity_paired"]
    interval = valid["ci95"]
    table_rows.append(["Native valid solid (%)", number(valid["baseline_mean"], 100), number(valid["dpo_5000_mean"], 100),
                       str(summary["intended_count"]), number(valid["dpo_minus_baseline"], 100),
                       f"[{interval[0] * 100:.2f}, {interval[1] * 100:.2f}] pp"])
    labels = {
        "raw_face_cd_sq": "Raw face CD² ×1000", "raw_edge_cd_sq": "Raw edge CD² ×1000",
        "raw_face_f1": "Raw face F1 (%)", "raw_face_count_abs_error": "Raw face count MAE",
        "post_face_cd_sq": "Post face CD² ×1000", "post_edge_cd_sq": "Post edge CD² ×1000",
        "post_face_f1": "Post face F1 (%)",
    }
    for key, label in labels.items():
        paired = summary["metrics"][key]["paired_common"]
        scale = 1000 if "cd_sq" in key else 100 if "f1" in key else 1
        interval = paired["ci95"]
        ci = "n/a" if interval is None else f"[{interval[0] * scale:.3f}, {interval[1] * scale:.3f}]"
        table_rows.append([label, number(paired["baseline_mean"], scale), number(paired["dpo_5000_mean"], scale),
                           str(paired["n"]), number(paired["dpo_minus_baseline"], scale), ci])
    table = ax_table.table(cellText=table_rows, colLabels=["Metric", "Original SVR", "DPO 5000", "Paired n", "DPO − SVR", "Paired 95% CI"],
                           cellLoc="center", colWidths=[.25, .14, .14, .09, .14, .24], bbox=[0, 0, 1, .95])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d9dde3")
        cell.set_facecolor("#e8edf3" if row == 0 else "white" if row % 2 else "#f2f5f8")
        if row == 0:
            cell.set_text_props(weight="bold")
        if col == 0:
            cell.set_text_props(ha="left")

    ax_valid = fig.add_subplot(grid[1, 0])
    counts = [summary["methods"][method]["counts"]["validity_valid"] for method in METHODS]
    total = summary["intended_count"]
    ax_valid.bar([0, 1], np.array(counts) / total * 100, color=colors, width=.55)
    for i, method in enumerate(METHODS):
        unknown = summary["methods"][method]["counts"]["validity_unknown"]
        ax_valid.text(i, counts[i] / total * 100 + 2, f"{counts[i]}/{total}\nunknown: {unknown}", ha="center", fontsize=10)
    ax_valid.set(xticks=[0, 1], xticklabels=["Original SVR", "DPO 5000"], ylim=(0, 113), ylabel="Native valid solid (%)", title="Validity: full intended list")

    def delta_plot(ax, keys, scale, xlabel, title):
        for index, key in enumerate(keys):
            paired = summary["metrics"][key]["paired_common"]
            if paired["n"]:
                delta = paired["dpo_minus_baseline"] * scale
                low, high = np.array(paired["ci95"]) * scale
                ax.plot([low, high], [index, index], color=colors[1], linewidth=3)
                ax.scatter([delta], [index], color=colors[1], s=45, zorder=3)
            else:
                ax.text(0, index, "n/a", va="center")
        ax.axvline(0, color="#555", linewidth=1, linestyle="--")
        names = [labels[key].replace(" ×1000", "").replace(" (%)", "") + f" (n={summary['metrics'][key]['paired_common']['n']})" for key in keys]
        ax.set(yticks=range(len(keys)), yticklabels=names, xlabel=xlabel, title=title)
        ax.invert_yaxis()
        ax.grid(axis="x", color="#e3e6e9")
        ax.margins(y=.3)

    delta_plot(fig.add_subplot(grid[1, 1]), ["raw_face_cd_sq", "raw_edge_cd_sq", "post_face_cd_sq", "post_edge_cd_sq"],
               1000, "DPO − SVR, CD² ×1000 (lower is better)", "Paired geometry differences + 95% CI")
    delta_plot(fig.add_subplot(grid[1, 2]), ["raw_face_f1", "post_face_f1"], 100,
               "DPO − SVR, percentage points (higher is better)", "Paired face F1 differences + 95% CI")
    for ax in fig.axes[1:]:
        ax.spines[["top", "right"]].set_visible(False)
    own_counts = []
    for method, label in zip(METHODS, ("SVR", "DPO")):
        raw_n = summary["metrics"]["raw_face_cd_sq"]["own"][method]["n"]
        post_n = summary["metrics"]["post_face_cd_sq"]["own"][method]["n"]
        own_counts.append(f"{label}: raw n={raw_n}, complete post n={post_n}")
    fig.text(.055, .035, " | ".join(own_counts) + "\nK=1, matched seeds; original AE; recover_v1; post = mean of 3 repeats per CAD. Missing geometry is not zero.\nTable geometry means use paired common CADs; own-subset means are in summary.json. Paired bootstrap: 10,000 resamples, seed 42.", fontsize=9, color="#444")
    fig.subplots_adjust(left=.055, right=.98, top=.91, bottom=.16)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"comparison.{suffix}", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_report(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir).resolve()
    ids, rows = load_rows(run_dir)
    summary = summarize(ids, rows)
    summary["run_dir"] = str(run_dir)
    output = run_dir / "report"
    output.mkdir(exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    write_csv(output / "per_object.csv", ids, rows)
    plot_comparison(summary, output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = write_report(args.run_dir)
    print(json.dumps({"report": str(args.run_dir / "report"), "intended_count": summary["intended_count"],
                      "validity_paired": summary["validity_paired"]}, indent=2))


if __name__ == "__main__":
    main()
