"""
Case Selection Script for Visualization.

Reads evaluation results (eval_condition.csv + validity) from a prediction directory,
classifies cases by failure mode, and outputs selected case IDs for visualization.

Usage:
    python -m src.brepnet.eval.select_cases \
        --pred-root /path/to/predictions \
        --num 5 \
        --output selected_cases.json

Output JSON format:
{
    "best": ["00001234", "00005678", ...],
    "median": [...],
    "worst_valid": [...],
    "invalid": [...],
    "face_count_error": [...],
    "topo_wrong_geom_right": [...]
}
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_eval_results(pred_root: Path) -> list[dict[str, Any]]:
    """
    Load evaluation results from eval_condition.csv and validity info.

    Each row in eval_condition.csv has columns like:
        prefix, face_cd, edge_cd, vertex_cd, face_f_score, fe_f_score, ev_f_score,
        num_faces_pred, num_faces_gt, ...

    We also check if success.txt exists (valid) or not (invalid).
    """
    results = []

    # Try CSV first (most complete)
    csv_path = pred_root / "eval_condition.csv"
    if csv_path.exists():
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entry = {"id": row.get("prefix", row.get("name", "unknown"))}
                # Parse numeric fields
                for key in ["face_cd", "edge_cd", "vertex_cd",
                            "face_f_score", "fe_f_score", "ev_f_score",
                            "num_faces_pred", "num_faces_gt",
                            "abs_face_count_error"]:
                    if key in row and row[key]:
                        try:
                            entry[key] = float(row[key])
                        except ValueError:
                            pass
                # Check validity
                sample_dir = pred_root / entry["id"]
                entry["valid"] = (sample_dir / "success.txt").exists()
                results.append(entry)
        return results

    # Fallback: scan directories and load per-sample npz
    for sample_dir in sorted(pred_root.iterdir()):
        if not sample_dir.is_dir():
            continue
        if sample_dir.name.startswith("_") or sample_dir.name == "reports":
            continue

        entry = {"id": sample_dir.name}
        entry["valid"] = (sample_dir / "success.txt").exists()

        # Load condition metrics
        for npz_name in ("eval_condition.npz", "eval.npz"):
            npz_path = sample_dir / npz_name
            if npz_path.exists():
                data = np.load(npz_path, allow_pickle=True)
                if "results" in data:
                    r = data["results"].item()
                    entry.update({
                        "face_cd": r.get("face_cd", r.get("face_chamfer")),
                        "edge_cd": r.get("edge_cd", r.get("edge_chamfer")),
                        "vertex_cd": r.get("vertex_cd", r.get("vertex_chamfer")),
                        "face_f_score": r.get("face_f_score"),
                        "fe_f_score": r.get("fe_f_score"),
                        "ev_f_score": r.get("ev_f_score"),
                        "num_faces_pred": r.get("num_faces_pred"),
                        "num_faces_gt": r.get("num_faces_gt"),
                    })
                break

        # Load validity metrics
        validity_path = sample_dir / "eval_validity.npz"
        if validity_path.exists():
            vdata = np.load(validity_path, allow_pickle=True)
            if "valid" in vdata:
                entry["valid"] = bool(vdata["valid"])
            if "num_faces" in vdata:
                entry["num_faces_pred"] = int(vdata["num_faces"])

        results.append(entry)

    return results


def select_cases(results: list[dict], num_per_category: int = 5) -> dict[str, list[str]]:
    """
    Systematically select visualization cases by failure mode.

    Categories:
        best:                   Lowest Chamfer (valid only) — prove the method works
        median:                 Median Chamfer — typical performance
        worst_valid:            Highest Chamfer (valid only) — hardest valid cases
        invalid:                Invalid cases — topology/assembly failures
        face_count_error:       Valid but face count is wrong — padding/dedup issue
        topo_wrong_geom_right:  Low Chamfer but low F1 — topology errors
        geom_wrong_topo_right:  High F1 but high Chamfer — geometry errors
    """
    valid = [r for r in results if r.get("valid", False) and "face_cd" in r]
    invalid = [r for r in results if not r.get("valid", False)]

    # Sort valid by Chamfer Distance
    valid_sorted = sorted(valid, key=lambda x: x.get("face_cd", 999))

    n = num_per_category
    cases = {}

    # Best (lowest Chamfer)
    cases["best"] = [r["id"] for r in valid_sorted[:n]]

    # Median
    mid = len(valid_sorted) // 2
    cases["median"] = [r["id"] for r in valid_sorted[max(0, mid - n // 2): mid + n // 2 + 1]][:n]

    # Worst valid (highest Chamfer among valid)
    cases["worst_valid"] = [r["id"] for r in valid_sorted[-n:]]

    # Invalid (sorted by face_cd if available, else arbitrary)
    invalid_sorted = sorted(invalid, key=lambda x: x.get("face_cd", 0), reverse=True)
    cases["invalid"] = [r["id"] for r in invalid_sorted[:n]]

    # Face count error: valid but |pred_faces - gt_faces| > 2
    face_count_wrong = [
        r for r in valid
        if "num_faces_pred" in r and "num_faces_gt" in r
        and abs(r["num_faces_pred"] - r["num_faces_gt"]) > 2
    ]
    face_count_wrong.sort(key=lambda x: abs(x["num_faces_pred"] - x["num_faces_gt"]), reverse=True)
    cases["face_count_error"] = [r["id"] for r in face_count_wrong[:n]]

    # Topology wrong but geometry right: face_cd < median but fe_f_score low
    median_cd = valid_sorted[len(valid_sorted) // 2]["face_cd"] if valid_sorted else 0.01
    median_f1 = np.median([r.get("fe_f_score", 0) for r in valid]) if valid else 0.5
    topo_wrong = [
        r for r in valid
        if r.get("face_cd", 999) < median_cd
        and r.get("fe_f_score", 1.0) < median_f1 * 0.6
    ]
    topo_wrong.sort(key=lambda x: x.get("fe_f_score", 0))
    cases["topo_wrong_geom_right"] = [r["id"] for r in topo_wrong[:n]]

    # Geometry wrong but topology right: good fe_f_score but face_cd > median
    geom_wrong = [
        r for r in valid
        if r.get("fe_f_score", 0) > median_f1 * 1.3
        and r.get("face_cd", 0) > median_cd
    ]
    geom_wrong.sort(key=lambda x: x.get("face_cd", 0), reverse=True)
    cases["geom_wrong_topo_right"] = [r["id"] for r in geom_wrong[:n]]

    return cases


def print_summary(results: list[dict], cases: dict[str, list[str]]):
    """Print a human-readable summary."""
    valid_count = sum(1 for r in results if r.get("valid", False))
    total = len(results)

    print(f"\n{'=' * 60}")
    print(f" Evaluation Summary: {valid_count}/{total} valid ({valid_count/total*100:.1f}%)")
    print(f"{'=' * 60}")

    valid_with_cd = [r for r in results if r.get("valid") and "face_cd" in r]
    if valid_with_cd:
        cds = [r["face_cd"] for r in valid_with_cd]
        f1s = [r.get("face_f_score", 0) for r in valid_with_cd]
        print(f" Face CD:  mean={np.mean(cds):.4f}  median={np.median(cds):.4f}")
        print(f" Face F1:  mean={np.mean(f1s):.4f}  median={np.median(f1s):.4f}")

    print(f"\n{'─' * 60}")
    print(f" Selected Cases ({sum(len(v) for v in cases.values())} total):")
    print(f"{'─' * 60}")

    for category, ids in cases.items():
        if not ids:
            continue
        # Show metrics for selected cases
        selected = [r for r in results if r["id"] in ids]
        if selected and "face_cd" in selected[0]:
            avg_cd = np.mean([r.get("face_cd", 0) for r in selected])
            avg_f1 = np.mean([r.get("face_f_score", 0) for r in selected])
            print(f"\n  [{category}] ({len(ids)} cases, avg CD={avg_cd:.4f}, avg F1={avg_f1:.4f})")
        else:
            print(f"\n  [{category}] ({len(ids)} cases)")
        for sample_id in ids:
            r = next((x for x in results if x["id"] == sample_id), {})
            cd = r.get("face_cd", "?")
            f1 = r.get("face_f_score", "?")
            nf = r.get("num_faces_pred", "?")
            nf_gt = r.get("num_faces_gt", "?")
            valid_str = "✓" if r.get("valid") else "✗"
            cd_str = f"{cd:.4f}" if isinstance(cd, float) else cd
            f1_str = f"{f1:.4f}" if isinstance(f1, float) else f1
            print(f"    {valid_str} {sample_id}  CD={cd_str}  F1={f1_str}  faces={nf}/{nf_gt}")


def main():
    parser = argparse.ArgumentParser(description="Select visualization cases from eval results")
    parser.add_argument("--pred-root", required=True, help="Prediction root directory")
    parser.add_argument("--num", type=int, default=5, help="Number of cases per category")
    parser.add_argument("--output", default=None, help="Output JSON path (default: pred_root/selected_cases.json)")
    args = parser.parse_args()

    pred_root = Path(args.pred_root)
    results = load_eval_results(pred_root)

    if not results:
        print(f"No results found in {pred_root}")
        return

    cases = select_cases(results, num_per_category=args.num)
    print_summary(results, cases)

    # Save
    output_path = Path(args.output) if args.output else pred_root / "selected_cases.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Also save with metrics for each case
    detailed = {}
    for category, ids in cases.items():
        detailed[category] = []
        for sample_id in ids:
            r = next((x for x in results if x["id"] == sample_id), {"id": sample_id})
            detailed[category].append(r)

    with open(output_path, "w") as f:
        json.dump(detailed, f, indent=2, default=str)

    print(f"\n  Saved to: {output_path}")


if __name__ == "__main__":
    main()
