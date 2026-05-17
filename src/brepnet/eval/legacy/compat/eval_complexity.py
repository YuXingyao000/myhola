"""Compatibility entry for complexity evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.brepnet.eval.io import flatten_scalar_result, save_npz_result, write_csv, write_json
from src.brepnet.eval.metrics.complexity import evaluate_sample, evaluate_step_file
from src.brepnet.eval.protocol import iter_eval_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BREP complexity")
    parser.add_argument("--eval_root", type=str, required=True)
    parser.add_argument("--use_ray", action="store_true")
    parser.add_argument("--num_cpus", type=int, default=16)
    args = parser.parse_args()

    samples = iter_eval_samples(args.eval_root)
    rows = []
    for sample in samples:
        result = evaluate_sample(sample)
        if result is None:
            continue
        save_npz_result(sample.pred_dir / "eval_complexity.npz", result)
        rows.append(flatten_scalar_result(sample.name, result))
    summary = {"num_eval": len(rows), "num_total": len(samples)}
    for key in ("num_faces", "num_edges", "num_vertices", "cyclomatic_complexity", "mean_curvature"):
        values = [row[key] for row in rows if key in row]
        if values:
            summary[f"{key}_mean"] = float(sum(values) / len(values))
    out_root = Path(args.eval_root)
    write_csv(out_root / "eval_complexity.csv", rows)
    write_json(out_root / "eval_complexity_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
