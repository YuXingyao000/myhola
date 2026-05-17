"""Compatibility entry for validity evaluation.

新实现见 `src.brepnet.eval.metrics.validity`。保留这个文件是为了兼容旧命令：

python -m src.brepnet.eval.legacy.compat.check_valid --data_root PRED_ROOT
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.brepnet.eval.io import flatten_scalar_result, save_npz_result, write_csv, write_json
from src.brepnet.eval.metrics.validity import (
    check_step_valid_soild,
    check_step_valid_solid,
    evaluate_sample,
    load_data_with_prefix,
    save_step_file,
)
from src.brepnet.eval.protocol import EvalSample, iter_eval_samples

__all__ = [name for name in globals() if not name.startswith("_")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check STEP validity")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--only_success", action="store_true", default=False)
    args = parser.parse_args()

    samples = iter_eval_samples(args.data_root, sample=args.prefix or None)
    rows = []
    for sample in samples:
        result = evaluate_sample(sample)
        if args.only_success and not result["has_success_marker"]:
            continue
        save_npz_result(sample.pred_dir / "eval_validity.npz", result)
        rows.append(flatten_scalar_result(sample.name, result))

    total = len(samples)
    valid = sum(1 for row in rows if row.get("is_valid_solid"))
    summary = {
        "num_samples": total,
        "num_checked": len(rows),
        "num_valid_solid": valid,
        "valid_rate": valid / total if total else 0.0,
    }
    out_root = Path(args.data_root)
    write_csv(out_root / "eval_validity.csv", rows)
    write_json(out_root / "eval_validity_summary.json", summary)

    try:
        import matplotlib.pyplot as plt

        num_faces = [row["num_faces"] for row in rows if row.get("is_valid_solid")]
        num_edges = [row["num_edges"] for row in rows if row.get("is_valid_solid")]
        if num_faces:
            report_dir = out_root / "reports"
            report_dir.mkdir(exist_ok=True)
            fig, ax = plt.subplots(1, 2, layout="constrained")
            ax[0].set_title("Num. faces")
            ax[1].set_title("Num. edges")
            ax[0].hist(num_faces, bins=5, range=(0, max(30, int(np.max(num_faces)))))
            ax[1].hist(num_edges, bins=5, range=(0, max(50, int(np.max(num_edges)))))
            fig.savefig(report_dir / "validity_faces_edges.png", dpi=600)
            plt.close(fig)
    except Exception:
        pass

    print(f"Number of valid CAD solids: {valid}")
    print(f"Valid rate: {summary['valid_rate'] * 100:.2f}%")


if __name__ == "__main__":
    main()
