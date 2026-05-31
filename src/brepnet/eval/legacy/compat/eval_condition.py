"""Compatibility entry for conditioned BREP evaluation.

新实现见 `src.brepnet.eval.metrics.condition`，统一入口见：

python -m src.brepnet.eval.run --metrics condition ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from src.brepnet.eval.metrics.condition import *  # noqa: F401,F403
from src.brepnet.eval.metrics.condition import (
    compute_statistics,
    eval_one,
    eval_one_with_try,
    evaluate_condition,
    get_chamfer,
    get_data,
    get_detection,
    get_match_ids,
    get_model_normalize,
    get_topo_detection,
    get_topology,
)
from src.brepnet.eval.protocol import condition_result_path, iter_eval_samples

__all__ = [name for name in globals() if not name.startswith("_")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated BREP conditions")
    parser.add_argument("--eval_root", "--pred-root", dest="eval_root", type=str, required=True)
    parser.add_argument("--gt_root", "--gt-root", dest="gt_root", type=str, required=True)
    parser.add_argument("--use_ray", action="store_true")
    parser.add_argument("--num_cpus", type=int, default=16)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--list", type=str, default="")
    parser.add_argument("--from_scratch", action="store_true")
    parser.add_argument("--is_point2cad", action="store_true")
    parser.add_argument("--is_complexgen", action="store_true")
    parser.add_argument("--is_nvdnet", action="store_true")
    parser.add_argument("--only_valid", action="store_true")
    args = parser.parse_args()

    samples = iter_eval_samples(args.eval_root, args.gt_root, split_list=args.list, sample=args.prefix or None)
    jobs = [sample for sample in samples if args.from_scratch or load_condition_result(sample.pred_dir) is None]

    if args.use_ray:
        import ray

        ray.init(ignore_reinit_error=True, num_cpus=args.num_cpus)
        remote_eval = ray.remote(max_retries=0)(eval_one_with_try)
        refs = [
            remote_eval.remote(
                args.eval_root,
                args.gt_root,
                sample.name,
                args.is_point2cad,
                args.is_complexgen,
                args.is_nvdnet,
                100,
                True,
            )
            for sample in jobs
        ]
        for ref in tqdm(refs, desc="Condition"):
            ray.get(ref)
        ray.shutdown()
    else:
        for sample in tqdm(jobs, desc="Condition"):
            eval_one(
                args.eval_root,
                args.gt_root,
                sample.name,
                args.is_point2cad,
                args.is_complexgen,
                args.is_nvdnet,
                write_legacy_eval=True,
            )

    compute_statistics(Path(args.eval_root), args.only_valid, args.list)


if __name__ == "__main__":
    main()
