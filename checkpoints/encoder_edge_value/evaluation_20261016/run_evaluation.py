"""Frozen continuation snapshots; reuse the validated 2026-10-15 evaluator."""
from pathlib import Path
import argparse
import importlib.util


ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "evaluation_20261015/run_evaluation.py"
spec = importlib.util.spec_from_file_location("previous_evaluation", PREVIOUS)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)
evaluation.ROOT = ROOT
evaluation.ORCHESTRATION = ROOT / "metadata/orchestration"
ARCHIVES = evaluation.REPO / "checkpoints/encoder_edge_value/candidate/20261015T073112.923985Z/weights/evaluated_snapshots"
SELECTIONS = {
    "last": "040c05a45dd7edae825f2e6dd8e2cb3489a418146f9a7b2122cc54d38070618c",
    "resume_best": "e1e579e7bc6bdf64b3c078d6b5414e33630222de989b180ca8f3eb22b59c6ccf",
}
evaluation.SELECTIONS = {
    name: {"method": "encoder_edge_value", "checkpoint": ARCHIVES / (digest + ".ckpt"), "sha256": digest}
    for name, digest in SELECTIONS.items()
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("smoke", "inference", "metrics", "post"))
    parser.add_argument("--selection", choices=tuple(SELECTIONS), required=True)
    parser.add_argument("--post-mode", choices=("both", "fit_only", "optimized"), default="both")
    args = parser.parse_args()
    name = args.selection
    if args.phase in {"smoke", "inference"}:
        evaluation.run_jobs(f"{args.phase}-{name}", evaluation.inference(name, smoke=args.phase == "smoke"))
        return
    pair_root = evaluation.prepare(name)
    common = ["--baseline-method", "conditioned", "--candidate-method", "encoder_edge_value",
              "--comparison-root", str(pair_root), "--data-root", "/mnt/d/data/deepcad_v7",
              "--split-list", str(evaluation.SPLIT)]
    if args.phase == "metrics":
        commands = [(f"metrics-{name}", [evaluation.PYTHON, "-m", "tools.vae_compare_metrics", *common,
                     "--mode", "posterior_mean", "--input-contract", "same_geometry_source", "--workers", "16"])]
    else:
        modes = ("fit_only", "optimized") if args.post_mode == "both" else (args.post_mode,)
        commands = [(f"post-{name}-{mode}", [evaluation.PYTHON, "-m", "tools.vae_compare_post", *common,
                     "--methods", "encoder_edge_value", "--reference-root", str(evaluation.REFERENCE),
                     "--mode", mode, "--workers", "32", "--timeout", "180",
                     "--max-optimize-iter", "200", "--original-test-count", "2424"]) for mode in modes]
    label = f"{args.phase}-{name}" + (f"-{args.post_mode}" if args.phase == "post" else "")
    evaluation.run_jobs(label, commands)


if __name__ == "__main__":
    main()
