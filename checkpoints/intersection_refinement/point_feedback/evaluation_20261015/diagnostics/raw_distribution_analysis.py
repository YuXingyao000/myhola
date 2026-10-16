"""Explicitly reuse the audited statistics with new method identities and paths."""
from pathlib import Path
import hashlib
import importlib.util
import json

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "diagnostics"
SOURCE = ROOT.parents[1] / "curve_conditioning/evaluation_20261014/diagnostics"
spec = importlib.util.spec_from_file_location("historical_raw_distribution", SOURCE / "raw_distribution_analysis.py")
implementation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(implementation)
implementation.ROOT = ROOT
implementation.OUT = OUT
implementation.METHODS = ("conditioned", "point_feedback")
implementation.LABELS = {"conditioned": "上一版曲线条件调制", "point_feedback": "本版逐点几何反馈"}
historical_report = implementation.build_report


def build_report(*args):
    return historical_report(*args).replace(
        "# 曲线条件调制：raw 分布与同 ID 变化", "# 逐点几何反馈：raw 分布与同 ID 变化", 1,
    )


implementation.build_report = build_report
# The reciprocal reader imports exactly these established numerical helpers.
METHODS = implementation.METHODS
SPLIT, SPLIT_SHA = implementation.SPLIT, implementation.SPLIT_SHA
digest, stats, value = implementation.digest, implementation.stats, implementation.value
paired, verify_stats, number = implementation.paired, implementation.verify_stats, implementation.number


if __name__ == "__main__":
    paths = [Path(__file__), SOURCE / "raw_distribution_analysis.py", SOURCE / "reciprocal_distribution_analysis.py"]
    provenance = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    with (OUT / "reuse_provenance.json").open("x") as handle:
        json.dump({"methods": METHODS, "root": str(ROOT), "sources_sha256": provenance,
                   "changes": "Explicit output root, method identities and Chinese labels only; statistical functions reused unchanged."}, handle, indent=2)
    implementation.main()
