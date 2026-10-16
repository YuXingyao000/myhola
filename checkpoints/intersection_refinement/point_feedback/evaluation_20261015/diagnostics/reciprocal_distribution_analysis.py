"""Reuse audited reciprocal statistics on this run's explicit paired methods."""
import importlib.util
from raw_distribution_analysis import SOURCE

spec = importlib.util.spec_from_file_location(
    "historical_reciprocal_distribution", SOURCE / "reciprocal_distribution_analysis.py",
)
implementation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(implementation)
historical_report = implementation.report


def report(*args):
    return historical_report(*args).replace("不能据此归因于有序 conditioning", "不能据此单独归因于新增逐点反馈")


implementation.report = report

if __name__ == "__main__":
    implementation.main()
