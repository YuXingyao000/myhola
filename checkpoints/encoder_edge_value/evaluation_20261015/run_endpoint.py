"""Evaluate the predeclared final checkpoint separately from the matched-step study."""
from pathlib import Path
import run_evaluation as evaluation


if __name__ == "__main__":
    evaluation.ROOT = Path(__file__).resolve().parent / "endpoint_14520"
    evaluation.ORCHESTRATION = evaluation.ROOT / "metadata/orchestration"
    evaluation.SELECTIONS = {
        "candidate": {
            "method": "encoder_edge_value",
            "checkpoint": evaluation.REPO / (
                "checkpoints/encoder_edge_value/candidate/"
                "20261015T073112.923985Z/weights/evaluated_snapshots/"
                "e9120c78254d9b94aaaef88333c0f0656fd6a01346225d092fb8341297bff2bf.ckpt"
            ),
            "sha256": "e9120c78254d9b94aaaef88333c0f0656fd6a01346225d092fb8341297bff2bf",
        }
    }
    evaluation.main()
