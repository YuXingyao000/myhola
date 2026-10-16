"""Keep aggregate results in Git; leave repeated per-object lists in local diagnostics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def compact(value):
    if isinstance(value, list) and len(value) > 30:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
        return {
            "omitted_local_list_count": len(value),
            "canonical_json_sha256": hashlib.sha256(encoded).hexdigest(),
        }
    if isinstance(value, dict):
        return {key: compact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [compact(item) for item in value]
    return value


def main():
    source = ROOT / "diagnostics.json"
    source_bytes = source.read_bytes()
    report = {
        "schema": "encoder_resumed_evaluation_compact_v1",
        "source": {
            "path": str(source),
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
        },
        "policy": "Lists longer than 30 items become count and canonical-JSON SHA256; all other values are copied exactly. Full lists remain in local diagnostics.json. No metric is recomputed.",
        "results": compact(json.loads(source_bytes)),
    }
    destination = ROOT / "compact_summary.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {destination.name}: {destination.stat().st_size} bytes")


if __name__ == "__main__":
    main()
