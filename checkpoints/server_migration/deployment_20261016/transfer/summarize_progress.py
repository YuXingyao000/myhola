#!/usr/bin/env python3
"""Summarize migration state and completed rsync tails without scanning assets.

Optional watch mode writes a small external summary and only logs meaningful
state changes. It never changes the transfer queue or sends external messages.
"""

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import re
import time


BASE = Path(__file__).resolve().parent
RUNTIME = Path("/mnt/d/migration-transfer-logs")
FIELDS = {
    "Number of files": "listed_items",
    "Number of regular files transferred": "regular_files_transferred",
    "Total file size": "listed_logical_bytes",
    "Total transferred file size": "transferred_logical_bytes",
    "Total bytes sent": "rsync_sent_bytes",
    "Total bytes received": "rsync_received_bytes",
}


def tail_stats(log):
    with Path(log).open("rb") as stream:
        stream.seek(max(0, os.fstat(stream.fileno()).st_size - 8192))
        tail = stream.read().decode("utf-8", errors="replace")
    result = {}
    for label, key in FIELDS.items():
        matches = re.findall(r"^" + re.escape(label) + r": ([\d,]+)", tail, re.M)
        if matches:
            result[key] = int(matches[-1].replace(",", ""))
    return result


def summarize():
    state = json.loads((RUNTIME / "status.json").read_text())
    manifest = json.loads((BASE / "manifest.json").read_text())
    groups = []
    for asset in manifest["queue"]:
        entry = state["assets"].get(asset["id"], {})
        row = {"id": asset["id"], "status": entry.get("status", "pending")}
        for key in ("attempt", "started_at_utc", "finished_at_utc", "elapsed_seconds", "exit_code"):
            if key in entry:
                row[key] = entry[key]
        if row["status"] == "completed":
            row.update(tail_stats(entry["log"]))
        elif entry.get("progress"):
            row["current_attempt_logical_bytes"] = entry["progress"]["transferred_bytes"]
            row["progress_observed_at_utc"] = entry["progress"]["observed_at_utc"]
        groups.append(row)
    now = dt.datetime.now(dt.timezone.utc)
    age = (now - dt.datetime.fromisoformat(state["updated_at_utc"])).total_seconds()
    completed = [row for row in groups if row["status"] == "completed"]
    current = next((row for row in groups if row["id"] == state.get("current_asset")), None)
    return {
        "observed_at_utc": now.isoformat(),
        "state_updated_at_utc": state["updated_at_utc"],
        "state_age_seconds": round(age, 1),
        "state_stale": state["status"] == "running" and age > 180,
        "overall_status": state["status"],
        "current_asset": state.get("current_asset"),
        "completed_groups": len(completed),
        "total_groups": len(groups),
        "failed_groups": [row["id"] for row in groups if row["status"] == "failed"],
        "completed_groups_transferred_logical_bytes": sum(row.get("transferred_logical_bytes", 0) for row in completed),
        "completed_groups_rsync_sent_bytes": sum(row.get("rsync_sent_bytes", 0) for row in completed),
        "current_attempt_logical_bytes": current.get("current_attempt_logical_bytes", 0) if current else 0,
        "accounting": "Rsync file-content transfer bytes, not allocated disk space. Prior development samples are skipped by later full groups; copies in distinct worktrees remain distinct. Retry history and SSH framing are not included in the completed-attempt wire counters. Current transfer bytes are uncompressed logical bytes, not wire speed. Completion records transfer status, not a full independent checksum audit.",
        "groups": groups,
    }


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RUNTIME / "verification_summary.json")
    parser.add_argument("--watch", type=float, metavar="SECONDS")
    args = parser.parse_args()
    if args.watch is not None and args.watch < 10:
        parser.error("watch interval must be at least 10 seconds")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / "verification_monitor.lock").open("w") as lock:
        if args.watch is not None:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            (RUNTIME / "verification_monitor.pid").write_text(str(os.getpid()) + "\n")
        previous = None
        while True:
            report = summarize()
            save(args.output, report)
            signature = (report["overall_status"], report["current_asset"], report["state_stale"],
                         tuple((r["id"], r["status"], r.get("attempt")) for r in report["groups"]))
            if signature != previous:
                event = {key: report[key] for key in (
                    "observed_at_utc", "overall_status", "current_asset", "completed_groups",
                    "total_groups", "failed_groups", "state_stale",
                    "completed_groups_transferred_logical_bytes", "current_attempt_logical_bytes")}
                print(json.dumps(event, ensure_ascii=False), flush=True)
                if args.watch is not None:
                    with (RUNTIME / "verification_events.jsonl").open("a") as stream:
                        stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            previous = signature
            if args.watch is None or report["overall_status"] in {"completed", "needs_attention", "interrupted"}:
                return
            time.sleep(args.watch)


if __name__ == "__main__":
    main()
