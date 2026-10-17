#!/usr/bin/env python3
"""Read-only, bounded verification of selected migration assets on both hosts.

Uses the existing inventory for checkpoint links and weights; samples eight
training CAD identities without enumerating the large data directories. Does
not inspect ABC geometry or evaluation results. This is not a full hash audit.
"""

import collections
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess


BASE = Path(__file__).resolve().parent
INVENTORY = BASE.parents[1] / "20261017"
REMOTE_PROBE = '''
import hashlib,json,os,stat,sys
rows=[]
for request in REQUESTS:
    p=request['path']; row={'path':p}
    try:
        s=os.lstat(p)
        row.update(size=s.st_size,mtime_ns=s.st_mtime_ns,
                   is_link=stat.S_ISLNK(s.st_mode),exists=os.path.exists(p))
        if row['is_link']:
            row['link_target']=os.readlink(p)
        if request.get('sha256'):
            h=hashlib.sha256()
            with open(p,'rb') as f:
                for b in iter(lambda:f.read(1024*1024),b''):
                    h.update(b)
            row['sha256']=h.hexdigest()
    except OSError as e:
        row['error']=str(e)
    rows.append(row)
print(json.dumps(rows))
'''


def main():
    manifest = json.loads((BASE / "manifest.json").read_text())
    weights = json.loads((INVENTORY / "weights.json").read_text())
    state = json.loads(Path("/mnt/d/migration-transfer-logs/status.json").read_text())
    requests = {}
    excluded_links = []

    def add(path, category, sha256=False):
        row = requests.setdefault(path, {"path": path, "categories": []})
        if category not in row["categories"]:
            row["categories"].append(category)
        row["sha256"] = row.get("sha256", False) or sha256

    def excluded(path):
        if "/diffusion_layerwise_svr/" in path:
            return "active diffusion deferred"
        if "/browser_profile/" in path or Path(path).name in {
            "SingletonCookie", "SingletonSocket", "SingletonLock"
        }:
            return "browser profile excluded"
        if Path(path).name == "debug-core.log":
            return "global W&B debug cache excluded"
        return None

    for root in weights["roots"]:
        for row in root["symlinks"]:
            reason = excluded(row["path"])
            if reason:
                excluded_links.append({"path": row["path"], "reason": reason})
            else:
                add(row["path"], "checkpoint_symlink")
        for row in root["weights"]:
            if not row["temporary_validation_weight"] and not excluded(row["path"]):
                add(row["path"], "stable_checkpoint_metadata")

    for asset in manifest["queue"]:
        if state["assets"].get(asset["id"], {}).get("status") == "completed":
            for path in asset["paths"]:
                add(path, "completed_asset_root")

    dino = Path("/root/.cache/huggingface/hub/models--facebook--dinov2-large")
    for path in dino.rglob("*"):
        if path.is_symlink():
            add(str(path), "dino_symlink")

    split = Path("/mnt/d/python/src/brepnet/data/list/deduplicated_deepcad_training_7_30.txt")
    ids = random.Random(20261017).sample(split.read_text().splitlines(), 8)
    for model_id in ids:
        for name in ("data.npz", "normalized_shape.step", "mesh.ply"):
            add(f"/mnt/d/data/deepcad_v7/{model_id}/{name}", "training_sample", True)
        for name in ("imgs.npz", "real_photo.npz", "text.npz", "pc.ply",
                     "rotation_meta.json", "img_feature_dinov2.npy"):
            add(f"/mnt/d/data/deepcad_v7_cond/{model_id}/{name}", "training_sample", True)
        for rotation in (0, 23):
            add(f"/mnt/d/data/ae_cache/1119_deepcad_aug1_11k_24/{model_id}_{rotation}/features.npy",
                "training_sample", True)
            add(f"/mnt/d/data/deepcad_v7_visibility/{model_id}/view_{rotation:03d}/visibility.npz",
                "training_sample", True)

    # Additional stable weight hash complements the previously verified DeepCAD AE.
    add("/mnt/d/data/ae_checkpoints/1225_abc_aug1.ckpt", "ae_abc_weight", True)
    requests = list(requests.values())
    script = "REQUESTS=" + repr(requests) + "\n" + REMOTE_PROBE
    local = json.loads(subprocess.check_output(["python", "-c", script], text=True))
    command = ["ssh", "-S", manifest["control_path"], "-o", "BatchMode=yes",
               "-p", str(manifest["ssh_port"]), "root@172.31.224.142", "python3 -"]
    remote = json.loads(subprocess.check_output(command, input=script, text=True))
    results = []
    for request, source, target in zip(requests, local, remote):
        issue = []
        if source.get("error"):
            issue.append("source_error")
        if target.get("error"):
            issue.append("target_error")
        if source.get("is_link") != target.get("is_link"):
            issue.append("link_type_mismatch")
        if source.get("is_link"):
            if source.get("link_target") != target.get("link_target"):
                issue.append("link_target_mismatch")
            if source.get("exists") and not target.get("exists"):
                issue.append("new_broken_link")
        elif not Path(request["path"]).is_dir():
            if source.get("size") != target.get("size"):
                issue.append("size_mismatch")
        if request["sha256"] and source.get("sha256") != target.get("sha256"):
            issue.append("sha256_mismatch")
        results.append({**request, "source": source, "target": target, "issues": issue})
    counts = collections.Counter(category for r in requests for category in r["categories"])
    selected_roots = [p.rstrip("/") for asset in manifest["queue"] for p in asset["paths"]]
    links = []
    for row in results:
        if row["source"].get("is_link"):
            resolved = os.path.realpath(row["path"])
            covered = [p for p in selected_roots if resolved == p or resolved.startswith(p + "/")]
            links.append({"path": row["path"], "resolved_target": resolved,
                          "in_selected_root": bool(covered),
                          "matched_root": max(covered, key=len) if covered else None})
    report = {
        "checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "Inventory links and stable weights; completed roots; 8 training IDs, rotations 0/23; ABC AE hash only, no ABC geometry inspection",
        "limits": "Not a full-file integrity audit; retained pre-existing missing data is not repaired; successful rsync groups are the transfer-completion evidence.",
        "training_sample_seed": 20261017,
        "training_sample_ids": ids,
        "training_split_sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
        "counts": dict(counts),
        "request_count": len(requests),
        "issue_count": sum(bool(r["issues"]) for r in results),
        "symlink_closure": {"count": len(links),
                            "outside_selected_roots": [r for r in links if not r["in_selected_root"]],
                            "results": links},
        "excluded_links": excluded_links,
        "results": results,
    }
    output = BASE / "asset_verification.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"report": str(output), "counts": dict(counts),
                      "request_count": len(requests), "issue_count": report["issue_count"],
                      "issues": [{"path": r["path"], "issues": r["issues"]}
                                 for r in results if r["issues"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
