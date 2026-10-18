"""冻结旧无条件模型的采样清单，并调度独立推理与后处理进程。"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[2]
CHECKPOINT = Path("/mnt/d/ckpt/HoLa/Diffusion_uncond_1100k.ckpt")
SAMPLE_COUNT = 1000
SEED_PREFIX = "hola-uncond-20261018-v1:"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sample_rows() -> list[dict]:
    """清单只由编号决定，不根据是否生成成功筛选。"""
    rows = []
    for index in range(SAMPLE_COUNT):
        sample_id = f"uncond_{index:06d}"
        seed = int.from_bytes(hashlib.sha256((SEED_PREFIX + sample_id).encode()).digest()[:4], "little")
        rows.append({"sample_id": sample_id, "seed": seed})
    return rows


def prepare(root: Path, checkpoint: Path) -> dict:
    """保存完整权重身份；已存在协议必须逐项相同才允许继续。"""
    import torch

    root.mkdir(parents=True, exist_ok=True)
    # 新输出目录也保存同一份协议，后续阶段直接核验该副本。
    plan_path = root / "PLAN.md"
    plan_bytes = (ROOT / "PLAN.md").read_bytes()
    if plan_path.exists() and plan_path.read_bytes() != plan_bytes:
        raise ValueError("Output directory contains a different PLAN.md")
    if not plan_path.exists():
        plan_path.write_bytes(plan_bytes)
    report = root / "report"
    report.mkdir(exist_ok=True)
    samples_path = report / "seeds.csv"
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=["sample_id", "seed"], lineterminator="\n")
    writer.writeheader()
    writer.writerows(sample_rows())
    sample_bytes = stream.getvalue().encode()
    if samples_path.exists() and samples_path.read_bytes() != sample_bytes:
        raise ValueError("Frozen seed list changed; use a new experiment directory")
    samples_path.write_bytes(sample_bytes)

    saved = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    manifest = {
        "schema_version": 1,
        "checkpoint": {
            "path": str(checkpoint), "sha256": sha256_file(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "global_step": int(saved["global_step"]), "epoch": int(saved["epoch"]),
        },
        "sample_count": SAMPLE_COUNT,
        "seed_prefix": SEED_PREFIX,
        "samples_sha256": hashlib.sha256(sample_bytes).hexdigest(),
        "plan_sha256": sha256_file(plan_path),
        "sampling": {
            "batch_size": 1, "steps": 1000, "dtype": "float32", "tf32": False,
            "initial_noise_device": "cpu", "prediction_type": "epsilon",
            "beta_schedule": "linear", "proposal_count_per_seed": 1,
        },
    }
    target = root / "metadata" / "manifest.json"
    if target.exists() and json.loads(target.read_text()) != manifest:
        raise ValueError("Checkpoint or protocol differs from the frozen manifest")
    atomic_json(target, manifest)
    atomic_json(report / "protocol.json", manifest)
    # 记录源机状态；不读取或提交其他研究目录里的未提交内容。
    provenance_path = root / "metadata" / "orchestration.json"
    if not provenance_path.exists():
        atomic_json(provenance_path, {
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--short"], cwd=REPOSITORY, text=True),
            "command": sys.argv,
            "python": sys.executable,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        })
    with (root / "metadata" / "commands.jsonl").open("a") as log:
        log.write(json.dumps({
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "argv": [sys.executable, *sys.argv],
            "orchestrator_sha256": sha256_file(Path(__file__)),
        }, ensure_ascii=False) + "\n")
    print(json.dumps({"phase": "prepared", "samples": SAMPLE_COUNT,
                      "checkpoint_step": manifest["checkpoint"]["global_step"],
                      "checkpoint_sha256": manifest["checkpoint"]["sha256"]}), flush=True)
    return manifest


def inference(root: Path, checkpoint: Path, gpus: list[str], limit: int | None) -> None:
    """每GPU仅一个进程；等所有已启动worker退出后再检查失败。"""
    logs = root / "metadata" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    workers = []
    for rank, gpu in enumerate(gpus):
        command = [sys.executable, str(ROOT / "infer.py"), "--root", str(root),
                   "--checkpoint", str(checkpoint), "--rank", str(rank), "--shards", str(len(gpus))]
        if limit is not None:
            command.extend(["--limit", str(limit)])
        handle = (logs / f"inference_gpu{gpu}.log").open("a")
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
                           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        process = subprocess.Popen(command, cwd=REPOSITORY, env=environment,
                                   stdout=handle, stderr=subprocess.STDOUT)
        workers.append((process, handle))
    codes = []
    for process, handle in workers:
        codes.append(process.wait())
        handle.close()
    if any(codes):
        raise RuntimeError(f"Inference workers exited with {codes}; inspect {logs}")
    print(json.dumps({"phase": "inference_terminal", "limit": limit, "gpus": gpus}), flush=True)


def evaluation(root: Path, phase: str, limit: int | None) -> None:
    command = [sys.executable, str(ROOT / "evaluate.py"), "--root", str(root), "--phase", phase]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    subprocess.run(command, cwd=REPOSITORY, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "infer", "post", "summarize", "all"), default="all")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--gpus", nargs="+", default=[str(i) for i in range(8)])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if len(args.gpus) != len(set(args.gpus)):
        parser.error("GPU IDs must be unique")
    if args.limit is not None and not 1 <= args.limit <= SAMPLE_COUNT:
        parser.error(f"limit must be between 1 and {SAMPLE_COUNT}")
    root, checkpoint = args.root.resolve(), args.checkpoint.resolve()
    prepare(root, checkpoint)
    if args.phase in {"infer", "all"}:
        inference(root, checkpoint, args.gpus, args.limit)
    if args.phase in {"post", "summarize", "all"}:
        evaluation(root, "all" if args.phase == "all" else args.phase, args.limit)


if __name__ == "__main__":
    main()
