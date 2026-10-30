"""固定 1000 个 seed 的无条件推理；原生采样、逐件溯源、失败不补样。"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[2]
sys.path.insert(0, str(REPOSITORY))
SAMPLE_COUNT = 1000
SEED_PREFIX = "hola-uncond-20261018-v1:"
SAMPLING = {
    "batch_size": 1, "steps": 1000, "dtype": "float32", "tf32": False,
    "initial_noise_device": "cpu", "prediction_type": "epsilon", "beta_schedule": "linear",
    "proposal_count_per_seed": 1,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def sample_seed(sample_id: str) -> int:
    return int.from_bytes(hashlib.sha256((SEED_PREFIX + sample_id).encode()).digest()[:4], "little")


def sample_ids(limit: int | None = None) -> list[str]:
    if limit is not None and not 1 <= limit <= SAMPLE_COUNT:
        raise ValueError("--limit must be between 1 and 1000; it selects the frozen list prefix.")
    return [f"uncond_{index:06d}" for index in range(SAMPLE_COUNT if limit is None else limit)]


def prepare_identity(root: Path, manifest_path: Path, checkpoint: Path) -> tuple[dict, dict]:
    """完整验证权重与协议；来源不同的结果不能混在一次比较中。"""
    from legacy_runtime import SHARED_RUNTIME, SOURCE_COMMIT

    manifest = json.loads(manifest_path.read_text())
    if (manifest["schema_version"] != 1 or manifest["sample_count"] != SAMPLE_COUNT
            or manifest["seed_prefix"] != SEED_PREFIX):
        raise ValueError("Manifest differs from the frozen 1000-sample protocol.")
    record = manifest["checkpoint"]
    if (Path(record["path"]).resolve() != checkpoint.resolve()
            or int(record["global_step"]) != 1_000_000 or int(record["epoch"]) != 324
            or checkpoint.stat().st_size != record["bytes"] or sha256(checkpoint) != record["sha256"]):
        raise ValueError("Checkpoint identity differs from the prepared manifest.")
    plan_hash = sha256(root / "PLAN.md")
    if plan_hash != manifest["plan_sha256"]:
        raise ValueError("PLAN.md changed after preparing the frozen manifest.")
    if "sampling" in manifest and manifest["sampling"] != SAMPLING:
        raise ValueError("Manifest sampling settings differ from the native protocol.")
    identity = {
        "manifest_sha256": sha256(manifest_path), "checkpoint_sha256": record["sha256"],
        "plan_sha256": plan_hash, "infer_source_sha256": sha256(Path(__file__)),
        "runtime_source_sha256": sha256(ROOT / "legacy_runtime.py"),
        "shared_runtime_source_sha256": sha256(SHARED_RUNTIME),
        "source_commit": SOURCE_COMMIT, "sampling": SAMPLING,
    }
    protocol_path = root / "metadata/inference/protocol.json"
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol = {"schema_version": 1, "identity": identity}
    with protocol_path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if protocol_path.exists():
            if json.loads(protocol_path.read_text()) != protocol:
                raise ValueError("Existing inference protocol differs; preserve results and use a new run.")
        else:
            write_json(protocol_path, protocol)
    return manifest, identity


def pending_samples(root: Path, selected: list[str], identity: dict) -> list[str]:
    """已有成功验证 hash；未完成标记 interrupted，失败不会自动重采样。"""
    pending = []
    for sample_id in selected:
        status_path = root / "metadata/inference/samples" / f"{sample_id}.json"
        raw = root / "raw_predictions" / sample_id / "data.npz"
        if not status_path.exists():
            if raw.exists():
                raise ValueError(f"Untracked raw prediction requires an explicit recovery audit: {raw}")
            pending.append(sample_id)
            continue
        record = json.loads(status_path.read_text())
        if (record.get("identity") != identity or record.get("sample_id") != sample_id
                or record.get("seed") != sample_seed(sample_id)):
            raise ValueError(f"Existing sample has different provenance: {status_path}")
        if record["status"] == "running":
            history = root / "metadata/inference/interrupted" / f"{sample_id}_{time.time_ns()}.json"
            write_json(history, record)
            record.update(status="interrupted", finished_at_utc=utc_now(),
                          error_type="InterruptedRun", error="Previous invocation did not finish; no automatic retry.")
            if raw.exists():
                record.update(output_path=str(raw.resolve()), output_sha256=sha256(raw))
            write_json(status_path, record)
        elif record["status"] not in {"ok", "error", "interrupted"}:
            raise ValueError(f"Unknown saved inference status: {status_path}")
        if record["status"] == "ok" and not record.get("output_sha256"):
            raise ValueError(f"Successful record has no raw hash: {status_path}")
        if record.get("output_sha256"):
            if not raw.is_file() or sha256(raw) != record["output_sha256"]:
                raise ValueError(f"Saved raw output changed: {raw}")
        elif raw.exists():
            raise ValueError(f"Raw output has no recorded hash: {raw}")
    return pending


@contextmanager
def capture_initial_noise(torch, expected_shape: tuple[int, ...]):
    """只记录原生首次 CPU randn 的结果，不额外消耗随机数。"""
    original = torch.randn
    captured = {}

    def recording_randn(*args, **kwargs):
        value = original(*args, **kwargs)
        if not captured:
            if tuple(value.shape) != expected_shape or value.device.type != "cpu" or value.dtype != torch.float32:
                raise ValueError("Native initial noise no longer matches CPU FP32 [1,30,32].")
            captured.update(sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest(),
                            shape=list(value.shape), dtype="float32", device="cpu")
        return value

    try:
        torch.randn = recording_randn
        yield captured
    finally:
        torch.randn = original


def save_prediction(root: Path, sample_id: str, prediction: dict, record: dict, np, torch) -> None:
    """原样保存所有数组；整数 connectivity 适配留给独立后处理入口。"""
    arrays = {key: value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
              for key, value in prediction.items()}
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError("Native output contains an unexpected object array.")
    raw = root / "raw_predictions" / sample_id / "data.npz"
    raw.parent.mkdir(parents=True, exist_ok=True)
    if raw.exists():
        raise FileExistsError(f"Refusing to overwrite raw prediction: {raw}")
    temporary = raw.with_name(f".data.{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(raw)
    record.update(output_path=str(raw.resolve()), output_sha256=sha256(raw),
                  shapes={key: list(value.shape) for key, value in arrays.items()},
                  dtypes={key: str(value.dtype) for key, value in arrays.items()})
    for key in ("pred_face", "pred_edge", "pred_edge_face_connectivity", "face_features"):
        if key not in arrays or not np.isfinite(arrays[key]).all():
            raise ValueError(f"Missing/nonfinite native prediction: {key}")
    faces, edges, connectivity = (arrays[key] for key in ("pred_face", "pred_edge", "pred_edge_face_connectivity"))
    record.update(raw_face_count=len(faces), raw_directed_edge_count=len(edges))
    if faces.ndim != 4 or faces.shape[1:] != (16, 16, 3) or not len(faces):
        raise ValueError(f"Empty or malformed face geometry: {faces.shape}")
    if edges.ndim != 3 or edges.shape[1:] != (16, 3) or not len(edges):
        raise ValueError(f"Empty or malformed edge geometry: {edges.shape}")
    if connectivity.shape != (len(edges), 3):
        raise ValueError(f"Malformed native connectivity: {connectivity.shape}")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--checkpoint", type=Path, default=Path("/mnt/d/ckpt/HoLa/Diffusion_uncond_1100k.ckpt"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--gpu", help="Physical GPU ID; otherwise inherit CUDA_VISIBLE_DEVICES and use cuda:0.")
    parser.add_argument("--limit", type=int, help="Select the first N frozen IDs before applying rank/shards.")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.shards < 1 or not 0 <= args.rank < args.shards:
        raise ValueError("Require 0 <= rank < shards.")
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    root = args.root.resolve()
    manifest_path = (args.manifest or root / "metadata/manifest.json").resolve()
    selected = sample_ids(args.limit)[args.rank::args.shards]
    _, identity = prepare_identity(root, manifest_path, args.checkpoint)
    # 按样本加锁，改变 shards/limit 也不能误判另一个活动 worker 已中断。
    lock_root = root / "metadata/inference/locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    with ExitStack() as locks:
        for sample_id in selected:
            lock = locks.enter_context((lock_root / f"{sample_id}.lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_worker(args, root, selected, identity, np, torch)


def run_worker(args, root: Path, selected: list[str], identity: dict, np, torch) -> None:
    from legacy_runtime import load_legacy_model

    pending = pending_samples(root, selected, identity)
    if not pending:
        print(f"unconditional shard {args.rank}: all selected records already terminal", flush=True)
        return
    worker_path = root / "metadata/inference/workers" / f"rank{args.rank:03d}_{time.time_ns()}.json"
    worker = {"schema_version": 1, "identity": identity, "rank": args.rank, "shards": args.shards,
              "selected_count": len(selected), "pending_count": len(pending), "started_at_utc": utc_now(),
              "status": "loading", "python": sys.version, "torch": torch.__version__, "numpy": np.__version__}
    write_json(worker_path, worker)
    load_started = time.perf_counter()
    try:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        model, provenance = load_legacy_model(args.checkpoint, archive_root=root / "metadata/legacy_source")
        model = model.float().to(device).eval().requires_grad_(False)
        if len(model.noise_scheduler.timesteps) != 1000:
            raise ValueError("The native scheduler must contain exactly 1000 sampling steps.")
        torch.cuda.synchronize(device)
        worker.update(status="sampling", gpu=torch.cuda.get_device_name(device),
                      load_seconds=time.perf_counter() - load_started, provenance=provenance)
        write_json(worker_path, worker)
    except Exception as error:
        failure = {"error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()}
        worker.update(status="error", finished_at_utc=utc_now(), **failure)
        write_json(worker_path, worker)
        for sample_id in pending:
            write_json(root / "metadata/inference/samples" / f"{sample_id}.json", {
                "schema_version": 1, "sample_id": sample_id, "seed": sample_seed(sample_id),
                "identity": identity, "status": "error", "stage": "model_load",
                "started_at_utc": worker["started_at_utc"], "finished_at_utc": utc_now(),
                "inference_seconds": None, "total_seconds": None, **failure,
            })
        raise

    for sample_id in pending:
        started = time.perf_counter()
        seed = sample_seed(sample_id)
        status_path = root / "metadata/inference/samples" / f"{sample_id}.json"
        record = {"schema_version": 1, "sample_id": sample_id, "seed": seed, "identity": identity,
                  "status": "running", "stage": "sampling", "started_at_utc": utc_now(),
                  "inference_seconds": None, "total_seconds": None}
        write_json(status_path, record)
        noise = {}
        try:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.cuda.synchronize(device)
            inference_started = time.perf_counter()
            with capture_initial_noise(torch, (1, 30, 32)) as noise, torch.inference_mode():
                prediction = model.inference(1, device, v_data=None, v_log=False)[0]
            torch.cuda.synchronize(device)
            record["inference_seconds"] = time.perf_counter() - inference_started
            if not noise:
                raise RuntimeError("The native inference did not produce the expected CPU initial noise.")
            record["stage"] = "export"
            save_prediction(root, sample_id, prediction, record, np, torch)
            record.update(status="ok", stage="complete")
        except Exception as error:
            record.update(status="error", error_type=type(error).__name__, error=str(error),
                          traceback=traceback.format_exc())
        record.update(initial_noise=noise or None, finished_at_utc=utc_now(),
                      total_seconds=time.perf_counter() - started)
        write_json(status_path, record)
        print(json.dumps({key: record.get(key) for key in
                          ("sample_id", "seed", "status", "inference_seconds", "raw_face_count", "error")}), flush=True)
    worker.update(status="complete", finished_at_utc=utc_now())
    write_json(worker_path, worker)


if __name__ == "__main__":
    main()
