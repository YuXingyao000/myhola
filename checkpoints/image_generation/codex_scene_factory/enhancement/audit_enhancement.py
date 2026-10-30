"""CPU 独立验收 Step 4 的磁盘产物；不导入生产代码、不调用模型。

逐项核对冻结输入、实际 PNG 和合成结果。耗时/显存属于运行时仪表记录，
这里只检查其一致性，不把这些记录或背景像素变化当作真实感证明。
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, distance_transform_edt


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_png(path, mode):
    with Image.open(path) as image:
        assert image.format == "PNG" and image.size == (512, 512) and image.mode == mode, str(path)
        return np.asarray(image).copy()


def audit_job(root, request, sample, job, record):
    """检查一个预定任务；失败由外层保留在完整分母中。"""
    assert record["status"] == "success", record.get("error", "pipeline_failed")
    assert all(record[key] == value for key, value in job.items()), "job identity changed"
    directory = root / job["job_id"]
    assert json.loads((directory / "metadata.json").read_text()) == record, "metadata differs from results.jsonl"
    for kind in ("rgb", "mask"):
        assert sha256(sample[f"{kind}_file"]) == sample[f"{kind}_sha256"], f"input {kind} hash changed"
    assert json.loads(Path(sample["metadata_file"]).read_text()) == sample["source"], "Step 3 metadata changed"
    original = read_png(sample["rgb_file"], "RGB")
    object_mask = read_png(sample["mask_file"], "L")
    assert set(np.unique(object_mask)) == {0, 255}, "input mask is not nonempty binary"
    protected = read_png(directory / "protected_mask.png", "L")
    edit = read_png(directory / "edit_mask.png", "L")
    blend = read_png(directory / "blend_mask.png", "L")
    generated = read_png(directory / "generated.png", "RGB")
    enhanced = read_png(directory / "enhanced.png", "RGB")
    assert sha256(directory / "enhanced.png") == record["output_sha256"], "final image hash mismatch"

    # 独立按约定计算保护范围：原 mask 的方形邻域扩张，羽化只能发生在其外侧。
    padding = request["config"]["protect_padding_px"]
    expected = binary_dilation(object_mask > 0, structure=np.ones((2 * padding + 1,) * 2))
    assert np.array_equal(protected, expected.astype(np.uint8) * 255), "protected mask differs from contract"
    outside = ~expected
    assert outside.any(), "no background remains"
    assert np.array_equal(edit, outside.astype(np.uint8) * 255), "edit mask is not complement"
    feather = request["config"]["feather_px"]
    weights = np.minimum(distance_transform_edt(outside) / feather, 1) if feather else outside
    assert np.array_equal(blend, np.rint(weights * 255).astype(np.uint8)), "blend mask differs from contract"
    difference = np.abs(enhanced.astype(np.int16) - original.astype(np.int16))
    protected_error = int(difference[expected].max())
    object_error = int(difference[object_mask > 0].max())
    assert protected_error == object_error == 0, "protected/object pixels changed"
    alpha = blend.astype(np.uint32)[..., None]
    recomposed = (generated.astype(np.uint32) * alpha + original.astype(np.uint32) * (255 - alpha) + 127) // 255
    assert np.array_equal(enhanced, recomposed.astype(np.uint8)), "saved output differs from masked composition"

    inference = record["inference"]
    expected_steps = int(request["config"]["num_inference_steps"] * job["strength"])
    assert inference["actual_steps"] == expected_steps, "recorded steps differ from strength schedule"
    assert 0 < inference["elapsed_seconds"] <= record["elapsed_seconds"], "inconsistent elapsed time"
    allocated = inference["peak_memory_allocated_bytes"]
    reserved = inference["peak_memory_reserved_bytes"]
    assert 0 < allocated <= reserved, "inconsistent GPU peak memory"
    return {
        "job_id": job["job_id"], "status": "pass", "strength": job["strength"],
        "protected_max_abs_difference": protected_error, "object_max_abs_difference": object_error,
        "background_mean_abs_difference": float(difference[outside].mean()),
        "actual_steps_recorded": inference["actual_steps"],
        "inference_seconds_recorded": inference["elapsed_seconds"],
        "peak_allocated_bytes_recorded": allocated, "peak_reserved_bytes_recorded": reserved,
    }


def audit_run(root, log_file=None):
    request = json.loads((root / "request.json").read_text())
    results_file = root / "results.jsonl"
    records = [json.loads(line) for line in results_file.read_text().splitlines()] if results_file.exists() else []
    jobs, samples = request["jobs"], request["samples"]
    expected_ids = [job["job_id"] for job in jobs]
    record_counts = Counter(record["job_id"] for record in records)
    sample_ids = [sample["model_id"] for sample in samples]
    errors = []
    if not jobs or not samples:
        errors.append("empty request")
    if len(set(expected_ids)) != len(expected_ids) or len(set(sample_ids)) != len(sample_ids):
        errors.append("request contains duplicate job/sample IDs")
    planned_pairs = {(job["model_id"], job["strength"]) for job in jobs}
    required_pairs = {(model_id, strength) for model_id in sample_ids for strength in request["config"]["strengths"]}
    if planned_pairs != required_pairs or len(jobs) != len(required_pairs):
        errors.append("jobs do not cover every requested sample/strength pair exactly once")
    duplicate_records = [key for key, count in record_counts.items() if count > 1]
    unexpected = sorted(set(record_counts) - set(expected_ids))
    if duplicate_records or unexpected:
        errors.append("result IDs duplicated or outside the planned list")
    source_request = Path(request["render_dir"]) / "request.json"
    if sha256(source_request) != request["source_request_sha256"]:
        errors.append("Step 3 request hash changed")
    for name, digest in request["source_hashes"].items():
        if sha256(root / "source" / name) != digest:
            errors.append(f"frozen source hash changed: {name}")
    prompt_name = Path(request["prompt_file"]).name
    # 内置提示词随源码冻结；独立实验提示词只有 run/prompt.md，仍须与原摘要逐字节相同。
    source_prompt = root / "source" / "prompts" / prompt_name
    prompt_snapshot = source_prompt if source_prompt.is_file() else root / "prompt.md"
    if sha256(prompt_snapshot) != request["prompt_sha256"]:
        errors.append("frozen prompt hash changed")
    if (root / "prompt.md").read_text().strip() != request["prompt"]:
        errors.append("saved prompt differs from request")

    by_id = {record["job_id"]: record for record in records}
    by_sample = {sample["model_id"]: sample for sample in samples}
    findings = []
    for job in jobs:
        if job["job_id"] not in by_id:
            findings.append({"job_id": job["job_id"], "status": "missing"})
            continue
        try:
            findings.append(audit_job(root, request, by_sample[job["model_id"]], job, by_id[job["job_id"]]))
        except (AssertionError, OSError, ValueError, KeyError) as error:
            findings.append({"job_id": job["job_id"], "status": "failed", "error": f"{type(error).__name__}: {error}"})

    # 与日志逐次核对步数、耗时及显存峰值；日志保留小数，使用相应的舍入容差。
    log_path = log_file if log_file is not None else root / "run.log"
    log_checked = False
    if log_path.exists():
        logged = re.findall(r"\[FLUX\] 完成：(\d+) 步，([\d.]+)s，显存峰值 ([\d.]+) GiB", log_path.read_text())
        completed = [record for record in records if "inference" in record]
        log_checked = len(logged) == len(completed)
        for (steps, seconds, gib), record in zip(logged, completed):
            info = record["inference"]
            log_checked &= int(steps) == info["actual_steps"]
            log_checked &= abs(float(seconds) - info["elapsed_seconds"]) <= 0.051
            log_checked &= abs(float(gib) - info["peak_memory_allocated_bytes"] / 1024**3) <= 0.0051
        if not log_checked:
            errors.append("runtime metrics disagree with the per-job log")
    counts = Counter(item["status"] for item in findings)
    passed = [item for item in findings if item["status"] == "pass"]
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "run_dir": str(root),
        "status": "pass" if not errors and counts["pass"] == len(jobs) else "failed",
        "planned": len(jobs), "unique_input_models": len(set(sample_ids)), "records": len(records),
        "reported_success": sum(record["status"] == "success" for record in records),
        "reported_failed": sum(record["status"] != "success" for record in records),
        "audit_pass": counts["pass"], "audit_failed": counts["failed"], "missing": counts["missing"],
        "duplicate_result_ids": duplicate_records, "unexpected_result_ids": unexpected,
        "errors": errors, "runtime_log_checked": log_checked, "runtime_log_file": str(log_path),
        "prompt_snapshot_checked": str(prompt_snapshot),
        "actual_steps_by_strength": {str(strength): sorted({item["actual_steps_recorded"] for item in passed if item["strength"] == strength}) for strength in request["config"]["strengths"]},
        "inference_seconds_recorded_total": sum(item["inference_seconds_recorded"] for item in passed),
        "peak_allocated_gib_recorded": max((item["peak_allocated_bytes_recorded"] / 1024**3 for item in passed), default=None),
        "peak_reserved_gib_recorded": max((item["peak_reserved_bytes_recorded"] / 1024**3 for item in passed), default=None),
        "human_review": "pending", "findings": findings,
        "limitations": ["Timing, GPU peaks and step counts are recorded instrumentation, not an independent GPU replay.",
                        "Model weight hashes were not re-read by this audit.",
                        "Background pixel change does not measure photorealism; edited background depth is not certified."],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--report_copy", type=Path, help="可选：另存一份紧凑研究记录")
    parser.add_argument("--log_file", type=Path, help="可选：明确指定不在 run 目录内的运行日志")
    args = parser.parse_args()
    report = audit_run(args.run_dir.resolve(), args.log_file)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    (args.run_dir / "validation.json").write_text(text)
    if args.report_copy:
        args.report_copy.write_text(text)
    print(json.dumps({key: value for key, value in report.items() if key != "findings"}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
