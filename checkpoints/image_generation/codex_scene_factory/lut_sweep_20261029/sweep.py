"""固定 paired LUT，一次浮点推理后扫描 mix=0.0..1.0；不训练、不渲染。

输入 JSON 每项：scene_id、scene_label、model_id、rgb、metadata。
每张输入生成 11 张独立 PNG；逐件对照保留 512×512 像素，不裁切。
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont
import torch


REPOSITORY = Path("/mnt/d/python_room_generation")
MODEL_CODE = REPOSITORY / "packages/cad_scene_factory/step4/retouch/model.py"
MODEL_DIR = Path("/mnt/d/model/Image_Adaptive_3DLUT")
FONT = "/root/.cache/cad_scene_figures/SourceHanSansSC-Regular.otf"
MIXES = [index / 10 for index in range(11)]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def load_model(snapshot):
    """执行本次保存的 model.py 副本，与记录的源码摘要保持一致。"""
    spec = importlib.util.spec_from_file_location("paired_lut_snapshot", snapshot)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AdaptiveLUT(MODEL_DIR, "paired")


def freeze_request(inputs, output_dir):
    samples = json.loads(inputs.read_text())
    assert samples, "输入清单不能为空"
    output_dir.mkdir(parents=True, exist_ok=False)
    source_dir = output_dir / "source"
    source_dir.mkdir()
    shutil.copyfile(__file__, source_dir / "sweep.py")
    shutil.copyfile(MODEL_CODE, source_dir / "model.py")
    shutil.copyfile(MODEL_DIR / "source.json", source_dir / "weights_source.json")
    shutil.copyfile(inputs, source_dir / "samples.json")
    for sample in samples:
        sample["rgb"] = str(Path(sample["rgb"]).resolve())
        sample["rgb_sha256"] = sha256(sample["rgb"])
        if isinstance(sample["metadata"], str):
            sample["metadata_sha256"] = sha256(sample["metadata"])
    weight_files = [MODEL_DIR / "pretrained_models/sRGB" / name
                    for name in ("classifier.pth", "LUTs.pth")]
    request = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "samples": samples, "variant": "paired", "mixes": MIXES,
        "device": "cpu", "threads": 4,
        "input_contract": "512x512 RGB PNG, display-referred sRGB",
        "mix_contract": "source.lerp(unquantized_retouched, mix); no interpolation from mix_1.0.png",
        "quantization": "mul(255).add(0.5).clamp(0,255).uint8; mix 0 copies original PNG",
        "model_dir": str(MODEL_DIR),
        "weights_source": json.loads((MODEL_DIR / "source.json").read_text()),
        "weights_source_sha256": sha256(MODEL_DIR / "source.json"),
        "weight_sha256": {str(path): sha256(path) for path in weight_files},
        "source_sha256": {name: sha256(source_dir / name) for name in ("sweep.py", "model.py")},
        "input_manifest_sha256": sha256(inputs),
        "versions": {"torch": torch.__version__, "Pillow": PIL.__version__, "numpy": np.__version__},
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True).strip(),
        "command": [sys.executable, *sys.argv],
        "human_review": "pending",
    }
    write_json(output_dir / "request.json", request)
    return request


def sweep_sample(sample, model, request, output_dir, stream):
    """只在这里调用一次模型；11 档都使用同一个未量化的浮点结果。"""
    source_file = Path(sample["rgb"])
    assert sha256(source_file) == sample["rgb_sha256"]
    source = np.asarray(Image.open(source_file).convert("RGB"))
    assert source.shape == (512, 512, 3)
    image = torch.from_numpy(source.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255
    started = time.perf_counter()
    retouched, weights = model(image)
    inference_seconds = time.perf_counter() - started
    assert torch.isfinite(retouched).all()
    directory = output_dir / sample["scene_id"] / sample["model_id"]
    directory.mkdir(parents=True, exist_ok=False)
    records = []
    for mix in MIXES:
        mixed = image.lerp(retouched, mix)
        encoded = mixed.squeeze(0).mul(255).add(0.5).clamp(0, 255)
        encoded = encoded.permute(1, 2, 0).to(torch.uint8).numpy()
        target = directory / f"mix_{mix:.1f}.png"
        if mix == 0:
            shutil.copyfile(source_file, target)
            assert sha256(target) == sample["rgb_sha256"]
            assert np.array_equal(np.asarray(Image.open(target).convert("RGB")), source)
            assert np.array_equal(encoded, source)
        else:
            Image.fromarray(encoded).save(target)
        record = {
            "scene_id": sample["scene_id"], "scene_label": sample["scene_label"],
            "model_id": sample["model_id"], "metadata": sample["metadata"],
            "variant": "paired", "mix": mix, "status": "success",
            "weights": weights, "rgb": str(source_file), "rgb_sha256": sample["rgb_sha256"],
            "source_sha256": request["source_sha256"],
            "weight_sha256": request["weight_sha256"],
            "weights_source": "source/weights_source.json",
            "weights_source_sha256": request["weights_source_sha256"],
            "output": str(target.relative_to(output_dir)), "output_sha256": sha256(target),
            "inference_seconds": inference_seconds, "inference_reused_for_all_mixes": True,
            "rgb_mae_255": float(np.abs(encoded.astype(float) - source).mean()),
            "source_channel_zero_fraction": float((source == 0).mean()),
            "source_channel_255_fraction": float((source == 255).mean()),
            "output_channel_zero_fraction": float((encoded == 0).mean()),
            "output_channel_255_fraction": float((encoded == 255).mean()),
            "preclip_low_fraction": float((mixed < 0).float().mean()),
            "preclip_high_fraction": float((mixed > 1).float().mean()),
            "mix_zero_copy_verified": mix == 0,
        }
        records.append(record)
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
    return records


def make_sample_comparison(sample, output_dir):
    """4×3 网格：11 张全尺寸图，最后一格说明同一输入与同一次推理。"""
    directory = output_dir / sample["scene_id"] / sample["model_id"]
    page = Image.new("RGB", (2048, 1728), "#fafafa")
    draw = ImageDraw.Draw(page)
    title_font, label_font = [ImageFont.truetype(FONT, size) for size in (32, 23)]
    draw.text((16, 8), f"{sample['scene_label']} · CAD {sample['model_id']}", font=title_font, fill="#202a35")
    draw.text((16, 49), "paired 3D LUT · 0.0 为原图 · 512×512 原尺寸 · 各档独立混合未量化结果",
              font=label_font, fill="#55616e")
    for index, mix in enumerate(MIXES):
        x, y = (index % 4) * 512, 84 + (index // 4) * 548
        draw.rectangle((x, y, x + 511, y + 35), fill="#e4edf4" if mix == 0 else "#eef1f4")
        label = "0.0 · 原图" if mix == 0 else f"mix {mix:.1f}"
        draw.text((x + 12, y + 2), label, font=label_font, fill="#202a35")
        with Image.open(directory / f"mix_{mix:.1f}.png") as image:
            page.paste(image.convert("RGB"), (x, y + 36))
    x, y = 3 * 512 + 22, 84 + 2 * 548 + 40
    for index, line in enumerate([sample["scene_label"], f"CAD {sample['model_id']}",
                                  "每图一次 paired LUT 推理", "11 档 mix：0.0 → 1.0",
                                  "原图 0.0 为 PNG 原文件复制", "原始几何和像素坐标不变",
                                  "仅比较颜色与明暗变化"]):
        draw.text((x, y + index * 47), line, font=label_font, fill="#394656")
    target = directory / "comparison.png"
    page.save(target)
    return target


def make_scene_overview(samples, output_dir):
    """每场景按输入次序排 CAD 行；11 列共用 mix 标签，0.0 明确标原图。"""
    thumb, header, row_height = 256, 100, 292
    page = Image.new("RGB", (11 * thumb, header + len(samples) * row_height), "#fafafa")
    draw = ImageDraw.Draw(page)
    title_font, label_font = [ImageFont.truetype(FONT, size) for size in (32, 22)]
    draw.text((14, 8), f"{samples[0]['scene_label']} · paired LUT · 同行是同一输入的 11 档",
              font=title_font, fill="#202a35")
    for column, mix in enumerate(MIXES):
        label = "0.0 · 原图" if mix == 0 else f"mix {mix:.1f}"
        draw.text((column * thumb + 12, 58), label, font=label_font, fill="#202a35")
    for row, sample in enumerate(samples):
        y = header + row * row_height
        draw.text((12, y), f"CAD {sample['model_id']}", font=label_font, fill="#394656")
        directory = output_dir / sample["scene_id"] / sample["model_id"]
        for column, mix in enumerate(MIXES):
            with Image.open(directory / f"mix_{mix:.1f}.png") as image:
                page.paste(image.convert("RGB").resize((thumb, thumb), Image.Resampling.LANCZOS),
                           (column * thumb, y + 36))
    target = output_dir / samples[0]["scene_id"] / "overview.png"
    page.save(target)
    return target


def run_workflow(args):
    torch.set_num_threads(4)
    request = freeze_request(args.inputs, args.output_dir)
    started = time.perf_counter()
    model = load_model(args.output_dir / "source/model.py")
    records, comparison_files = [], []
    by_scene = defaultdict(list)
    with (args.output_dir / "records.jsonl").open("x") as stream:
        for sample in request["samples"]:
            records.extend(sweep_sample(sample, model, request, args.output_dir, stream))
            comparison_files.append(make_sample_comparison(sample, args.output_dir))
            by_scene[sample["scene_id"]].append(sample)
            print(f"{sample['scene_id']} / {sample['model_id']}：11 档已保存", flush=True)
    overview_files = [make_scene_overview(samples, args.output_dir) for samples in by_scene.values()]
    pages = [Image.open(path).convert("RGB") for path in comparison_files]
    pdf = args.output_dir / "all_comparisons.pdf"
    pages[0].save(pdf, save_all=True, append_images=pages[1:], resolution=150,
                  quality=96, subsampling=0, title="Paired 3D LUT · Mix 0.0–1.0")
    for page in pages:
        page.close()
    manifest = {
        "status": "completed", "planned_inputs": len(request["samples"]),
        "planned_outputs": len(request["samples"]) * 11, "success": len(records),
        "failed": 0, "missing": 0, "inference_calls": len(request["samples"]),
        "elapsed_seconds": time.perf_counter() - started,
        "inference_seconds": sum(row["inference_seconds"] for row in records if row["mix"] == 0),
        "request": "request.json", "records": "records.jsonl", "samples": request["samples"],
        "variant": "paired", "mixes": MIXES, "pdf": pdf.name, "pdf_pages": len(pages),
        "comparisons": [str(path.relative_to(args.output_dir)) for path in comparison_files],
        "scene_overviews": [str(path.relative_to(args.output_dir)) for path in overview_files],
        "human_review": "pending", "selected_mix": None,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in ("status", "success", "pdf_pages", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    args.inputs = args.inputs.resolve()
    args.output_dir = args.output_dir.resolve()
    run_workflow(args)
