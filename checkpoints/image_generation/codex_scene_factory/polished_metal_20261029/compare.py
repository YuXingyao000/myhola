"""汇总全部预定样本，原尺寸拼图，不对比较图再次调色。"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def make_board(spec, output):
    """512像素原尺寸贴入画布，所有标签位于图像之外。"""
    font = "/root/.cache/cad_scene_figures/SourceHanSansSC-Regular.otf"
    title, label, small = [ImageFont.truetype(font, size) for size in (30, 23, 19)]
    page = Image.new("RGB", (512 * len(spec["columns"]), 132 + 552 * len(spec["rows"])), "#f6f7f9")
    draw = ImageDraw.Draw(page)
    draw.text((16, 10), spec["title"], font=title, fill="#172a3a")
    draw.text((16, 55), spec["subtitle"], font=small, fill="#526575")
    for column, text in enumerate(spec["columns"]):
        draw.text((column * 512 + 16, 95), text, font=label, fill="#172a3a")
    for row, item in enumerate(spec["rows"]):
        y = 132 + row * 552
        draw.text((16, y), item["label"], font=label, fill="#172a3a")
        for column, path in enumerate(item["images"]):
            image = Image.open(path).convert("RGB")
            assert image.size == (512, 512)
            page.paste(image, (column * 512, y + 40))
    page.save(output)
    output.with_suffix(".json").write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n")


def summarize(root):
    request = json.loads((root / "request.json").read_text())
    source = json.loads(Path(request["source_request"]).read_text())
    labels = {"office": "办公室", "factory": "机加工车间", "assembly": "机械装配车间"}
    columns = ["原材质：roughness 0.32", "均匀抛光：素材均值", "抛光：素材粗糙度贴图"]
    records, overview = [], []
    for scene_index, scene in enumerate(request["scenes"]):
        example = request["models"][scene_index % len(request["models"])]
        rows = []
        for model in request["models"]:
            row = {"label": f"{labels[scene]} · {model}", "images": []}
            for profile in request["profiles"]:
                folder = root / scene / model / profile
                metadata = folder / "metadata.json"
                if not metadata.exists():
                    records.append({"scene": scene, "model": model, "profile": profile, "status": "missing"})
                    continue
                record = json.loads(metadata.read_text())
                # 先导三张只误命名了该诊断值；它来自保存的PNG，不是线性辐射。
                if "target_mean_abs_linear_rgb_difference" in record:
                    record["target_mean_abs_saved_png_rgb_difference"] = record.pop("target_mean_abs_linear_rgb_difference")
                image = np.asarray(Image.open(folder / "rgb.png").convert("RGB"))
                assert image.shape == (512, 512, 3)
                if profile == "baseline":
                    original = np.asarray(Image.open(Path(source["output_dir"]) / "raw" / scene / model / "rgb.png").convert("RGB"))
                    error = np.abs(image.astype(float) - original.astype(float))
                    record["rerender_vs_old_png"] = {"mean_abs_rgb_255": float(error.mean()), "max_abs_rgb_255": float(error.max())}
                records.append(record)
                row["images"].append(str(folder / "rgb.png"))
            if len(row["images"]) == 3:
                rows.append(row)
                if model == example:
                    overview.append(row)
        if rows:
            make_board({"title": f"抛光金属对照 · {labels[scene]}",
                        "subtitle": "固定几何、底色、灯光和相机 · 512×512 · CPU / 128 samples / seed 8",
                        "columns": columns, "rows": rows}, root / f"comparison_{scene}.png")
    if overview:
        make_board({"title": "抛光金属对照 · 三场景概览", "subtitle": "只改变粗糙度；不改变网格、法线、底色或灯光。",
                    "columns": columns, "rows": overview}, root / "comparison.png")
    complete = [r for r in records if r["status"] == "completed"]
    checks_pass = all(r["checks"]["signature_unchanged"] and r["checks"]["mask_differing_pixels"] == 0
                      and r["checks"]["depth_max_abs_error_m"] == 0 for r in complete)
    result = {"planned": request["planned"], "completed": len(complete),
              "missing": sum(r["status"] == "missing" for r in records),
              "failed": sum(r["status"] not in {"completed", "missing"} for r in records),
              "geometry_checks_pass": checks_pass, "human_review": "pending",
              "selected_default": None, "records": records}
    (root / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    summarize(parser.parse_args().output_dir)
