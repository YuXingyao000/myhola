"""固定 CAD 裁剪与结构警报；只帮助审图，不判定生成图的几何合格。

使用原始 mask 定位，不能把它当作新图的分割真值。梯度也可能来自高光、
阴影或纹理，因此边缘支持率与小范围平移只作为提示，所有候选仍需实际看图。
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter, maximum_filter, sobel


def read_rgb(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def crop_box(mask, padding=12):
    """用真实零件像素求包围框；返回 Pillow 的右下角不包含式坐标。"""
    ys, xs = np.where(mask)
    height, width = mask.shape
    return [max(0, int(xs.min()) - padding), max(0, int(ys.min()) - padding),
            min(width, int(xs.max()) + padding + 1), min(height, int(ys.max()) + padding + 1)]


def gradient_map(image):
    """各裁剪独立归一化亮度范围，减小整体曝光变化对边缘位置比较的影响。"""
    rgb = np.asarray(image, dtype=np.float32) / 255
    gray = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    low, high = np.percentile(gray, [5, 95])
    normalized = np.clip((gray - low) / max(float(high - low), 0.05), 0, 1)
    smooth = gaussian_filter(normalized, sigma=0.7)
    return np.hypot(sobel(smooth, axis=0), sobel(smooth, axis=1)) / 8


def shifted_cosine(source, candidate, region, dx, dy):
    """在固定 ROI 中比较梯度形状；偏移方向指新图相对原图的位置。"""
    height, width = source.shape
    y0, y1 = max(0, -dy), min(height, height - dy)
    x0, x1 = max(0, -dx), min(width, width - dx)
    selected = region[y0:y1, x0:x1]
    a = source[y0:y1, x0:x1][selected]
    b = candidate[y0 + dy:y1 + dy, x0 + dx:x1 + dx][selected]
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator > 0 else 0.0


def structure_signals(original, candidate, mask):
    """报告原轮廓附近梯度、原 CAD 内截断与 ±4 px 梯度平移搜索。"""
    source_gradient = gradient_map(original)
    candidate_gradient = gradient_map(candidate)
    region = binary_dilation(mask, iterations=4)
    contour = mask & ~binary_erosion(mask)
    edge_support = []
    for threshold in [0.05, 0.10]:
        reference_edges = (source_gradient >= threshold) & region
        candidate_edges = (candidate_gradient >= threshold) & region
        nearby_reference = maximum_filter(source_gradient >= threshold, size=5)
        nearby_candidate = maximum_filter(candidate_gradient >= threshold, size=5)
        edge_support.append({
            "normalized_gradient_threshold": threshold,
            "reference_edge_pixels": int(reference_edges.sum()),
            "reference_edges_supported_within_2px": (
                float(nearby_candidate[reference_edges].mean()) if reference_edges.any() else None
            ),
            "candidate_edge_pixels": int(candidate_edges.sum()),
            "candidate_edges_unmatched_within_2px": (
                float((~nearby_reference[candidate_edges]).mean()) if candidate_edges.any() else None
            ),
            "original_contour_supported_within_2px": float(nearby_candidate[contour].mean()),
        })

    offsets = [(dx, dy) for dy in range(-4, 5) for dx in range(-4, 5)]
    offsets.sort(key=lambda pair: abs(pair[0]) + abs(pair[1]))
    scores = [(shifted_cosine(source_gradient, candidate_gradient, region, dx, dy), dx, dy)
              for dx, dy in offsets]
    best_score, dx, dy = max(scores, key=lambda item: item[0])
    zero_score = shifted_cosine(source_gradient, candidate_gradient, region, 0, 0)
    rgb = np.asarray(candidate)[mask]
    gray = rgb @ np.array([0.2126, 0.7152, 0.0722])
    return {
        "edge_support": edge_support,
        "cad_clipping": {
            "any_channel_at_or_below_1_fraction": float((rgb <= 1).any(axis=1).mean()),
            "any_channel_at_or_above_254_fraction": float((rgb >= 254).any(axis=1).mean()),
            "luminance_at_or_below_1_fraction": float((gray <= 1).mean()),
            "luminance_at_or_above_254_fraction": float((gray >= 254).mean()),
        },
        "gradient_shift_search": {
            "best_dx_px": dx, "best_dy_px": dy,
            "zero_shift_cosine": zero_score, "best_cosine": best_score,
            "improvement_over_zero": best_score - zero_score,
            "nonzero_best_shift": bool(dx or dy),
            "search_boundary_reached": abs(dx) == 4 or abs(dy) == 4,
        },
    }


def contour_overlay(image, mask):
    pixels = np.asarray(image).copy()
    pixels[mask & ~binary_erosion(mask)] = [255, 0, 190]
    return Image.fromarray(pixels)


def save_page(run_dir, strength, rows, page_index):
    """每行四栏共用同一个裁剪与缩放比例；缺失/失败保留占位。"""
    panel_size, gap, title_height, row_height = 272, 8, 64, 318
    canvas = Image.new("RGB", (4 * panel_size + 5 * gap, title_height + len(rows) * row_height), "#eef1f5")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    small_font = ImageFont.load_default(size=13)
    draw.text((gap, 7), f"CAD crops | strength={strength} | magenta = original mask contour", fill="#172238", font=font)
    for column, title in enumerate(["Original", "Raw edit", "Original CAD pasted back", "Raw + original contour"]):
        draw.text((gap + column * (panel_size + gap), 35), title, fill="#172238", font=font)
    for index, row in enumerate(rows):
        y = title_height + index * row_height
        draw.text((gap, y), row["label"], fill="#172238", font=small_font)
        for column, image in enumerate(row["panels"]):
            x = gap + column * (panel_size + gap)
            draw.rectangle((x, y + 24, x + panel_size, y + 24 + panel_size), fill="#d9dfe7")
            if image is None:
                draw.text((x + 12, y + 130), "Not available", fill="#9b2435", font=font)
                continue
            scale = min(panel_size / image.width, panel_size / image.height)
            enlarged = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.NEAREST)
            canvas.paste(enlarged, (x + (panel_size - enlarged.width) // 2,
                                    y + 24 + (panel_size - enlarged.height) // 2))
    strength_name = str(strength).replace(".", "_")
    output = run_dir / f"inspection_{strength_name}_{page_index:02d}.jpg"
    canvas.save(output, quality=94)
    return str(output)


def inspect_run(run_dir, samples_file):
    samples = json.loads(samples_file.read_text())
    records = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines() if line.strip()]
    strengths = list(dict.fromkeys(record["strength"] for record in records)) or [None]
    report = {
        "purpose": "visual_review_signals_only_not_geometry_acceptance",
        "limitations": [
            "Original mask locates the ROI; it is not segmentation ground truth for the edited image.",
            "Gradients include reflections and texture. Missing gradients can mean low contrast, not missing geometry.",
            "A small best shift can reflect shading changes. A boundary optimum may miss a larger displacement.",
            "No signal in this report certifies holes, topology, geometry, photorealism or training eligibility.",
        ],
        "protocol": {"bbox_padding_px": 12, "support_radius_px": 2, "shift_radius_px": 4,
                     "display_resampling": "nearest", "gradient_roi": "original CAD mask dilated 4px"},
        "record_count": len(records), "records": [], "pages": [],
    }
    for strength in strengths:
        rows = []
        for sample in samples:
            matches = [record for record in records if record["model_id"] == sample["model_id"]
                       and record["strength"] == strength]
            record = matches[0] if matches else {"status": "missing"}
            original = read_rgb(sample["rgb_file"])
            with Image.open(sample["mask_file"]) as image:
                mask = np.asarray(image.convert("L")) > 0
            box = crop_box(mask)
            x0, y0, x1, y1 = box
            cropped_mask = mask[y0:y1, x0:x1]
            original_crop = original.crop(box)
            panels = [original_crop, None, None, None]
            result = {"model_id": sample["model_id"], "strength": strength, "status": record["status"],
                      "duplicate_records": max(0, len(matches) - 1), "crop_box_px": box}
            for key, column in [("generated_file", 1), ("protected_file", 2)]:
                path = Path(record[key]) if record.get(key) else None
                if path is not None and not path.is_absolute():
                    path = run_dir / path
                if path is None or not path.exists():
                    result[key] = {"available": False}
                    continue
                candidate = read_rgb(path)
                if candidate.size != original.size:
                    result[key] = {"available": False, "reason": "resolution_mismatch", "size": candidate.size}
                    continue
                crop = candidate.crop(box)
                panels[column] = crop
                result[key] = {"available": True, "path": str(path),
                               "signals": structure_signals(original_crop, crop, cropped_mask)}
                if column == 1:
                    panels[3] = contour_overlay(crop, cropped_mask)
            label = f'{sample["model_id"]} | {record["status"]} | crop {x1-x0} x {y1-y0} px'
            rows.append({"label": label, "panels": panels})
            report["records"].append(result)
        for page_start in range(0, len(rows), 3):
            report["pages"].append(save_page(run_dir, strength, rows[page_start:page_start + 3], page_start // 3))
    output = run_dir / "structure_signals.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--samples_file", type=Path, default=Path(__file__).with_name("samples.json"))
    args = parser.parse_args()
    inspect_run(args.run_dir, args.samples_file)
