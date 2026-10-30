"""独立解析输入核对分层合成；只用 CPU，不读模型或渲染场景。"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from compose_layers import compose_layers


HERE = Path(__file__).resolve().parent
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


def linear_rgb(values):
    values = np.asarray(values, dtype=np.float64) / 255.0
    output = np.empty_like(values)
    low = values <= 0.04045
    output[low] = values[low] / 12.92
    output[~low] = ((values[~low] + 0.055) / 1.055) ** 2.4
    return output


def quantize(values):
    values = np.clip(values, 0.0, 1.0)
    output = np.empty_like(values)
    low = values <= 0.0031308
    output[low] = 12.92 * values[low]
    output[~low] = 1.055 * values[~low] ** (1.0 / 2.4) - 0.055
    return np.rint(output * 255.0).astype(np.uint8)


def rgb(color):
    return np.broadcast_to(np.asarray(color, dtype=np.uint8), (512, 512, 3)).copy()


def foreground():
    """一个不透明核心及三个实际部分覆盖像素；其余隐藏 RGB 故意非零。"""
    values = np.empty((512, 512, 4), dtype=np.uint8)
    values[..., :3] = [180, 70, 25]
    values[..., 3] = 0
    values[255:257, 255:257, 3] = 255
    values[254, 255, 3] = 64
    values[255, 257, 3] = 128
    values[257, 256, 3] = 192
    return values


def check_case(name, b8, g8, s8, e8):
    f8 = foreground()
    images = [Image.fromarray(values) for values in (f8, b8, g8, s8, e8)]
    main, reference, metrics, diagnostics = compose_layers(*images)
    alpha = f8[..., 3].astype(np.float64) / 255.0
    f, b, g, s, e = [linear_rgb(values) for values in (f8[..., :3], b8, g8, s8, e8)]
    yb, yg, ys = [values @ LUMA for values in (b, g, s)]
    denominator = np.maximum(yb, 1e-6)
    # 主背景独立使用 Y(G)/Y(B)，检查分开的两项确实望远镜相消。
    gain = yg / denominator
    background = e * gain[..., None]
    expected = alpha[..., None] * f + (1.0 - alpha[..., None]) * background
    # 旧 RGB 转移的独立分段表达，保留近黑补偿，不调用生产 helper。
    delta = g - b
    safe_b = np.maximum(b, 1e-4)
    dark = e + delta * (e / safe_b + 1.0 - b / safe_b)
    reference_background = np.where(delta >= 0, e + delta, dark)
    expected_reference = alpha[..., None] * f + (1.0 - alpha[..., None]) * reference_background
    actual = np.asarray(main)
    actual_reference = np.asarray(reference)
    assert np.array_equal(actual, quantize(expected)), name
    assert np.array_equal(actual_reference, quantize(expected_reference)), name

    expected_diagnostics = {
        "alpha": alpha,
        "shadow_gain": ys / denominator,
        "indirect_gain": (yg - ys) / denominator,
        "total_gain": gain,
        "background": background,
        "rgb_background": reference_background,
        "linear": expected,
    }
    for key, value in expected_diagnostics.items():
        np.testing.assert_allclose(diagnostics[key], value, rtol=2e-14, atol=2e-14, err_msg=f"{name}/{key}")
    expected_metrics = {
        "before_clip_min": expected.min(), "before_clip_max": expected.max(),
        "preclip_negative_fraction": (expected < 0).mean(),
        "preclip_above_one_fraction": (expected > 1).mean(),
        "rgb_reference_negative_fraction": (expected_reference < 0).mean(),
        "rgb_reference_above_one_fraction": (expected_reference > 1).mean(),
        "gain_min": gain.min(), "gain_max": gain.max(),
        "indirect_gain_min": ((yg - ys) / denominator).min(),
        "near_black_background_pixels": (yb < 1e-6).sum(),
        "no_enhancement_background_rgb_mae": np.abs(b * gain[..., None] - g).mean(),
    }
    for key, value in expected_metrics.items():
        np.testing.assert_allclose(metrics[key], value, rtol=2e-14, atol=2e-14, err_msg=f"{name}/{key}")

    opaque, clear = alpha == 1, alpha == 0
    assert np.array_equal(actual[opaque], f8[..., :3][opaque]), name
    assert np.array_equal(actual_reference[opaque], f8[..., :3][opaque]), name
    assert np.array_equal(actual[clear], quantize(background)[clear]), name
    reference_identity_checked = np.array_equal(e8, b8)
    if reference_identity_checked:
        original_interaction_over = alpha[..., None] * f + (1.0 - alpha[..., None]) * g
        assert np.array_equal(actual_reference, quantize(original_interaction_over)), name
    # 这些 alpha=0 点紧邻前景，旧 2px 硬回贴会改掉它们。
    ring_points = [(255, 254), (256, 257), (258, 256)]
    for row, column in ring_points:
        assert alpha[row, column] == 0
        assert np.array_equal(actual[row, column], quantize(background)[row, column])
    assert metrics["alpha_partial_pixels"] == 3
    assert metrics["alpha_opaque_pixels"] == 4
    assert metrics["alpha_nonzero_pixels"] == 7
    assert metrics["opaque_max_rgb_error"] == 0
    positive = e > 0
    scalar_field = np.broadcast_to(gain[..., None], e.shape)
    np.testing.assert_allclose(diagnostics["background"][positive] / e[positive], scalar_field[positive], rtol=2e-14)
    return {
        "case": name, "status": "pass",
        "main_matches_independent_formula": True,
        "rgb_reference_matches_independent_formula": True,
        "opaque_and_clear_limits_exact": True,
        "adjacent_clear_pixels_have_no_old_background_ring": True,
        "background_chromaticity_preserved_before_clip": True,
        "rgb_reference_identity_checked": bool(reference_identity_checked),
        "partial_alpha_values": [64, 128, 192],
        "gain_max": float(metrics["gain_max"]),
        "indirect_gain_min": float(metrics["indirect_gain_min"]),
        "before_clip_max": float(metrics["before_clip_max"]),
        "preclip_above_one_fraction": float(metrics["preclip_above_one_fraction"]),
        "no_enhancement_background_rgb_mae": float(metrics["no_enhancement_background_rgb_mae"]),
    }


def validate():
    cases = [
        check_case("straight_alpha_and_no_ring", rgb([110]*3), rgb([110]*3), rgb([110]*3), rgb([20, 80, 170])),
        check_case("colored_interaction_not_rgb_identity", rgb([120, 100, 80]), rgb([70, 90, 120]), rgb([60, 95, 75]), rgb([120, 100, 80])),
        check_case("negative_indirect_gain_not_clamped", rgb([180]*3), rgb([100]*3), rgb([150]*3), rgb([60, 100, 140])),
    ]
    near_black = rgb([96]*3)
    near_black[:256] = 0
    cases.append(check_case("near_black_extreme_gain_and_visible_clipping", near_black, rgb([255]*3), rgb([128]*3), rgb([180]*3)))
    assert cases[1]["no_enhancement_background_rgb_mae"] > 0.01
    assert cases[2]["indirect_gain_min"] < 0
    assert cases[3]["gain_max"] >= 999999
    assert cases[3]["before_clip_max"] > 100000
    assert cases[3]["preclip_above_one_fraction"] > 0.49
    return {
        "status": "pass", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                          for name in ["compose_layers.py", "validate_composite.py"]},
        "device": "CPU", "model_loaded": False, "render_called": False,
        "cases": cases,
        "limits": "解析输入只验证显示域合成合同；实际 PNG straight alpha、holdout 遮挡、EXR 分解及十件几何另需实际产物审核。",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=HERE / "composite_validation.json")
    args = parser.parse_args()
    report = validate()
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "cases": len(report["cases"]), "report": str(args.report)}, ensure_ascii=False))
