"""配对合成的 CPU 恒等性、阴影/反射解析值和保护区域验证。"""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from composite import compose, transfer_delta


def main():
    rng = np.random.default_rng(8)
    empty = rng.integers(20, 220, size=(128, 128, 3), dtype=np.uint8)
    with_part = empty.copy()
    target = np.zeros((128, 128), dtype=np.uint8)
    target[30:90, 30:90] = 255
    target[45:75, 45:75] = 0
    with_part[target > 0] = [180, 165, 130]
    with_part[94:105, 35:95] = (empty[94:105, 35:95] * .7).astype(np.uint8)
    with_part[15:25, 35:95] = np.minimum(empty[15:25, 35:95].astype(int) + 20, 255).astype(np.uint8)
    enhanced = np.minimum(empty.astype(int) + 15, 255).astype(np.uint8)
    images = [Image.fromarray(image) for image in (with_part, empty, enhanced, target)]

    for index in range(3):
        invalid = images.copy()
        invalid[index] = Image.fromarray(np.full(target.shape, np.nan, dtype=np.float32))
        try:
            compose(*invalid, "additive")
        except ValueError as error:
            assert "8 位 RGB" in str(error)
        else:
            raise AssertionError("浮点输入未被拒绝")

    results = {}
    for mode in ("additive", "relative_darkening"):
        output, identity_metrics, _ = compose(images[0], images[1], images[1], images[3], mode)
        assert np.array_equal(np.asarray(output), with_part)
        assert identity_metrics["identity_max_abs_uint8_error"] == 0
        assert identity_metrics["identity_max_abs_float_error"] < 1e-12

        output, edited_metrics, diagnostics = compose(*images, mode)
        output = np.asarray(output)
        protected = np.asarray(diagnostics["protected_mask"]) > 0
        assert np.array_equal(output[protected], with_part[protected])
        assert np.array_equal(output[60, 60], enhanced[60, 60])  # 孔隙显示新背景。
        assert np.array_equal(np.asarray(diagnostics["hull_protected"])[60, 60], with_part[60, 60])

        zero_delta = transfer_delta(np.array([.4]), np.array([.4]), np.array([.8]), mode)
        assert np.allclose(zero_delta, .8)
        shadow = transfer_delta(np.array([.2]), np.array([.4]), np.array([.6]), mode)
        assert np.allclose(shadow, .4 if mode == "additive" else .3)
        reflection = transfer_delta(np.array([.6]), np.array([.4]), np.array([.6]), mode)
        assert np.allclose(reflection, .8)
        black_b = np.array([0, 1e-8, 5e-5, 1e-4, .5])
        black_i = np.array([.01, 0, 1e-5, .2, .3])
        assert np.max(np.abs(transfer_delta(black_i, black_b, black_b, mode) - black_i)) < 1e-12
        results[mode] = {
            "identity": identity_metrics,
            "edited_background": edited_metrics,
            "strict_target_copy": "pass",
            "hole_background_not_filled": "pass",
            "near_black_identity": "pass",
            "analytic_shadow": float(shadow[0]),
            "analytic_positive_delta": float(reflection[0]),
        }

    # 大幅亮度变化必须暴露裁剪，不能在指标里只看量化后的有效范围。
    i = np.full_like(with_part, 230)
    b = np.full_like(with_part, 60)
    e = np.full_like(with_part, 235)
    _, clipping, _ = compose(Image.fromarray(i), Image.fromarray(b), Image.fromarray(e), images[3], "additive")
    assert clipping["before_clip_max"] > 1 and clipping["above_one_channel_fraction"] > 0

    # 位移后的边缘信号应恶化；这是定位行为测试，不是 geometry-safe 断言。
    flat = np.full_like(empty, 100)
    flat[:, 40:70] = 180
    shifted = np.roll(flat, 12, axis=1)
    _, aligned, _ = compose(Image.fromarray(flat), Image.fromarray(flat), Image.fromarray(flat), images[3], "additive")
    _, moved, _ = compose(Image.fromarray(flat), Image.fromarray(flat), Image.fromarray(shifted), images[3], "additive")
    key = "source_supported_within_2px"
    assert aligned["alignment_signals_not_a_gate"]["full_frame"][key] > moved["alignment_signals_not_a_gate"]["full_frame"][key]

    directory = Path(__file__).parent
    report = {
        "status": "pass", "device": "cpu", "seed": 8,
        "rgb_input_contract": "pass: each of the three inputs rejects float PIL images",
        "source_sha256": hashlib.sha256((directory / "composite.py").read_bytes()).hexdigest(),
        "cases": results, "clipping_detection": clipping,
        "edge_shift_check": {"aligned": aligned["alignment_signals_not_a_gate"], "moved_12px": moved["alignment_signals_not_a_gate"]},
        "limits": "Synthetic CPU behavior checks; no photographic-quality or physically correct relighting conclusion.",
    }
    output = directory / "composite_validation.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
