"""先转移背景光影，再按真实覆盖alpha叠前景；没有mask外扩回贴。"""

import numpy as np
from PIL import Image


def decode(rgb):
    values = np.asarray(rgb, dtype=np.float64) / 255
    return np.where(values <= .04045, values / 12.92, ((values + .055) / 1.055) ** 2.4)


def encode(values):
    values = np.clip(values, 0, 1)
    rgb = np.where(values <= .0031308, 12.92 * values, 1.055 * values ** (1 / 2.4) - .055)
    return Image.fromarray(np.rint(rgb * 255).astype(np.uint8))


def relative_darkening(g, b, e):
    delta = g - b
    negative, positive = np.minimum(delta, 0), np.maximum(delta, 0)
    ratio = 1 + negative / np.maximum(b, 1e-4)
    correction = negative - b * (ratio - 1)
    return ratio * e + positive + correction


def compose_layers(foreground_rgba, empty_rgb, interaction_rgb, shadow_rgb, enhanced_rgb):
    """PNG前景RGB是straight，不再次除alpha；不向前景叠加背景差分。

    主版本只搬运亮度比例，保留增强背景色度。S为shadow-ray干预，G-S包含
    反射、色溢和间接遮挡，并非纯反射。RGB版本作为固定诊断保存。
    """
    rgba = np.asarray(foreground_rgba)
    assert rgba.shape == (512, 512, 4)
    alpha = rgba[..., 3:4].astype(np.float64) / 255
    f = decode(rgba[..., :3])
    b, g, s, e = [decode(image) for image in (empty_rgb, interaction_rgb, shadow_rgb, enhanced_rgb)]
    luminance = np.array([.2126, .7152, .0722])
    yb, yg, ys = [image @ luminance for image in (b, g, s)]
    denominator = np.maximum(yb, 1e-6)
    shadow_gain = ys / denominator
    indirect_gain = (yg - ys) / denominator
    gain = shadow_gain + indirect_gain
    background = e * gain[..., None]
    rgb_background = relative_darkening(g, b, e)
    main = alpha * f + (1 - alpha) * background
    reference = alpha * f + (1 - alpha) * rgb_background
    assert np.isfinite(main).all() and np.isfinite(reference).all()
    image, rgb_image = encode(main), encode(reference)
    opaque = alpha[..., 0] == 1
    metrics = dict(
        working_domain='display-linear; AgX remains baked',
        alpha_partial_pixels=int(((alpha > 0) & (alpha < 1)).sum()),
        alpha_nonzero_pixels=int((alpha > 0).sum()), alpha_opaque_pixels=int(opaque.sum()),
        opaque_max_rgb_error=int(np.abs(np.asarray(image)[opaque].astype(int) - rgba[..., :3][opaque]).max()),
        gain_min=float(gain.min()), gain_max=float(gain.max()),
        shadow_gain_min=float(shadow_gain.min()), shadow_gain_max=float(shadow_gain.max()),
        indirect_gain_min=float(indirect_gain.min()), indirect_gain_max=float(indirect_gain.max()),
        near_black_background_pixels=int((yb < 1e-6).sum()),
        before_clip_min=float(main.min()), before_clip_max=float(main.max()),
        preclip_negative_fraction=float((main < 0).mean()), preclip_above_one_fraction=float((main > 1).mean()),
        rgb_reference_negative_fraction=float((reference < 0).mean()),
        rgb_reference_above_one_fraction=float((reference > 1).mean()),
        no_enhancement_background_rgb_mae=float(np.abs(b * gain[..., None] - g).mean()),
        identity_note='Neutral transfer intentionally does not preserve colored interaction; no RGB identity claim.',
    )
    diagnostics = dict(alpha=alpha[..., 0], linear=main, shadow_gain=shadow_gain, indirect_gain=indirect_gain,
                       total_gain=gain, background=background, rgb_background=rgb_background)
    return image, rgb_image, metrics, diagnostics
