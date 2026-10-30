"""Numerical checks for the modern CPU implementation against official formulas."""

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from model import AdaptiveLUT, Classifier, apply_lut


def direct_interpolation(lut: np.ndarray, image: np.ndarray) -> np.ndarray:
    """Independent NumPy translation of official TriLinearForwardCpu."""
    size = lut.shape[-1]
    binsize = np.float32(1.0001 / (size - 1))
    colors = image[0]
    indices = np.floor(colors / binsize).astype(int)
    delta = np.fmod(colors, binsize) / binsize
    result = np.zeros_like(image)
    for blue in (0, 1):
        for green in (0, 1):
            for red in (0, 1):
                offsets = (red, green, blue)
                weight = np.ones_like(delta[0])
                for channel, offset in enumerate(offsets):
                    weight *= delta[channel] if offset else 1 - delta[channel]
                result[0] += weight * lut[:, indices[2] + blue, indices[1] + green, indices[0] + red]
    return result


def validate(model_dir: Path) -> dict:
    torch.set_num_threads(4)
    torch.manual_seed(8)
    image = torch.rand(1, 3, 31, 29)
    image[:, :, 0, 0] = 0
    image[:, :, 0, 1] = 1
    ramp = torch.linspace(0, 1, 33)
    blue, green, red = torch.meshgrid(ramp, ramp, ramp, indexing="ij")
    identity = torch.stack([red, green, blue])
    random = torch.randn(3, 33, 33, 33)
    report = {
        "interpolation_binsize": "1.0001 / (dim - 1), matching models.py",
        "models_x_difference": "models_x.py uses 1.000001; this port explicitly follows original demo_eval.py/models.py",
        "identity_max_abs_error_from_input_div_1_0001": float((apply_lut(identity, image) - image / 1.0001).abs().max()),
        "random_lut_max_abs_error_from_direct_formula": float(np.max(np.abs(apply_lut(random, image).numpy() - direct_interpolation(random.numpy(), image.numpy())))),
        "classifiers": {},
    }
    # Load only the original class definitions, avoiding its obsolete extension imports.
    source = ast.parse((model_dir / "models.py").read_text())
    names = {"discriminator_block", "Classifier", "Classifier_unpaired"}
    definitions = ast.Module(body=[node for node in source.body if getattr(node, "name", None) in names], type_ignores=[])
    scope = {"nn": nn}
    exec(compile(definitions, str(model_dir / "models.py"), "exec"), scope)
    with torch.inference_mode():
        for variant in ("paired", "unpaired"):
            model = AdaptiveLUT(model_dir, variant)
            official = scope["Classifier" if variant == "paired" else "Classifier_unpaired"]().eval()
            official.load_state_dict(model.classifier.state_dict())
            own = model.classifier(image)
            other = official(image)
            weights = own.reshape(3)
            lut = (weights[:, None, None, None, None] * model.tables).sum(0)
            expected = direct_interpolation(lut.numpy(), image.numpy())
            actual = apply_lut(lut, image).numpy()
            report["classifiers"][variant] = {
                "official_class_max_abs_error": float((own - other).abs().max()),
                "real_lut_max_abs_error_from_direct_formula": float(np.abs(actual - expected).max()),
                "parameter_count": sum(p.numel() for p in model.classifier.parameters()) + model.tables.numel(),
            }
    assert report["identity_max_abs_error_from_input_div_1_0001"] < 1e-6
    assert report["random_lut_max_abs_error_from_direct_formula"] < 2e-5
    for checks in report["classifiers"].values():
        assert checks["official_class_max_abs_error"] == 0
        assert checks["real_lut_max_abs_error_from_direct_formula"] < 2e-5
    report["status"] = "pass"
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.model_dir)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
