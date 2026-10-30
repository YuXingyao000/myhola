"""CPU photo retouching with the official Image-Adaptive-3DLUT sRGB weights.

Classifier architecture follows HuiZeng/Image-Adaptive-3DLUT (Apache-2.0).
The legacy interpolation extension is replaced by PyTorch grid_sample.
"""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class Classifier(nn.Module):
    """Predict three LUT mixing weights from a 256-pixel image thumbnail."""

    def __init__(self, paired: bool):
        super().__init__()
        layers = [
            nn.Upsample(size=(256, 256), mode="bilinear", align_corners=False),
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(16, affine=True),
        ]
        for source, target in [(16, 32), (32, 64), (64, 128), (128, 128)]:
            layers.extend([
                nn.Conv2d(source, target, 3, stride=2, padding=1),
                nn.LeakyReLU(0.2),
            ])
            if paired and source != 128:
                layers.append(nn.InstanceNorm2d(target, affine=True))
        if paired:
            layers.append(nn.Dropout(p=0.5))
        layers.append(nn.Conv2d(128, 3, 8, padding=0))
        self.model = nn.Sequential(*layers)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)


def apply_lut(lut: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    """Map RGB values, preserving the image's pixel positions.

    LUT axes are [output_channel, B, G, R]; grid_sample coordinates are R,G,B.
    1.0001 reproduces models.py's binsize=1.0001/(33-1), including endpoints.
    This slight boundary offset is retained rather than silently changing the
    pretrained model's interpolation convention.
    """
    grid = (image.permute(0, 2, 3, 1) / 1.0001 * 2 - 1).unsqueeze(1)
    return F.grid_sample(
        lut.unsqueeze(0), grid, mode="bilinear", padding_mode="border",
        align_corners=True,
    ).squeeze(2)


class AdaptiveLUT:
    """One deterministic sRGB color transform per image; CPU inference only."""

    def __init__(self, model_dir: Path, variant: str):
        suffix = "" if variant == "paired" else "_unpaired"
        weights_dir = model_dir / "pretrained_models" / "sRGB"
        self.classifier = Classifier(paired=variant == "paired").eval()
        self.classifier.load_state_dict(torch.load(
            weights_dir / f"classifier{suffix}.pth", map_location="cpu",
            weights_only=True,
        ))
        tables = torch.load(weights_dir / f"LUTs{suffix}.pth",
                            map_location="cpu", weights_only=True)
        self.tables = torch.stack([tables[str(index)]["LUT"] for index in range(3)])

    @torch.inference_mode()
    def __call__(self, image: torch.Tensor) -> tuple[torch.Tensor, list[float]]:
        weights = self.classifier(image).reshape(3)
        lut = (weights[:, None, None, None, None] * self.tables).sum(0)
        return apply_lut(lut, image), weights.tolist()
