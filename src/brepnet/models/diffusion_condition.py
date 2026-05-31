from __future__ import annotations

from torch import nn

from src.brepnet.models.condition_encoders import ConditionExtractor


def build_condition_encoder(condition_cfg: dict) -> nn.Module | None:
    condition_type = condition_cfg["type"]
    if condition_type == "none":
        return None
    if condition_type in {"multi_img", "text"}:
        raise NotImplementedError(f"{condition_type} conditioning is disabled in the single-view runtime.")

    modality_by_type = {
        "single_img": ["single_img"],
        "sketch": ["sketch"],
        "point_cloud": ["pc"],
    }
    modalities = modality_by_type[condition_type]
    _cfg = {
        "condition": modalities,
        "backbone": condition_cfg["image"]["backbone"],
        "depth_anything_v2_ckpt": condition_cfg["image"]["depth_anything_v2_ckpt"],
        "is_aug": condition_cfg["point_cloud"]["augment_probability"] > 0,
        "aug_points_prob": condition_cfg["point_cloud"]["augment_probability"],
        "point_encoder": condition_cfg["point_cloud"]["encoder"],
    }
    return ConditionExtractor(
        config=_cfg,
        projection_dim=condition_cfg["output_dim"],
    )
