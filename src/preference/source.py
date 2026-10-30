"""Explicit image and checkpoint contracts for grey-render and photo pairs."""

from pathlib import Path

from tools.vae_compare_post import sha256_file


def is_photo(config: dict) -> bool:
    kind = config.get("model_source", "legacy_svr")
    if kind not in {"legacy_svr", "photo"}:
        raise ValueError(f"Unsupported preference model_source: {kind}")
    return kind == "photo"


def image_path(config: dict, image_id: str) -> Path:
    filename = "real_photo.npz" if is_photo(config) else "imgs.npz"
    return Path(config["condition_data_root"]) / image_id / filename


def load_condition(config: dict, image_id: str):
    import numpy as np
    from src.datasets.photo import photo_transform
    from src.models.diffusion.image_loading import load_image_view

    path = image_path(config, image_id)
    if is_photo(config):
        with np.load(path, allow_pickle=False) as data:
            image = data["flux"]
        if image.shape != (512, 512, 3) or image.dtype != np.uint8:
            raise ValueError(f"Expected uint8 flux[512,512,3]: {path}")
    else:
        image = load_image_view(path, "svr_imgs", 0, read_single_view=True)
    return photo_transform()(image)


def source_protocol(config: dict) -> dict:
    """Bind photo policy and original frozen AE/DINO before starting workers."""
    if not is_photo(config):
        return {}
    import torch

    original = Path(config["original_checkpoint"]).resolve()
    original_hash = sha256_file(original)
    saved = torch.load(config["checkpoint"], map_location="cpu", mmap=True, weights_only=False)
    hp = saved["hyper_parameters"]
    if hp["model"]["name"] != "PhotoBaselineDiffusion":
        raise ValueError("Photo preference generation requires PhotoBaselineDiffusion")
    if hp["dataset"]["provenance"]["source_checkpoint_sha256"] != original_hash:
        raise ValueError("Photo checkpoint used a different original DINO/AE")
    if saved["global_step"] != config["expected_global_step"]:
        raise ValueError("Photo checkpoint global_step differs from the explicit selection")
    return {"model_source": "photo", "original_checkpoint": str(original),
            "original_checkpoint_sha256": original_hash, "photo_global_step": saved["global_step"],
            "photo_training_provenance": dict(hp["dataset"]["provenance"]),
            "condition": {"filename": "real_photo.npz", "key": "flux", "rotation_id": 0,
                          "shape": [512, 512, 3], "dtype": "uint8"},
            "stop_if_no_validation_pairs": config.get("stop_if_no_validation_pairs", True),
            "validity": "retain completed native check if later geometry measurement fails"}


def load_model(config: dict, source_cache: Path, device: str):
    if is_photo(config):
        from src.models.diffusion.photo_baseline import load_photo_svr
        return load_photo_svr(config["original_checkpoint"], config["checkpoint"], device=device,
                              source_cache=source_cache, dino_repo=config["dino_repo"])
    from src.models.diffusion.legacy import load_legacy_svr
    return load_legacy_svr(config["checkpoint"], device, dino_repo=config["dino_repo"],
                           source_cache=source_cache)


def retain_native_validity(record: dict) -> dict:
    """A later sampling error does not erase an already completed OCC check."""
    native = record.get("native", {}).get("native_valid_solid")
    if record.get("validity") == "unknown" and isinstance(native, bool):
        return {**record, "validity": "valid" if native else "invalid",
                "validity_basis": "completed native check before later evaluation failure"}
    return record
