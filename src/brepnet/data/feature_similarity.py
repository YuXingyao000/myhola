import importlib
from pathlib import Path
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
from lightning_fabric import seed_everything
from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Dinov2Model

from src.brepnet.dataset import Diffusion_dataset


COMBINED_IMAGE_KEYS = ("sketch_img", "sketch_imgs")
IMAGE_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)
IMAGE_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)
IMAGE_WEIGHT_PREFIXES = ("img_model.", "img_fc.", "camera_embedding.")


class Diffusion_dataset_fidelity(Diffusion_dataset):
    def __init__(self, v_training_mode, v_conf):
        if OmegaConf.is_config(v_conf):
            dataset_conf = OmegaConf.to_container(v_conf, resolve=False)
        else:
            dataset_conf = dict(v_conf)
        dataset_conf["condition_names"] = []
        dataset_conf["condition_root"] = None
        super().__init__(v_training_mode, dataset_conf)

        self.conditional_data_root = Path(v_conf["condition_root"]) if v_conf["condition_root"] is not None else None
        self.combined_filename = v_conf.get("combined_filename", "combined_imgs.npz")
        if self.conditional_data_root is None:
            raise ValueError("condition_root must point to the combined image root.")

        base_folders = self.data_folders
        filtered_folders = [
            item for item in base_folders
            if (self.conditional_data_root / item / self.combined_filename).exists()
        ]
        print("Filter out {} folders without combined imgs".format(len(base_folders) - len(filtered_folders)))
        self.data_folders = filtered_folders
        print("Total fidelity data num:", len(self.data_folders))

    def load_combined_condition(self, prefix: str) -> dict[str, torch.Tensor]:
        combined_path = self.conditional_data_root / prefix / self.combined_filename
        with np.load(combined_path) as data:
            missing_keys = [key for key in COMBINED_IMAGE_KEYS if key not in data]
            if missing_keys:
                raise KeyError(f"{combined_path} missing keys: {missing_keys}")
            return {
                key: torch.from_numpy(data[key].copy())
                for key in COMBINED_IMAGE_KEYS
            }

    def __getitem__(self, idx):
        prefix, cached_latent_stats, face_mask, _, id_aug, face_adj = super().__getitem__(idx)
        condition = self.load_combined_condition(prefix)
        return prefix, cached_latent_stats, face_mask, condition, id_aug, face_adj


class DiffusionImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim_condition = 1024
        self.img_model = Dinov2Model.from_pretrained('facebook/dinov2-large')
        for param in self.img_model.parameters():
            param.requires_grad = False
        self.img_model.eval()

        self.img_fc = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, self.dim_condition),
        )
        self.camera_embedding = nn.Sequential(
            nn.Embedding(8, 256),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, self.dim_condition),
        )

    @staticmethod
    def normalize_image_tensor(images: torch.Tensor) -> torch.Tensor:
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        else:
            images = images.float()
            if images.max() > 1.0:
                images = images / 255.0

        if images.ndim == 4:
            images = images.permute(0, 3, 1, 2)
        elif images.ndim == 5:
            images = images.permute(0, 1, 4, 2, 3)
        else:
            raise ValueError(f"Unsupported image tensor shape: {tuple(images.shape)}")

        mean = IMAGE_MEAN.to(device=images.device, dtype=images.dtype)
        std = IMAGE_STD.to(device=images.device, dtype=images.dtype)
        if images.ndim == 4:
            mean = mean.view(1, 3, 1, 1)
            std = std.view(1, 3, 1, 1)
        else:
            mean = mean.view(1, 1, 3, 1, 1)
            std = std.view(1, 1, 3, 1, 1)
        return (images - mean) / std

    def encode_image_batch(self, images: torch.Tensor) -> torch.Tensor:
        images = self.normalize_image_tensor(images)
        if images.ndim == 4:
            dino_feature = self.img_model(images).last_hidden_state
            # return self.img_fc(dino_feature)
            return dino_feature
        elif images.ndim == 5:
            batch_size = images.shape[0]
            projected_chunks = []
            for i in range(0, batch_size):
                dino_feature = self.img_model(images[i]).last_hidden_state
                # projected_chunks.append(self.img_fc(dino_feature))
                projected_chunks.append(dino_feature)
            return torch.stack(projected_chunks, dim=0)
        else:
            raise ValueError(f"Unsupported image tensor shape: {tuple(images.shape)}")

    def extract_features(self, conditions: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        missing_keys = [key for key in COMBINED_IMAGE_KEYS if key not in conditions]
        if missing_keys:
            raise KeyError(f"Missing combined image keys: {missing_keys}")

        features = {}
        for key in COMBINED_IMAGE_KEYS:
            features[key] = self.encode_image_batch(conditions[key])
        return features

    def forward(self, batch_or_conditions: dict[str, Any]) -> dict[str, torch.Tensor]:
        if "conditions" in batch_or_conditions:
            conditions = batch_or_conditions["conditions"]
        else:
            conditions = batch_or_conditions
        return self.extract_features(conditions)


def move_to_device(data: Any, device: torch.device) -> Any:
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    if isinstance(data, list):
        return [move_to_device(value, device) for value in data]
    if isinstance(data, tuple):
        return tuple(move_to_device(value, device) for value in data)
    return data


def resolve_dataset_cls(dataset_name: str):
    if dataset_name in globals():
        return globals()[dataset_name]

    dataset_mod = importlib.import_module("src.brepnet.dataset")
    return getattr(dataset_mod, dataset_name)


def build_dataloader(v_cfg: DictConfig, split: str) -> DataLoader:
    dataset_cls = resolve_dataset_cls(v_cfg["dataset"]["name"])
    dataset = dataset_cls(split, v_cfg["dataset"])

    return DataLoader(
        dataset,
        batch_size=v_cfg["trainer"]["batch_size"],
        collate_fn=dataset_cls.collate_fn,
        num_workers=v_cfg["trainer"]["num_workers"],
        pin_memory=True,
    )


def build_image_encoder(v_cfg: DictConfig, device: torch.device) -> DiffusionImageEncoder:
    model = DiffusionImageEncoder()

    checkpoint_path = v_cfg["trainer"].get("resume_from_checkpoint")
    if checkpoint_path is not None and checkpoint_path != "none":
        print(f"Resuming from {checkpoint_path}")
        weights = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["state_dict"]
        if any(key.startswith("model.") for key in weights):
            weights = {
                key[len("model."):]: value
                for key, value in weights.items()
                if key.startswith("model.")
            }
        image_weights = {
            key: value
            for key, value in weights.items()
            if key.startswith(IMAGE_WEIGHT_PREFIXES)
        }
        missing_keys, unexpected_keys = model.load_state_dict(image_weights, strict=False)
        if unexpected_keys:
            raise RuntimeError(f"Unexpected image encoder keys: {unexpected_keys}")
        missing_non_img_keys = [key for key in missing_keys if not key.startswith(IMAGE_WEIGHT_PREFIXES)]
        if missing_non_img_keys:
            raise RuntimeError(f"Unexpected missing non-image keys: {missing_non_img_keys}")

    model.to(device)
    model.eval()
    return model


def extract_image_features_from_batch(
        model: DiffusionImageEncoder,
        batch: dict[str, Any],
        device: torch.device,
) -> dict[str, torch.Tensor]:
    conditions = move_to_device(batch["conditions"], device)
    with torch.inference_mode():
        return model(conditions)


def compute_max_sketch_similarity(features: dict[str, torch.Tensor]) -> torch.Tensor:
    sketch_img = features["sketch_img"].reshape(features["sketch_img"].shape[0], -1)
    sketch_imgs = features["sketch_imgs"].reshape(features["sketch_imgs"].shape[0], features["sketch_imgs"].shape[1], -1)

    sketch_img = torch.nn.functional.normalize(sketch_img, dim=-1)
    sketch_imgs = torch.nn.functional.normalize(sketch_imgs, dim=-1)
    cosine = torch.einsum("bd,bkd->bk", sketch_img, sketch_imgs)
    return cosine.max(dim=1).values


def collect_similarity_statistics(
        v_cfg: DictConfig,
        splits: tuple[str, ...] = ("training", "validation", "testing"),
        device: torch.device | None = None,
) -> dict[str, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
    model = build_image_encoder(v_cfg, device)

    split_results = {}
    for split in splits:
        dataloader = build_dataloader(v_cfg, split)
        split_scores = []
        for batch in tqdm(dataloader, desc=f"{split}", leave=True):
            features = extract_image_features_from_batch(model, batch, device)
            split_scores.append(compute_max_sketch_similarity(features).cpu())
        split_results[split] = torch.cat(split_scores, dim=0).numpy() if split_scores else np.empty((0,), dtype=np.float32)
    return split_results


def plot_similarity_histogram(split_results: dict[str, np.ndarray], output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "sketch_similarity_hist.png"
    bins = np.arange(-1.0, 1.05, 0.05)

    plt.figure(figsize=(10, 6))
    for split, values in split_results.items():
        if values.size == 0:
            continue
        plt.hist(values, bins=bins, alpha=0.5, label=f"{split} ({values.size})")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Frequency")
    plt.title("Max similarity between sketch_img and sketch_imgs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


@hydra.main(config_name="train_diffusion.yaml", config_path="../../../configs/brepnet/", version_base="1.1")
def main(v_cfg: DictConfig):
    seed_everything(0)
    split_results = collect_similarity_statistics(v_cfg)
    output_root = Path.cwd() / "fidelity_outputs_2"
    hist_path = plot_similarity_histogram(split_results, output_root)
    for split, values in split_results.items():
        print(f"{split}: count={values.size}, mean={values.mean() if values.size > 0 else float('nan'):.4f}, std={values.std() if values.size > 0 else float('nan'):.4f}")
    print(f"Histogram saved to {hist_path}")
    return split_results


if __name__ == "__main__":
    main()
