"""原版 SVR：保留 GT 去噪训练，用离线同图偏好对更新相对生成倾向。"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import math
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .preference_geometry import geometry_settings, predicted_geometry_loss


def per_sample_mse(prediction: Tensor, target: Tensor) -> Tensor:
    """每个完整 [30,32] latent 等权，不按去重后的面数加权。"""
    return (prediction.float() - target.float()).square().flatten(1).mean(1)


def diffusion_dpo_loss(
    policy_winner: Tensor,
    policy_loser: Tensor,
    reference_winner: Tensor,
    reference_loser: Tensor,
    beta: float,
) -> tuple[Tensor, Tensor]:
    """输入为逐样本 epsilon MSE；正 logit 表示偏好较 reference 改善。

    沿用 Diffusion-DPO 官方实现的 uniform-t、mean-MSE 与 -beta/2 约定：
    https://github.com/SalesforceAIResearch/DiffusionDPO/blob/main/train.py
    beta 已是该简化目标的有效系数，不再额外乘 diffusion 步数。
    """
    policy_gap = policy_winner - policy_loser
    reference_gap = reference_winner.detach() - reference_loser.detach()
    logits = -0.5 * beta * (policy_gap - reference_gap)
    return -F.logsigmoid(logits).mean(), logits


class LegacySVRDenoiser(nn.Module):
    """仅保留实际参与单图去噪的五个模块；不复制 DINO / VAE。"""

    def __init__(self, legacy: nn.Module):
        super().__init__()
        self.width = legacy.dim_total
        for name in ("p_embed", "time_embed", "net1", "fc_out", "img_fc"):
            setattr(self, name, getattr(legacy, name))

    def image_condition(self, dino_features: Tensor) -> Tensor:
        return self.img_fc(dino_features)[:, None]

    def forward(self, noisy_latent: Tensor, timestep: Tensor, condition: Tensor) -> Tensor:
        # 与历史 sincos_embedding 的 dtype、sin/cos 顺序和 10000 周期一致。
        half = self.width // 2
        frequency = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=timestep.dtype, device=timestep.device) / half,
        )
        phase = timestep[:, None].float() * frequency[None]
        time = self.time_embed(torch.cat((phase.cos(), phase.sin()), -1))[:, None]
        latent = self.p_embed(noisy_latent)
        tokens = torch.cat((latent, condition.repeat(1, noisy_latent.shape[1], 1)), -1)
        return self.fc_out(self.net1(tokens + time))


@contextmanager
def deterministic_policy(policy: nn.Module):
    """偏好比较关闭 dropout，但保留 policy 梯度，防止随机 mask 伪造优劣。"""
    was_training = policy.training
    policy.eval()
    try:
        yield
    finally:
        policy.train(was_training)


class PreferenceDiffusion(nn.Module):
    """共享一个冻结 DINO；policy 可训练，reference 固定；训练不加载 OCC。"""

    def __init__(self, cfg: dict):
        super().__init__()
        from .legacy import build_legacy_svr, load_legacy_svr

        self.geometry = geometry_settings(cfg.get("geometry"))
        loader_options = dict(source_cache=cfg.get("legacy_source_cache"), dino_repo=cfg.get("dino_repo"))
        if cfg.get("restoring_checkpoint", False):
            legacy, self.provenance = build_legacy_svr(
                cfg["legacy_model_config"], with_autoencoder=self.geometry["enabled"], **loader_options)
        elif cfg.get("model_source") == "photo":
            from .photo_baseline import load_photo_svr
            legacy, self.provenance = load_photo_svr(
                cfg["legacy_checkpoint"], cfg["photo_checkpoint"], **loader_options)
            if self.provenance["photo"]["global_step"] != cfg["photo_global_step"]:
                raise ValueError("Photo policy/reference initialization step differs from the selected baseline")
        else:
            legacy, self.provenance = load_legacy_svr(cfg["legacy_checkpoint"], **loader_options)
        self.legacy_model_config = self.provenance["model_config"]
        self.noise_scheduler = legacy.noise_scheduler
        self.image_encoder = legacy.img_model.requires_grad_(False).eval()
        self.policy = LegacySVRDenoiser(legacy).requires_grad_(True)
        self.reference = deepcopy(self.policy).requires_grad_(False).eval()
        self.beta = float(cfg["beta_dpo"])
        self.preference_weight = float(cfg["preference_weight"])
        if self.beta <= 0 or self.preference_weight < 0:
            raise ValueError("beta_dpo must be positive; preference_weight must be nonnegative.")
        if self.geometry["enabled"]:
            # Existing original VAE, including all original weights; no new head.
            self.geometry_decoder = legacy.ae_model.requires_grad_(False).eval()
        # No geometry: the historical DPO state is unchanged. Geometry: keep the
        # original frozen AE in the checkpoint for a self-contained strict resume.
        del legacy

    def initialize_preference_weights(self, path: str | Path, data_provenance: dict) -> dict:
        """Start a new run from pre-geometry DPO; keep the original frozen reference.

        This is weight initialization only. Adam state, step count and RNG are not
        restored. Adding the original frozen decoder is the only permitted state
        extension; malformed/missing policy or changed frozen weights are errors.
        """
        from omegaconf import OmegaConf
        from src.brepnet.data.preference_dataset import sha256_file

        path = Path(path).expanduser().resolve()
        checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        hp = checkpoint["hyper_parameters"]
        if OmegaConf.is_config(hp):
            hp = OmegaConf.to_container(hp, resolve=True)
        model_config = hp["model"]
        if (model_config["name"] != "PreferenceDiffusion" or
                model_config["legacy_model_config"] != self.legacy_model_config or
                geometry_settings(model_config.get("geometry"))["enabled"]):
            raise ValueError("Initialize from the same pre-geometry PreferenceDiffusion architecture")
        if float(model_config["beta_dpo"]) != self.beta or float(model_config["preference_weight"]) != self.preference_weight:
            raise ValueError("Preference objective differs from the initialization checkpoint")
        source_data = {key: value for key, value in data_provenance.items() if key != "geometry_supervision"}
        if hp["dataset"]["provenance"] != source_data:
            raise ValueError("Initialization requires the same pair pool, GT split and source checkpoint")
        expected = {f"model.{key}": value for key, value in self.state_dict().items()
                    if not key.startswith("geometry_decoder.")}
        weights = checkpoint["state_dict"]
        if set(weights) != set(expected):
            raise ValueError("Preference initialization state keys differ")
        for key, current in expected.items():
            value = weights[key]
            if value.shape != current.shape or value.dtype != current.dtype:
                raise ValueError(f"Preference initialization shape/dtype differs: {key}")
            if not key.startswith("model.policy.") and not torch.equal(value, current):
                raise ValueError(f"Initialization frozen reference/DINO differs from original SVR: {key}")
        prefix = "model.policy."
        self.policy.load_state_dict({key[len(prefix):]: value for key, value in weights.items()
                                     if key.startswith(prefix)}, strict=True)
        return {"checkpoint": str(path), "sha256": sha256_file(path),
                "source_global_step": checkpoint["global_step"], "source_epoch": checkpoint["epoch"],
                "policy_strict_load": True, "reference_and_dino_equal_original": True,
                "optimizer_restored": False, "additional_trainable_parameters": 0}

    def train(self, mode: bool = True):
        super().train(mode)
        self.image_encoder.eval()
        self.reference.eval()
        if self.geometry["enabled"]:
            self.geometry_decoder.eval()
        return self

    def _add_geometry(self, result: dict[str, Tensor], noisy: Tensor, prediction: Tensor,
                      timestep: Tensor, batch: dict, v_test: bool) -> dict[str, Tensor]:
        # Keep the existing validation loss and its checkpoint ranking unchanged.
        if self.geometry["enabled"] and not v_test:
            geometry = predicted_geometry_loss(
                self.geometry_decoder, noisy, prediction, timestep, self.noise_scheduler.alphas_cumprod,
                batch, min_snr=self.geometry["min_snr"],
                primitive_chunk_size=self.geometry["primitive_chunk_size"])
            result.update(geometry)
            result["geometry_weighted_loss"] = self.geometry["weight"] * geometry["geometry_loss"]
            result["total_loss"] = result["total_loss"] + result["geometry_weighted_loss"]
        return result

    def _noise(self, clean: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        timestep = torch.randint(
            self.noise_scheduler.config.num_train_timesteps, (len(clean),), device=clean.device,
        )
        noise = torch.randn_like(clean)
        return self.noise_scheduler.add_noise(clean, noise, timestep), timestep, noise

    def forward(self, batch: dict, v_test: bool = False) -> dict[str, Tensor]:
        # GT 与 pair 从不同对象流抽样；一次 DINO forward，冻结特征由双方共享。
        has_pairs = not v_test and "winner_latent" in batch
        if "image_features" in batch:
            # 无图像增强时复用冻结DINO的FP32输出；img_fc仍由policy训练。
            image_features = (torch.cat((batch["image_features"], batch["pair_image_features"]))
                              if has_pairs else batch["image_features"]).detach()
        else:
            images = torch.cat((batch["gt_image"], batch["pair_image"])) if has_pairs else batch["gt_image"]
            with torch.no_grad():
                image_features = self.image_encoder(images)
        batch_size = len(batch["gt_latent"])
        gt_features = image_features[:batch_size]
        noisy_gt, gt_timestep, gt_noise = self._noise(batch["gt_latent"])
        gt_prediction = self.policy(noisy_gt, gt_timestep, self.policy.image_condition(gt_features))
        ddpm_loss = per_sample_mse(gt_prediction, gt_noise).mean()
        result = {"ddpm_loss": ddpm_loss, "total_loss": ddpm_loss}
        if not has_pairs:
            return self._add_geometry(result, noisy_gt, gt_prediction, gt_timestep, batch, v_test)

        winner, loser = batch["winner_latent"], batch["loser_latent"]
        noisy_winner, timestep, noise = self._noise(winner)
        noisy_loser = self.noise_scheduler.add_noise(loser, noise, timestep)
        noisy_pairs = torch.cat((noisy_winner, noisy_loser))
        pair_timestep = timestep.repeat(2)
        targets = noise.repeat(2, 1, 1)
        pair_features = image_features[batch_size:]
        with deterministic_policy(self.policy):
            condition = self.policy.image_condition(pair_features).repeat(2, 1, 1)
            policy_losses = per_sample_mse(self.policy(noisy_pairs, pair_timestep, condition), targets)
        with torch.no_grad():
            condition = self.reference.image_condition(pair_features).repeat(2, 1, 1)
            reference_losses = per_sample_mse(self.reference(noisy_pairs, pair_timestep, condition), targets)
        policy_winner, policy_loser = policy_losses.chunk(2)
        reference_winner, reference_loser = reference_losses.chunk(2)
        preference_loss, logits = diffusion_dpo_loss(
            policy_winner, policy_loser, reference_winner, reference_loser, self.beta,
        )
        result.update({
            "preference_loss": preference_loss,
            "preference_logit": logits.detach().mean(),
            "preference_accuracy": (logits.detach() > 0).float().mean(),
            "winner_mse": policy_winner.detach().mean(),
            "loser_mse": policy_loser.detach().mean(),
            "total_loss": ddpm_loss + self.preference_weight * preference_loss,
        })
        return self._add_geometry(result, noisy_gt, gt_prediction, gt_timestep, batch, v_test)

    @torch.no_grad()
    def sample_latents(
        self,
        images: Tensor,
        *,
        initial_generator: torch.Generator | None = None,
        step_generator: torch.Generator | None = None,
    ) -> Tensor:
        """原生 CPU 初始噪声 + device 上逐步噪声；返回去重前的 latent。

        省略 generator 时沿用各设备的全局 RNG；显式复现需分别为 CPU 和
        images.device 建立同 seed generator。原始协议比较使用 batch=1。
        """
        if self.training:
            raise ValueError("Call eval() before sampling.")
        device = images.device
        condition = self.policy.image_condition(self.image_encoder(images))
        latent = torch.randn((len(images), 30, 32), generator=initial_generator).to(device)
        self.noise_scheduler.set_timesteps(1000, device=device)
        for timestep in self.noise_scheduler.timesteps:
            time_batch = timestep.expand(len(images))
            prediction = self.policy(latent, time_batch, condition)
            latent = self.noise_scheduler.step(
                prediction, timestep, latent, generator=step_generator,
            ).prev_sample
        return latent
