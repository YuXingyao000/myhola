"""Differentiable XYZ supervision through the unchanged, frozen legacy decoder.

GT owner pairs are training correspondences, never inference inputs. No topology
classification, thresholding, deduplication, OCC or decoded numpy arrays enter
this loss. Decoder parameters are frozen, but the input latent retains gradients.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn


def geometry_settings(config: Mapping | None) -> dict:
    config = {} if config is None else dict(config)
    unknown = set(config) - {"enabled", "weight", "min_snr", "primitive_chunk_size"}
    if unknown:
        raise ValueError(f"Unknown preference geometry settings: {sorted(unknown)}")
    enabled = config.get("enabled", False)
    weight, min_snr = float(config.get("weight", 0.0)), float(config.get("min_snr", 1.0))
    chunk = config.get("primitive_chunk_size", 16)
    if not isinstance(enabled, bool):
        raise ValueError("geometry.enabled must be a boolean")
    if not math.isfinite(weight) or weight < 0 or not math.isfinite(min_snr) or min_snr < 1:
        raise ValueError("Geometry requires finite weight>=0 and min_snr>=1")
    if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1:
        raise ValueError("geometry.primitive_chunk_size must be a positive integer")
    return {"enabled": enabled, "weight": weight, "min_snr": min_snr, "primitive_chunk_size": chunk}


def predicted_clean_latent(noisy: Tensor, epsilon: Tensor, timestep: Tensor, alphas: Tensor) -> Tensor:
    """Epsilon prediction -> unclipped x0, preserving the gradient to epsilon."""
    alpha = alphas.to(device=noisy.device, dtype=torch.float32)[timestep].view(-1, 1, 1)
    return (noisy.float() - (1 - alpha).sqrt() * epsilon.float()) / alpha.sqrt()


def _global_xyz(local: Tensor, center_scale: Tensor) -> Tensor:
    # Exactly the XYZ part of historical denormalize_coord1112; normals are unused.
    shape = (len(local),) + (1,) * (local.ndim - 2)
    return local[..., :3] * center_scale[:, 3:4].reshape(*shape, 1) + center_scale[:, :3].reshape(*shape, 3)


def decode_supervised_points(autoencoder: nn.Module, face_z: Tensor, edge_faces: Tensor) -> tuple[Tensor, Tensor]:
    """Use the native decoder heads at known training correspondences.

    Process one CAD at a time: face self-attention must not connect different
    objects. ``edge_faces`` keeps the original ordered half-edge owner pairs.
    """
    if autoencoder.training or any(parameter.requires_grad for parameter in autoencoder.parameters()):
        raise ValueError("Geometry supervision requires a frozen decoder in eval mode")
    if face_z.ndim != 2 or face_z.shape[1] != 32 or len(face_z) < 1:
        raise ValueError("Expected a nonempty [N,32] face latent")
    if edge_faces.ndim != 2 or edge_faces.shape[1] != 2 or edge_faces.dtype != torch.long:
        raise ValueError("Expected int64 [E,2] ordered owner pairs")
    face_features = autoencoder.face_attn_proj_in2(face_z)
    mask = torch.zeros((len(face_z), len(face_z)), dtype=torch.bool, device=face_z.device)
    face_features = autoencoder.face_attn2(face_features, mask)
    face_features = autoencoder.face_attn_proj_out2(face_features)
    faces = _global_xyz(autoencoder.face_points_decoder(face_features),
                        autoencoder.face_center_scale_decoder(face_features))
    if len(edge_faces):
        edge_features = autoencoder.inter(face_features[edge_faces])
        edges = _global_xyz(autoencoder.edge_points_decoder(edge_features),
                            autoencoder.edge_center_scale_decoder(edge_features))
    else:
        edges = face_z.new_empty((0, 16, 3))
    return faces, edges


def primitive_chamfer_squared(prediction: Tensor, target: Tensor, chunk_size: int = 16) -> Tensor:
    """One symmetric squared CD per corresponding primitive, without a 1/2."""
    if (len(prediction) != len(target) or prediction.shape[-1] != 3 or target.shape[-1] != 3
            or prediction.ndim < 3 or target.ndim < 3):
        raise ValueError("Expected corresponding primitive point clouds with XYZ channels")
    if not len(prediction):
        return prediction.new_empty((0,))
    prediction, target = prediction.flatten(1, -2).float(), target.flatten(1, -2).float()
    if not prediction.shape[1] or not target.shape[1]:
        raise ValueError("A geometric primitive cannot contain zero sample points")
    distances = []
    for start in range(0, len(prediction), chunk_size):
        # Direct distance avoids cancellation at coincident points in the MM path.
        distance = torch.cdist(prediction[start:start + chunk_size], target[start:start + chunk_size],
                               compute_mode="donot_use_mm_for_euclid_dist").square()
        distances.append(distance.min(-1).values.mean(-1) + distance.min(-2).values.mean(-1))
    return torch.cat(distances)


def predicted_geometry_loss(autoencoder: nn.Module, noisy: Tensor, epsilon: Tensor,
                            timestep: Tensor, alphas: Tensor, batch: dict, *,
                            min_snr: float = 1.0, primitive_chunk_size: int = 16) -> dict[str, Tensor]:
    """Mean over the full GT batch, including zero contributions above the gate."""
    batch_size = len(epsilon)
    if batch_size == 0 or len(batch["gt_geometry"]) != batch_size:
        raise ValueError("Geometry labels must match the nonempty GT batch")
    alpha = alphas.to(device=epsilon.device, dtype=torch.float32)[timestep]
    active = alpha >= min_snr / (1.0 + min_snr)
    surface_loss = epsilon.float().sum() * 0.0
    curve_loss = surface_loss
    # Geometry math is FP32 even when called inside an outer autocast context.
    with torch.autocast(device_type=epsilon.device.type, enabled=False):
        for index in active.nonzero(as_tuple=False).flatten().tolist():
            target = batch["gt_geometry"][index]
            slots = target["face_slots"]
            clean = predicted_clean_latent(noisy[index:index + 1], epsilon[index:index + 1],
                                            timestep[index:index + 1], alphas)[0, slots]
            faces, edges = decode_supervised_points(autoencoder, clean, target["edge_faces"])
            face_value = primitive_chamfer_squared(faces, target["face_points"], primitive_chunk_size).mean()
            edge_value = (primitive_chamfer_squared(edges, target["edge_points"], primitive_chunk_size).mean()
                          if len(edges) else face_value * 0.0)
            if not torch.isfinite(face_value + edge_value):
                raise FloatingPointError(f"Nonfinite geometry loss for {batch['v_prefix'][index]}")
            surface_loss = surface_loss + face_value / batch_size
            curve_loss = curve_loss + edge_value / batch_size
    return {"geometry_surface_loss": surface_loss, "geometry_curve_loss": curve_loss,
            "geometry_loss": surface_loss + curve_loss,
            "geometry_active_fraction": active.float().mean()}
