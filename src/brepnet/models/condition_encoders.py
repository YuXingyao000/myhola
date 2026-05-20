"""
Condition Encoders for HoLa-BRep Diffusion Model.

Provides modular condition encoding for image, point cloud, and text modalities:
  - DINOv2ImageEncoder: Frozen DINOv2 ViT-L backbone with learnable projection
  - CameraEmbedding: Learnable camera view embeddings for multi-view conditioning
  - PointNetEncoder: Point cloud encoder with multiple backend options
  - TextEncoder: Frozen sentence-transformers encoder with projection
  - ConditionExtractor: Orchestrator that unifies all modalities into a single tensor
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.spatial.transform import Rotation
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# DINOv2 Image Encoder
# ---------------------------------------------------------------------------

class DINOv2ImageEncoder(nn.Module):
    """Frozen DINOv2 ViT-L backbone with a learnable projection head.

    Supports two backbone variants controlled by `backbone`:
      - "dinov2" (default): HuggingFace `facebook/dinov2-large`
      - "depth_anything_v2": Depth Anything V2 ViT-L encoder (fine-tuned DINOv2)

    Both produce [B, 257, 1024] tokens (1 CLS + 16x16 patches).
    The learnable projection maps these to [B, 257, projection_dim].

    Parameters
    ----------
    projection_dim : int
        Output feature dimension (default 1024).
    backbone : str
        Which backbone to load: "dinov2" or "depth_anything_v2".
    depth_anything_v2_ckpt : str or None
        Path to Depth Anything V2 checkpoint (required if backbone="depth_anything_v2").
    """

    def __init__(
        self,
        projection_dim: int = 1024,
        backbone: str = "dinov2",
        depth_anything_v2_ckpt: Optional[str] = None,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.projection_dim = projection_dim

        # ----- Backbone loading -----
        if backbone == "depth_anything_v2":
            from thirdparty.Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2

            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
            }
            encoder = 'vitl'
            _da_v2 = DepthAnythingV2(**model_configs[encoder])
            if depth_anything_v2_ckpt is None:
                depth_anything_v2_ckpt = (
                    f'thirdparty/Depth_Anything_V2/checkpoints/depth_anything_v2_{encoder}.pth'
                )
            _da_v2.load_state_dict(
                torch.load(depth_anything_v2_ckpt, map_location='cpu')
            )
            self.img_model = _da_v2.pretrained
            del _da_v2
        elif backbone == "dinov2":
            from transformers import Dinov2Model
            self.img_model = Dinov2Model.from_pretrained('facebook/dinov2-large')
        else:
            raise ValueError(
                f"Unknown backbone '{backbone}'. Choose 'dinov2' or 'depth_anything_v2'."
            )

        # Freeze backbone
        for param in self.img_model.parameters():
            param.requires_grad = False

        # ----- Learnable projection head -----
        self.projection = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, projection_dim),
        )

    def train(self, mode: bool = True):
        """Keep the frozen backbone in eval mode regardless of parent state."""
        super().train(mode)
        self.img_model.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract image features and project them.

        Parameters
        ----------
        images : Tensor of shape [B, 3, 224, 224]
            Input images (normalized for DINOv2).

        Returns
        -------
        Tensor of shape [B, 257, projection_dim]
            Projected image features (1 CLS + 256 patch tokens).
        """
        with torch.no_grad():
            if self.backbone_name == "depth_anything_v2":
                backbone_out = self.img_model.forward_features(images)
                img_feature = torch.cat([
                    backbone_out["x_norm_clstoken"].unsqueeze(1),
                    backbone_out["x_norm_patchtokens"],
                ], dim=1)
            else:
                # HuggingFace Dinov2Model
                img_feature = self.img_model(images).last_hidden_state

        # img_feature: [B, 257, 1024]
        return self.projection(img_feature)


# ---------------------------------------------------------------------------
# Camera Embedding
# ---------------------------------------------------------------------------

class CameraEmbedding(nn.Module):
    """Learnable camera view embedding for multi-view conditioning.

    Maps discrete camera view indices (0..7) to dense embeddings.

    Parameters
    ----------
    num_views : int
        Number of discrete camera views (default 8).
    output_dim : int
        Output embedding dimension (default 1024).
    """

    def __init__(self, num_views: int = 8, output_dim: int = 1024):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Embedding(num_views, 256),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, view_ids: torch.Tensor) -> torch.Tensor:
        """Compute camera embeddings.

        Parameters
        ----------
        view_ids : LongTensor of shape [B] or [B, num_views]
            Camera view indices.

        Returns
        -------
        Tensor of shape matching input with last dim = output_dim
        """
        return self.embedding(view_ids)


# ---------------------------------------------------------------------------
# PointNet Encoder
# ---------------------------------------------------------------------------

class PointNetEncoder(nn.Module):
    """Point cloud encoder with augmentation and multiple backend support.

    Supports backends:
      - "pointnet": PointNet++ (CUDA) via thirdparty/Pointnet2_PyTorch
      - "pointnet2": PointNet++ deeper variant (CUDA) via thirdparty/Pointnet2_PyTorch
      - "pointnet_noncuda": Pure PyTorch PointNet++ via thirdparty/Pointnet_Pointnet2_pytorch

    Parameters
    ----------
    v_conf : dict
        Configuration dictionary containing:
          - point_encoder: str, backend name
          - is_aug: bool, whether to apply rotation augmentation
          - aug_points_prob: float, probability of applying point augmentations
    output_dim : int
        Output feature dimension (default 1024).
    """

    def __init__(self, v_conf: dict, output_dim: int = 1024):
        super().__init__()
        self.conf = v_conf
        self.aug_points_prob = v_conf["aug_points_prob"]
        self.aug_rotate = v_conf["is_aug"]
        self.backend = v_conf["point_encoder"]

        if self.backend == "pointnet":
            from thirdparty.Pointnet2_PyTorch.pointnet2_ops_lib.pointnet2_ops.pointnet2_modules import (
                PointnetSAModuleMSG, PointnetSAModule
            )

            self.point_model = nn.ModuleList()
            c_in = 6
            with_bn = False
            self.point_model.append(
                PointnetSAModuleMSG(
                    npoint=1024,
                    radii=[0.05, 0.1],
                    nsamples=[16, 32],
                    mlps=[[c_in, 32], [c_in, 64]],
                    use_xyz=True,
                    bn=with_bn
                )
            )
            c_out_0 = 32 + 64

            c_in = c_out_0
            self.point_model.append(
                PointnetSAModuleMSG(
                    npoint=256,
                    radii=[0.1, 0.2],
                    nsamples=[16, 32],
                    mlps=[[c_in, 64], [c_in, 128]],
                    use_xyz=True,
                    bn=with_bn
                )
            )
            c_out_1 = 64 + 128
            c_in = c_out_1
            self.point_model.append(
                PointnetSAModuleMSG(
                    npoint=64,
                    radii=[0.2, 0.4],
                    nsamples=[16, 32],
                    mlps=[[c_in, 128], [c_in, 256]],
                    use_xyz=True,
                    bn=with_bn
                )
            )
            c_out_2 = 128 + 256

            c_in = c_out_2
            self.point_model.append(
                PointnetSAModule(
                    mlp=[c_in, 256, 512, 1024],
                    use_xyz=True,
                    bn=with_bn
                )
            )
            self.fc_layer = nn.Sequential(
                nn.Linear(1024, 1024),
                nn.LayerNorm(1024),
                nn.SiLU(),
                nn.Linear(1024, output_dim),
            )

        elif self.backend == "pointnet2":
            from thirdparty.Pointnet2_PyTorch.pointnet2_ops_lib.pointnet2_ops.pointnet2_modules import (
                PointnetSAModuleMSG, PointnetSAModule
            )

            self.point_model = nn.ModuleList()
            c_in = 6
            with_bn = False
            self.point_model.append(
                PointnetSAModuleMSG(
                    npoint=1024,
                    radii=[0.05, 0.1, 0.2],
                    nsamples=[16, 32, 128],
                    mlps=[[c_in, 16, 16], [c_in, 32, 32], [c_in, 64, 64]],
                    use_xyz=True,
                    bn=with_bn
                )
            )
            c_out_0 = 16 + 32 + 64

            c_in = c_out_0
            self.point_model.append(
                PointnetSAModuleMSG(
                    npoint=256,
                    radii=[0.1, 0.2, 0.4],
                    nsamples=[16, 32, 128],
                    mlps=[[c_in, 32, 32], [c_in, 64, 64], [c_in, 128, 128]],
                    use_xyz=True,
                    bn=with_bn
                )
            )
            c_out_1 = 32 + 64 + 128
            c_in = c_out_1
            self.point_model.append(
                PointnetSAModuleMSG(
                    npoint=64,
                    radii=[0.2, 0.4, 0.8],
                    nsamples=[16, 32, 128],
                    mlps=[[c_in, 64, 64], [c_in, 128, 128], [c_in, 128, 128]],
                    use_xyz=True,
                    bn=with_bn
                )
            )
            c_out_2 = 64 + 128 + 128

            c_in = c_out_2
            self.point_model.append(
                PointnetSAModuleMSG(
                    npoint=16,
                    radii=[0.4, 0.8, 1.6],
                    nsamples=[16, 32, 128],
                    mlps=[[c_in, 128, 128], [c_in, 128, 128], [c_in, 128, 128]],
                    use_xyz=True,
                    bn=with_bn
                )
            )

            c_out_3 = 128 + 128 + 128
            c_in = c_out_3
            self.point_model.append(
                PointnetSAModule(
                    mlp=[c_in, 1024, 1024, 1024],
                    use_xyz=True,
                    bn=with_bn
                )
            )
            self.fc_layer = nn.Sequential(
                nn.Linear(1024, 1024),
                nn.LayerNorm(1024),
                nn.SiLU(),
                nn.Linear(1024, output_dim),
            )

        elif self.backend == "pointnet_noncuda":
            from thirdparty.Pointnet_Pointnet2_pytorch.models.pointnet2_utils import (
                PointNetSetAbstractionMsg, PointNetSetAbstraction
            )

            self.sa1 = PointNetSetAbstractionMsg(
                1024, [0.05, 0.1], [16, 32], 6,
                [[16, 16, 32], [32, 32, 64]]
            )
            self.sa2 = PointNetSetAbstractionMsg(
                256, [0.1, 0.2], [16, 32], 32 + 64,
                [[64, 64, 128], [64, 96, 128]]
            )
            self.sa3 = PointNetSetAbstractionMsg(
                64, [0.2, 0.4], [16, 32], 128 + 128,
                [[128, 196, 256], [128, 196, 256]]
            )
            self.sa4 = PointNetSetAbstractionMsg(
                16, [0.4, 0.8], [16, 32], 256 + 256,
                [[256, 256, 512], [256, 384, 512]]
            )
            self.sa5 = PointNetSetAbstraction(
                None, None, None, 512 + 512 + 3,
                [512, 512, 1024], True
            )

            self.fc1 = nn.Linear(1024, 512)
            self.drop1 = nn.Dropout(0.4)
            self.fc2 = nn.Linear(512, 256)
            self.drop2 = nn.Dropout(0.5)
            self.fc3 = nn.Linear(256, output_dim)

        else:
            raise ValueError(
                f"Unknown point_encoder backend '{self.backend}'. "
                "Choose from: 'pointnet', 'pointnet2', 'pointnet_noncuda'."
            )

    def augment(self, v_points: torch.Tensor) -> torch.Tensor:
        """Apply stochastic augmentations to point clouds.

        Augmentations (applied with probability aug_points_prob):
          - Random crop around a seed point
          - Random downsample (to 1000..N points)
          - Gaussian noise (sigma=0.02)
          - Random normal dropout (50% chance to zero normals)

        Parameters
        ----------
        v_points : Tensor of shape [B, N, 6]
            Point cloud with XYZ + normals.

        Returns
        -------
        Tensor of shape [B, M, 6] where M <= N
        """
        pc = v_points

        if np.random.rand() < self.aug_points_prob:
            # Crop
            bs = pc.shape[0]
            num_points = pc.shape[1]
            pc_index = torch.randint(0, pc.shape[1], (bs,), device=pc.device)
            center_pos = torch.gather(
                pc, 1, pc_index[:, None, None].repeat(1, 1, 6)
            )[..., :3]
            length_xyz = torch.rand((bs, 3), device=pc.device) * 1.0
            bbox_min = center_pos - length_xyz[:, None, :]
            bbox_max = center_pos + length_xyz[:, None, :]
            mask = torch.logical_not(
                ((pc[:, :, :3] > bbox_min) & (pc[:, :, :3] < bbox_max)).all(dim=-1)
            )

            sort_results = torch.sort(mask.long(), descending=True)
            mask = sort_results.values
            pc_sorted = torch.gather(
                pc, 1, sort_results.indices[:, :, None].repeat(1, 1, 6)
            )
            num_valid = mask.sum(dim=-1)
            index1 = torch.rand((bs, num_points), device=pc.device) * num_valid[:, None]
            index2 = torch.arange(num_points, device=pc.device)[None].repeat(bs, 1)
            index = torch.where(mask.bool(), index2, index1)
            pc = pc_sorted[
                torch.arange(bs)[:, None].repeat(1, num_points), index.long()
            ]

            # Downsample
            index = np.arange(num_points)
            np.random.shuffle(index)
            num_points = np.random.randint(1000, num_points)
            pc = pc[:, index[:num_points]]

            # Noise
            noise = torch.randn_like(pc) * 0.02
            pc = pc + noise

            # Mask normal
            pc[..., 3:] = 0. if torch.rand(1) > 0.5 else pc[..., 3:]

        return pc

    def forward(
        self,
        v_points: torch.Tensor,
        euler_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode a point cloud into a feature vector.

        Parameters
        ----------
        v_points : Tensor of shape [B, N, 6]
            Point cloud with XYZ coordinates and normals.
        euler_id : LongTensor of shape [B] or None
            Discrete rotation index for deterministic augmentation.
            Encodes rotation as: x = id%4 * 90deg, y = (id//4)%4 * 90deg,
            z = (id//16) * 90deg.

        Returns
        -------
        Tensor of shape [B, output_dim]
        """
        # Apply rotation augmentation based on euler_id
        if euler_id is not None:
            angles = torch.stack([
                euler_id % 4 * torch.pi / 2,
                euler_id // 4 % 4 * torch.pi / 2,
                euler_id // 16 * torch.pi / 2,
            ], dim=1)
            matrix = Rotation.from_euler('xyz', angles.cpu().numpy()).as_matrix()
            rotation_3d_matrix = torch.tensor(
                matrix, device=v_points.device, dtype=v_points.dtype
            )
            points = v_points[..., :3]
            normals = v_points[..., 3:6]

            pc2 = (rotation_3d_matrix @ points.permute(0, 2, 1)).permute(0, 2, 1)
            tpc2 = (rotation_3d_matrix @ (points + normals).permute(0, 2, 1)).permute(0, 2, 1)
            normals2 = tpc2 - pc2
            v_points = torch.cat([pc2, normals2], dim=-1)

        # Apply stochastic augmentations
        aug_pc = self.augment(v_points)

        if self.backend in ("pointnet", "pointnet2"):
            l_xyz = [aug_pc[:, :, :3].contiguous().float()]
            l_features = [aug_pc.permute(0, 2, 1).contiguous().float()]
            with torch.autocast(device_type=aug_pc.device.type, dtype=torch.float32):
                for i in range(len(self.point_model)):
                    li_xyz, li_features = self.point_model[i](l_xyz[i], l_features[i])
                    l_xyz.append(li_xyz)
                    l_features.append(li_features)
            features = self.fc_layer(l_features[-1][..., 0])

        elif self.backend == "pointnet_noncuda":
            aug_pc = aug_pc.permute(0, 2, 1)
            l0_points = aug_pc
            l0_xyz = aug_pc[:, :3, :]

            l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
            l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
            l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
            l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
            l5_xyz, l5_points = self.sa5(l4_xyz, l4_points)

            x = l5_points.view(aug_pc.shape[0], 1024)
            x = self.drop1(F.relu(self.fc1(x)))
            x = self.drop2(F.relu(self.fc2(x)))
            features = self.fc3(x)

        else:
            raise RuntimeError(f"Unsupported backend: {self.backend}")

        return features


# ---------------------------------------------------------------------------
# Text Encoder
# ---------------------------------------------------------------------------

class TextEncoder(nn.Module):
    """Frozen sentence-transformers text encoder with learnable projection.

    Uses 'Alibaba-NLP/gte-large-en-v1.5' as the backbone.

    Parameters
    ----------
    output_dim : int
        Output feature dimension (default 1024).
    model_path : str
        HuggingFace model path for sentence-transformers.
    """

    def __init__(
        self,
        output_dim: int = 1024,
        model_path: str = 'Alibaba-NLP/gte-large-en-v1.5',
    ):
        super().__init__()
        from sentence_transformers import SentenceTransformer

        self.txt_model = SentenceTransformer(model_path, trust_remote_code=True)
        for param in self.txt_model.parameters():
            param.requires_grad = False
        self.txt_model.eval()

        self.projection = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, output_dim),
        )

    def train(self, mode: bool = True):
        """Keep the frozen text model in eval mode regardless of parent state."""
        super().train(mode)
        self.txt_model.eval()
        return self

    def forward(self, texts: List[str]) -> torch.Tensor:
        """Encode a batch of text strings.

        Parameters
        ----------
        texts : List[str]
            Batch of text descriptions.

        Returns
        -------
        Tensor of shape [B, output_dim]
        """
        with torch.no_grad():
            txt_feat = self.txt_model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=False,
                device=self.txt_model.device,
            )
            txt_feat = torch.stack(txt_feat, dim=0)

        return self.projection(txt_feat)


# ---------------------------------------------------------------------------
# Condition Extractor (Orchestrator)
# ---------------------------------------------------------------------------

class ConditionExtractor(nn.Module):
    """Orchestrates multiple condition encoders into a unified condition tensor.

    Instantiates the appropriate encoder(s) based on the configured modalities
    and fuses their outputs into a single [B, 1, seq_len, dim] tensor consumed
    by the diffusion model.

    Supports:
      - Raw input data (images, point clouds, text)
      - Pre-cached features (img_features, txt_features in data["conditions"])
      - Multi-view image averaging with camera embeddings

    Parameters
    ----------
    v_conf : dict
        Configuration dictionary. Expected keys:
          - condition: list of modality strings, e.g. ["single_img"], ["multi_img"],
            ["sketch"], ["pc"], ["txt"]
          - is_aug: bool, augmentation flag for point clouds
          - aug_points_prob: float, augmentation probability for point clouds
          - point_encoder: str, backend for PointNetEncoder (if "pc" in condition)
          - backbone: str, "dinov2" or "depth_anything_v2" (optional, default "dinov2")
          - depth_anything_v2_ckpt: str (optional)
    projection_dim : int
        Unified output dimension for all encoders (default 1024).
    """

    def __init__(self, v_conf: dict, projection_dim: int = 1024):
        super().__init__()
        self.projection_dim = projection_dim
        self.with_img = False
        self.with_pc = False
        self.with_txt = False

        condition = v_conf.get("condition", [])

        # --- Image encoder ---
        if ("single_img" in condition or "multi_img" in condition
                or "sketch" in condition):
            self.with_img = True
            backbone = v_conf.get("backbone", "dinov2")
            da_ckpt = v_conf.get("depth_anything_v2_ckpt", None)
            self.image_encoder = DINOv2ImageEncoder(
                projection_dim=projection_dim,
                backbone=backbone,
                depth_anything_v2_ckpt=da_ckpt,
            )
            self.camera_embedding = CameraEmbedding(
                num_views=8,
                output_dim=projection_dim,
            )

        # --- Point cloud encoder ---
        if "pc" in condition:
            self.with_pc = True
            self.point_encoder = PointNetEncoder(v_conf, output_dim=projection_dim)

        # --- Text encoder ---
        if "txt" in condition:
            self.with_txt = True
            self.text_encoder = TextEncoder(output_dim=projection_dim)

    def train(self, mode: bool = True):
        """Keep all frozen sub-encoders in eval mode."""
        super().train(mode)
        if self.with_img:
            self.image_encoder.train(mode)  # delegates to its own override
        if self.with_txt:
            self.text_encoder.train(mode)  # delegates to its own override
        return self

    def forward(self, data: Dict) -> Optional[torch.Tensor]:
        """Extract condition features from input data.

        Parameters
        ----------
        data : dict
            Must contain a "conditions" sub-dict with modality-specific data:
              - For images: "imgs" ([B, V, 3, 224, 224]) or "img_features" (cached)
              - For images: "img_id" ([B, V]) camera view indices
              - For point clouds: "points" ([B, 1, N, 6])
              - For text: "txt" (List[str]) or "txt_features" (cached)
            May also contain "id_aug" for point cloud rotation augmentation.

        Returns
        -------
        Tensor of shape [B, 1, seq_len, projection_dim] or None
            Unified condition tensor. seq_len depends on the modality:
              - Image: 257 (CLS + patches) or 1 (after multi-view averaging to global)
              - Point cloud: 1 (global feature)
              - Text: 1 (global feature)
            Returns None if no condition modalities are active.
        """
        condition = None

        if self.with_img:
            conditions = data["conditions"]

            if "img_features" in conditions:
                # Pre-cached image features: [B, num_imgs, 257, 1024] or [B*num_imgs, 257, 1024]
                img_feature = conditions["img_features"]
                num_imgs = img_feature.shape[1]
            else:
                imgs = conditions["imgs"]
                num_imgs = imgs.shape[1]
                imgs = imgs.reshape(-1, 3, 224, 224)
                img_feature = self.image_encoder(imgs)

            img_idx = conditions["img_id"]

            # img_feature is already projected by the encoder
            # For pre-cached features, apply projection manually
            if "img_features" in conditions:
                img_feature = self.image_encoder.projection(img_feature)

            if img_idx.shape[-1] > 1:
                # Multi-view: add camera embeddings and average
                camera_emb = self.camera_embedding(img_idx)
                img_feature = (
                    img_feature.reshape(-1, num_imgs, self.projection_dim) + camera_emb
                ).mean(dim=1)
            else:
                img_feature = img_feature

            condition = img_feature[:, None]

        elif self.with_pc:
            pc = data["conditions"]["points"]
            euler_id = data.get("id_aug", None)
            feat = self.point_encoder(pc[:, 0, :, :], euler_id)
            condition = feat[:, None]

        elif self.with_txt:
            conditions = data["conditions"]

            if "txt_features" in conditions:
                txt_feat = conditions["txt_features"]
                # Apply projection to pre-cached features
                txt_feat = self.text_encoder.projection(txt_feat)
            else:
                txt = conditions["txt"]
                txt_feat = self.text_encoder(txt)

            condition = txt_feat[:, None]

        # Reshape to [B, 1, seq_len, dim] for consistency
        if condition is not None and condition.dim() == 3:
            # condition is [B, 1, dim] -> [B, 1, 1, dim]
            condition = condition.unsqueeze(2)
        elif condition is not None and condition.dim() == 2:
            # condition is [B, dim] -> [B, 1, 1, dim]
            condition = condition.unsqueeze(1).unsqueeze(1)

        return condition
