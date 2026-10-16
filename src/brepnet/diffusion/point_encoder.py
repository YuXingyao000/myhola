"""保留点云条件分支；PointNet 后端只在构造时加载。"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.brepnet.data.rotations import cube24_rotation_matrix


class PointNetEncoder(nn.Module):
    """Point cloud encoder with augmentation and multiple backend support.

    Supports backends:
      - "pointnet": PointNet++ (CUDA) via thirdparty/Pointnet2_PyTorch
      - "pointnet2": PointNet++ deeper variant (CUDA) via thirdparty/Pointnet2_PyTorch
      - "pointnet_noncuda": Pure PyTorch PointNet++ via thirdparty/Pointnet_Pointnet2_pytorch

    Parameters
    ----------
    config : dict
        Configuration dictionary containing:
          - point_encoder: str, backend name
          - is_aug: bool, retained configuration flag (rotation uses rotation_id)
          - aug_points_prob: float, probability of applying point augmentations
    output_dim : int
        Output feature dimension (default 1024).
    """

    def __init__(self, config: dict, output_dim: int = 1024):
        super().__init__()
        self.conf = config
        self.aug_points_prob = config["aug_points_prob"]
        self.aug_rotate = config["is_aug"]
        self.backend = config["point_encoder"]

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
        rotation_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode a point cloud into a feature vector.

        Parameters
        ----------
        v_points : Tensor of shape [B, N, 6]
            Point cloud with XYZ coordinates and normals.
        rotation_id : LongTensor of shape [B] or None
            Identity-first cube-24 rotation id for deterministic augmentation.

        Returns
        -------
        Tensor of shape [B, output_dim]
        """
        if rotation_id is not None:
            rotation_ids = rotation_id.detach().cpu().numpy().astype(np.int64)
            matrix = np.stack([cube24_rotation_matrix(int(item)) for item in rotation_ids], axis=0)
            rotation_3d_matrix = torch.tensor(
                matrix, device=v_points.device, dtype=v_points.dtype
            )
            points = v_points[..., :3]
            normals = v_points[..., 3:6]

            pc2 = (rotation_3d_matrix @ points.permute(0, 2, 1)).permute(0, 2, 1)
            tpc2 = (rotation_3d_matrix @ (points + normals).permute(0, 2, 1)).permute(0, 2, 1)
            normals2 = tpc2 - pc2
            v_points = torch.cat([pc2, normals2], dim=-1)

        # 保留原有策略：增广由 aug_points_prob 控制，不依赖 train/eval。
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
