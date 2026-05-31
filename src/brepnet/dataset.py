import math
import os.path
from pathlib import Path
import random
import sys
import time

import h5py
import numpy as np
import plyfile
import torch
import trimesh
from torch.utils.data import DataLoader
from tqdm import tqdm
import open3d as o3d
from scipy.spatial.transform import Rotation
from shared.common_utils import export_point_cloud, check_dir

from einops import rearrange
import torchvision.transforms as T
import networkx as nx

from torch.nn.utils.rnn import pad_sequence

import torch.nn.functional as F
from PIL import Image

from src.brepnet.data.rotations import NUM_CUBE24_VIEWS, cube24_rotation_matrix


def normalize_coord1112(v_points):
    points = v_points[..., :3]
    normals = v_points[..., 3:]
    shape = points.shape
    num_items = shape[0]
    points = points.reshape(num_items, -1, 3)
    target_points = points + normals.reshape(num_items, -1, 3)

    center = points.mean(dim=1, keepdim=True)
    scale = (torch.linalg.norm(points - center, dim=-1)).max(dim=1, keepdims=True)[0]
    assert scale.min() > 1e-3
    points = (points - center) / (scale[:, None] + 1e-6)
    target_points = (target_points - center) / (scale[:, None] + 1e-6)
    normals = target_points - points
    normals = normals / (1e-6 + torch.linalg.norm(normals, dim=-1, keepdim=True))

    points = points.reshape(shape)
    normals = normals.reshape(shape)

    return points, normals, center[:, 0], scale


def denormalize_coord1112(points, bbox):
    normal = points[..., 3:]
    points = points[..., :3]
    target_points = points + normal
    center = bbox[..., :3]
    scale = bbox[..., 3:4]
    while len(points.shape) > len(center.shape):
        center = center.unsqueeze(1)
        scale = scale.unsqueeze(1)
    points = points * scale + center
    target_points = target_points * scale + center
    normal = target_points - points
    normal = normal / (1e-6 + torch.linalg.norm(normal, dim=-1, keepdim=True))
    points = torch.cat((points, normal), dim=-1)
    return points


class Dummy_dataset(torch.utils.data.Dataset):
    def __init__(self, v_mode, v_conf):
        self.length = v_conf["length"]

    def __len__(self, ):
        return self.length

    def __getitem__(self, idx):
        return "{:08d}".format(idx)

    @staticmethod
    def collate_fn(batch):
        return {
            "v_prefix": batch,
        }


# Input pc range from [-1,1]
def crop_pc(v_pc, v_min_points=1000):
    while True:
        num_points = v_pc.shape[0]
        index = np.arange(num_points)
        np.random.shuffle(index)
        pc_index = np.random.randint(0, num_points - 1)
        center_pos = v_pc[pc_index, :3]
        length_xyz = np.random.rand(3) * 1.0
        obb = o3d.geometry.AxisAlignedBoundingBox(center_pos - length_xyz / 2, center_pos + length_xyz / 2)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(v_pc[:, :3])
        pcd.normals = o3d.utility.Vector3dVector(v_pc[:, 3:])
        inliers_indices = obb.get_point_indices_within_bounding_box(pcd.points)
        cropped = pcd.select_by_index(inliers_indices, invert=True)
        result = np.concatenate((np.asarray(cropped.points), np.asarray(cropped.normals)), axis=-1)
        if result.shape[0] > v_min_points:
            result = result[index % result.shape[0]]
            return result


def rotate_pc(v_pc, v_angle):
    matrix = Rotation.from_euler('xyz', v_angle).as_matrix()
    points = v_pc[:, :3]
    normals = v_pc[:, 3:]
    points1 = (matrix @ points.T).T

    ft = points + normals
    ft1 = (matrix @ ft.T).T

    fn1 = ft1 - points1

    fn1 = fn1 / (1e-6 + np.linalg.norm(fn1, axis=-1, keepdims=True))
    points = points1
    normals = fn1
    return np.concatenate((points, normals), axis=-1)


def noisy_pc(v_pc, v_length=0.02):
    noise = np.random.randn(*v_pc.shape) * v_length
    return v_pc + noise


def downsample_pc(v_pc, v_num_points):
    index = np.arange(v_pc.shape[0])
    np.random.shuffle(index)
    return v_pc[index[:v_num_points]]


def _load_real_photo_flux(cond_dir: Path, rotation_id: int) -> np.ndarray:
    real_photo_path = cond_dir / "real_photo.npz"
    with np.load(real_photo_path) as real_photo_data:
        flux = real_photo_data["flux"]

    # Temporary bridge for the current migrated FLUX data. The intended format
    # is [24, H, W, 3]; the existing single-image data can only be used for the
    # identity rotation.
    # ============ Bridge Block ============
    if flux.ndim == 3:
        if rotation_id != 0:
            raise ValueError(
                f"{real_photo_path} contains a single flux image but rotation_id={rotation_id}. "
                "Set real_photo_ratio=0 for rotation augmentation, or provide flux with shape [24,H,W,3]."
            )
        return flux
    # ============ ~Bridge Block ============
    return flux[int(rotation_id)]


def has_required_condition_files(condition_name, cond_dir: Path, cached_condition: bool, real_photo_ratio: float = 0.0) -> bool:
    if not cond_dir.exists():
        return False

    if condition_name == "multi_img":
        raise NotImplementedError("multi_img is legacy HoLa-BRep multi-view conditioning and is disabled in this runtime.")
    if condition_name == "txt":
        raise NotImplementedError("txt conditioning is disabled until the new dataset format is defined.")
    if condition_name == "natural_img":
        raise NotImplementedError("natural_img/natural.npz is legacy data; use real_photo.npz with single_img or sketch.")

    if condition_name == "pc" and not (cond_dir / "pc.ply").is_file():
        return False

    if condition_name == "single_img" or condition_name == "sketch":
        if cached_condition:
            if real_photo_ratio > 0:
                raise ValueError("real_photo_ratio > 0 is not supported with cached_condition=True")
            return (cond_dir / "img_feature_dinov2.npy").is_file()

        if not (cond_dir / "imgs.npz").is_file():
            return False
        if real_photo_ratio > 0 and not (cond_dir / "real_photo.npz").is_file():
            return False

    return True


def prepare_condition(v_condition_names, v_cond_root, v_folder_path, rotation_id,
                      v_cache_data=None, v_transform=None,
                      v_num_points=None, v_real_photo_ratio=0.0):
    condition = {

    }
    if "multi_img" in v_condition_names:
        raise NotImplementedError("multi_img is legacy HoLa-BRep multi-view conditioning and is disabled in this runtime.")
    if "txt" in v_condition_names:
        raise NotImplementedError("txt conditioning is disabled until the new dataset format is defined.")
    if "natural_img" in v_condition_names:
        raise NotImplementedError("natural_img/natural.npz is legacy data; use real_photo.npz with single_img or sketch.")

    rotation_id = int(rotation_id)
    if rotation_id < 0 or rotation_id >= NUM_CUBE24_VIEWS:
        raise ValueError(f"rotation_id must be in [0, 23], got {rotation_id}")

    if "single_img" in v_condition_names or "sketch" in v_condition_names:
        cond_dir = v_cond_root / v_folder_path

        if v_cache_data:
            if v_real_photo_ratio > 0:
                raise ValueError("real_photo_ratio > 0 is not supported with cached_condition=True")
            ori_data = np.load(v_cond_root / v_folder_path / "img_feature_dinov2.npy")
            if "single_img" in v_condition_names:
                img_features = torch.from_numpy(ori_data[rotation_id][None, :]).float()
                img_id = np.array([0], dtype=np.int64)
            else:
                img_features = torch.from_numpy(ori_data[rotation_id + NUM_CUBE24_VIEWS][None, :]).float()
                img_id = np.array([0], dtype=np.int64)

            condition["img_id"] = torch.from_numpy(img_id)
            condition["img_features"] = img_features
        else:
            ori_data = np.load(v_cond_root / v_folder_path / "imgs.npz")
            if "single_img" in v_condition_names:
                imgs = ori_data["svr_imgs"][rotation_id]
                img_id = np.array([0], dtype=np.int64)
            else:
                imgs = ori_data["sketch_imgs"][rotation_id]
                img_id = np.array([0], dtype=np.int64)

            if v_real_photo_ratio > 0:
                real_photo = _load_real_photo_flux(cond_dir, rotation_id)
                if np.random.rand() < v_real_photo_ratio:
                    imgs = real_photo

            transformed_imgs = v_transform(imgs)
            condition["ori_imgs"] = torch.from_numpy(imgs)
            condition["imgs"] = transformed_imgs
        condition["img_id"] = torch.from_numpy(img_id)
    if "pc" in v_condition_names:
        pc = o3d.io.read_point_cloud(str(v_cond_root / v_folder_path / "pc.ply"))
        points = np.concatenate((np.asarray(pc.points), np.asarray(pc.normals)), axis=-1)
        assert points.shape[0] == 10000
        condition["points"] = torch.from_numpy(points).float()[None,]
    return condition


class AutoEncoder_dataset3(torch.utils.data.Dataset):
    def __init__(self, v_training_mode, v_conf):
        super(AutoEncoder_dataset3, self).__init__()
        self.mode = v_training_mode
        self.conf = v_conf
        # self.max_intersection = 500
        self.scale_factor = int(v_conf["scale_factor"])
        if v_training_mode == "testing":
            listfile = v_conf['test_dataset']
        elif v_training_mode == "training":
            listfile = v_conf['train_dataset']
        elif v_training_mode == "validation":
            listfile = v_conf['val_dataset']
        else:
            raise

        self.data_folders = [item.strip() for item in open(listfile).readlines()]
        self.root = Path(v_conf["data_root"])

        # Cond related
        self.is_aug = v_conf["is_aug"]
        self.condition = list(v_conf["condition_names"])
        self.conditional_data_root = Path(v_conf["condition_root"]) if self.condition else None
        self.cached_condition = v_conf["cached_condition"]
        if v_training_mode == "validation":
            self.is_aug = 0
            self.cached_condition = False

        # Check data
        data_folders = []
        for item in self.data_folders:
            if os.path.exists(self.root / item / "data.npz"):
                data_folders.append(item)
        self.data_folders = data_folders

        # Check cond data
        if len(self.condition) > 0:
            data_folders = []
            for item in self.data_folders:
                if has_required_condition_files(self.condition, self.conditional_data_root / item, self.cached_condition):
                    data_folders.append(item)
            print("Filter out {} folders without feat".format(len(self.data_folders) - len(data_folders)))
            self.data_folders = data_folders

        if v_conf["is_overfit"]:
            self.data_folders = self.data_folders[:100]
            if v_training_mode == "training":
                self.data_folders = self.data_folders * 100

        self.ori_length = len(self.data_folders)
        if v_training_mode == "testing" and self.is_aug == 1:
            self.data_folders = self.data_folders * NUM_CUBE24_VIEWS
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        # self.disable_half = v_conf["disable_half"]
        print(len(self.data_folders))

    def __len__(self):
        # return 100
        return len(self.data_folders) * self.scale_factor

    def __getitem__(self, idx):
        # idx = 0
        folder_path = self.data_folders[idx % len(self.data_folders)]
        output_prefix = folder_path
        data_npz = np.load(str(self.root / folder_path / "data.npz"))
        face_points = torch.from_numpy(data_npz['sample_points_faces'])
        edge_points = torch.from_numpy(data_npz['sample_points_lines'])

        condition_rotation_id = 0
        if self.is_aug == 0:
            matrix = np.identity(3)
        elif self.is_aug == 1:
            if self.mode == "testing":
                rotation_id = (idx // self.ori_length) % NUM_CUBE24_VIEWS
                condition_rotation_id = rotation_id
                output_prefix = "{}_{}".format(folder_path, rotation_id)
            else:
                rotation_id = int(np.random.randint(0, NUM_CUBE24_VIEWS))
                condition_rotation_id = rotation_id
            matrix = cube24_rotation_matrix(rotation_id)
        elif self.is_aug == 2:
            matrix = Rotation.from_euler('xyz', np.random.rand(3) * np.pi * 2).as_matrix()
        if self.is_aug != 0:
            matrix = torch.as_tensor(matrix, dtype=face_points.dtype)
            fp = face_points[..., :3].reshape(-1, 3)
            lp = edge_points[..., :3].reshape(-1, 3)
            fp1 = (matrix @ fp.T).T
            lp1 = (matrix @ lp.T).T

            if face_points.shape[-1] > 3 and edge_points.shape[-1] > 3:
                fn = face_points[..., 3:].reshape(-1, 3)
                ln = edge_points[..., 3:].reshape(-1, 3)
                ft = fp + fn
                lt = lp + ln
                ft1 = (matrix @ ft.T).T
                lt1 = (matrix @ lt.T).T

                fn1 = ft1 - fp1
                ln1 = lt1 - lp1

                fn1 = fn1 / (1e-6 + torch.linalg.norm(fn1, dim=-1, keepdim=True))
                ln1 = ln1 / (1e-6 + torch.linalg.norm(ln1, dim=-1, keepdim=True))
                face_points[..., 3:] = fn1.reshape(face_points[..., 3:].shape)
                edge_points[..., 3:] = ln1.reshape(edge_points[..., 3:].shape)

            face_points[..., :3] = fp1.reshape(face_points[..., :3].shape)
            edge_points[..., :3] = lp1.reshape(edge_points[..., :3].shape)

        num_faces = face_points.shape[0]
        num_edges = edge_points.shape[0]

        face_adj = torch.from_numpy(data_npz['face_adj'])
        edge_face_connectivity = torch.from_numpy(data_npz['edge_face_connectivity'])
        edge_face_connectivity = edge_face_connectivity[edge_face_connectivity[:, 1] != edge_face_connectivity[:, 2]]

        zero_positions = torch.from_numpy(data_npz['zero_positions'])
        if zero_positions.shape[0] > edge_face_connectivity.shape[0]:
            index = np.random.choice(zero_positions.shape[0], edge_face_connectivity.shape[0], replace=False)
            zero_positions = zero_positions[index]

        face_points_norm, face_normal_norm, face_center, face_scale = normalize_coord1112(face_points)
        edge_points_norm, edge_normal_norm, edge_center, edge_scale = normalize_coord1112(edge_points)

        face_norm = torch.cat((face_points_norm, face_normal_norm), dim=-1)
        edge_norm = torch.cat((edge_points_norm, edge_normal_norm), dim=-1)

        face_bbox = torch.cat((face_center, face_scale), dim=-1)
        edge_bbox = torch.cat((edge_center, edge_scale), dim=-1)

        condition = prepare_condition(self.condition, self.conditional_data_root, folder_path, condition_rotation_id,
                                      self.cached_condition, self.transform, self.conf["num_points"])

        return (
            output_prefix,
            face_points, edge_points,
            face_norm, edge_norm,
            face_bbox, edge_bbox,
            edge_face_connectivity, zero_positions, face_adj,
            condition
        )

    @staticmethod
    def collate_fn(batch):
        (
            prefix,
            face_points, edge_points,
            face_norm, edge_norm,
            face_bbox, edge_bbox,
            edge_face_connectivity, zero_positions, face_adj,
            conditions
        ) = zip(*batch)
        bs = len(prefix)

        flat_zero_positions = []
        face_counts = []

        num_faces = 0
        num_edges = 0
        edge_conn_num = []
        for i in range(bs):
            edge_face_connectivity[i][:, 0] += num_edges
            edge_face_connectivity[i][:, 1:] += num_faces
            edge_conn_num.append(edge_face_connectivity[i].shape[0])
            flat_zero_positions.append(zero_positions[i] + num_faces)
            num_faces += face_norm[i].shape[0]
            num_edges += edge_norm[i].shape[0]
            face_counts.append(face_norm[i].shape[0])
        face_counts = torch.tensor(face_counts, dtype=torch.long)
        num_sum_edges = sum(edge_conn_num)
        edge_attn_mask = torch.ones((num_sum_edges, num_sum_edges), dtype=bool)
        id_cur = 0
        for i in range(bs):
            edge_attn_mask[id_cur:id_cur + edge_conn_num[i], id_cur:id_cur + edge_conn_num[i]] = False
            id_cur += edge_conn_num[i]

        max_faces_in_batch = face_counts.max()
        valid_mask = torch.zeros((bs, max_faces_in_batch), dtype=bool)
        for i in range(bs):
            valid_mask[i, :face_counts[i]] = True
        attn_mask = torch.ones((num_faces, num_faces), dtype=bool)
        id_cur = 0
        for i in range(bs):
            attn_mask[id_cur:id_cur + face_norm[i].shape[0], id_cur: id_cur + face_norm[i].shape[0]] = False
            id_cur += face_norm[i].shape[0]

        dtype = torch.float32
        flat_zero_positions = torch.cat(flat_zero_positions, dim=0)

        keys = conditions[0].keys()
        condition_out = {key: [] for key in keys}
        for idx in range(len(conditions)):
            for key in keys:
                condition_out[key].append(conditions[idx][key])

        for key in keys:
            condition_out[key] = torch.stack(condition_out[key], dim=0) if isinstance(condition_out[key][0], torch.Tensor) else \
                condition_out[key]

        return {
            "v_prefix"              : prefix,
            "face_points"           : torch.cat(face_points, dim=0).to(dtype),
            "face_norm"             : torch.cat(face_norm, dim=0).to(dtype),
            "edge_points"           : torch.cat(edge_points, dim=0).to(dtype),
            "edge_norm"             : torch.cat(edge_norm, dim=0).to(dtype),
            "face_bbox"             : torch.cat(face_bbox, dim=0).to(dtype),
            "edge_bbox"             : torch.cat(edge_bbox, dim=0).to(dtype),

            "edge_face_connectivity": torch.cat(edge_face_connectivity, dim=0),
            "zero_positions"        : flat_zero_positions,
            "attn_mask"             : attn_mask,
            "edge_attn_mask"        : edge_attn_mask,

            "face_counts"           : face_counts,
            "valid_mask"            : valid_mask,
            "conditions"            : condition_out
        }


class Diffusion_dataset(torch.utils.data.Dataset):
    def __init__(self, training_mode, config):
        super(Diffusion_dataset, self).__init__()
        self.mode = training_mode
        self.conf = config
        epoch_scale_factor      = int(self.conf["epoch_scale_factor"])
        self.cached_latent_root = Path(self.conf["cached_latent_root"])
        self.raw_data_root      = Path(self.conf["raw_data_root"]) if self.conf["raw_data_root"] is not None else None
        self.load_topology      = bool(self.conf["load_topology"])

        # --- Data split ---
        if training_mode == "testing":
            self.data_split = Path(self.conf['test_dataset'])
        elif training_mode == "training":
            self.data_split = Path(self.conf['train_dataset'])
        elif training_mode == "validation":
            self.data_split = Path(self.conf['val_dataset'])
        else:
            raise ValueError(f"Invalid training mode: {training_mode}")
        epoch_scale_factor = 1 if training_mode != "training" else epoch_scale_factor
        print("Use deduplicate list ", self.data_split)
        filelist = [item.strip() for item in open(self.data_split).readlines()]
        filelist.sort()

        # --- Padding Settings ---
        self.padding            = self.conf["padding"]
        self.max_faces          = self.conf["max_faces"]

        # --- Condition Settings ---
        self.is_aug             = self.conf["is_aug"]           # Whether to apply rotation augmentation
        self.cached_condition   = self.conf["cached_condition"] # Whether to use cached condition features
        self.real_photo_ratio   = self.conf["real_photo_ratio"] # Ratio of real photo to use for conditioning
        
        if training_mode == "validation":
            self.is_aug = False
            self.cached_condition = False
            self.real_photo_ratio = 0.0

        # --- Condition Names ---
        self.condition_name         = self.conf["condition_name"]
        self.condition_data_root  = Path(self.conf["condition_data_root"])
        if self.condition_name == "single_img" or self.condition_name == "sketch":
            self.img_transform = T.Compose([
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

        if self.condition_name is not None:
            data_folders = []
            for item in filelist:
                if has_required_condition_files(self.condition_name, self.condition_data_root / item, self.cached_condition, self.real_photo_ratio):
                    data_folders.append(item)
            print("Filter out {} folders without feat".format(len(filelist) - len(data_folders)))
            filelist = data_folders

        # Display which local GPU/device (rank) this dataset is being used on.
        # Try to get local rank for multi-GPU DDP setups; fallback to 0 if not found.
        print("=======================================================")
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
        print(f"===== {self.mode} mode dataset : local_rank={local_rank} ======")
        print("Total data count:", len(filelist))
        # --- Overfitting Training Mode ---
        if self.conf["is_overfit"]:
            print("\033[91m[!] OVERFIT TRAINING MODE [!]\033[0m")
            self.data_folders = filelist[:100] * epoch_scale_factor
        else:
            self.data_folders = filelist * epoch_scale_factor

        print("Epoch scale factor:", epoch_scale_factor)
        print("Total data count:", len(self.data_folders))
        print("=======================================================")

    def __len__(self):
        return len(self.data_folders)

    def _load_pc_points(self, folder_path):
        pc = o3d.io.read_point_cloud(str(self.condition_data_root / folder_path / "pc.ply"))
        points = np.concatenate((np.asarray(pc.points), np.asarray(pc.normals)), axis=-1)
        assert points.shape[0] == 10000
        return torch.from_numpy(points).float()[None,]

    def _load_imgs(self, folder_path, rotation_id):
        occ_imgs = np.load(self.condition_data_root / folder_path / "imgs.npz")
        if self.condition_name == "single_img":
            imgs = occ_imgs["svr_imgs"][rotation_id]
        else:
            imgs = occ_imgs["sketch_imgs"][rotation_id]
        img_id = np.array([0], dtype=np.int64)

        if self.real_photo_ratio > 0:
            real_photo = _load_real_photo_flux(self.condition_data_root / folder_path, rotation_id)
            if np.random.rand() < self.real_photo_ratio:
                imgs = real_photo

        return imgs, img_id

    def __getitem__(self, idx):
        model_id = self.data_folders[idx]
        if self.is_aug:
            rotation_id = int(np.random.randint(0, NUM_CUBE24_VIEWS))
        else:
            rotation_id = 0
        latent_np = np.load(self.cached_latent_root / (model_id + f"_{rotation_id}") / "features.npy")
        latent_tensor = torch.from_numpy(latent_np)
        num_faces = latent_tensor.shape[0]

        if self.padding == "zero":
            # Zero padding
            latent_tensor_padded = torch.zeros((self.max_faces, latent_tensor.shape[-1]), dtype=latent_tensor.dtype)
            latent_tensor_padded[:num_faces] = latent_tensor
            face_mask = torch.zeros((self.max_faces,), dtype=torch.bool)
            face_mask[:num_faces] = True
        elif self.padding == "random":
            # Random padding
            positions = torch.arange(self.max_faces, device=latent_tensor.device)
            mandatory_mask = positions < num_faces
            random_indices = (torch.rand((self.max_faces,), device=latent_tensor.device) * num_faces).long()
            indices = torch.where(mandatory_mask, positions, random_indices)
            r_indices = torch.argsort(torch.rand((self.max_faces,), device=latent_tensor.device), dim=0)
            index = indices.gather(0, r_indices)
            latent_tensor_padded = latent_tensor[index]
            face_mask = torch.ones((self.max_faces,), dtype=torch.bool)
        else:
            raise ValueError(f"Invalid padding method: {self.padding}")

        # --- Topology: load GT adjacency and map to padded space ---
        padded_adj = None
        if self.load_topology and self.raw_data_root is not None:
            brep_path = self.raw_data_root / model_id / "data.npz"
            if brep_path.exists():
                brep_data = np.load(str(brep_path))
                face_adj = torch.from_numpy(brep_data["face_adj"]).float()  # [num_faces_orig, num_faces_orig]
                # Clip to actual num_faces (in case data has more faces than features)
                n = min(face_adj.shape[0], num_faces)
                face_adj_clipped = torch.zeros(num_faces, num_faces)
                face_adj_clipped[:n, :n] = face_adj[:n, :n]
                # Map to padded space using the same index used for latent stats.
                if self.padding == "random":
                    padded_adj = face_adj_clipped[index][:, index]  # [max_faces, max_faces]
                else:  # zero padding
                    padded_adj = torch.zeros(self.max_faces, self.max_faces)
                    padded_adj[:num_faces, :num_faces] = face_adj_clipped
            if padded_adj is None:
                padded_adj = torch.zeros(self.max_faces, self.max_faces)

        if self.condition_name is None:
            condition = {}
        elif self.condition_name == "pc":
            condition = {
                "points": self._load_pc_points(model_id)
            }
        elif self.condition_name == "single_img" or self.condition_name == "sketch":
            imgs, img_id = self._load_imgs(model_id, rotation_id)
            condition = {
                "ori_imgs" : torch.from_numpy(imgs),
                "imgs" : self.img_transform(imgs),
                "img_id" : torch.from_numpy(img_id)
            }
        else:
            raise NotImplementedError(f"Invalid condition name: {self.condition_name}")

        return (
            model_id,
            latent_tensor_padded,
            face_mask,
            condition,
            rotation_id,
            padded_adj,
        )

    @staticmethod
    def collate_fn(batch):
        (
            model_id, latent_tensor_padded, face_mask, batch_condition, rotation_id, padded_adj
        ) = zip(*batch)

        latent_tensor_padded = torch.stack(latent_tensor_padded, dim=0)
        face_mask = torch.stack(face_mask, dim=0)
        rotation_id = torch.tensor(rotation_id)

        # Topology: stack adjacency matrices (None if not loaded)
        if padded_adj[0] is not None:
            face_adj_batch = torch.stack(padded_adj, dim=0)  # [B, max_faces, max_faces]
        else:
            face_adj_batch = None

        keys = batch_condition[0].keys()
        condition_out = {key: [] for key in keys}
        for one_condition in batch_condition:
            for key in keys:
                condition_out[key].append(one_condition[key])

        for key in keys:
            if isinstance(condition_out[key][0], torch.Tensor):
                condition_out[key] = torch.stack(condition_out[key], dim=0)

        result = {
            "v_prefix"           : model_id,
            "cached_latent_stats": latent_tensor_padded,
            "face_mask"          : face_mask,
            "conditions"         : condition_out,
            "rotation_id"         : rotation_id,
        }
        if face_adj_batch is not None:
            result["face_adj"] = face_adj_batch
        return result


class Diffusion_dataset_mm(Diffusion_dataset):
    def __init__(self, training_mode, config):
        super(Diffusion_dataset_mm, self).__init__(training_mode, config)
        self.cond_prob = list(self.conf["cond_prob"])
        self.cond_prob_acc = np.cumsum(self.cond_prob)
        return

    def __getitem__(self, idx):
        # idx = 0
        folder_path = self.data_folders[idx]
        if self.is_aug != 0:
            rotation_id = int(np.random.randint(0, NUM_CUBE24_VIEWS))
        else:
            rotation_id = 0
        data_npz = np.load(self.latent_root / (folder_path + f"_{rotation_id}") / "features.npy")
        latent_stats = torch.from_numpy(data_npz)
        num_faces = latent_stats.shape[0]

        if self.padding == "zero":
            cached_latent_stats = torch.zeros((self.max_faces, latent_stats.shape[-1]), dtype=latent_stats.dtype)
            cached_latent_stats[:num_faces] = latent_stats
            face_mask = torch.zeros((self.max_faces,), dtype=torch.bool)
            face_mask[:num_faces] = True
        elif self.padding == "random":
            positions = torch.arange(self.max_faces, device=latent_stats.device)
            mandatory_mask = positions < num_faces
            random_indices = (
                    torch.rand((self.max_faces,), device=latent_stats.device) * num_faces).long()
            indices = torch.where(mandatory_mask, positions, random_indices)
            r_indices = torch.argsort(torch.rand((self.max_faces,), device=latent_stats.device), dim=0)
            index = indices.gather(0, r_indices)
            cached_latent_stats = latent_stats[index]
            face_mask = torch.ones((self.max_faces,), dtype=torch.bool)
        else:
            raise ValueError("Invalid padding method")

        sampled_prob = np.random.rand()
        idx = self.cond_prob_acc.shape[0] - (sampled_prob < self.cond_prob_acc).sum(axis=-1)
        used_condition = []
        if self.condition_names[idx] == "mm":
            available_condition = [item for item in self.condition_names if item != "uncond" and item != "mm"]
            num_condition = len(available_condition)
            rand_onehot = np.random.rand(num_condition) > 0.5
            used_condition = [available_condition[i] for i in range(num_condition) if rand_onehot[i]]
        else:
            used_condition.append(self.condition_names[idx])

        condition = prepare_condition(used_condition, self.conditional_data_root, folder_path, rotation_id,
                                      self.cached_condition, self.transform, self.conf["num_points"],
                                      v_real_photo_ratio=self.real_photo_ratio)
        condition["name"] = self.condition_names[idx]

        return (
            folder_path,
            cached_latent_stats,
            face_mask,
            condition,
            rotation_id
        )

    @staticmethod
    def collate_fn(batch):
        (
            v_prefix, v_cached_latent_stats, face_mask, conditions, rotation_id
        ) = zip(*batch)

        cached_latent_stats = torch.stack(v_cached_latent_stats, dim=0)
        face_mask = torch.stack(face_mask, dim=0)
        rotation_id = torch.tensor(rotation_id)

        condition_out = {
            "names"       : [],
            "points"      : [],

            "img_features": [],
            "img_id"      : [],
            "ori_imgs"    : [],
            "imgs"        : [],
        }

        id_condition = {
            "pc"            : [],
            "single_img"    : [],
            "sketch"        : [],
            "single_img_rec": [],
            "sketch_rec"    : [],
        }
        for id_batch, condition in enumerate(conditions):
            condition_out["names"].append(condition["name"])
            if condition["name"] == "pc":
                condition_out["points"].append(condition["points"])
                id_condition["pc"].append(id_batch)
            elif condition["name"] in ["single_img", "sketch"]:
                if "img_features" in condition:
                    id_cur = sum([item.shape[0] for item in condition_out["img_features"]])
                    condition_out["img_features"].append(condition["img_features"])
                else:
                    id_cur = sum([item.shape[0] for item in condition_out["ori_imgs"]])
                    condition_out["ori_imgs"].append(condition["ori_imgs"])
                    condition_out["imgs"].append(condition["imgs"])
                if condition["name"] == "single_img":
                    id_condition["single_img"].append(id_batch)
                    id_condition["single_img_rec"].append(id_cur)
                elif condition["name"] == "sketch":
                    id_condition["sketch"].append(id_batch)
                    id_condition["sketch_rec"].append(id_cur)
                condition_out["img_id"].append(condition["img_id"])
            elif condition["name"] == "mm":
                continue

        if len(condition_out["points"]) > 0:
            condition_out["points"] = torch.concatenate(condition_out["points"], dim=0)
        if len(condition_out["img_features"]) > 0:
            condition_out["img_features"] = torch.concatenate(condition_out["img_features"], dim=0)
        if len(condition_out["ori_imgs"]) > 0:
            condition_out["ori_imgs"] = torch.concatenate(condition_out["ori_imgs"], dim=0)
        if len(condition_out["imgs"]) > 0:
            condition_out["imgs"] = torch.concatenate(condition_out["imgs"], dim=0)
        if len(condition_out["img_id"]) > 0:
            condition_out["img_id"] = torch.concatenate(condition_out["img_id"], dim=0)
        condition_out["id_batch"] = id_condition
        return {
            "v_prefix"           : v_prefix,
            "cached_latent_stats": cached_latent_stats,
            "face_mask"          : face_mask,
            "conditions"         : condition_out,
            "rotation_id"         : rotation_id,
        }
