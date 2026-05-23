import argparse
import sys
import os.path
from pathlib import Path
import numpy as np
import open3d as o3d
import ray
from PIL import Image
from tqdm import tqdm

sys.path.append('../../../')
from src.brepnet.post.utils import export_edges
from src.brepnet.models.diffusion import Diffusion
from src.brepnet.post.construct_brep import construct_brep_from_datanpz

import torch
from lightning_fabric import seed_everything

os.environ["HTTP_PROXY"] = "http://172.31.178.126:7890"
os.environ["HTTPS_PROXY"] = "http://172.31.178.126:7890"


def normalize_condition_type(condition: str) -> str:
    aliases = {
        "pc": "point_cloud",
        "txt": "text",
    }
    return aliases.get(condition, condition)


def condition_dataset_name(condition_type: str) -> str:
    names = {
        "point_cloud": "pc",
        "text": "txt",
    }
    return names.get(condition_type, condition_type)


def build_condition_conf(condition_type: str) -> dict:
    return {
        "type": condition_type,
        "dataset_names": [condition_dataset_name(condition_type)],
        "cached_features": False,
        "output_dim": 1024,
        "image": {
            "backbone": "dinov2",
            "depth_anything_v2_ckpt": None,
            "augment_probability": 0.0,
        },
        "point_cloud": {
            "encoder": "pointnet",
            "augment_probability": 0.0,
        },
    }


def build_model_conf(autoencoder_checkpoint: str) -> dict:
    return {
        "name": "Diffusion",
        "stage": "diffusion",
        "loss": "l2",
        "latent": {
            "dim": 32,
            "use_cached_latents": False,
            "use_mean": True,
        },
        "padding": {
            "type": "random",
            "max_faces": 30,
            "valid_loss_weight": 0.01,
        },
        "noise": {
            "prediction_type": "epsilon",
            "beta_schedule": "squaredcos_cap_v2",
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "variance_type": "fixed_small",
            "num_train_timesteps": 1000,
        },
        "denoiser": {
            "hidden_dim": 768,
            "num_layers": 24,
            "nhead_divisor": 64,
            "feedforward_dim": 2048,
            "dropout": 0.1,
        },
        "condition_fuser": {
            "type": "cross_attention",
            "condition_dim": 1024,
            "hidden_dim": 1024,
            "num_layers": 4,
        },
        "topology_bias": {
            "enabled": False,
            "scale": 2.0,
        },
        "autoencoder": {
            "name": "AutoEncoder_light",
            "stage": "vae",
            "mode": "frozen_inference",
            "checkpoint": autoencoder_checkpoint,
            "in_channels": 6,
            "latent_channels": 8,
            "hidden_channels": 768,
            "norm": "layer",
            "gaussian_weights": 1e-6,
            "sigmoid": False,
            "num_gat_layers": 5,
            "bottleneck_dim": 768,
            "num_encoder_layers": 8,
            "num_decoder_layers": 8,
            "nhead": 16,
            "with_intersection": True,
            "intersection_dim": 512,
            "intersection_layers": 8,
            "intersection_noise_std": 0.0,
            "trainable_scope": "all",
            "trainable_module_prefixes": ["inter", "classifier"],
            "loss": "l1",
        },
    }


def load_diffusion_weights(model: Diffusion, checkpoint_path: str, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    weights = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            key = key[len("model."):]
        if key.startswith("autoencoder.") or key.startswith("latent_codec.autoencoder.") or key.startswith("ae_model."):
            continue
        weights[key] = value
    model.load_state_dict(weights, strict=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Inference')
    parser.add_argument('--autoencoder-weights', type=str, required=True)
    parser.add_argument('--diffusion-weights', type=str, required=True)
    parser.add_argument('--condition', type=str, required=True)
    parser.add_argument('--input', nargs='+', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default="./inference_output")
    parser.add_argument('--num-samples', type=int, default=32)

    args = parser.parse_args()
    condition_type = normalize_condition_type(args.condition)
    model_conf = build_model_conf(args.autoencoder_weights)
    condition_conf = build_condition_conf(condition_type)

    seed_everything(0)
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("medium")

    model = Diffusion(model_conf, condition_conf)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_diffusion_weights(model, args.diffusion_weights, device)
    model.to(device)
    model.eval()

    print("We have {} data".format(len(args.input)))
    output_dir = Path(args.output_dir)
    (output_dir / "network_pred").mkdir(parents=True, exist_ok=True)

    for id_item, fileitem in enumerate(tqdm(args.input)):
        data = {
            "conditions": {
            }
        }
        num_proposals = args.num_samples
        if condition_type == "point_cloud":
            input_file = Path(fileitem)
            name = input_file.stem
            if not input_file.exists():
                print(f"File {input_file} not found.")
                exit(1)

            pcd = o3d.io.read_point_cloud(str(input_file))
            points = np.array(pcd.points)
            if pcd.has_normals():
                normals = np.array(pcd.normals)
            else:
                normals = np.zeros_like(points)

            # Normalize
            bbox_min = np.min(points, axis=0)
            bbox_max = np.max(points, axis=0)
            center = (bbox_min + bbox_max) / 2
            points -= center
            scale = np.max(bbox_max - bbox_min)
            points /= scale
            points *= 0.9 * 2

            points = np.concatenate([points, normals], axis=1)
            num_sample = 8192
            index = np.random.choice(points.shape[0], num_sample, replace=False)
            points = points[index]
            points_tensor = torch.tensor(points, dtype=torch.float32).to(device)
            data["conditions"]["points"] = points_tensor[None, None, :, :].repeat(num_proposals, 1, 1, 1)
        elif condition_type == "text":
            data["conditions"]["txt"] = [fileitem for item in range(num_proposals)]
            name = f"{id_item:02d}"
        elif condition_type in ("single_img", "sketch"):
            input_file = Path(fileitem)
            name = input_file.stem
            if not input_file.exists():
                print(f"File {input_file} not found.")
                exit(1)

            import torchvision.transforms as T
            transform = T.Compose([
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
            img = np.array(Image.open(input_file))
            if img.shape[2] == 4:
                img = img[:, :, :3]
            # img[img>100]=255
            img = transform(img).to(device)
            img = img[None, None, :].repeat(num_proposals, 1, 1, 1, 1)
            data["conditions"]["imgs"] = img
            data["conditions"]["img_id"] = torch.zeros((num_proposals, 1), dtype=torch.long, device=device)
        else:
            raise ValueError(f"Unknown condition type: {condition_type}")

        with torch.no_grad():
            network_preds = model.inference(num_proposals, device, v_data=data, v_log=True)

        for idx in range((len(network_preds))):
            prefix = f"{name}_{idx:02d}"
            (output_dir/"network_pred"/prefix).mkdir(parents=True, exist_ok=True)
            recon_data = network_preds[idx]
            export_edges(recon_data["pred_edge"], str(output_dir / "network_pred" / prefix / f"edge.obj"))
            np.savez_compressed(str(output_dir / "network_pred" / prefix / f"data.npz"),
                                pred_face_adj_prob=recon_data["pred_face_adj_prob"],
                                pred_face_adj=recon_data["pred_face_adj"].cpu().numpy(),
                                pred_face=recon_data["pred_face"],
                                pred_edge=recon_data["pred_edge"],
                                pred_edge_face_connectivity=recon_data["pred_edge_face_connectivity"],
                                )

    print("Start post processing")
    num_cpus = 8
    ray.init(
        dashboard_host="0.0.0.0",
        dashboard_port=8080,
        num_cpus=num_cpus,
    )
    construct_brep_from_datanpz_ray = ray.remote(num_cpus=1, max_retries=0)(construct_brep_from_datanpz)

    all_folders = os.listdir(output_dir / "network_pred")
    all_folders.sort()

    tasks = []
    for i in range(len(all_folders)):
        tasks.append(construct_brep_from_datanpz_ray.remote(
            output_dir / "network_pred", output_dir/"after_post",
            all_folders[i],
            v_drop_num=3,
            use_cuda=False, from_scratch=True,
            is_log=False, is_ray=True, is_optimize_geom=True, isdebug=False,
        ))
    results = []
    for i in tqdm(range(len(all_folders))):
        try:
            results.append(ray.get(tasks[i]))
        except:
            results.append(None)
    print("Done.")
