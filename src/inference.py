import argparse
import sys
import os.path
from pathlib import Path
import numpy as np
import ray
from PIL import Image
from tqdm import tqdm

sys.path.append('../../../')
from src.postprocess.utils import export_edges
from src.models.diffusion import Diffusion
from src.postprocess.construct_brep import construct_brep_from_datanpz

import torch
from lightning_fabric import seed_everything



def build_condition_conf(condition_type: str) -> dict:
    return {
        "type": condition_type,
        "dataset_names": [condition_type],
        "cached_features": False,
        "output_dim": 1024,
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
            "conditioning": "layerwise_cross_attention",
            "hidden_dim": 768,
            "num_layers": 24,
            "nhead_divisor": 64,
            "feedforward_dim": 2048,
            "dropout": 0.1,
        },
        "topology_bias": {
            "enabled": False,
            "source": "gt_adjacency",
            "target": "self_attention",
            "mode": "soft_bias",
            "timestep_weight": "linear_high_noise",
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
        weights[key] = value
    # VAE 权重来源独立指定；其余 key 必须匹配当前逐层条件架构。
    model.load_diffusion_state_dict(weights)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Inference')
    parser.add_argument('--autoencoder-weights', type=str, required=True)
    parser.add_argument('--diffusion-weights', type=str, required=True)
    parser.add_argument('--condition', type=str, required=True, choices=['single_img'])
    parser.add_argument('--input', nargs='+', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default="./inference_output")
    parser.add_argument('--num-samples', type=int, default=32)

    args = parser.parse_args()
    condition_type = args.condition
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
        if condition_type == "single_img":
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
            img = np.array(Image.open(input_file).convert("RGB"))
            img = transform(img).to(device)
            # 单图编码器直接接收 [B,3,224,224]，不添加 view 轴。
            img = img[None].repeat(num_proposals, 1, 1, 1)
            data["conditions"]["imgs"] = img
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
