"""
Decoder Robustness Diagnostic.

Tests how sensitive the VAE decoder is to perturbations in latent space.
This tells us: if diffusion generates latents that are "close but not exact",
how badly does the decoded geometry degrade?

Usage:
    python -m src.brepnet.experiments.decoder_robustness \
        --vae-checkpoint /path/to/vae.ckpt \
        --data-root /path/to/data \
        --test-list src/brepnet/data/list/deduplicated_deepcad_testing_7_30_sample_100.txt \
        --output ./outputs/decoder_robustness

Output:
    - A table showing CD vs noise level
    - Per-sample breakdown for visualization
    - A JSON summary
"""

import argparse
import json
import importlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def load_vae(checkpoint_path, device="cuda"):
    """Load pretrained VAE from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract model config from checkpoint
    if "hyper_parameters" in ckpt:
        cfg = ckpt["hyper_parameters"]
        model_cfg = cfg.get("model", cfg)
    else:
        # Fallback: assume AutoEncoder_1119_light with default config
        model_cfg = {
            "name": "AutoEncoder_1119_light",
            "dim_latent": 8,
            "dim_shape": 768,
            "norm": "layer",
            "in_channels": 6,
            "with_intersection": True,
            "sigmoid": False,
            "gaussian_weights": 1e-6,
            "loss": "l1",
        }

    # Load model class
    model_name = model_cfg.get("name", "AutoEncoder_1119_light")
    model_mod = importlib.import_module("src.brepnet.models.vae")
    model_cls = getattr(model_mod, model_name)
    model = model_cls(model_cfg)

    # Load weights
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()
                  if k.startswith("model.") or "." in k}
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    return model


def compute_chamfer_distance(pred_points, gt_points):
    """
    Compute Chamfer Distance between two point sets.

    Args:
        pred_points: [N, 3] or [N, H, W, 3]
        gt_points: same shape

    Returns:
        float: mean chamfer distance
    """
    # Flatten to [M, 3]
    pred = pred_points.reshape(-1, 3)
    gt = gt_points.reshape(-1, 3)

    # pred → gt
    diff_p2g = pred.unsqueeze(1) - gt.unsqueeze(0)  # [M, M, 3]
    dist_p2g = (diff_p2g ** 2).sum(-1)  # [M, M]
    min_p2g = dist_p2g.min(dim=1)[0].mean()

    # gt → pred
    min_g2p = dist_p2g.min(dim=0)[0].mean()

    return (min_p2g + min_g2p).item() / 2


def compute_chamfer_batch(pred_faces, gt_faces):
    """
    Compute CD for a batch of faces efficiently.
    pred_faces, gt_faces: [num_faces, 16, 16, 3]
    """
    # Per-face CD then average
    num_faces = pred_faces.shape[0]
    cds = []
    for i in range(num_faces):
        pred_pts = pred_faces[i].reshape(-1, 3)
        gt_pts = gt_faces[i].reshape(-1, 3)
        # Efficient: use torch cdist
        dists = torch.cdist(pred_pts.unsqueeze(0), gt_pts.unsqueeze(0)).squeeze(0)
        cd = dists.min(dim=1)[0].mean() + dists.min(dim=0)[0].mean()
        cds.append(cd.item() / 2)
    return np.mean(cds)


@torch.no_grad()
def run_robustness_test(model, dataloader, noise_levels, device="cuda"):
    """
    For each sample:
      1. Encode → get clean face_z
      2. Add noise at various levels
      3. Decode perturbed face_z
      4. Compute CD between decoded and GT
    """
    from src.brepnet.dataset import denormalize_coord1112

    results = {level: [] for level in noise_levels}
    results["clean"] = []  # reconstruction without noise (baseline)

    for batch in tqdm(dataloader, desc="Testing decoder robustness"):
        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Encode
        encoding_result = model.encode(batch, v_test=True)
        face_z_clean, _, _ = model.sample(encoding_result["face_features"], v_is_test=True)
        # face_z_clean: [total_faces_in_batch, 32]

        # GT face points
        gt_face_points = batch["face_points"][..., :3]  # [total_faces, 16, 16, 3]

        # Decode clean (baseline)
        encoding_result_clean = {"face_z": face_z_clean, "edge_features": encoding_result["edge_features"]}
        decoded_clean = model.decode(encoding_result_clean, batch)
        pred_clean = denormalize_coord1112(
            decoded_clean["face_points_local"],
            decoded_clean["face_center_scale"]
        )[..., :3]
        cd_clean = compute_chamfer_batch(pred_clean, gt_face_points)
        results["clean"].append(cd_clean)

        # Test each noise level
        for noise_level in noise_levels:
            noise = torch.randn_like(face_z_clean) * noise_level
            face_z_noisy = face_z_clean + noise

            encoding_result_noisy = {"face_z": face_z_noisy, "edge_features": encoding_result["edge_features"]}
            decoded_noisy = model.decode(encoding_result_noisy, batch)
            pred_noisy = denormalize_coord1112(
                decoded_noisy["face_points_local"],
                decoded_noisy["face_center_scale"]
            )[..., :3]
            cd_noisy = compute_chamfer_batch(pred_noisy, gt_face_points)
            results[noise_level].append(cd_noisy)

    # Average across all samples
    summary = {}
    for key, values in results.items():
        summary[key] = {
            "mean_cd": float(np.mean(values)),
            "median_cd": float(np.median(values)),
            "std_cd": float(np.std(values)),
            "max_cd": float(np.max(values)),
            "num_samples": len(values),
        }
    return summary


def print_results(summary, noise_levels):
    """Print a nice table."""
    print("\n" + "=" * 70)
    print(" DECODER ROBUSTNESS DIAGNOSTIC")
    print("=" * 70)
    print(f"\n {'Noise σ':<12} {'Mean CD':<14} {'Median CD':<14} {'Max CD':<14} {'Degradation'}")
    print(" " + "─" * 66)

    baseline_cd = summary["clean"]["mean_cd"]
    print(f" {'clean':<12} {baseline_cd:<14.6f} {summary['clean']['median_cd']:<14.6f} {summary['clean']['max_cd']:<14.6f} {'(baseline)'}")

    for level in noise_levels:
        s = summary[level]
        degradation = s["mean_cd"] / baseline_cd if baseline_cd > 0 else float("inf")
        print(f" {level:<12.4f} {s['mean_cd']:<14.6f} {s['median_cd']:<14.6f} {s['max_cd']:<14.6f} {degradation:.1f}x")

    print()

    # Interpretation
    # Find the noise level where CD doubles
    for level in noise_levels:
        if summary[level]["mean_cd"] > 2 * baseline_cd:
            print(f" ⚠️  CD doubles at noise σ = {level}")
            print(f"    Decoder 对 latent 扰动的容忍度: σ < {level}")
            break
    else:
        print(f" ✓  Decoder 在所有测试噪声下都比较鲁棒")

    # Check if the diffusion generation gap can be explained
    diffusion_cd = 0.21  # user's current real-photo CD
    print(f"\n 参考: 你当前 diffusion 生成的 CD ≈ {diffusion_cd}")
    for level in noise_levels:
        if summary[level]["mean_cd"] >= diffusion_cd * 0.8:
            print(f" → 相当于 diffusion 生成的 latent 偏离了约 σ={level} 的距离")
            break

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="VAE Decoder Robustness Test")
    parser.add_argument("--vae-checkpoint", required=True, help="Path to VAE checkpoint")
    parser.add_argument("--data-root", required=True, help="Data root directory")
    parser.add_argument("--test-list", required=True, help="Test split list file")
    parser.add_argument("--output", default="./outputs/decoder_robustness", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--noise-levels", type=str, default="0.01,0.05,0.1,0.2,0.5,1.0,2.0",
                        help="Comma-separated noise standard deviations to test")
    args = parser.parse_args()

    # Parse noise levels
    noise_levels = [float(x) for x in args.noise_levels.split(",")]

    # Load model
    print(f"Loading VAE from: {args.vae_checkpoint}")
    model = load_vae(args.vae_checkpoint, device=args.device)

    # Load dataset
    print(f"Loading test data from: {args.data_root}")
    dataset_mod = importlib.import_module("src.brepnet.dataset")
    dataset_cls = getattr(dataset_mod, "AutoEncoder_dataset3")
    dataset_cfg = {
        "data_root": args.data_root,
        "test_dataset": args.test_list,
        "is_aug": 0,
        "condition": [],
        "in_channels": 6,
    }
    dataset = dataset_cls("testing", dataset_cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dataset_cls.collate_fn,
        num_workers=args.num_workers,
    )

    # Run test
    print(f"Testing noise levels: {noise_levels}")
    summary = run_robustness_test(model, dataloader, noise_levels, device=args.device)

    # Print results
    print_results(summary, noise_levels)

    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "robustness_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to: {output_dir / 'robustness_summary.json'}")


if __name__ == "__main__":
    main()
