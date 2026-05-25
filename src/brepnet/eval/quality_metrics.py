"""Dataset quality metrics for FLUX-generated images.

Metrics:
  - Silhouette IoU: SAM2 segments FLUX foreground → compare with OCC mask (ground truth)
  - DINO Cosine: DINOv2 CLS token similarity between OCC render and FLUX image
  - CLIP Realism: CLIP score of FLUX image vs "a real photograph"

Usage:
    # IoU only (needs GPU for SAM2)
    python -m src.brepnet.eval.quality_metrics \
        --condition-root /mnt/d/data/deepcad_v6_cond \
        --model-list src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt \
        --output quality_report.json

    # All metrics
    python -m src.brepnet.eval.quality_metrics \
        --condition-root /mnt/d/data/deepcad_v6_cond \
        --model-list src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt \
        --compute-dino --compute-clip \
        --output quality_report.json

    # SAM2 checkpoint: download sam2_hiera_large.pt from
    # https://github.com/facebookresearch/sam2
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------

def occ_mask(image: np.ndarray, threshold: int = 250) -> np.ndarray:
    """Extract foreground mask from OCC render (gray on pure white, specular OFF).

    Any pixel with luminance < threshold is foreground.
    """
    gray = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    return gray < threshold


def sam2_mask(image: np.ndarray, predictor, point: np.ndarray) -> np.ndarray:
    """Segment foreground from FLUX image using SAM2 with a point prompt.

    Args:
        image: (H, W, 3) uint8 RGB
        predictor: SAM2ImagePredictor (already loaded)
        point: (2,) array [x, y] — prompt point on the object

    Returns:
        (H, W) bool mask
    """
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(
        point_coords=point.reshape(1, 2),
        point_labels=np.array([1]),
        multimask_output=True,
    )
    # Pick highest confidence mask
    return masks[scores.argmax()]


def mask_centroid(mask: np.ndarray) -> np.ndarray:
    """Compute centroid (x, y) of a binary mask."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        h, w = mask.shape
        return np.array([w // 2, h // 2])
    return np.array([int(xs.mean()), int(ys.mean())])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """IoU between two boolean masks. Resizes mask_b to match mask_a if needed."""
    if mask_a.shape != mask_b.shape:
        h, w = mask_a.shape
        pil = Image.fromarray(mask_b.astype(np.uint8) * 255)
        pil = pil.resize((w, h), Image.NEAREST)
        mask_b = np.array(pil) > 127

    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def compute_dino_cosine(img1: np.ndarray, img2: np.ndarray, model, processor, device) -> float:
    """DINOv2 CLS token cosine similarity."""
    import torch
    inputs1 = processor(images=Image.fromarray(img1), return_tensors="pt").to(device)
    inputs2 = processor(images=Image.fromarray(img2), return_tensors="pt").to(device)
    with torch.no_grad():
        feat1 = model(**inputs1).last_hidden_state[:, 0]
        feat2 = model(**inputs2).last_hidden_state[:, 0]
    return float(torch.nn.functional.cosine_similarity(feat1, feat2, dim=-1).item())


def compute_clip_realism(image: np.ndarray, clip_model, clip_processor, device,
                         text: str = "a real photograph of a machined metal part on a desk") -> float:
    """CLIP image-text cosine similarity (realism score)."""
    import torch
    inputs = clip_processor(images=Image.fromarray(image), return_tensors="pt").to(device)
    text_inputs = clip_processor(text=[text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        img_feat = clip_model.get_image_features(**inputs)
        txt_feat = clip_model.get_text_features(**text_inputs)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
    return float(torch.nn.functional.cosine_similarity(img_feat, txt_feat, dim=-1).item())


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    condition_root: Path,
    model_id: str,
    sam_predictor,
    view_idx: int = 0,
    dino_components: tuple | None = None,
    clip_components: tuple | None = None,
) -> dict | None:
    """Evaluate one model.

    IoU pipeline:
      1. OCC svr_imgs[view_idx] (224×224, gray on white) → threshold → gt_mask
      2. Compute gt_mask centroid → scale to 512×512 → SAM2 prompt point
      3. SAM2(flux_img, prompt_point) → pred_mask
      4. IoU(gt_mask, pred_mask)
    """
    occ_path = condition_root / model_id / "imgs.npz"
    flux_path = condition_root / model_id / "single_view.npz"

    if not occ_path.is_file() or not flux_path.is_file():
        return None

    try:
        occ_img = np.load(occ_path)["svr_imgs"][view_idx]       # (224, 224, 3)
        flux_data = np.load(flux_path)
        flux_img = flux_data["flux"]                             # (512, 512, 3)
    except Exception as e:
        logger.warning("model=%s load error: %s", model_id, e)
        return None

    # Ground truth mask from OCC (clean: gray on white, no shadows, no specular)
    gt_mask = occ_mask(occ_img, threshold=250)  # (224, 224)

    # SAM2 prompt: use OCC mask centroid, scaled to FLUX resolution
    centroid_occ = mask_centroid(gt_mask)  # (x, y) in 224×224
    scale_x = flux_img.shape[1] / occ_img.shape[1]
    scale_y = flux_img.shape[0] / occ_img.shape[0]
    prompt_point = np.array([centroid_occ[0] * scale_x, centroid_occ[1] * scale_y])

    # Segment FLUX image with SAM2
    pred_mask = sam2_mask(flux_img, sam_predictor, prompt_point)  # (512, 512)

    # IoU (resize gt_mask to 512×512 to match)
    iou = compute_iou(pred_mask, gt_mask)

    result = {"model_id": model_id, "silhouette_iou": iou}

    # DINO cosine: OCC vs FLUX (semantic similarity)
    if dino_components is not None:
        model, processor, device = dino_components
        result["dino_cosine"] = compute_dino_cosine(occ_img, flux_img, model, processor, device)

    # CLIP realism: FLUX vs text anchor
    if clip_components is not None:
        clip_model, clip_processor, device = clip_components
        result["clip_realism"] = compute_clip_realism(flux_img, clip_model, clip_processor, device)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="FLUX dataset quality metrics (SAM2 IoU + DINO + CLIP).")
    parser.add_argument("--condition-root", type=Path, required=True)
    parser.add_argument("--model-list", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, default=None, help="Path to sam2_hiera_large.pt")
    parser.add_argument("--view-idx", type=int, default=0)
    parser.add_argument("--compute-dino", action="store_true")
    parser.add_argument("--compute-clip", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--iou-threshold", type=float, default=0.70)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    # Load model list
    model_ids = [l.strip() for l in args.model_list.read_text().splitlines() if l.strip() and not l.startswith("#")]
    logger.info("Models: %d", len(model_ids))

    # Load SAM2
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam2_ckpt = str(args.sam2_checkpoint) if args.sam2_checkpoint else "sam2.1_hiera_large.pt"
    sam2_model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", sam2_ckpt, device=str(device))
    sam_predictor = SAM2ImagePredictor(sam2_model)
    logger.info("SAM2 loaded on %s", device)

    # Optional: DINO
    dino_components = None
    if args.compute_dino:
        from transformers import AutoImageProcessor, AutoModel
        proc = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
        dino = AutoModel.from_pretrained("facebook/dinov2-large").to(device).eval()
        dino_components = (dino, proc, device)
        logger.info("DINOv2 loaded")

    # Optional: CLIP
    clip_components = None
    if args.compute_clip:
        from transformers import CLIPModel, CLIPProcessor
        clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
        clip_components = (clip_model, clip_proc, device)
        logger.info("CLIP loaded")

    # Evaluate
    results = []
    skipped = 0
    for i, model_id in enumerate(model_ids):
        result = evaluate_model(
            args.condition_root, model_id, sam_predictor,
            args.view_idx, dino_components, clip_components,
        )
        if result is None:
            skipped += 1
            continue
        results.append(result)
        if (i + 1) % 100 == 0:
            logger.info("Progress: %d/%d, skipped %d", len(results), i + 1, skipped)

    if not results:
        logger.warning("No models evaluated.")
        return

    # Report
    ious = [r["silhouette_iou"] for r in results]
    pass_count = sum(1 for x in ious if x >= args.iou_threshold)

    report = {
        "num_evaluated": len(results),
        "num_skipped": skipped,
        "iou_threshold": args.iou_threshold,
        "pass_rate": pass_count / len(results),
        "iou_mean": float(np.mean(ious)),
        "iou_std": float(np.std(ious)),
        "iou_median": float(np.median(ious)),
        "iou_p10": float(np.percentile(ious, 10)),
        "iou_p90": float(np.percentile(ious, 90)),
    }
    if args.compute_dino:
        d = [r["dino_cosine"] for r in results if "dino_cosine" in r]
        report["dino_mean"] = float(np.mean(d))
        report["dino_std"] = float(np.std(d))
    if args.compute_clip:
        c = [r["clip_realism"] for r in results if "clip_realism" in r]
        report["clip_mean"] = float(np.mean(c))
        report["clip_std"] = float(np.std(c))

    print("\n" + "=" * 60)
    print("FLUX Quality Report")
    print("=" * 60)
    print(f"  Evaluated:   {report['num_evaluated']} ({skipped} skipped)")
    print(f"  Pass rate:   {report['pass_rate']:.1%} (IoU >= {args.iou_threshold})")
    print(f"  IoU:         {report['iou_mean']:.4f} ± {report['iou_std']:.4f} (median {report['iou_median']:.4f})")
    print(f"  IoU P10/P90: [{report['iou_p10']:.4f}, {report['iou_p90']:.4f}]")
    if "dino_mean" in report:
        print(f"  DINO:        {report['dino_mean']:.4f} ± {report['dino_std']:.4f}")
    if "clip_mean" in report:
        print(f"  CLIP:        {report['clip_mean']:.4f} ± {report['clip_std']:.4f}")
    print("=" * 60)

    # Worst cases
    worst = sorted(results, key=lambda r: r["silhouette_iou"])[:10]
    print("\nWorst 10:")
    for r in worst:
        print(f"  {r['model_id']}: IoU={r['silhouette_iou']:.4f}", end="")
        if "dino_cosine" in r:
            print(f" DINO={r['dino_cosine']:.4f}", end="")
        if "clip_realism" in r:
            print(f" CLIP={r['clip_realism']:.4f}", end="")
        print()

    # Save
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump({"report": report, "per_model": results}, f, indent=2)
        logger.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()
