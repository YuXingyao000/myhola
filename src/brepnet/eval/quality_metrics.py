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

# Hardcoded CLIP realism prompt. TODO: later read a per-model prompt from disk.
CLIP_PROMPT = """
Positive: Generate a natural-looking photo placing this CAD part on a desk, as if shot on iPhone. Make it look realistic, and ensure the material and surface texture match a real-world aluminum alloy CAD part that has been used, including believable wear and aging.

Negative: deformed geometry, distorted shape, incorrect proportions, warped structure, missing parts, extra parts, altered topology	
"""


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


def _clip_embed(out):
    """Extract the projected embedding tensor from get_image/text_features output.

    transformers <5 returns a Tensor; transformers >=5 returns a model output
    object whose `pooler_output` holds the projected embedding.
    """
    import torch
    if isinstance(out, torch.Tensor):
        return out
    if getattr(out, "pooler_output", None) is not None:
        return out.pooler_output
    for attr in ("image_embeds", "text_embeds"):
        if getattr(out, attr, None) is not None:
            return getattr(out, attr)
    raise TypeError(f"Unexpected CLIP feature output type: {type(out)}")


def compute_clip_realism(image: np.ndarray, clip_model, clip_processor, device,
                         text: str = CLIP_PROMPT) -> float:
    """CLIP image-text cosine similarity (realism score)."""
    import torch
    inputs = clip_processor(images=Image.fromarray(image), return_tensors="pt").to(device)
    text_inputs = clip_processor(text=[text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        img_feat = _clip_embed(clip_model.get_image_features(**inputs))
        txt_feat = _clip_embed(clip_model.get_text_features(**text_inputs))
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
    flux_path = condition_root / model_id / "real_photo.npz"

    if not occ_path.is_file() or not flux_path.is_file():
        return None

    try:
        occ_img = np.load(occ_path)["svr_imgs"][view_idx]       # (224, 224, 3)
        flux_data = np.load(flux_path)
        flux = flux_data["flux"]
        flux_img = flux[view_idx] if flux.ndim == 4 else flux     # (512, 512, 3)
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
# Top-K visualization export (run after full evaluation)
# ---------------------------------------------------------------------------

def _load_occ_flux(condition_root: Path, model_id: str, view_idx: int):
    """Reload (occ_img, flux_img) for a single model, or None if unavailable."""
    occ_path = condition_root / model_id / "imgs.npz"
    flux_path = condition_root / model_id / "real_photo.npz"
    if not occ_path.is_file() or not flux_path.is_file():
        return None
    try:
        occ_img = np.load(occ_path)["svr_imgs"][view_idx]
        flux = np.load(flux_path)["flux"]
        flux_img = flux[view_idx] if flux.ndim == 4 else flux
    except Exception as e:
        logger.warning("vis load error model=%s: %s", model_id, e)
        return None
    return occ_img, flux_img


def _save_image(arr: np.ndarray, path: Path) -> None:
    a = np.asarray(arr)
    if a.dtype == bool:
        a = a.astype(np.uint8) * 255
    Image.fromarray(a.astype(np.uint8)).save(path)


def _overlay_mask(flux_img: np.ndarray, mask: np.ndarray, prompt_point: np.ndarray) -> Image.Image:
    """Green semi-transparent overlay of `mask` on `flux_img` + red prompt point."""
    from PIL import ImageDraw
    h, w = flux_img.shape[:2]
    mask = np.asarray(mask).astype(bool)
    if mask.shape != (h, w):
        pil = Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)
        mask = np.array(pil) > 127
    overlay = flux_img.copy()
    overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    img = Image.fromarray(overlay)
    draw = ImageDraw.Draw(img)
    px, py = int(prompt_point[0]), int(prompt_point[1])
    draw.ellipse([px - 6, py - 6, px + 6, py + 6], outline=(255, 0, 0), width=3)
    return img


def _rank_groups(results: list[dict], key: str, topk: int) -> list[tuple[str, int, dict]]:
    """Return [(group, rank, row)] for the best topk and worst topk by `key`, high→low."""
    rows = [r for r in results if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)]
    rows.sort(key=lambda r: r[key], reverse=True)
    best = rows[:topk]
    best_ids = {r["model_id"] for r in best}
    worst = [r for r in rows[-topk:] if r["model_id"] not in best_ids]  # already high→low
    out = [("best", i, r) for i, r in enumerate(best)]
    out += [("worst", i, r) for i, r in enumerate(worst)]
    return out


def export_topk_visualizations(
    condition_root: Path,
    results: list[dict],
    out_dir: Path,
    topk: int = 10,
    view_idx: int = 0,
    clip_prompt: str = CLIP_PROMPT,
    sam_predictor=None,
    sam2_checkpoint=None,
) -> None:
    """For each metric, dump the best/worst `topk` FLUX samples (sorted high→low).

    iou         : flux overlay (mask+prompt) + flux + occ + gt_mask
    dino_cosine : occ + flux
    clip_realism: flux + prompt.txt
    """
    out_dir = Path(out_dir)

    # --- IoU ---
    iou_groups = _rank_groups(results, "silhouette_iou", topk)
    if iou_groups:
        if sam_predictor is None:
            import torch
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            sam_predictor, _, _ = load_components(sam2_checkpoint, False, False, dev)
        iou_dir = out_dir / "iou"
        iou_dir.mkdir(parents=True, exist_ok=True)
        for group, rank, row in iou_groups:
            mid = row["model_id"]
            val = row["silhouette_iou"]
            pair = _load_occ_flux(condition_root, mid, view_idx)
            if pair is None:
                continue
            occ_img, flux_img = pair
            gt_mask = occ_mask(occ_img, threshold=250)
            centroid = mask_centroid(gt_mask)
            sx = flux_img.shape[1] / occ_img.shape[1]
            sy = flux_img.shape[0] / occ_img.shape[0]
            prompt_point = np.array([centroid[0] * sx, centroid[1] * sy])
            pred_mask = sam2_mask(flux_img, sam_predictor, prompt_point)
            prefix = f"{group}_{rank:02d}_iou{val:.3f}_{mid}"
            _overlay_mask(flux_img, pred_mask, prompt_point).save(iou_dir / f"{prefix}_overlay.png")
            _save_image(flux_img, iou_dir / f"{prefix}_flux.png")
            _save_image(occ_img, iou_dir / f"{prefix}_occ.png")
            _save_image(gt_mask, iou_dir / f"{prefix}_gtmask.png")
        logger.info("IoU vis: %d samples -> %s", len(iou_groups), iou_dir)

    # --- DINO cosine ---
    dino_groups = _rank_groups(results, "dino_cosine", topk)
    if dino_groups:
        dino_dir = out_dir / "dino_cosine"
        dino_dir.mkdir(parents=True, exist_ok=True)
        for group, rank, row in dino_groups:
            mid = row["model_id"]
            val = row["dino_cosine"]
            pair = _load_occ_flux(condition_root, mid, view_idx)
            if pair is None:
                continue
            occ_img, flux_img = pair
            prefix = f"{group}_{rank:02d}_dino{val:.3f}_{mid}"
            _save_image(occ_img, dino_dir / f"{prefix}_occ.png")
            _save_image(flux_img, dino_dir / f"{prefix}_flux.png")
        logger.info("DINO vis: %d samples -> %s", len(dino_groups), dino_dir)

    # --- CLIP realism ---
    clip_groups = _rank_groups(results, "clip_realism", topk)
    if clip_groups:
        clip_dir = out_dir / "clip_realism"
        clip_dir.mkdir(parents=True, exist_ok=True)
        for group, rank, row in clip_groups:
            mid = row["model_id"]
            val = row["clip_realism"]
            pair = _load_occ_flux(condition_root, mid, view_idx)
            if pair is None:
                continue
            _, flux_img = pair
            prefix = f"{group}_{rank:02d}_clip{val:.3f}_{mid}"
            _save_image(flux_img, clip_dir / f"{prefix}_flux.png")
            (clip_dir / f"{prefix}_prompt.txt").write_text(
                f"model_id: {mid}\nclip_realism: {val:.6f}\nprompt: {clip_prompt}\n"
            )
        logger.info("CLIP vis: %d samples -> %s", len(clip_groups), clip_dir)


# ---------------------------------------------------------------------------
# Component loading
# ---------------------------------------------------------------------------

def load_components(sam2_checkpoint, compute_dino: bool, compute_clip: bool, device):
    """Load SAM2 (+ optional DINOv2, CLIP) onto `device`.

    Returns (sam_predictor, dino_components, clip_components).
    """
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    sam2_ckpt = str(sam2_checkpoint) if sam2_checkpoint else "sam2.1_hiera_large.pt"
    sam2_model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", sam2_ckpt, device=str(device))
    sam_predictor = SAM2ImagePredictor(sam2_model)

    dino_components = None
    if compute_dino:
        from transformers import AutoImageProcessor, AutoModel
        proc = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
        dino = AutoModel.from_pretrained("facebook/dinov2-large").to(device).eval()
        dino_components = (dino, proc, device)

    clip_components = None
    if compute_clip:
        from transformers import CLIPModel, CLIPProcessor
        clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
        clip_components = (clip_model, clip_proc, device)

    return sam_predictor, dino_components, clip_components


# ---------------------------------------------------------------------------
# Ray parallel execution (multi-GPU actor pool)
# ---------------------------------------------------------------------------

def evaluate_parallel(
    condition_root: Path,
    model_ids: list[str],
    sam2_checkpoint,
    view_idx: int,
    compute_dino: bool,
    compute_clip: bool,
    num_gpus: int,
    actors_per_gpu: int,
) -> list[dict]:
    """Distribute model evaluation across a pool of GPU actors via Ray."""
    import ray
    from ray.util import ActorPool

    ray.init(ignore_reinit_error=True)

    gpus_per_actor = 1.0 / actors_per_gpu
    num_actors = num_gpus * actors_per_gpu

    @ray.remote(num_gpus=gpus_per_actor)
    class QualityWorker:
        def __init__(self):
            import torch
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.condition_root = condition_root
            self.view_idx = view_idx
            self.sam_predictor, self.dino_components, self.clip_components = load_components(
                sam2_checkpoint, compute_dino, compute_clip, dev,
            )

        def evaluate(self, model_id: str):
            try:
                return evaluate_model(
                    self.condition_root, model_id, self.sam_predictor,
                    self.view_idx, self.dino_components, self.clip_components,
                )
            except Exception as e:  # never let one bad model kill the actor
                logger.warning("model=%s eval error: %s", model_id, e)
                return None

    actors = [QualityWorker.remote() for _ in range(num_actors)]
    pool = ActorPool(actors)
    logger.info("Ray: %d actors across %d GPUs (%d actors/GPU)", num_actors, num_gpus, actors_per_gpu)

    from tqdm import tqdm
    results: list[dict] = []
    skipped = 0
    for res in tqdm(pool.map_unordered(lambda a, mid: a.evaluate.remote(mid), model_ids),
                    total=len(model_ids), desc="SAM2 IoU"):
        if res is None:
            skipped += 1
        else:
            results.append(res)
    logger.info("Parallel done: %d evaluated, %d skipped", len(results), skipped)
    ray.shutdown()
    return results


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
    parser.add_argument("--use-ray", action="store_true", help="Parallelize across GPUs with Ray")
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="GPUs to use with --use-ray (default: all visible)")
    parser.add_argument("--actors-per-gpu", type=int, default=1,
                        help="SAM2 actors per GPU with --use-ray (fractional GPU sharing)")
    parser.add_argument("--topk-vis-dir", type=Path, default=None,
                        help="If set, after eval export best/worst topk samples per metric here")
    parser.add_argument("--topk", type=int, default=10,
                        help="Number of best and worst samples per metric for --topk-vis-dir")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    # Load model list
    model_ids = [l.strip() for l in args.model_list.read_text().splitlines() if l.strip() and not l.startswith("#")]
    logger.info("Models: %d", len(model_ids))

    sam_predictor = None
    if args.use_ray:
        import torch
        num_gpus = args.num_gpus if args.num_gpus is not None else max(1, torch.cuda.device_count())
        results = evaluate_parallel(
            args.condition_root, model_ids, args.sam2_checkpoint, args.view_idx,
            args.compute_dino, args.compute_clip, num_gpus, args.actors_per_gpu,
        )
        skipped = len(model_ids) - len(results)
    else:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sam_predictor, dino_components, clip_components = load_components(
            args.sam2_checkpoint, args.compute_dino, args.compute_clip, device,
        )
        logger.info("SAM2 loaded on %s (dino=%s clip=%s)", device, args.compute_dino, args.compute_clip)

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

    # Top-K visualization export (best/worst per metric)
    if args.topk_vis_dir:
        export_topk_visualizations(
            args.condition_root, results, args.topk_vis_dir,
            topk=args.topk, view_idx=args.view_idx, clip_prompt=CLIP_PROMPT,
            sam_predictor=sam_predictor, sam2_checkpoint=args.sam2_checkpoint,
        )


if __name__ == "__main__":
    main()
