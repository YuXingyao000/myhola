from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
from tqdm import tqdm

try:
    import ray
except ImportError:
    ray = None

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPECTED_OUTPUT_FILES = (
    "natural_fidelity_metrics.npz",
    "summary.json",
    "failures.json",
    "mask_precision_hist.png",
    "mask_recall_hist.png",
    "dice_hist.png",
    "iou_hist.png",
    "boundary_iou_hist.png",
    "extra_error_hist.png",
    "missing_error_hist.png",
)


class EvaluationError(Exception):
    """Raised for expected per-model validation failures."""


@dataclass(frozen=True)
class ViewMetrics:
    mask_precision: float
    mask_recall: float
    dice: float
    iou: float
    boundary_iou: float
    extra_error: float
    missing_error: float


@dataclass(frozen=True)
class ModelMetrics:
    model_id: str
    best_view_index: int
    view_count: int
    mask_precision: float
    mask_recall: float
    dice: float
    iou: float
    boundary_iou: float
    extra_error: float
    missing_error: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate natural-image structural fidelity against multi-view GT renders."
    )
    parser.add_argument("--data_root", required=True, type=Path)
    parser.add_argument("--output_root", required=True, type=Path)
    parser.add_argument("--input_filename", default="combined_imgs.npz")
    parser.add_argument("--boundary_d_ratio", default=0.02, type=float)
    parser.add_argument("--hist_bin_width", default=0.05, type=float)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def list_model_dirs(data_root: Path) -> list[Path]:
    if not data_root.exists():
        raise FileNotFoundError(f"data_root does not exist: {data_root}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"data_root is not a directory: {data_root}")
    return sorted((path for path in data_root.iterdir() if path.is_dir()), key=lambda path: path.name)


def ensure_output_root(output_root: Path, overwrite: bool) -> dict[str, Path]:
    output_paths = {name: output_root / name for name in EXPECTED_OUTPUT_FILES}
    existing_outputs = [str(path) for path in output_paths.values() if path.exists()]
    if existing_outputs and not overwrite:
        joined = ", ".join(existing_outputs)
        raise FileExistsError(
            f"output files already exist and --overwrite was not provided: {joined}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in output_paths.values():
            if not path.exists():
                continue
            if not path.is_file():
                raise IsADirectoryError(f"expected output path is not a file: {path}")
            path.unlink()
    return output_paths


def validate_cli_args(boundary_d_ratio: float, hist_bin_width: float) -> None:
    if boundary_d_ratio <= 0.0:
        raise ValueError("boundary_d_ratio must be > 0")
    if not (0.0 < hist_bin_width <= 1.0):
        raise ValueError("hist_bin_width must satisfy 0 < hist_bin_width <= 1")


def ensure_numeric_finite(array: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(array)
    if result.dtype == np.dtype("O"):
        raise EvaluationError(f"{name} must be numeric, got object dtype")
    if not np.issubdtype(result.dtype, np.number):
        raise EvaluationError(f"{name} must be numeric, got dtype={result.dtype}")
    if not np.all(np.isfinite(result)):
        raise EvaluationError(f"{name} contains NaN or Inf")
    return result


def convert_to_gray_float32(image: np.ndarray, name: str) -> np.ndarray:
    array = ensure_numeric_finite(image, name)
    if array.ndim == 2:
        gray = array.astype(np.float32, copy=False)
    elif array.ndim == 3 and array.shape[-1] == 1:
        gray = np.squeeze(array, axis=-1).astype(np.float32, copy=False)
    elif array.ndim == 3 and array.shape[-1] == 3:
        rgb = np.ascontiguousarray(array.astype(np.float32, copy=False))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    else:
        raise EvaluationError(
            f"{name} has unsupported shape {array.shape}; expected [H, W], [H, W, 1], or [H, W, 3]"
        )

    if gray.size == 0:
        raise EvaluationError(f"{name} is empty")

    max_value = float(np.max(gray))
    if max_value > 1.0:
        gray = gray / 255.0
    gray = np.clip(gray, 0.0, 1.0).astype(np.float32, copy=False)
    return gray


def convert_reference_stack(reference_images: np.ndarray) -> np.ndarray:
    array = ensure_numeric_finite(reference_images, "svr_imgs")
    if array.ndim == 3:
        view_count = int(array.shape[0])
    elif array.ndim == 4 and array.shape[-1] in (1, 3):
        view_count = int(array.shape[0])
    else:
        raise EvaluationError(
            "svr_imgs has unsupported shape "
            f"{array.shape}; expected [V, H, W], [V, H, W, 1], or [V, H, W, 3]"
        )

    if view_count <= 0:
        raise EvaluationError("svr_imgs has zero valid views")

    gray_views = [convert_to_gray_float32(array[index], f"svr_imgs[{index}]") for index in range(view_count)]
    reference_shape = gray_views[0].shape
    if any(view.shape != reference_shape for view in gray_views):
        raise EvaluationError("reference views do not share a consistent resolution")
    return np.stack(gray_views, axis=0)


def resize_to_shape(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if image.shape == target_shape:
        return image.astype(np.float32, copy=False)
    target_height, target_width = target_shape
    interpolation = cv2.INTER_AREA
    if image.shape[0] < target_height or image.shape[1] < target_width:
        interpolation = cv2.INTER_LINEAR
    resized = cv2.resize(image, (target_width, target_height), interpolation=interpolation)
    return resized.astype(np.float32, copy=False)


def extract_foreground_mask(gray_image: np.ndarray) -> np.ndarray:
    gray_uint8 = np.clip(np.rint(gray_image * 255.0), 0, 255).astype(np.uint8)

    white_threshold = 245
    near_white_background = gray_uint8 >= white_threshold

    _, otsu_inv = cv2.threshold(
        gray_uint8,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )
    candidate_foreground = otsu_inv > 0

    mask = np.logical_and(~near_white_background, candidate_foreground)

    kernel = np.ones((3, 3), dtype=np.uint8)
    mask_uint8 = mask.astype(np.uint8)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask_uint8.astype(bool)


def boundary_width_from_shape(shape: tuple[int, int], boundary_d_ratio: float) -> int:
    height, width = shape
    diagonal = math.sqrt(float(height * height + width * width))
    return max(1, int(round(boundary_d_ratio * diagonal)))


def make_disk_kernel(radius: int) -> np.ndarray:
    diameter = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))


def build_boundary_band(mask: np.ndarray, distance: int) -> np.ndarray:
    mask_uint8 = mask.astype(np.uint8)
    if not np.any(mask_uint8):
        return np.zeros_like(mask_uint8, dtype=bool)

    contour_kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(mask_uint8, contour_kernel, iterations=1)
    contour = mask_uint8 ^ eroded
    band_kernel = make_disk_kernel(distance)
    band = cv2.dilate(contour, band_kernel, iterations=1)
    return band.astype(bool)


def compute_view_metrics(generated_mask: np.ndarray, reference_mask: np.ndarray, boundary_distance: int) -> ViewMetrics:
    generated_count = int(np.count_nonzero(generated_mask))
    reference_count = int(np.count_nonzero(reference_mask))
    intersection_count = int(np.count_nonzero(np.logical_and(generated_mask, reference_mask)))
    union_count = int(np.count_nonzero(np.logical_or(generated_mask, reference_mask)))

    if generated_count == 0 and reference_count == 0:
        return ViewMetrics(1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0)
    if generated_count == 0 or reference_count == 0:
        return ViewMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0)

    mask_precision = intersection_count / float(generated_count)
    mask_recall = intersection_count / float(reference_count)
    dice = 2.0 * intersection_count / float(generated_count + reference_count)
    iou = intersection_count / float(union_count)
    extra_error = (generated_count - intersection_count) / float(generated_count)
    missing_error = (reference_count - intersection_count) / float(reference_count)

    generated_band = build_boundary_band(generated_mask, boundary_distance)
    reference_band = build_boundary_band(reference_mask, boundary_distance)
    boundary_union = int(np.count_nonzero(np.logical_or(generated_band, reference_band)))
    if boundary_union == 0:
        boundary_iou = 1.0
    else:
        boundary_intersection = int(np.count_nonzero(np.logical_and(generated_band, reference_band)))
        boundary_iou = boundary_intersection / float(boundary_union)

    return ViewMetrics(
        mask_precision=float(mask_precision),
        mask_recall=float(mask_recall),
        dice=float(dice),
        iou=float(iou),
        boundary_iou=float(boundary_iou),
        extra_error=float(extra_error),
        missing_error=float(missing_error),
    )


def load_model_arrays(model_dir: Path, input_filename: str) -> tuple[np.ndarray, np.ndarray]:
    input_path = model_dir / input_filename
    if not input_path.is_file():
        raise EvaluationError(f"missing input file: {input_path.name}")

    try:
        with np.load(input_path, allow_pickle=False) as npz_file:
            if "natural_img" not in npz_file:
                raise EvaluationError("missing required key: natural_img")
            if "svr_imgs" not in npz_file:
                raise EvaluationError("missing required key: svr_imgs")
            natural_img = npz_file["natural_img"]
            svr_imgs = npz_file["svr_imgs"]
    except EvaluationError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise EvaluationError(f"failed to read npz: {exc}") from exc

    natural_gray = convert_to_gray_float32(natural_img, "natural_img")
    reference_grays = convert_reference_stack(svr_imgs)
    natural_gray = resize_to_shape(natural_gray, tuple(reference_grays.shape[1:]))
    return natural_gray, reference_grays


def evaluate_model(model_dir: Path, input_filename: str, boundary_d_ratio: float) -> ModelMetrics:
    natural_gray, reference_grays = load_model_arrays(model_dir, input_filename)
    natural_mask = extract_foreground_mask(natural_gray)
    boundary_distance = boundary_width_from_shape(natural_mask.shape, boundary_d_ratio)

    per_view_metrics: list[ViewMetrics] = []
    for view_index, reference_gray in enumerate(reference_grays):
        try:
            reference_mask = extract_foreground_mask(reference_gray)
            metrics = compute_view_metrics(natural_mask, reference_mask, boundary_distance)
        except Exception as exc:  # noqa: BLE001
            raise EvaluationError(f"failed while evaluating view {view_index}: {exc}") from exc
        per_view_metrics.append(metrics)

    best_view_index = max(
        range(len(per_view_metrics)),
        key=lambda index: (
            per_view_metrics[index].dice,
            per_view_metrics[index].boundary_iou,
            per_view_metrics[index].iou,
            -index,
        ),
    )
    best_metrics = per_view_metrics[best_view_index]
    return ModelMetrics(
        model_id=model_dir.name,
        best_view_index=best_view_index,
        view_count=int(reference_grays.shape[0]),
        mask_precision=best_metrics.mask_precision,
        mask_recall=best_metrics.mask_recall,
        dice=best_metrics.dice,
        iou=best_metrics.iou,
        boundary_iou=best_metrics.boundary_iou,
        extra_error=best_metrics.extra_error,
        missing_error=best_metrics.missing_error,
    )


def build_failure_record(model_id: str, reason: str, exception: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {"model_id": model_id, "reason": reason}
    if exception is not None:
        record["exception"] = exception
    return record


def evaluate_model_safe(model_dir: Path, input_filename: str, boundary_d_ratio: float) -> dict[str, Any]:
    try:
        result = evaluate_model(model_dir, input_filename=input_filename, boundary_d_ratio=boundary_d_ratio)
    except EvaluationError as exc:
        return {"ok": False, "failure": build_failure_record(model_dir.name, str(exc))}
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "failure": build_failure_record(
                model_dir.name,
                "unexpected evaluation error",
                repr(exc),
            ),
        }
    return {"ok": True, "result": result}


if ray is not None:

    @ray.remote
    def evaluate_model_remote(model_dir: str, input_filename: str, boundary_d_ratio: float) -> dict[str, Any]:
        cv2.setNumThreads(1)
        return evaluate_model_safe(Path(model_dir), input_filename=input_filename, boundary_d_ratio=boundary_d_ratio)

else:
    evaluate_model_remote = None


def evaluate_models_serial(
    model_dirs: list[Path],
    input_filename: str,
    boundary_d_ratio: float,
) -> list[dict[str, Any]]:
    if not model_dirs:
        return []
    if ray is None or evaluate_model_remote is None:
        raise ImportError("ray is required for parallel evaluation but is not installed")

    max_workers = len(model_dirs)
    started_here = False
    if not ray.is_initialized():
        ray.init(
            num_cpus=max_workers,
            ignore_reinit_error=True,
            include_dashboard=False,
            log_to_driver=False,
            logging_level="ERROR",
        )
        started_here = True

    try:
        refs = [
            evaluate_model_remote.remote(str(model_dir), input_filename, boundary_d_ratio)
            for model_dir in model_dirs
        ]
        ref_to_index = {ref: index for index, ref in enumerate(refs)}
        outcomes: list[dict[str, Any] | None] = [None] * len(refs)
        pending = list(refs)

        with tqdm(total=len(model_dirs), desc="Evaluating", unit="model") as progress:
            while pending:
                ready, pending = ray.wait(pending, num_returns=1)
                ref = ready[0]
                index = ref_to_index[ref]
                try:
                    outcomes[index] = ray.get(ref)
                except Exception as exc:  # noqa: BLE001
                    outcomes[index] = {
                        "ok": False,
                        "failure": build_failure_record(
                            model_dirs[index].name,
                            "ray worker execution error",
                            repr(exc),
                        ),
                    }
                progress.update(1)

        return [outcome for outcome in outcomes if outcome is not None]
    finally:
        if started_here:
            ray.shutdown()


def split_outcomes(outcomes: list[dict[str, Any]]) -> tuple[list[ModelMetrics], list[dict[str, Any]]]:
    results: list[ModelMetrics] = []
    failures: list[dict[str, Any]] = []
    for outcome in outcomes:
        if outcome["ok"]:
            results.append(outcome["result"])
        else:
            failures.append(outcome["failure"])
    return results, failures


def validate_success_results(results: list[ModelMetrics]) -> None:
    if not results:
        return

    metric_names = (
        "mask_precision",
        "mask_recall",
        "dice",
        "iou",
        "boundary_iou",
        "extra_error",
        "missing_error",
    )
    for metric_name in metric_names:
        values = np.asarray([getattr(result, metric_name) for result in results], dtype=np.float64)
        if not np.all((0.0 <= values) & (values <= 1.0)):
            raise RuntimeError(f"{metric_name} values fall outside [0, 1]")

    for result in results:
        if not 0 <= result.best_view_index < result.view_count:
            raise RuntimeError(
                f"best_view_index out of range for model {result.model_id}: "
                f"{result.best_view_index} vs view_count={result.view_count}"
            )
        if abs(result.extra_error - (1.0 - result.mask_precision)) > 1e-6:
            raise RuntimeError(f"extra_error mismatch for model {result.model_id}")
        if abs(result.missing_error - (1.0 - result.mask_recall)) > 1e-6:
            raise RuntimeError(f"missing_error mismatch for model {result.model_id}")


def save_metrics_npz(output_path: Path, results: list[ModelMetrics]) -> None:
    payload = {
        "model_ids": np.asarray([result.model_id for result in results], dtype=str),
        "best_view_indices": np.asarray([result.best_view_index for result in results], dtype=np.int64),
        "mask_precision": np.asarray([result.mask_precision for result in results], dtype=np.float32),
        "mask_recall": np.asarray([result.mask_recall for result in results], dtype=np.float32),
        "dice": np.asarray([result.dice for result in results], dtype=np.float32),
        "iou": np.asarray([result.iou for result in results], dtype=np.float32),
        "boundary_iou": np.asarray([result.boundary_iou for result in results], dtype=np.float32),
        "extra_error": np.asarray([result.extra_error for result in results], dtype=np.float32),
        "missing_error": np.asarray([result.missing_error for result in results], dtype=np.float32),
    }
    lengths = {len(value) for value in payload.values()}
    if len(lengths) != 1:
        raise RuntimeError("natural_fidelity_metrics.npz arrays do not share the same length")
    np.savez(output_path, **payload)


def save_json(output_path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def make_hist_bins(hist_bin_width: float) -> np.ndarray:
    bins = np.arange(0.0, 1.0 + hist_bin_width, hist_bin_width, dtype=np.float64)
    if bins.size == 0 or not np.isclose(bins[0], 0.0):
        bins = np.insert(bins, 0, 0.0)
    bins = np.unique(np.clip(bins, 0.0, 1.0))
    if bins.size < 2:
        bins = np.array([0.0, 1.0], dtype=np.float64)
    elif bins[-1] < 1.0:
        bins = np.append(bins, 1.0)
    else:
        bins[-1] = 1.0
    return bins


def plot_histogram(values: np.ndarray, bins: np.ndarray, title: str, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.hist(values, bins=bins, range=(0.0, 1.0), color="#4477AA", edgecolor="black")
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel(title)
    axis.set_ylabel("Count")
    axis.set_title(f"{title} Histogram")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def summarize_failures(failures: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(failure["reason"]) for failure in failures))


def build_summary(
    data_root: Path,
    output_root: Path,
    input_filename: str,
    boundary_d_ratio: float,
    hist_bin_width: float,
    model_dirs: list[Path],
    results: list[ModelMetrics],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    num_total_models = len(model_dirs)
    num_valid_models = len(results)
    num_failed_models = len(failures)
    if num_valid_models + num_failed_models != num_total_models:
        raise RuntimeError("num_valid_models + num_failed_models does not equal num_total_models")

    non_standard_view_models = [
        {"model_id": result.model_id, "view_count": result.view_count}
        for result in results
        if result.view_count != 64
    ]

    implementation_notes = [
        "Boundary band is discretized by extracting a 1-pixel inner contour from the binary mask and dilating that contour with an elliptical structuring element of radius d.",
        "Foreground mask extraction uses a deterministic near-white suppression threshold at 245/255 combined with inverse Otsu thresholding, followed by fixed 3x3 open-close morphology.",
    ]

    if results:
        mask_precision = np.asarray([result.mask_precision for result in results], dtype=np.float64)
        mask_recall = np.asarray([result.mask_recall for result in results], dtype=np.float64)
        dice = np.asarray([result.dice for result in results], dtype=np.float64)
        iou = np.asarray([result.iou for result in results], dtype=np.float64)
        boundary_iou = np.asarray([result.boundary_iou for result in results], dtype=np.float64)
        extra_error = np.asarray([result.extra_error for result in results], dtype=np.float64)
        missing_error = np.asarray([result.missing_error for result in results], dtype=np.float64)
        sample_results = [
            {
                "model_id": result.model_id,
                "best_view_index": result.best_view_index,
                "mask_precision": result.mask_precision,
                "mask_recall": result.mask_recall,
                "dice": result.dice,
                "boundary_iou": result.boundary_iou,
            }
            for result in results[:3]
        ]
    else:
        mask_precision = mask_recall = dice = iou = boundary_iou = extra_error = missing_error = np.empty(0)
        sample_results = []

    return {
        "data_root": str(data_root),
        "output_root": str(output_root),
        "input_filename": input_filename,
        "boundary_d_ratio": boundary_d_ratio,
        "hist_bin_width": hist_bin_width,
        "num_total_models": num_total_models,
        "num_valid_models": num_valid_models,
        "num_failed_models": num_failed_models,
        "mean_mask_precision": None if mask_precision.size == 0 else float(np.mean(mask_precision)),
        "mean_mask_recall": None if mask_recall.size == 0 else float(np.mean(mask_recall)),
        "mean_dice": None if dice.size == 0 else float(np.mean(dice)),
        "mean_iou": None if iou.size == 0 else float(np.mean(iou)),
        "mean_boundary_iou": None if boundary_iou.size == 0 else float(np.mean(boundary_iou)),
        "mean_extra_error": None if extra_error.size == 0 else float(np.mean(extra_error)),
        "mean_missing_error": None if missing_error.size == 0 else float(np.mean(missing_error)),
        "failure_reason_counts": summarize_failures(failures),
        "non_standard_view_models": non_standard_view_models,
        "sample_results": sample_results,
        "implementation_notes": implementation_notes,
    }


def print_terminal_summary(summary: dict[str, Any]) -> None:
    print(f"Input root: {summary['data_root']}")
    print(f"Output root: {summary['output_root']}")
    print(
        "Models: "
        f"total={summary['num_total_models']} "
        f"valid={summary['num_valid_models']} "
        f"failed={summary['num_failed_models']}"
    )
    print(
        "Means: "
        f"P={summary['mean_mask_precision']} "
        f"R={summary['mean_mask_recall']} "
        f"Dice={summary['mean_dice']} "
        f"IoU={summary['mean_iou']} "
        f"BIoU={summary['mean_boundary_iou']} "
        f"Extra={summary['mean_extra_error']} "
        f"Missing={summary['mean_missing_error']}"
    )
    if summary["failure_reason_counts"]:
        print(f"Failure reasons: {summary['failure_reason_counts']}")
    for sample in summary["sample_results"][:3]:
        print(
            f"{sample['model_id']} / {sample['best_view_index']} / "
            f"{sample['mask_precision']:.4f} / {sample['mask_recall']:.4f} / "
            f"{sample['dice']:.4f} / {sample['boundary_iou']:.4f}"
        )


def save_histograms(results: list[ModelMetrics], output_paths: dict[str, Path], hist_bin_width: float) -> None:
    bins = make_hist_bins(hist_bin_width)
    if bins[0] != 0.0 or bins[-1] != 1.0:
        raise RuntimeError("histogram bins do not cover [0, 1]")

    metric_to_output = {
        "mask_precision": output_paths["mask_precision_hist.png"],
        "mask_recall": output_paths["mask_recall_hist.png"],
        "dice": output_paths["dice_hist.png"],
        "iou": output_paths["iou_hist.png"],
        "boundary_iou": output_paths["boundary_iou_hist.png"],
        "extra_error": output_paths["extra_error_hist.png"],
        "missing_error": output_paths["missing_error_hist.png"],
    }
    for metric_name, output_path in metric_to_output.items():
        values = np.asarray([getattr(result, metric_name) for result in results], dtype=np.float64)
        plot_histogram(values, bins, metric_name, output_path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_cli_args(args.boundary_d_ratio, args.hist_bin_width)

    model_dirs = list_model_dirs(args.data_root)
    output_paths = ensure_output_root(args.output_root, overwrite=args.overwrite)

    outcomes = evaluate_models_serial(
        model_dirs=model_dirs,
        input_filename=args.input_filename,
        boundary_d_ratio=args.boundary_d_ratio,
    )
    results, failures = split_outcomes(outcomes)

    validate_success_results(results)
    save_metrics_npz(output_paths["natural_fidelity_metrics.npz"], results)
    save_json(output_paths["failures.json"], failures)

    summary = build_summary(
        data_root=args.data_root,
        output_root=args.output_root,
        input_filename=args.input_filename,
        boundary_d_ratio=args.boundary_d_ratio,
        hist_bin_width=args.hist_bin_width,
        model_dirs=model_dirs,
        results=results,
        failures=failures,
    )
    save_json(output_paths["summary.json"], summary)
    print_terminal_summary(summary)

    if len(results) == 0:
        return 1

    save_histograms(results, output_paths, hist_bin_width=args.hist_bin_width)
    return 0


if __name__ == "__main__":
    sys.exit(main())
