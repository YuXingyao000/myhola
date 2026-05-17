from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from tqdm import tqdm

try:
    import ray
except ImportError:
    ray = None

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPECTED_OUTPUT_FILES = (
    "fidelity_metrics.npz",
    "summary.json",
    "failures.json",
    "precision_hist.png",
    "recall_hist.png",
    "f1_hist.png",
)


class EvaluationError(Exception):
    """Raised for expected per-model validation failures."""


@dataclass(frozen=True)
class ModelMetrics:
    model_id: str
    best_view_index: int
    precision: float
    recall: float
    f1: float
    view_count: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate sketch structural fidelity against multi-view references."
    )
    parser.add_argument("--data_root", required=True, type=Path)
    parser.add_argument("--output_root", required=True, type=Path)
    parser.add_argument("--input_filename", default="combined_imgs.npz")
    parser.add_argument("--tau", default=4.0, type=float)
    parser.add_argument("--hist_bin_width", default=0.05, type=float)
    parser.add_argument("--num_workers", default=1, type=int)
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


def ensure_numeric_finite(array: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(array)
    if result.dtype == np.dtype("O"):
        raise EvaluationError(f"{name} must be a numeric ndarray, got object dtype")
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
        rgb = np.ascontiguousarray(array.astype(np.float32))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    else:
        raise EvaluationError(
            f"{name} has unsupported shape {array.shape}; expected [H, W], [H, W, 1], or [H, W, 3]"
        )

    if np.issubdtype(array.dtype, np.floating):
        if gray.size and float(np.max(gray)) > 1.0:
            gray = gray / 255.0
    elif gray.size and float(np.max(gray)) > 1.0:
        gray = gray / 255.0

    gray = np.clip(gray, 0.0, 1.0).astype(np.float32, copy=False)
    return gray


def convert_reference_stack(reference_images: np.ndarray) -> np.ndarray:
    array = ensure_numeric_finite(reference_images, "sketch_imgs")
    if array.ndim == 3:
        view_count = int(array.shape[0])
    elif array.ndim == 4 and array.shape[-1] in (1, 3):
        view_count = int(array.shape[0])
    else:
        raise EvaluationError(
            "sketch_imgs has unsupported shape "
            f"{array.shape}; expected [V, H, W], [V, H, W, 1], or [V, H, W, 3]"
        )

    if view_count <= 0:
        raise EvaluationError("sketch_imgs has zero valid views")

    gray_views = [convert_to_gray_float32(array[index], f"sketch_imgs[{index}]") for index in range(view_count)]
    reference_shape = gray_views[0].shape
    if any(view.shape != reference_shape for view in gray_views):
        raise EvaluationError("reference sketch views do not share a consistent resolution")
    return np.stack(gray_views, axis=0)


def resize_to_shape(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    target_height, target_width = target_shape
    if image.shape == target_shape:
        return image
    interpolation = cv2.INTER_AREA
    if image.shape[0] < target_height or image.shape[1] < target_width:
        interpolation = cv2.INTER_LINEAR
    resized = cv2.resize(image, (target_width, target_height), interpolation=interpolation)
    return resized.astype(np.float32, copy=False)


def otsu_dark_foreground_mask(gray_image: np.ndarray) -> np.ndarray:
    gray_uint8 = np.clip(np.rint(gray_image * 255.0), 0, 255).astype(np.uint8)
    _, binary = cv2.threshold(
        gray_uint8,
        0,
        1,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )
    return binary.astype(bool)


def skeleton_points_from_mask(mask: np.ndarray) -> np.ndarray:
    skeleton = skeletonize(mask)
    points = np.argwhere(skeleton)
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    return points.astype(np.float32, copy=False)


def compute_view_metrics(
    generated_points: np.ndarray,
    reference_points: np.ndarray,
    tau: float,
) -> tuple[float, float, float]:
    if generated_points.size == 0 and reference_points.size == 0:
        return 1.0, 1.0, 1.0
    if generated_points.size == 0 or reference_points.size == 0:
        return 0.0, 0.0, 0.0

    reference_tree = cKDTree(reference_points)
    generated_tree = cKDTree(generated_points)

    generated_distances, _ = reference_tree.query(
        generated_points, k=1, distance_upper_bound=tau
    )
    reference_distances, _ = generated_tree.query(
        reference_points, k=1, distance_upper_bound=tau
    )

    supported_generated = int(np.count_nonzero(np.isfinite(generated_distances)))
    supported_reference = int(np.count_nonzero(np.isfinite(reference_distances)))

    precision = supported_generated / float(len(generated_points))
    recall = supported_reference / float(len(reference_points))
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return float(precision), float(recall), float(f1)


def load_model_arrays(model_dir: Path, input_filename: str) -> tuple[np.ndarray, np.ndarray]:
    input_path = model_dir / input_filename
    if not input_path.is_file():
        raise EvaluationError(f"missing input file: {input_path.name}")

    try:
        with np.load(input_path, allow_pickle=False) as npz_file:
            if "sketch_img" not in npz_file:
                raise EvaluationError("missing required key: sketch_img")
            if "sketch_imgs" not in npz_file:
                raise EvaluationError("missing required key: sketch_imgs")
            sketch_img = npz_file["sketch_img"]
            sketch_imgs = npz_file["sketch_imgs"]
    except EvaluationError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise EvaluationError(f"failed to read npz: {exc}") from exc

    generated_gray = convert_to_gray_float32(sketch_img, "sketch_img")
    reference_grays = convert_reference_stack(sketch_imgs)
    reference_shape = tuple(reference_grays.shape[1:])
    generated_gray = resize_to_shape(generated_gray, reference_shape)
    return generated_gray, reference_grays


def evaluate_model(model_dir: Path, input_filename: str, tau: float) -> ModelMetrics:
    generated_gray, reference_grays = load_model_arrays(model_dir, input_filename)
    generated_points = skeleton_points_from_mask(otsu_dark_foreground_mask(generated_gray))

    per_view_metrics: list[tuple[float, float, float]] = []
    for view_index, reference_gray in enumerate(reference_grays):
        try:
            reference_points = skeleton_points_from_mask(otsu_dark_foreground_mask(reference_gray))
            metrics = compute_view_metrics(generated_points, reference_points, tau)
        except Exception as exc:  # noqa: BLE001
            raise EvaluationError(
                f"failed while evaluating view {view_index}: {exc}"
            ) from exc
        per_view_metrics.append(metrics)

    best_view_index = max(
        range(len(per_view_metrics)),
        key=lambda index: (
            per_view_metrics[index][2],
            per_view_metrics[index][0],
            per_view_metrics[index][1],
            -index,
        ),
    )
    precision, recall, f1 = per_view_metrics[best_view_index]
    return ModelMetrics(
        model_id=model_dir.name,
        best_view_index=best_view_index,
        precision=precision,
        recall=recall,
        f1=f1,
        view_count=int(reference_grays.shape[0]),
    )


def build_failure_record(model_id: str, reason: str, exception: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {"model_id": model_id, "reason": reason}
    if exception is not None:
        record["exception"] = exception
    return record


def evaluate_model_safe(model_dir: Path, input_filename: str, tau: float) -> dict[str, Any]:
    try:
        result = evaluate_model(model_dir, input_filename=input_filename, tau=tau)
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
    def evaluate_model_remote(model_dir: str, input_filename: str, tau: float) -> dict[str, Any]:
        cv2.setNumThreads(1)
        return evaluate_model_safe(Path(model_dir), input_filename=input_filename, tau=tau)

else:
    evaluate_model_remote = None


def evaluate_models_serial(
    model_dirs: list[Path],
    input_filename: str,
    tau: float,
) -> list[dict[str, Any]]:
    outcomes: list[dict[str, Any]] = []
    for model_dir in tqdm(model_dirs, desc="Evaluating", unit="model"):
        outcomes.append(evaluate_model_safe(model_dir, input_filename=input_filename, tau=tau))
    return outcomes


def evaluate_models_with_ray(
    model_dirs: list[Path],
    input_filename: str,
    tau: float,
    num_workers: int,
) -> list[dict[str, Any]]:
    if ray is None or evaluate_model_remote is None:
        raise ImportError("ray is required for parallel execution but is not installed")

    started_here = False
    if not ray.is_initialized():
        ray.init(
            num_cpus=num_workers,
            ignore_reinit_error=True,
            include_dashboard=False,
            log_to_driver=False,
            logging_level="ERROR",
        )
        started_here = True

    try:
        refs = [
            evaluate_model_remote.remote(str(model_dir), input_filename, tau)
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

        finalized: list[dict[str, Any]] = []
        for index, outcome in enumerate(outcomes):
            if outcome is None:
                finalized.append(
                    {
                        "ok": False,
                        "failure": build_failure_record(
                            model_dirs[index].name,
                            "ray worker execution error",
                            "missing worker result",
                        ),
                    }
                )
                continue
            finalized.append(outcome)
        return finalized
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

    precision = np.asarray([result.precision for result in results], dtype=np.float64)
    recall = np.asarray([result.recall for result in results], dtype=np.float64)
    f1 = np.asarray([result.f1 for result in results], dtype=np.float64)
    if not np.all((0.0 <= precision) & (precision <= 1.0)):
        raise RuntimeError("precision values fall outside [0, 1]")
    if not np.all((0.0 <= recall) & (recall <= 1.0)):
        raise RuntimeError("recall values fall outside [0, 1]")
    if not np.all((0.0 <= f1) & (f1 <= 1.0)):
        raise RuntimeError("f1 values fall outside [0, 1]")

    for result in results:
        if not 0 <= result.best_view_index < result.view_count:
            raise RuntimeError(
                f"best_view_index out of range for model {result.model_id}: "
                f"{result.best_view_index} vs view_count={result.view_count}"
            )


def save_metrics_npz(output_path: Path, results: list[ModelMetrics]) -> None:
    model_ids = np.asarray([result.model_id for result in results], dtype=str)
    best_view_indices = np.asarray([result.best_view_index for result in results], dtype=np.int64)
    precision = np.asarray([result.precision for result in results], dtype=np.float32)
    recall = np.asarray([result.recall for result in results], dtype=np.float32)
    f1 = np.asarray([result.f1 for result in results], dtype=np.float32)

    lengths = {len(model_ids), len(best_view_indices), len(precision), len(recall), len(f1)}
    if len(lengths) != 1:
        raise RuntimeError("fidelity_metrics.npz arrays do not share the same length")

    np.savez(
        output_path,
        model_ids=model_ids,
        best_view_indices=best_view_indices,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def save_json(output_path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def make_hist_bins(hist_bin_width: float) -> np.ndarray:
    bins = np.arange(0.0, 1.0 + hist_bin_width, hist_bin_width, dtype=np.float64)
    if bins.size == 0 or not np.isclose(bins[0], 0.0):
        bins = np.insert(bins, 0, 0.0)
    bins = np.clip(bins, 0.0, 1.0)
    bins = np.unique(bins)
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
    tau: float,
    hist_bin_width: float,
    num_workers: int,
    parallel_backend: str,
    model_dirs: list[Path],
    results: list[ModelMetrics],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    num_valid_models = len(results)
    num_failed_models = len(failures)
    num_total_models = len(model_dirs)
    if num_valid_models + num_failed_models != num_total_models:
        raise RuntimeError("num_valid_models + num_failed_models does not equal num_total_models")

    non_standard_view_models = [
        {"model_id": result.model_id, "view_count": result.view_count}
        for result in results
        if result.view_count != 64
    ]

    if results:
        precision = np.asarray([result.precision for result in results], dtype=np.float64)
        recall = np.asarray([result.recall for result in results], dtype=np.float64)
        f1 = np.asarray([result.f1 for result in results], dtype=np.float64)
        mean_precision: float | None = float(np.mean(precision))
        mean_recall: float | None = float(np.mean(recall))
        mean_f1: float | None = float(np.mean(f1))
        sample_results = [
            {
                "model_id": result.model_id,
                "best_view_index": result.best_view_index,
                "precision": result.precision,
                "recall": result.recall,
                "f1": result.f1,
            }
            for result in results[:3]
        ]
    else:
        mean_precision = None
        mean_recall = None
        mean_f1 = None
        sample_results = []

    summary: dict[str, Any] = {
        "data_root": str(data_root),
        "output_root": str(output_root),
        "input_filename": input_filename,
        "tau": float(tau),
        "hist_bin_width": float(hist_bin_width),
        "num_workers": int(num_workers),
        "parallel_backend": parallel_backend,
        "num_total_models": num_total_models,
        "num_valid_models": num_valid_models,
        "num_failed_models": num_failed_models,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "mean_f1": mean_f1,
        "non_standard_view_models": non_standard_view_models,
        "failure_reason_counts": summarize_failures(failures),
        "sample_results": sample_results,
    }

    implementation_notes: list[str] = []
    if non_standard_view_models:
        implementation_notes.append(
            "Models with view counts other than 64 were evaluated with their actual number of reference views."
        )
    if parallel_backend == "ray":
        implementation_notes.append(
            "Parallel execution uses Ray to distribute per-model evaluation while preserving deterministic result order."
        )
    if implementation_notes:
        summary["implementation_notes"] = implementation_notes
    return summary


def print_terminal_summary(
    data_root: Path,
    output_root: Path,
    summary: dict[str, Any],
    failures: list[dict[str, Any]],
    results: list[ModelMetrics],
) -> None:
    print(f"Input root: {data_root}")
    print(f"Output root: {output_root}")
    print(f"Parallel backend: {summary['parallel_backend']}")
    print(f"Num workers: {summary['num_workers']}")
    print(f"Total models: {summary['num_total_models']}")
    print(f"Valid models: {summary['num_valid_models']}")
    print(f"Failed models: {summary['num_failed_models']}")
    print(f"Mean Precision: {summary['mean_precision']}")
    print(f"Mean Recall: {summary['mean_recall']}")
    print(f"Mean F1: {summary['mean_f1']}")

    failure_reason_counts = summary.get("failure_reason_counts", {})
    if failure_reason_counts:
        print("Failure reasons:")
        for reason, count in sorted(failure_reason_counts.items()):
            print(f"  {reason}: {count}")

    if results:
        print("Sample results:")
        for result in results[:3]:
            print(
                f"  {result.model_id} / {result.best_view_index} / "
                f"{result.precision:.4f} / {result.recall:.4f} / {result.f1:.4f}"
            )
    elif failures:
        first_failure = failures[0]
        detail = first_failure.get("exception")
        suffix = f" ({detail})" if detail else ""
        print(f"First failure: {first_failure['model_id']} / {first_failure['reason']}{suffix}")


def run_batch_evaluation(
    data_root: Path,
    output_root: Path,
    input_filename: str = "combined_imgs.npz",
    tau: float = 4.0,
    hist_bin_width: float = 0.05,
    num_workers: int = 1,
    overwrite: bool = False,
    print_summary: bool = False,
) -> int:
    if tau < 0.0:
        raise ValueError(f"tau must be non-negative, got {tau}")
    if hist_bin_width <= 0.0:
        raise ValueError(f"hist_bin_width must be positive, got {hist_bin_width}")
    if num_workers <= 0:
        raise ValueError(f"num_workers must be positive, got {num_workers}")

    output_paths = ensure_output_root(output_root, overwrite)
    model_dirs = list_model_dirs(data_root)

    parallel_backend = "serial"
    if num_workers > 1:
        parallel_backend = "ray"
        outcomes = evaluate_models_with_ray(
            model_dirs,
            input_filename=input_filename,
            tau=tau,
            num_workers=num_workers,
        )
    else:
        outcomes = evaluate_models_serial(
            model_dirs,
            input_filename=input_filename,
            tau=tau,
        )

    results, failures = split_outcomes(outcomes)
    validate_success_results(results)
    save_metrics_npz(output_paths["fidelity_metrics.npz"], results)
    save_json(output_paths["failures.json"], failures)

    summary = build_summary(
        data_root=data_root,
        output_root=output_root,
        input_filename=input_filename,
        tau=tau,
        hist_bin_width=hist_bin_width,
        num_workers=num_workers,
        parallel_backend=parallel_backend,
        model_dirs=model_dirs,
        results=results,
        failures=failures,
    )
    save_json(output_paths["summary.json"], summary)

    if results:
        bins = make_hist_bins(hist_bin_width)
        precision = np.asarray([result.precision for result in results], dtype=np.float64)
        recall = np.asarray([result.recall for result in results], dtype=np.float64)
        f1 = np.asarray([result.f1 for result in results], dtype=np.float64)
        plot_histogram(precision, bins, "Precision", output_paths["precision_hist.png"])
        plot_histogram(recall, bins, "Recall", output_paths["recall_hist.png"])
        plot_histogram(f1, bins, "F1", output_paths["f1_hist.png"])

    if print_summary:
        print_terminal_summary(data_root, output_root, summary, failures, results)

    return 0 if results else 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run_batch_evaluation(
            data_root=args.data_root,
            output_root=args.output_root,
            input_filename=args.input_filename,
            tau=args.tau,
            hist_bin_width=args.hist_bin_width,
            num_workers=args.num_workers,
            overwrite=args.overwrite,
            print_summary=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
