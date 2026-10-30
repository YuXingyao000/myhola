"""Optional Ray execution for batch evaluation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

from tqdm import tqdm

T = TypeVar("T")
R = TypeVar("R")


def run_tasks(
    items: Sequence[T],
    worker: Callable[[T], R],
    *,
    use_ray: bool = False,
    num_cpus: int | None = None,
    timeout: int | None = None,
    desc: str | None = None,
) -> list[R | None]:
    if not use_ray:
        return [worker(item) for item in tqdm(items, desc=desc)]

    import ray

    init_kwargs = {}
    if num_cpus is not None:
        init_kwargs["num_cpus"] = num_cpus
    ray.init(ignore_reinit_error=True, **init_kwargs)
    remote_worker = ray.remote(max_retries=0)(worker)
    refs = [remote_worker.remote(item) for item in items]
    results: list[R | None] = []
    for ref in tqdm(refs, desc=desc):
        try:
            results.append(ray.get(ref, timeout=timeout))
        except ray.exceptions.GetTimeoutError:
            ray.cancel(ref)
            results.append(None)
        except Exception:
            results.append(None)
    ray.shutdown()
    return results
