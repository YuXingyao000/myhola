"""HoLa-BRep diffusion；按需加载主模型，避免仅导入工具时加载 VAE。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .model import Diffusion


def __getattr__(name: str) -> Any:
    if name == "Diffusion":
        from .model import Diffusion

        globals()[name] = Diffusion
        return Diffusion
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Diffusion"]
