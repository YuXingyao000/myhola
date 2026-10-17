from .config import Img2BrepConfig


def __getattr__(name):
    if name == "Img2BrepPipeline":
        from .pipeline import Img2BrepPipeline

        return Img2BrepPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['Img2BrepConfig', 'Img2BrepPipeline']
