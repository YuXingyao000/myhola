"""Common VAE operations, independent of a particular geometry representation."""

from abc import ABC, abstractmethod
from typing import Any

from torch import nn


class BaseVAE(nn.Module, ABC):
    """A VAE owns encoding, posterior sampling, decoding and its training flow.

    Subclasses specify their own arguments and return structures. In particular,
    the legacy dictionary decoder and the refinement state decoder need not
    pretend to return the same object. ``forward`` stays with the concrete model
    because matching, teacher paths, losses and exports differ across versions.

    The base registers no layers: subclass attribute names remain the checkpoint
    contract. ``sample`` is the existing reparameterization API; cached statistics
    retain their established mean/std convention.
    """

    @abstractmethod
    def encode(self, *args: Any, **kwargs: Any) -> Any:
        """Convert the model's observed geometry/topology into features."""

    @abstractmethod
    def sample(self, *args: Any, **kwargs: Any) -> Any:
        """Reparameterize the posterior, or return its mean for evaluation."""

    @abstractmethod
    def decode(self, *args: Any, **kwargs: Any) -> Any:
        """Decode the representation required by this model variant."""

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the variant's encode → sample → decode and supervision flow."""

    @abstractmethod
    def decode_latents(self, *args: Any, **kwargs: Any) -> Any:
        """Decode latent inputs without ground-truth geometry/topology."""

    def inference(self, v_face_features):
        """Compatibility entry used by existing diffusion and evaluation code."""
        return self.decode_latents(v_face_features)
