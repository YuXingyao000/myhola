"""Checkpoint I/O and explicit component migrations; never model layers.

Loaders are short-lived helpers. They are not registered as nn.Module children,
so moving migration code cannot add a prefix or duplicate model parameters.
"""
from __future__ import annotations

from collections.abc import Mapping
import logging

import torch
from torch import Tensor

from .encoders import EdgeValueGATv2Conv

logger = logging.getLogger(__name__)


def _read_checkpoint(path, **load_options):
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False, **load_options)
    state = checkpoint.get("state_dict", checkpoint)
    weights = {key.removeprefix("model."): value for key, value in state.items()}
    if len(weights) != len(state):
        raise ValueError("Ambiguous checkpoint keys after removing model prefix")
    return checkpoint, weights


class IntersectionCheckpointLoader:
    """Strict migration policies for each accepted intersection experiment."""

    def __init__(self, model):
        self.model = model

    def init_shared_curve(self, path: str) -> None:
        """Initialize from the complete trained two-owner UV auxiliary model.

        Retain the original geometry/encoder/refiner and UV Transformer. Map
        the old first-owner UV output onto the new per-owner shared head.
        """
        checkpoint, source = _read_checkpoint(path)
        current = self.model.state_dict()
        fresh_prefixes = ("decoder.shared_curve.", "decoder.uv_boundary.curve_embedding.")
        fresh = {key: value for key, value in current.items() if key.startswith(fresh_prefixes)}
        expected = set(current) - fresh.keys()
        if not fresh or source.keys() != expected:
            raise ValueError(
                "Shared curve initialization requires the complete previous UV-boundary model; "
                f"missing={sorted(expected - source.keys())}, unexpected={sorted(source.keys() - expected)}"
            )
        prefix = "decoder.uv_boundary."
        edge_dim, face_dim = self.model.decoder.edge_dim, self.model.decoder.face_dim
        context_key, output_key, bias_key = (prefix + name for name in (
            "context.0.weight", "output.1.weight", "output.1.bias",
        ))
        old_context, old_output = source[context_key], source[output_key]
        hidden = current[context_key].shape[0]
        if old_context.shape != (hidden, edge_dim + 2 * face_dim) or old_output.shape != (4, hidden) or source[bias_key].shape != (4,):
            raise ValueError("Shared curve source must use the original two-owner UV head with matching backbone width")
        mapped_context = torch.zeros_like(current[context_key])
        mapped_context[:, :edge_dim] = old_context[:, :edge_dim]
        mapped_context[:, 2 * edge_dim:] = old_context[:, edge_dim:edge_dim + face_dim]
        mapped = {
            context_key: mapped_context,
            output_key: old_output[:2].clone(),
            bias_key: source[bias_key][:2].clone(),
        }
        self.model.load_state_dict({**source, **fresh, **mapped}, strict=True)
        self.model.initialization_report = {
            "source": str(path), "source_epoch": checkpoint.get("epoch"),
            "source_global_step": checkpoint.get("global_step"),
            "retained_state_keys": len(source) - len(mapped),
            "mapped_uv_keys": sorted(mapped), "initialized_keys": sorted(fresh),
            "initialized_parameters": sum(value.numel() for value in fresh.values()),
            "mapping": "context [h,F_i,F_j] -> [h,g,F_owner], g columns zero; output keeps old owner-i channels",
        }
        logger.info("Shared curve initialization: %s", self.model.initialization_report)

    def init_uv_boundary(self, path: str) -> None:
        """Add only an auxiliary UV head to the complete adopted conditioned VAE."""
        checkpoint, source = _read_checkpoint(path)
        prefix = "decoder.uv_boundary."
        current = self.model.state_dict()
        fresh = {key: value for key, value in current.items() if key.startswith(prefix)}
        if not fresh:
            raise ValueError("UV boundary migration requires enabled UV head parameters")
        expected = set(current) - set(fresh)
        missing = sorted(expected - source.keys())
        unexpected = sorted(source.keys() - expected)
        if missing or unexpected:
            raise ValueError(
                "UV boundary migration requires the complete conditioned CNN refinement model; "
                f"missing={missing}, unexpected={unexpected}"
            )
        self.model.load_state_dict({**source, **fresh}, strict=True)
        self.model.initialization_report = {
            "source": str(path), "source_epoch": checkpoint.get("epoch"),
            "source_global_step": checkpoint.get("global_step"),
            "retained_state_keys": len(source), "initialized_uv_boundary_keys": sorted(fresh),
            "initialized_uv_boundary_parameters": sum(value.numel() for value in fresh.values()),
        }
        logger.info("UV boundary initialization: %s", self.model.initialization_report)


    def init_face_encoder(self, path: str, legacy_face_state: Mapping[str, Tensor]) -> None:
        """Replace exactly the CNN; strictly retain the complete conditioned VAE.

        Check the discarded CNN key set too: a partial, already-UV, local-inter,
        point-feedback or edge-value checkpoint is not a migration source.
        Complete UV models must use checkpoint / resume instead.
        """
        checkpoint, source = _read_checkpoint(path)
        prefix = "encoder.face_coords."
        expected_old = {prefix + key: value for key, value in legacy_face_state.items()}
        old = {key: value for key, value in source.items() if key.startswith(prefix)}
        if old.keys() != expected_old.keys():
            raise ValueError("Face encoder source must contain the complete original face CNN state")
        current = self.model.state_dict()
        retained = {key: value for key, value in source.items() if not key.startswith(prefix)}
        expected = {key for key in current if not key.startswith(prefix)}
        missing = sorted(expected - retained.keys())
        unexpected = sorted(retained.keys() - expected)
        if missing or unexpected:
            raise ValueError(
                "Face encoder migration may replace only encoder.face_coords in the complete conditioned model; "
                f"missing={missing}, unexpected={unexpected}"
            )
        fresh = {key: value for key, value in current.items() if key.startswith(prefix)}
        self.model.load_state_dict({**retained, **fresh}, strict=True)
        self.model.initialization_report = {
            "source": str(path),
            "source_epoch": checkpoint.get("epoch"),
            "source_global_step": checkpoint.get("global_step"),
            "retained_state_keys": len(retained),
            "discarded_face_encoder_state_keys": len(old),
            "discarded_face_encoder_parameters": sum(value.numel() for value in old.values()),
            "initialized_face_encoder_state_keys": sorted(fresh),
            "initialized_face_encoder_parameters": sum(value.numel() for value in fresh.values()),
        }
        logger.info("UV face encoder initialization: %s", self.model.initialization_report)


    def init_from_refinement(self, path: str, legacy_inter_state: Mapping[str, Tensor]) -> None:
        """Replace exactly the legacy inter, strictly preserving every other weight.

        This is initialization of a new architecture/run, never optimizer resume.
        Check even the discarded keys so a partial or already-local checkpoint
        cannot be silently accepted as a legacy refinement source.
        """
        checkpoint, source = _read_checkpoint(path)
        prefix = "decoder.inter."
        expected_old = {prefix + key: value for key, value in legacy_inter_state.items()}
        old = {key: value for key, value in source.items() if key.startswith(prefix)}
        if old.keys() != expected_old.keys():
            raise ValueError("Refinement source must contain the complete legacy decoder.inter state")
        current = self.model.state_dict()
        retained = {key: value for key, value in source.items() if not key.startswith(prefix)}
        expected = {key for key in current if not key.startswith(prefix)}
        if retained.keys() != expected:
            missing = sorted(expected - retained.keys())
            unexpected = sorted(retained.keys() - expected)
            raise ValueError(
                f"Refinement migration may replace only decoder.inter; "
                f"missing={missing}, unexpected={unexpected}"
            )
        # strict=True also rejects any shape mismatch in the retained modules.
        fresh_inter = {key: value for key, value in current.items() if key.startswith(prefix)}
        self.model.load_state_dict({**retained, **fresh_inter}, strict=True)
        self.model.initialization_report = {
            "source": str(path),
            "source_epoch": checkpoint.get("epoch"),
            "source_global_step": checkpoint.get("global_step"),
            "retained_state_keys": len(retained),
            "discarded_legacy_inter_keys": len(old),
            "initialized_local_inter_keys": len(fresh_inter),
        }
        logger.info("Local intersection initialization: %s", self.model.initialization_report)


    def init_curve_conditioning(self, path: str) -> None:
        """Add only zero-initialized curve modulation to a complete refinement model.

        This migration retains the legacy inter and every pre-existing parameter.
        Local-inter and already-conditioned checkpoints are not valid sources;
        loading a complete conditioned model uses checkpoint / resume instead.
        """
        checkpoint, source = _read_checkpoint(path)
        prefix = "decoder.edge_geometry.conditioning."
        current = self.model.state_dict()
        fresh = {key: value for key, value in current.items() if key.startswith(prefix)}
        if not fresh:
            raise ValueError("Curve conditioning migration requires enabled conditioning parameters")
        expected = {key for key in current if not key.startswith(prefix)}
        missing = sorted(expected - source.keys())
        unexpected = sorted(source.keys() - expected)
        if missing or unexpected:
            raise ValueError(
                "Curve conditioning migration requires the complete original refinement model; "
                f"missing={missing}, unexpected={unexpected}"
            )
        self.model.load_state_dict({**source, **fresh}, strict=True)
        self.model.initialization_report = {
            "source": str(path),
            "source_epoch": checkpoint.get("epoch"),
            "source_global_step": checkpoint.get("global_step"),
            "retained_state_keys": len(source),
            "initialized_curve_conditioning_keys": sorted(fresh),
            "initialized_curve_conditioning_parameters": sum(value.numel() for value in fresh.values()),
        }
        logger.info("Curve conditioning initialization: %s", self.model.initialization_report)


    def init_curve_feedback(self, path: str) -> None:
        """Add only point feedback to a complete, already-conditioned refinement model.

        All source tensors, including learned conditioning, must be present with
        exactly their existing shapes. Only decoder.curve_point_feedback is new.
        Full feedback checkpoints use strict checkpoint / resume loading instead.
        """
        checkpoint, source = _read_checkpoint(path)
        prefix = "decoder.curve_point_feedback."
        current = self.model.state_dict()
        fresh = {key: value for key, value in current.items() if key.startswith(prefix)}
        if not fresh:
            raise ValueError("Curve feedback migration requires enabled curve_point_feedback parameters")
        expected = {key for key in current if not key.startswith(prefix)}
        missing = sorted(expected - source.keys())
        unexpected = sorted(source.keys() - expected)
        if missing or unexpected:
            raise ValueError(
                "Curve feedback migration requires the complete conditioned legacy-inter refinement model "
                "without existing point feedback; "
                f"missing={missing}, unexpected={unexpected}"
            )
        self.model.load_state_dict({**source, **fresh}, strict=True)
        self.model.initialization_report = {
            "source": str(path),
            "source_epoch": checkpoint.get("epoch"),
            "source_global_step": checkpoint.get("global_step"),
            "retained_state_keys": len(source),
            "initialized_curve_point_feedback_keys": sorted(fresh),
            "initialized_curve_point_feedback_parameters": sum(value.numel() for value in fresh.values()),
        }
        logger.info("Curve point feedback initialization: %s", self.model.initialization_report)


    def init_encoder_edge_value(self, path: str) -> None:
        """Initialize only new GAT messages; preserve the full conditioned model.

        This starts a new optimizer/run. Existing edge-value checkpoints must
        use strict full checkpoint loading or trainer resume instead.
        """
        checkpoint, source = _read_checkpoint(path)
        current = self.model.state_dict()
        fresh = {
            f"encoder.graph_face_edge.{index}.{key}": value
            for index, layer in enumerate(self.model.encoder.graph_face_edge)
            if isinstance(layer, EdgeValueGATv2Conv)
            for key, value in layer.state_dict().items()
            if key.startswith("value_net.")
        }
        if not fresh:
            raise ValueError("Encoder edge-value migration requires enabled value_net parameters")
        expected = current.keys() - fresh.keys()
        missing = sorted(expected - source.keys())
        unexpected = sorted(source.keys() - expected)
        if missing or unexpected:
            raise ValueError(
                "Encoder edge-value migration requires the complete conditioned model without value_net; "
                f"missing={missing}, unexpected={unexpected}"
            )
        self.model.load_state_dict({**source, **fresh}, strict=True)
        self.model.initialization_report = {
            "source": str(path),
            "source_epoch": checkpoint.get("epoch"),
            "source_global_step": checkpoint.get("global_step"),
            "retained_state_keys": len(source),
            "initialized_encoder_edge_value_keys": sorted(fresh),
            "initialized_encoder_edge_value_parameters": sum(value.numel() for value in fresh.values()),
        }
        logger.info("Encoder edge-value initialization: %s", self.model.initialization_report)



class JointCheckpointLoader:
    """Load a complete joint model or migrate only its legacy encoder."""

    def __init__(self, model):
        self.model = model

    @staticmethod
    def read_weights(path) -> dict:
        return _read_checkpoint(path, mmap=True)[1]


    def init_encoder(self, path) -> None:
        """Load only encoder/posterior modules with PyTorch strict loading."""
        weights = self.read_weights(path)
        migrated = 0
        for name in self.model.ENCODER_MODULES:
            module = getattr(self.model, name)
            prefix = name + "."
            incoming = {key[len(prefix):]: value for key, value in weights.items() if key.startswith(prefix)}
            module.load_state_dict(incoming, strict=True)
            migrated += len(incoming)
        logger.info("Migrated %d encoder/posterior tensors from %s; legacy decoder was not loaded", migrated, path)
