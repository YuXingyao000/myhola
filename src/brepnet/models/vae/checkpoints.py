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


class IntersectionCheckpointLoader:
    """Strict migration policies for each accepted intersection experiment."""

    def __init__(self, model):
        self.model = model

    def init_uv_boundary(self, path: str) -> None:
        """Add only an auxiliary UV head to the complete adopted conditioned VAE."""
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise ValueError("UV boundary source must contain a state_dict mapping")
        raw_state = checkpoint.get("state_dict", checkpoint)
        if not isinstance(raw_state, Mapping):
            raise ValueError("UV boundary source state_dict must be a mapping")
        source = {}
        for key, value in raw_state.items():
            if not isinstance(key, str):
                raise ValueError("UV boundary source state keys must be strings")
            normalized = key.removeprefix("model.")
            if normalized in source:
                raise ValueError(f"Ambiguous checkpoint key after removing model prefix: {normalized}")
            source[normalized] = value
        prefix = "decoder.uv_boundary."
        current = self.model.state_dict()
        fresh = {key: value for key, value in current.items() if key.startswith(prefix)}
        if not fresh:
            raise ValueError("UV boundary migration requires enabled UV head parameters")
        expected = set(current) - set(fresh)
        missing = sorted(expected - source.keys())
        unexpected = sorted(source.keys() - expected)
        mismatched = sorted(key for key in expected & source.keys()
                            if not isinstance(source[key], Tensor) or source[key].shape != current[key].shape)
        if missing or unexpected or mismatched:
            raise ValueError(
                "UV boundary migration requires the complete conditioned CNN refinement model; "
                f"missing={missing}, unexpected={unexpected}, shape_mismatch={mismatched}"
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

        Validate the discarded CNN too: a partial, already-UV, local-inter,
        point-feedback or edge-value checkpoint is not a migration source.
        Complete UV models must use checkpoint / resume instead.
        """
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise ValueError("Face encoder checkpoint must be a mapping")
        raw_state = checkpoint.get("state_dict", checkpoint)
        if not isinstance(raw_state, Mapping):
            raise ValueError("Face encoder state_dict must be a mapping")
        source = {}
        for key, value in raw_state.items():
            if not isinstance(key, str):
                raise ValueError("Face encoder source state keys must be strings")
            normalized = key.removeprefix("model.")
            if normalized in source:
                raise ValueError(f"Ambiguous checkpoint key after removing model prefix: {normalized}")
            source[normalized] = value

        prefix = "encoder.face_coords."
        expected_old = {prefix + key: value for key, value in legacy_face_state.items()}
        old = {key: value for key, value in source.items() if key.startswith(prefix)}
        if old.keys() != expected_old.keys() or any(
            not isinstance(value, Tensor) or value.shape != expected_old[key].shape
            for key, value in old.items()
        ):
            raise ValueError("Face encoder source must contain the complete original face CNN state")
        current = self.model.state_dict()
        retained = {key: value for key, value in source.items() if not key.startswith(prefix)}
        expected = {key for key in current if not key.startswith(prefix)}
        missing = sorted(expected - retained.keys())
        unexpected = sorted(retained.keys() - expected)
        mismatched = sorted(
            key for key in expected & retained.keys()
            if not isinstance(retained[key], Tensor) or retained[key].shape != current[key].shape
        )
        if missing or unexpected or mismatched:
            raise ValueError(
                "Face encoder migration may replace only encoder.face_coords in the complete conditioned model; "
                f"missing={missing}, unexpected={unexpected}, shape_mismatch={mismatched}"
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
        Validate even the discarded keys so a partial or already-local checkpoint
        cannot be silently accepted as a legacy refinement source.
        """
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        source = checkpoint.get("state_dict", checkpoint)
        source = {key.removeprefix("model."): value for key, value in source.items()}
        prefix = "decoder.inter."
        expected_old = {prefix + key: value for key, value in legacy_inter_state.items()}
        old = {key: value for key, value in source.items() if key.startswith(prefix)}
        if old.keys() != expected_old.keys():
            raise ValueError("Refinement source must contain the complete legacy decoder.inter state")
        for key, value in old.items():
            if value.shape != expected_old[key].shape:
                raise ValueError(f"Legacy intersection shape mismatch for {key}")

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
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise ValueError("Refinement checkpoint must contain a state_dict mapping")
        raw_state = checkpoint.get("state_dict", checkpoint)
        if not isinstance(raw_state, Mapping):
            raise ValueError("Refinement checkpoint state_dict must be a mapping")
        source = {}
        for key, value in raw_state.items():
            if not isinstance(key, str):
                raise ValueError("Refinement state keys must be strings")
            normalized = key.removeprefix("model.")
            if normalized in source:
                raise ValueError(f"Ambiguous checkpoint key after removing model prefix: {normalized}")
            source[normalized] = value

        prefix = "decoder.edge_geometry.conditioning."
        current = self.model.state_dict()
        fresh = {key: value for key, value in current.items() if key.startswith(prefix)}
        if not fresh:
            raise ValueError("Curve conditioning migration requires enabled conditioning parameters")
        expected = {key for key in current if not key.startswith(prefix)}
        missing = sorted(expected - source.keys())
        unexpected = sorted(source.keys() - expected)
        mismatched = sorted(
            key for key in expected & source.keys()
            if not isinstance(source[key], Tensor) or source[key].shape != current[key].shape
        )
        if missing or unexpected or mismatched:
            raise ValueError(
                "Curve conditioning migration requires the complete original refinement model; "
                f"missing={missing}, unexpected={unexpected}, shape_mismatch={mismatched}"
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
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise ValueError("Curve feedback source must contain a state_dict mapping")
        raw_state = checkpoint.get("state_dict", checkpoint)
        if not isinstance(raw_state, Mapping):
            raise ValueError("Curve feedback source state_dict must be a mapping")
        source = {}
        for key, value in raw_state.items():
            if not isinstance(key, str):
                raise ValueError("Curve feedback source state keys must be strings")
            normalized = key.removeprefix("model.")
            if normalized in source:
                raise ValueError(f"Ambiguous checkpoint key after removing model prefix: {normalized}")
            source[normalized] = value

        prefix = "decoder.curve_point_feedback."
        current = self.model.state_dict()
        fresh = {key: value for key, value in current.items() if key.startswith(prefix)}
        if not fresh:
            raise ValueError("Curve feedback migration requires enabled curve_point_feedback parameters")
        expected = {key for key in current if not key.startswith(prefix)}
        missing = sorted(expected - source.keys())
        unexpected = sorted(source.keys() - expected)
        mismatched = sorted(
            key for key in expected & source.keys()
            if not isinstance(source[key], Tensor) or source[key].shape != current[key].shape
        )
        if missing or unexpected or mismatched:
            raise ValueError(
                "Curve feedback migration requires the complete conditioned legacy-inter refinement model "
                "without existing point feedback; "
                f"missing={missing}, unexpected={unexpected}, shape_mismatch={mismatched}"
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
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise ValueError("Encoder edge-value checkpoint must be a mapping")
        raw_state = checkpoint.get("state_dict", checkpoint)
        if not isinstance(raw_state, Mapping):
            raise ValueError("Encoder edge-value state_dict must be a mapping")
        source = {}
        for key, value in raw_state.items():
            if not isinstance(key, str):
                raise ValueError("Encoder edge-value state keys must be strings")
            normalized = key.removeprefix("model.")
            if normalized in source:
                raise ValueError(f"Ambiguous checkpoint key after removing model prefix: {normalized}")
            source[normalized] = value
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
        mismatched = sorted(
            key for key in expected & source.keys()
            if not isinstance(source[key], Tensor) or source[key].shape != current[key].shape
        )
        if missing or unexpected or mismatched:
            raise ValueError(
                "Encoder edge-value migration requires the complete conditioned model without value_net; "
                f"missing={missing}, unexpected={unexpected}, shape_mismatch={mismatched}"
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
        checkpoint = torch.load(str(path), weights_only=False, map_location="cpu", mmap=True)
        state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        if not isinstance(state, Mapping):
            raise ValueError("Checkpoint must contain a state_dict mapping")
        weights = {}
        for key, value in state.items():
            normalized = key[len("model."):] if key.startswith("model.") else key
            if normalized in weights:
                raise ValueError(f"Ambiguous checkpoint key after removing model prefix: {normalized}")
            weights[normalized] = value
        return weights


    def init_encoder(self, path) -> None:
        """Validated component migration; old decoder keys are explicitly ignored."""
        weights = self.read_weights(path)
        migrated = 0
        validated = []
        for name in self.model.ENCODER_MODULES:
            module = getattr(self.model, name)
            expected = module.state_dict()
            prefix = name + "."
            incoming = {key[len(prefix):]: value for key, value in weights.items() if key.startswith(prefix)}
            missing, unexpected = set(expected) - set(incoming), set(incoming) - set(expected)
            mismatched = [
                key for key in set(expected) & set(incoming)
                if not isinstance(incoming[key], Tensor) or incoming[key].shape != expected[key].shape
            ]
            if missing or unexpected or mismatched:
                raise ValueError(
                    f"Encoder migration failed for {name}: missing={sorted(missing)}, "
                    f"unexpected={sorted(unexpected)}, shape_mismatch={sorted(mismatched)}. "
                    "Use the exact legacy encoder architecture and posterior configuration."
                )
            validated.append((module, incoming))
            migrated += len(incoming)
        # Validate every component before modifying any parameter.
        for module, incoming in validated:
            module.load_state_dict(incoming, strict=True)
        logger.info("Migrated %d validated encoder/posterior tensors from %s; legacy decoder was not loaded", migrated, path)
