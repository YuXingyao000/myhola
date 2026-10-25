"""VAE model registry and compatibility imports.

Read ``legacy``, ``intersection`` or ``joint`` for a concrete model. New code
imports components from their owning modules; only model classes are part of
this package's advertised API. Historical helper imports remain lazy aliases.
"""
from importlib import import_module

_MODELS = {
    "AutoEncoder": ".legacy",
    "AutoEncoder_light": ".legacy",
    "AutoEncoder_light_exp": ".legacy",
    "AutoEncoder_intersection": ".intersection",
    "AutoEncoder_joint": ".joint",
    "AutoEncoder_uv_intersection": ".uv_intersection_training",
}

_COMPATIBILITY = {
    "BaseVAE": (".base", "BaseVAE"),
    "HolisticEncoder": (".encoders", "HolisticEncoder"),
    "FacePosterior": (".encoders", "FacePosterior"),
    "EdgeValueGATv2Conv": (".encoders", "EdgeValueGATv2Conv"),
    "UVFaceEncoder": (".encoders", "UVFaceEncoder"),
    "AttnIntersection": (".decoders.intersection", "AttnIntersection"),
    "IntersectionDecoder": (".decoders.intersection", "IntersectionDecoder"),
    "JointBoundaryDecoder": (".decoders.joint", "JointBoundaryDecoder"),
    "DecoderStage": (".state", "DecoderStage"),
    "PrimitiveGeometry": (".state", "PrimitiveGeometry"),
    "RefinementOutput": (".state", "RefinementOutput"),
    "closest_polyline_points": (".geometry", "closest_polyline_points"),
    "closest_surface_points": (".geometry", "closest_surface_points"),
    "DirectedTargets": (".losses.intersection", "DirectedTargets"),
    "IntersectionLoss": (".losses.intersection", "IntersectionLoss"),
    "balanced_binary_cross_entropy": (".losses.intersection", "balanced_binary_cross_entropy"),
    "_mean_l1": (".losses.intersection", "_mean_l1"),
    "_map_directed_targets": (".losses.intersection", "_map_directed_targets"),
    "boundary_costs": (".losses.joint", "boundary_costs"),
    "_match_costs": (".losses.joint", "_match_costs"),
    "match_boundaries": (".losses.joint", "match_boundaries"),
    "surface_boundary_consistency": (".losses.joint", "surface_boundary_consistency"),
    "joint_boundary_loss": (".losses.joint", "joint_boundary_loss"),
    "add_timer": (".profiling", "add_timer"),
    "profile_time": (".profiling", "profile_time"),
}


def build_autoencoder(v_conf):
    """Keep existing config names and construction semantics."""
    name = v_conf["name"]
    return getattr(import_module(_MODELS[name], __name__), name)(v_conf)


def __getattr__(name):
    if name in _MODELS:
        value = getattr(import_module(_MODELS[name], __name__), name)
    elif name in _COMPATIBILITY:
        module, attribute = _COMPATIBILITY[name]
        value = import_module(module, __name__)
        for part in attribute.split("."):
            value = getattr(value, part)
    elif name == "ResBlockXD":
        # Imported by the historical module; defined in the shared block library.
        value = getattr(import_module("src.brepnet.models.blocks"), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


__all__ = ["BaseVAE", *_MODELS, "build_autoencoder"]
