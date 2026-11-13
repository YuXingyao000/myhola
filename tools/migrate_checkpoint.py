"""Migrate a Lightning checkpoint across repository layout and module renames.

    python tools/migrate_checkpoint.py OLD.ckpt NEW.ckpt [--dry-run]
    python tools/migrate_checkpoint.py OLD.ckpt INIT.ckpt --curve-head-mode independent

Every entry of MIGRATIONS is applied once, in order, and recorded in
checkpoint["hola_migrations"]. Layout/rename migrations leave tensor values unchanged:
  strings   ordered (old, new) substring replacements in every non-tensor
            field (hyper_parameters, callbacks, loops), e.g. dataset.name
            and paths that resume compares against the current config;
  prefixes  ordered (old, new) state_dict key prefix renames. Optimizer state
            is indexed by parameter order, not by name, so a pure rename keeps
            it valid; changing parameter registration order does not.
The output is a new file; the source checkpoint is never overwritten.
Add a new version for every rename and list it in AGENTS.md.
Optional curve-head migration clones the old CNN/bbox values into independent
heads. Its result is initialization-only: optimizer/training state is removed.
Decoder widening repeats face channels and partitions consumer weights, retaining
the initial prediction to floating-point tolerance. It also starts a new run.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

LAYOUT_20261030 = [
    # Run outputs moved out of the repository.
    ("/mnt/d/python/checkpoints/", "/mnt/d/hola_runs/"),
    # Module paths (dotted), most specific first.
    ("src.brepnet.data.spatial_photo_dataset", "src.datasets.spatial_photo"),
    ("src.brepnet.data.photo_preference_dataset", "src.datasets.photo_preference"),
    ("src.brepnet.data.preference_dataset", "src.datasets.preference"),
    ("src.brepnet.data.photo_dataset", "src.datasets.photo"),
    ("src.brepnet.dataset", "src.datasets.brep"),
    ("src.brepnet.diffusion", "src.models.diffusion"),
    ("src.brepnet.models", "src.models"),
    ("src.brepnet.post", "src.postprocess"),
    ("src.brepnet.eval.run", "src.evaluate"),
    ("src.brepnet.eval", "src.eval"),
    ("src.brepnet.run_management", "src.common.run_management"),
    ("src.brepnet.prediction_export", "src.common.prediction_export"),
    ("src.brepnet.train", "src.train"),
    ("src.brepnet.inference", "src.inference"),
    # Repository-relative file paths.
    ("src/brepnet/data/list/", "src/preprocess/splits/"),
]

MIGRATIONS = [
    {"version": "2026-10-30-layout", "strings": LAYOUT_20261030, "prefixes": []},
    {"version": "2026-10-30-curve-head-roles", "optional": "curve_head_mode"},
    {"version": "2026-11-09-decoder-face-width", "optional": "decoder_face_dim"},
    {"version": "2026-11-13-encoder-coedge", "optional": "encoder_coedge_hidden_dim"},
]


def rewrite_strings(value, pairs, changes: list, path: str = ""):
    if isinstance(value, str):
        new = value
        for old, replacement in pairs:
            new = new.replace(old, replacement)
        if new != value:
            changes.append((path, value, new))
        return new
    if isinstance(value, dict):
        return {key: rewrite_strings(item, pairs, changes, f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, list):
        return [rewrite_strings(item, pairs, changes, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, tuple):
        return tuple(rewrite_strings(item, pairs, changes, f"{path}[{index}]") for index, item in enumerate(value))
    return value


def rename_prefixes(state_dict: dict, pairs) -> tuple[dict, int]:
    renamed, count = {}, 0
    for key, tensor in state_dict.items():
        new = key
        for old, replacement in pairs:
            if new.startswith(old):
                new = replacement + new[len(old):]
        count += new != key
        if new in renamed:
            raise ValueError(f"Prefix rename collides on {new!r}")
        renamed[new] = tensor
    return renamed, count


def split_curve_heads(checkpoint: dict, mode: str) -> int:
    """Clone only the unconditioned geometry parameters for a new training run."""
    if mode not in {"teacher_split", "independent"}:
        raise ValueError(f"Unsupported curve head mode: {mode!r}")
    state = checkpoint["state_dict"]
    roots = [prefix for prefix in ("model.decoder.", "decoder.")
             if any(key.startswith(prefix + "edge_geometry.points_decoder.") for key in state)]
    if len(roots) != 1:
        raise ValueError("Curve-head migration requires one complete Intersection decoder state_dict")
    root = roots[0]
    if any(key.startswith(root + name) for key in state
           for name in ("teacher_edge_geometry.", "coarse_edge_geometry.")):
        raise ValueError("Source already contains split curve heads; use its matching full configuration")
    source = root + "edge_geometry."
    destinations = ["teacher_edge_geometry"]
    if mode == "independent":
        destinations.append("coarse_edge_geometry")
    copies = {}
    for part in ("points_decoder.", "bbox_decoder."):
        values = {key[len(source):]: value for key, value in state.items() if key.startswith(source + part)}
        if not values:
            raise ValueError(f"Missing source geometry component: {source + part}")
        for destination in destinations:
            copies.update({root + destination + "." + key: value.clone() for key, value in values.items()})
    state.update(copies)
    params = checkpoint.setdefault("hyper_parameters", {})
    model = params.setdefault("model", {})
    model.setdefault("geometry_heads", {}).update(curve_mode=mode)
    checkpoint["initialization_only"] = {
        "reason": "Curve head parameters changed; initialize a new optimizer/run.",
        "source_epoch": checkpoint.get("epoch"),
        "source_global_step": checkpoint.get("global_step"),
    }
    for key in ("optimizer_states", "lr_schedulers", "loops", "callbacks"):
        checkpoint.pop(key, None)
    return len(copies)


def _repeat_face_inputs(weight, segments, repeats):
    """Expand each face segment independently; other concatenated inputs stay put."""
    # Per-element cyclic shares break replica symmetry while summing to one.
    # Four-way widening uses exactly representable binary coefficients.
    shares = ([14 / 64, 15 / 64, 17 / 64, 18 / 64] if repeats == 4 else
              [i / sum(range(1, repeats + 1)) for i in range(1, repeats + 1)])
    shares = weight.new_tensor(shares)
    result = []
    for part, (_, expand) in zip(weight.split([n for n, _ in segments], dim=1), segments):
        if expand:
            rows = torch.arange(part.shape[0], device=part.device).reshape(-1, *([1] * (part.ndim - 1)))
            columns = torch.arange(part[0].numel(), device=part.device).reshape(1, *part.shape[1:])
            phase = rows + columns
            result.extend(part * shares[(phase + copy) % repeats] for copy in range(repeats))
        else:
            result.append(part)
    return torch.cat(result, dim=1)


def widen_decoder_faces(checkpoint: dict, target_dim: int) -> int:
    """Widen decoder faces only; latent, encoder and all edge widths are retained."""
    state = checkpoint["state_dict"]
    model = checkpoint["hyper_parameters"]["model"]
    if model.get("intersection", {}).get("kind", "legacy") not in {"legacy", "conditional_attention"}:
        raise ValueError("Decoder-width migration supports legacy/conditional attention; local attention needs its own mapping")
    root = "model.decoder." if "model.decoder.face_features.output_projection.weight" in state else "decoder."
    source_dim = state[root + "face_features.output_projection.weight"].shape[0]
    if target_dim <= source_dim or target_dim % source_dim:
        raise ValueError("Decoder widening requires a larger integer multiple of the source face width")
    bbox_hidden = state[root + "face_geometry.bbox_decoder.0.conv1.weight"].shape[0]
    cnn_hidden = state[root + "face_geometry.points_decoder.1.conv1.weight"].shape[0]
    if bbox_hidden in {source_dim, target_dim} or cnn_hidden in {source_dim // 4, target_dim // 4}:
        raise ValueError("This width changes the face ResBlock shortcut structure; a different migration is required")
    repeats = target_dim // source_dim
    edge_dim = state[root + "classifier.weight"].shape[1]
    inputs = {
        "face_geometry.points_decoder.1.conv1.weight": [(source_dim // 4, True)],
        "face_geometry.points_decoder.1.downsample.0.weight": [(source_dim // 4, True)],
        "face_geometry.bbox_decoder.0.conv1.weight": [(source_dim, True)],
        "face_geometry.bbox_decoder.0.downsample.0.weight": [(source_dim, True)],
        "inter.attn_proj_in.weight": [(source_dim, True)],
    }
    if root + "inter.reference_uv_head.layers.0.weight" in state:
        inputs["inter.reference_uv_head.layers.0.weight"] = [(source_dim, True)] * 2
    if root + "refiner.face_delta.weight" in state:
        hidden = state[root + "refiner.face_delta.weight"].shape[1]
        geometry = state[root + "refiner.geometry_read.pool_projection.weight"].shape[0]
        inputs["refiner.relation_encoder.0.weight"] = [
            (edge_dim + geometry, False), (source_dim, True), (source_dim, True), (2 * hidden, False),
        ]
        inputs["refiner.face_update.0.weight"] = [(source_dim, True), (hidden, False)]
    if root + "edge_geometry.conditioning.projection.0.weight" in state:
        inputs["edge_geometry.conditioning.projection.0.weight"] = [
            (edge_dim, False), (source_dim, True), (source_dim, True),
        ]
    if root + "uv_boundary.context.0.weight" in state:
        inputs["uv_boundary.context.0.weight"] = (
            [(2 * edge_dim, False), (source_dim, True)]
            if model.get("shared_curve", {}).get("enabled", False) else
            [(edge_dim, False), (source_dim, True), (source_dim, True)]
        )
    if root + "curve_point_feedback.point_encoder.0.weight" in state:
        inputs["curve_point_feedback.point_encoder.0.weight"] = [
            (17 + edge_dim, False), (source_dim, True), (source_dim, True),
        ]
    changed = {}
    for key, segments in inputs.items():
        changed[root + key] = _repeat_face_inputs(state[root + key], segments, repeats)
    for layer in ("face_features.output_projection", "refiner.face_delta"):
        for kind in ("weight", "bias"):
            key = root + layer + "." + kind
            if key in state:
                value = state[key]
                changed[key] = value.repeat(repeats, *([1] * (value.ndim - 1)))
    state.update(changed)
    model["decoder_face_dim"] = target_dim
    source_initializers = {}
    for key in list(model):
        if key == "checkpoint" or (key.startswith("init_") and key.endswith("checkpoint")):
            if model[key]:
                source_initializers[key] = model[key]
            model[key] = None
    if "restoring_checkpoint" in model:
        model["restoring_checkpoint"] = False
    checkpoint["initialization_only"] = {
        "reason": "Decoder face width changed; initialize a new optimizer/run.",
        "source_epoch": checkpoint.get("epoch"), "source_global_step": checkpoint.get("global_step"),
        "source_decoder_face_dim": source_dim, "target_decoder_face_dim": target_dim,
        "input_partition": "cyclic per-element unequal shares; four-way [14,15,17,18]/64",
        "changed_tensors": list(changed),
        "source_initializers": source_initializers,
    }
    for key in ("optimizer_states", "lr_schedulers", "loops", "callbacks"):
        checkpoint.pop(key, None)
    return len(changed)


def add_encoder_coedge(checkpoint: dict, hidden_dim: int) -> int:
    """Add a zero-output curve residual while keeping all original tensors."""
    from src.models.vae.coedge import CoedgeFeatureFusion

    state = checkpoint["state_dict"]
    root = "model.encoder." if "model.encoder.graph_face_edge.0.lin_edge.weight" in state else "encoder."
    edge_dim = state[root + "graph_face_edge.0.lin_edge.weight"].shape[1]
    if any(key.startswith(root + "coedge_fusion.") for key in state):
        raise ValueError("Source already contains coedge fusion; load its matching configuration")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        module = CoedgeFeatureFusion(edge_dim, hidden_dim)
    added = {root + "coedge_fusion." + key: value for key, value in module.state_dict().items()}
    state.update(added)
    model = checkpoint["hyper_parameters"]["model"]
    model["encoder_coedge"] = {"enabled": True, "hidden_dim": hidden_dim}
    source_initializers = {}
    for key in list(model):
        if key == "checkpoint" or (key.startswith("init_") and key.endswith("checkpoint")):
            if model[key]:
                source_initializers[key] = model[key]
            model[key] = None
    model["restoring_checkpoint"] = False
    checkpoint["initialization_only"] = {
        "reason": "New Encoder coedge parameters; initialize a new optimizer/run.",
        "source_epoch": checkpoint.get("epoch"), "source_global_step": checkpoint.get("global_step"),
        "added_tensors": list(added), "source_initializers": source_initializers,
        "initialization_seed": 42,
    }
    for key in ("optimizer_states", "lr_schedulers", "loops", "callbacks"):
        checkpoint.pop(key, None)
    return len(added)


def migrate(checkpoint: dict, curve_head_mode: str | None = None, decoder_face_dim: int | None = None,
            encoder_coedge_hidden_dim: int | None = None) -> list[str]:
    applied = list(checkpoint.get("hola_migrations", []))
    report = []
    options = {"curve_head_mode": curve_head_mode, "decoder_face_dim": decoder_face_dim,
               "encoder_coedge_hidden_dim": encoder_coedge_hidden_dim}
    for migration in MIGRATIONS:
        option = migration.get("optional")
        if option and options[option] is None:
            continue
        if migration["version"] in applied:
            if option:
                model = checkpoint["hyper_parameters"]["model"]
                saved_mode = (model["geometry_heads"]["curve_mode"] if option == "curve_head_mode"
                              else model["decoder_face_dim"] if option == "decoder_face_dim"
                              else model["encoder_coedge"]["hidden_dim"])
                if saved_mode != options[option]:
                    raise ValueError(f"Checkpoint was already migrated to {saved_mode!r}")
            report.append(f"{migration['version']}: already applied")
            continue
        if option:
            count = (split_curve_heads(checkpoint, curve_head_mode) if option == "curve_head_mode"
                     else widen_decoder_faces(checkpoint, decoder_face_dim) if option == "decoder_face_dim"
                     else add_encoder_coedge(checkpoint, encoder_coedge_hidden_dim))
            applied.append(migration["version"])
            report.append(f"{migration['version']}: {count} migrated tensors; initialization only")
            continue
        changes = []
        for field in list(checkpoint):
            if field not in ("state_dict", "optimizer_states"):
                checkpoint[field] = rewrite_strings(checkpoint[field], migration["strings"], changes, field)
        checkpoint["state_dict"], renamed = rename_prefixes(checkpoint["state_dict"], migration["prefixes"])
        applied.append(migration["version"])
        report.append(f"{migration['version']}: {len(changes)} strings, {renamed} state_dict keys")
        report.extend(f"  {path}: {old!r} -> {new!r}" for path, old, new in changes)
    checkpoint["hola_migrations"] = applied
    return report


def main() -> int:
    # Direct script invocation must import this checkout, not another editable install.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--curve-head-mode", choices=("teacher_split", "independent"))
    parser.add_argument("--decoder-face-dim", type=int)
    parser.add_argument("--encoder-coedge-hidden-dim", type=int)
    args = parser.parse_args()
    if args.target.exists():
        parser.error(f"Refusing to overwrite {args.target}")
    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False, mmap=True)
    print("\n".join(migrate(checkpoint, args.curve_head_mode, args.decoder_face_dim,
                           args.encoder_coedge_hidden_dim)))
    if args.curve_head_mode or args.decoder_face_dim or args.encoder_coedge_hidden_dim:
        checkpoint["initialization_only"]["source_checkpoint"] = str(args.source.resolve())
    if not args.dry_run:
        args.target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, args.target)
        print(f"wrote {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
