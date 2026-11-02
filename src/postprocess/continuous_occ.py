"""OCC initialization → fixed-basis optimization → same OCC spline templates.

Periodic poles remain unique optimization variables. Expanded knots are only
used for evaluation; no geometric matching or re-fitting is used on writeback.
"""
from __future__ import annotations

import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace
from OCC.Core.Geom import Geom_BSplineCurve, Geom_BSplineSurface
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnCurve, GeomAPI_ProjectPointOnSurf
from OCC.Core.gp import gp_Pnt

from src.postprocess.constants import TRANSFER_PRECISION
from src.postprocess.continuous_geometry import Axis, Config, Geometry, Observation, Spline, refine


def _array(values):
    return np.array([values.Value(i) for i in range(values.Lower(), values.Upper() + 1)], dtype=np.float64)


def _xyz(point):
    return [point.X(), point.Y(), point.Z()]


def from_occ_curve(curve):
    count = curve.NbPoles()
    periodic_count = count if curve.IsPeriodic() else None
    expanded = Geom_BSplineCurve.DownCast(curve.Copy())
    if periodic_count is not None:
        expanded.SetNotPeriodic()
    axis = Axis(_array(expanded.KnotSequence()), curve.Degree(), periodic_count)
    poles = np.array([_xyz(curve.Pole(i)) for i in range(1, count + 1)], dtype=np.float64)
    weights = np.array([curve.Weight(i) for i in range(1, count + 1)]) if curve.IsRational() else np.ones(count)
    return Spline((axis,), poles, weights)


def from_occ_surface(surface):
    nu, nv = surface.NbUPoles(), surface.NbVPoles()
    u_count = nu if surface.IsUPeriodic() else None
    v_count = nv if surface.IsVPeriodic() else None
    expanded = Geom_BSplineSurface.DownCast(surface.Copy())
    if u_count is not None:
        expanded.SetUNotPeriodic()
    if v_count is not None:
        expanded.SetVNotPeriodic()
    axes = (Axis(_array(expanded.UKnotSequence()), surface.UDegree(), u_count),
            Axis(_array(expanded.VKnotSequence()), surface.VDegree(), v_count))
    poles = np.array([_xyz(surface.Pole(i, j)) for i in range(1, nu + 1) for j in range(1, nv + 1)], dtype=np.float64)
    # OCCT's non-rational expanded surfaces may expose zero Weight() values.
    weights = (np.array([surface.Weight(i, j) for i in range(1, nu + 1) for j in range(1, nv + 1)])
               if surface.IsURational() or surface.IsVRational() else np.ones(nu * nv))
    return Spline(axes, poles, weights)


def write_occ_curve(template, spline):
    curve = Geom_BSplineCurve.DownCast(template.Copy())
    for i, xyz in enumerate(spline.poles, 1):
        curve.SetPole(i, gp_Pnt(*map(float, xyz)))
    return curve


def write_occ_surface(template, spline):
    surface = Geom_BSplineSurface.DownCast(template.Copy())
    nv = surface.NbVPoles()
    for index, xyz in enumerate(spline.poles):
        surface.SetPole(index // nv + 1, index % nv + 1, gp_Pnt(*map(float, xyz)))
    return surface


def curve_observation(curve, points):
    parameters = []
    first, last = curve.FirstParameter(), curve.LastParameter()
    for xyz in points:
        point = gp_Pnt(*map(float, xyz))
        projection = GeomAPI_ProjectPointOnCurve(point, curve, first, last)
        # Orthogonal extrema alone omit the bounded curve's endpoint minima.
        candidates = [first, last] + [projection.Parameter(i) for i in range(1, projection.NbPoints() + 1)]
        nearest = min(candidates, key=lambda value: curve.Value(value).SquareDistance(point))
        parameters.append([nearest])
    return Observation(np.asarray(parameters, dtype=np.float64), np.asarray(points, dtype=np.float64).copy())


def surface_observation(surface, points):
    parameters = []
    for xyz in points.reshape(-1, 3):
        projection = GeomAPI_ProjectPointOnSurf(gp_Pnt(*map(float, xyz)), surface, *surface.Bounds(), 1e-7)
        if projection.NbPoints() == 0:
            raise RuntimeError("Initial surface data projection has no solution")
        parameters.append(projection.LowerDistanceParameters())
    return Observation(np.asarray(parameters, dtype=np.float64), np.asarray(points, dtype=np.float64).reshape(-1, 3).copy())


def soft_endpoint_pairs(shape):
    """Fixed initial counterparts, from the same cues as the legacy penalties.

    These are soft geometric relations, never asserted to be vertex topology.
    No distance-only transitive clustering or GT native edges are introduced.
    """
    relations = set()

    def add(a, b):
        a, b = sorted((tuple(map(int, a)), tuple(map(int, b))))
        if a != b:
            relations.add((*a, *b))

    if shape.pair1 is not None:
        for edges, ends in zip(shape.pair1, shape.is_end_point):
            for a, b in combinations(zip(edges, ends.astype(int)), 2):
                add(a, b)
    for edge_ids in shape.face_edge_adj:
        endpoints = [(edge, end) for edge in edge_ids for end in (0, 1)]
        if len(endpoints) < 2:
            continue
        xyz = np.array([shape.recon_edge_points[edge, -1 if end else 0] for edge, end in endpoints])
        distances = np.square(xyz[:, None] - xyz[None]).sum(axis=-1)
        np.fill_diagonal(distances, np.inf)
        for i, j in enumerate(distances.argmin(axis=1)):
            add(endpoints[i], endpoints[j])
    return np.array(sorted(relations), dtype=int).reshape(-1, 4)


def _save_geometry(path, geometry, problem, uv, stage_points=None):
    arrays = {"uv": uv, "uv_reference": problem.uv_reference, "edge_faces": problem.edge_faces,
              "endpoint_pairs": problem.endpoint_pairs}
    for kind, splines, originals, observations in (
        ("curve", geometry.curves, problem.reference.curves, problem.curve_observations),
        ("surface", geometry.surfaces, problem.reference.surfaces, problem.surface_observations),
    ):
        for index, (spline, original, observation) in enumerate(zip(splines, originals, observations)):
            prefix = f"{kind}_{index}"
            arrays.update({prefix + "_poles": spline.poles, prefix + "_reference_poles": original.poles,
                           prefix + "_weights": spline.weights, prefix + "_observation_xyz": observation.xyz,
                           prefix + "_observation_parameters": observation.parameters})
            for axis_id, axis in enumerate(spline.axes):
                for name, value in (("knots", axis.knots), ("degree", axis.degree),
                                    ("periodic_count", axis.periodic_count or 0)):
                    arrays[f"{prefix}_axis_{axis_id}_{name}"] = np.asarray(value)
    if stage_points is not None:
        arrays.update(stage_points)
    np.savez_compressed(path, **arrays)


def _distance_summary(differences):
    lengths = np.linalg.norm(np.concatenate([item.reshape(-1, 3) for item in differences]), axis=1)
    return {"rms": float(np.sqrt(np.mean(lengths ** 2))),
            "mean": float(np.mean(lengths)), "max": float(np.max(lengths))}


def _cascade_diagnostics(initial, refined, observations, old_points, raw_points):
    """Data residuals use the same fixed parameters; they are not closest distances."""
    before = [spline.evaluate(obs.parameters) for spline, obs in zip(initial, observations)]
    after = [spline.evaluate(obs.parameters) for spline, obs in zip(refined, observations)]
    return {
        "legacy_sample_displacement_from_raw": _distance_summary([old_points - raw_points]),
        "initial_fit_residual_to_raw": _distance_summary([a - obs.xyz for a, obs in zip(before, observations)]),
        "final_fit_residual_to_raw": _distance_summary([a - obs.xyz for a, obs in zip(after, observations)]),
        "continuous_displacement_at_data_parameters": _distance_summary([a - b for a, b in zip(after, before)]),
    }


def refine_occ_geometry(shape, config, diagnostics_path, *, raw_face_points=None,
                        raw_edge_points=None, endpoint_pairs=None):
    """Input shape already fitted once, with natural-edge replacement disabled."""
    started = time.monotonic()
    config = Config(**(config or {}))
    curves = [from_occ_curve(curve) for curve in shape.recon_geom_curves]
    surfaces = [from_occ_surface(surface) for surface in shape.recon_geom_faces]
    geometry = Geometry(curves, surfaces, np.empty((0, 3), dtype=np.float64))
    face_points = shape.recon_face_points if raw_face_points is None else raw_face_points
    edge_points = shape.recon_edge_points if raw_edge_points is None else raw_edge_points
    curve_data = [curve_observation(curve, points) for curve, points in zip(shape.recon_geom_curves, edge_points)]
    surface_data = [surface_observation(surface, points) for surface, points in zip(shape.recon_geom_faces, face_points)]
    initialized = time.monotonic()
    owners = np.empty((len(curves), 2), dtype=int)
    for edge, a, b in shape.edge_face_connectivity:
        owners[edge] = a, b
    refined, uv, report, problem = refine(
        geometry, curve_data, surface_data, owners, np.full((len(curves), 2), -1, dtype=int),
        config, endpoint_pairs=soft_endpoint_pairs(shape) if endpoint_pairs is None else endpoint_pairs)
    shape.recon_geom_curves = [write_occ_curve(template, spline)
                              for template, spline in zip(shape.recon_geom_curves, refined.curves)]
    shape.recon_geom_faces = [write_occ_surface(template, spline)
                             for template, spline in zip(shape.recon_geom_faces, refined.surfaces)]
    shape.recon_topo_curves = [BRepBuilderAPI_MakeEdge(curve).Edge() for curve in shape.recon_geom_curves]
    shape.recon_topo_faces = [BRepBuilderAPI_MakeFace(surface, TRANSFER_PRECISION).Face() for surface in shape.recon_geom_faces]
    shape.replace_edge_idx = []
    report.update(scope="continuous OCC geometry before ShapeFix/trimming; not final STEP validity",
                  initialization_seconds=initialized - started, total_seconds=time.monotonic() - started,
                  endpoint_pair_count=len(problem.endpoint_pairs),
                  endpoint_relations="soft initial corner and per-face nearest endpoints; no inferred hard vertices",
                  periodic_curve_count=sum(c.axes[0].periodic_count is not None for c in curves),
                  periodic_surface_axes=sum(a.periodic_count is not None for s in surfaces for a in s.axes),
                  natural_boundary_replacement=False, refitted_after_optimization=False)
    stage_points = None
    if raw_face_points is not None:
        report.update(initial_geometry="legacy_optimized_then_fitted", data_reference="pre_legacy_raw_xyz",
                      raw_initial_stage="post_legacy_fit_before_continuous",
                      movement_box_reference="post_legacy_initial_poles; not total movement from raw",
                      endpoint_relations="frozen pre-legacy soft pairs; no re-pairing after legacy",
                      cascade={kind: _cascade_diagnostics(initial, final, observations, old, raw)
                               for kind, initial, final, observations, old, raw in (
                                   ("curve", curves, refined.curves, curve_data, shape.recon_edge_points, edge_points),
                                   ("surface", surfaces, refined.surfaces, surface_data, shape.recon_face_points, face_points))})
        stage_points = {"pre_legacy_face_points": face_points, "pre_legacy_edge_points": edge_points,
                        "post_legacy_face_points": shape.recon_face_points,
                        "post_legacy_edge_points": shape.recon_edge_points}
    path = Path(diagnostics_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    _save_geometry(path.with_suffix(".npz"), refined, problem, uv, stage_points)
    return shape.recon_geom_faces, shape.recon_topo_faces, shape.recon_geom_curves, shape.recon_topo_curves
