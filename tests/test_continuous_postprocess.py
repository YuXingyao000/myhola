"""Focused mathematical/OCC regressions for continuous geometry refinement."""
import copy

import numpy as np
import pytest

from src.postprocess.continuous_geometry import Axis, Config, Geometry, Observation, Spline, refine


def test_fixed_raw_observations_and_surface_movement_box():
    axis = Axis(np.array([0.0, 0.0, 1.0, 1.0]), 1)
    poles = np.array([[0., 0., 0.], [0., 1., 0.], [1., 0., 0.], [1., 1., 0.]])
    surface = Spline((axis, axis), poles.copy(), np.ones(4))
    geometry = Geometry([], [surface], np.empty((0, 3)))
    parameters = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.], [.5, .5]])
    raw_points = surface.evaluate(parameters) + [0., 0., .1]
    observation = Observation(parameters.copy(), raw_points.copy())
    config = Config(iterations=2, freeze_surface_iterations=0, surface_move_limit=.02)
    result, _, report, problem = refine(
        geometry, [], [observation], np.empty((0, 2), dtype=int),
        np.empty((0, 2), dtype=int), config)
    np.testing.assert_array_equal(geometry.surfaces[0].poles, poles)
    np.testing.assert_array_equal(problem.surface_observations[0].xyz, raw_points)
    np.testing.assert_array_equal(problem.surface_observations[0].parameters, parameters)
    np.testing.assert_allclose(result.surfaces[0].poles[:, 2], .02, rtol=0, atol=2e-7)
    assert report["final"]["terms"]["surface_data"] > .01
    assert report["final"]["objective"] < report["raw_initial"]["objective"]
    assert np.max(abs(result.surfaces[0].poles - poles)) <= .02 + config.constraint_tolerance
    assert all(not step["accepted"] or step["candidate"] <= step["before"] + config.descent_tolerance
               for step in report["history"])
    # The immutable problem must not alias caller-owned observations.
    observation.xyz[:] = 7
    np.testing.assert_array_equal(problem.surface_observations[0].xyz, raw_points)


def _occ_modules():
    pytest.importorskip("OCC")
    from src.postprocess import continuous_occ, occ_fit
    return continuous_occ, occ_fit


@pytest.mark.parametrize("kind", ["fitted_cylinder", "rational_torus"])
def test_periodic_unique_poles_survive_occ_writeback(kind):
    adapter, fitter = _occ_modules()
    from OCC.Core.Geom import Geom_ToroidalSurface
    from OCC.Core.GeomConvert import geomconvert
    from OCC.Core.gp import gp_Ax3
    if kind == "fitted_cylinder":
        angle, height = np.linspace(0, 2 * np.pi, 16), np.linspace(0, 1, 16)
        points = np.stack(np.broadcast_arrays(np.cos(angle)[:, None], np.sin(angle)[:, None], height[None]), axis=-1)
        source = fitter.create_surface(points)
    else:
        source = geomconvert.SurfaceToBSplineSurface(Geom_ToroidalSurface(gp_Ax3(), 1.2, .3))
    flags = source.IsUPeriodic(), source.IsVPeriodic()
    counts = source.NbUPoles(), source.NbVPoles()
    spline = adapter.from_occ_surface(source)
    assert tuple(axis.count for axis in spline.axes) == counts
    if not (source.IsURational() or source.IsVRational()):
        np.testing.assert_array_equal(spline.weights, np.ones(np.prod(counts)))
    original = spline.poles.copy()
    changed = copy.deepcopy(spline)
    changed.poles += np.random.default_rng(20261102).normal(size=changed.poles.shape) * .005
    written = adapter.write_occ_surface(source, changed)
    assert (written.IsUPeriodic(), written.IsVPeriodic()) == flags
    np.testing.assert_array_equal(adapter.from_occ_surface(source).poles, original)
    np.testing.assert_allclose(adapter.from_occ_surface(written).poles, changed.poles, rtol=0, atol=1e-14)
    parameters = np.stack(np.meshgrid(*[np.linspace(*axis.domain, 17) for axis in spline.axes], indexing="ij"), axis=-1).reshape(-1, 2)
    expected = np.array([written.Value(float(u), float(v)).Coord() for u, v in parameters])
    np.testing.assert_allclose(changed.evaluate(parameters), expected, rtol=0, atol=2e-12)
    for axis, periodic in enumerate(flags):
        if periodic:
            shifted = parameters.copy()
            shifted[:, axis] += 2 * np.ptp(spline.axes[axis].domain)
            np.testing.assert_allclose(changed.evaluate(shifted), expected, rtol=0, atol=2e-12)


def test_curve_observation_accepts_raw_fitted_endpoints_without_stationary_projection():
    adapter, fitter = _occ_modules()
    from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnCurve
    from OCC.Core.gp import gp_Pnt
    times = np.linspace(0, 1, 16)
    raw = np.column_stack((times, .2 * np.sin(5 * times), np.zeros_like(times)))
    raw += np.random.default_rng(2).normal(size=raw.shape) * .002
    curve = fitter.create_edge(raw)
    projections = [GeomAPI_ProjectPointOnCurve(gp_Pnt(*map(float, xyz)), curve,
                                              curve.FirstParameter(), curve.LastParameter()) for xyz in raw]
    assert any(projection.NbPoints() == 0 for projection in projections)
    observation = adapter.curve_observation(curve, raw)
    assert np.isfinite(observation.parameters).all()
    np.testing.assert_array_equal(observation.xyz, raw)
    for index, projection in enumerate(projections):
        if projection.NbPoints() == 0:
            assert observation.parameters[index, 0] in (curve.FirstParameter(), curve.LastParameter())


def test_curve_observation_compares_endpoints_even_when_stationary_points_exist():
    adapter, _ = _occ_modules()
    from OCC.Core.Geom import Geom_BSplineCurve
    from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnCurve
    from OCC.Core.TColgp import TColgp_Array1OfPnt
    from OCC.Core.TColStd import TColStd_Array1OfInteger, TColStd_Array1OfReal
    from OCC.Core.gp import gp_Pnt
    poles = TColgp_Array1OfPnt(1, 3)
    for index, xyz in enumerate([(0., 0., 0.), (.5, 0., 0.), (1., 1., 0.)], 1):
        poles.SetValue(index, gp_Pnt(*xyz))
    knots, multiplicities = TColStd_Array1OfReal(1, 2), TColStd_Array1OfInteger(1, 2)
    for index, value in enumerate((0., 1.), 1):
        knots.SetValue(index, value)
        multiplicities.SetValue(index, 3)
    curve = Geom_BSplineCurve(poles, knots, multiplicities, 2)
    point = gp_Pnt(-.25, 1., 0.)
    projection = GeomAPI_ProjectPointOnCurve(point, curve, 0., 1.)
    assert projection.NbPoints() > 0
    assert curve.Value(0.).Distance(point) < projection.LowerDistance()
    observation = adapter.curve_observation(curve, np.array([[-.25, 1., 0.]]))
    assert observation.parameters[0, 0] == 0.
