import numpy as np

from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeVertex
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline, GeomAPI_PointsToBSplineSurface
from OCC.Core.TColgp import TColgp_Array1OfPnt, TColgp_Array2OfPnt
from OCC.Core.gp import gp_Pnt

from src.brepnet.post.constants import (
    CONTINUITY,
    EDGE_FITTING_TOLERANCE,
    FACE_FITTING_TOLERANCE,
    ROUGH_FITTING_TOLERANCE,
    TRANSFER_PRECISION,
    USE_VARIATIONAL_SMOOTHING,
    edge_fitting_deg_max,
    edge_fitting_deg_min,
    face_fiting_deg_max,
    face_fiting_deg_min,
    weight_CurveLength,
    weight_Curvature,
    weight_Torsion,
)


def _fitting_sample_indices(size, stride):
    """Keep the existing stride, including the previously unchecked last index."""
    return np.unique(np.r_[np.arange(0, size, stride), size - 1])


def _fitting_metrics_v2(topo_shape, key_points):
    """Return RMS and maximum point-to-shape distances in input XYZ units."""
    errors = []
    for point in key_points:
        vertex = BRepBuilderAPI_MakeVertex(gp_Pnt(float(point[0]), float(point[1]), float(point[2]))).Vertex()
        distance = BRepExtrema_DistShapeShape(vertex, topo_shape)
        if not distance.IsDone():
            raise RuntimeError("Fitting distance calculation did not finish")
        errors.append(distance.Value())
    errors = np.asarray(errors, dtype=np.float64)
    if errors.size == 0 or not np.isfinite(errors).all():
        raise ValueError("Fitting distances must be nonempty and finite")
    return float(np.sqrt(np.mean(errors ** 2))), float(np.max(errors))


def create_surface(points, use_variational_smoothing=USE_VARIATIONAL_SMOOTHING, *, fitting_policy="legacy"):
    """Fit a surface; opt-in v2 checks a boundary-inclusive grid in XYZ units.

    V2 ranks candidates by RMS + maximum distance, keeping the original
    tolerance order for ties. Its 1e-3 early stop applies only to checked points,
    before the unchanged periodic conversion; it is not a continuous bound.
    """
    if fitting_policy not in ("legacy", "fit_score_v2"):
        raise ValueError(f"Unknown fitting_policy: {fitting_policy}")

    def fit_face(uv_points_array, precision, use_variational_smoothing=USE_VARIATIONAL_SMOOTHING):
        if use_variational_smoothing:
            return GeomAPI_PointsToBSplineSurface(uv_points_array, weight_CurveLength, weight_Curvature, weight_Torsion,
                                                  face_fiting_deg_max, CONTINUITY, precision).Surface()
        else:
            return GeomAPI_PointsToBSplineSurface(uv_points_array, face_fiting_deg_min, face_fiting_deg_max, CONTINUITY,
                                                  precision).Surface()

    def set_face_uv_periodic(geom_face, points, tol=2e-3):
        u_intervals = np.sqrt(np.sum((points - np.roll(points, axis=0, shift=1)) ** 2, axis=2)).mean(axis=1)
        v_intervals = np.sqrt(np.sum((points - np.roll(points, axis=1, shift=1)) ** 2, axis=2)).mean(axis=0)

        u_min, u_max, v_min, v_max = geom_face.Bounds()
        us = geom_face.Value(u_min, v_min)
        ue = geom_face.Value(u_max, v_min)
        if us.Distance(ue) < np.mean(u_intervals):
            geom_face.SetUPeriodic()

        vs = geom_face.Value(u_min, v_min)
        ve = geom_face.Value(u_min, v_max)
        if vs.Distance(ve) < np.mean(v_intervals):
            geom_face.SetVPeriodic()
        return geom_face

    def eval_fitting_face(approx_face, uv_points_array):
        errors = []
        key_points = uv_points_array[::4, ::4, :].reshape(-1, 3)
        for point in key_points:
            topo_face = BRepBuilderAPI_MakeFace(approx_face, TRANSFER_PRECISION).Face()
            vertex = BRepBuilderAPI_MakeVertex(gp_Pnt(float(point[0]), float(point[1]), float(point[2]))).Vertex()
            min_dist = BRepExtrema_DistShapeShape(vertex, topo_face).Value()
            errors.append(min_dist)
        rmse = np.sqrt(np.mean(np.array(errors)))
        max_error = np.max(np.array(errors))
        return rmse + max_error

    num_u_points, num_v_points = points.shape[0], points.shape[1]
    uv_points_array = TColgp_Array2OfPnt(1, num_u_points, 1, num_v_points)
    for u_index in range(1, num_u_points + 1):
        for v_index in range(1, num_v_points + 1):
            pt = points[u_index - 1, v_index - 1]
            point_3d = gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2]))
            uv_points_array.SetValue(u_index, v_index, point_3d)

    approx_face_list = []
    error_list = []
    if fitting_policy == "fit_score_v2":
        u_indices = _fitting_sample_indices(num_u_points, 4)
        v_indices = _fitting_sample_indices(num_v_points, 4)
        key_points = points[np.ix_(u_indices, v_indices)].reshape(-1, 3)
    for precision in FACE_FITTING_TOLERANCE:
        try:
            approx_face = fit_face(uv_points_array, precision, use_variational_smoothing)
            if fitting_policy == "legacy":
                error = eval_fitting_face(approx_face, points)
                acceptable = error < 1e-2
            else:
                topo_face = BRepBuilderAPI_MakeFace(approx_face, TRANSFER_PRECISION).Face()
                rmse, max_error = _fitting_metrics_v2(topo_face, key_points)
                error = rmse + max_error
                acceptable = max_error <= 1e-3
            approx_face_list.append(approx_face)
            error_list.append(error)
            if acceptable:
                break
        except Exception as e:
            continue

    if len(approx_face_list) > 0:
        approx_face = approx_face_list[np.argmin(error_list)]
    else:
        approx_face = fit_face(uv_points_array, ROUGH_FITTING_TOLERANCE, use_variational_smoothing=False)

    approx_face = set_face_uv_periodic(approx_face, points)
    return approx_face


def create_edge(points, use_variational_smoothing=USE_VARIATIONAL_SMOOTHING, *, fitting_policy="legacy"):
    """Fit a curve, optionally checking the final sample with true RMS + max."""
    if fitting_policy not in ("legacy", "fit_score_v2"):
        raise ValueError(f"Unknown fitting_policy: {fitting_policy}")

    def fit_edge(u_points_array, precision, use_variational_smoothing=USE_VARIATIONAL_SMOOTHING):
        if use_variational_smoothing:
            return GeomAPI_PointsToBSpline(u_points_array, weight_CurveLength, weight_Curvature, weight_Torsion,
                                           edge_fitting_deg_max, CONTINUITY, precision).Curve()
        else:
            return GeomAPI_PointsToBSpline(u_points_array, edge_fitting_deg_min, edge_fitting_deg_max, CONTINUITY, precision).Curve()

    def eval_fitting_edge(approx_edge, u_points_array):
        errors = []
        key_points = u_points_array[::2, :].reshape(-1, 3)
        for point in key_points:
            topo_edge = BRepBuilderAPI_MakeEdge(approx_edge).Edge()
            vertex = BRepBuilderAPI_MakeVertex(gp_Pnt(float(point[0]), float(point[1]), float(point[2]))).Vertex()
            min_dist = BRepExtrema_DistShapeShape(vertex, topo_edge).Value()
            errors.append(min_dist)
        rmse = np.sqrt(np.mean(np.array(errors)))
        max_error = np.max(np.array(errors))
        return rmse + max_error

    num_u_points = points.shape[0]
    u_points_array = TColgp_Array1OfPnt(1, num_u_points)
    for u_index in range(1, num_u_points + 1):
        pt = points[u_index - 1]
        point_2d = gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2]))
        u_points_array.SetValue(u_index, point_2d)

    approx_edge_list = []
    error_list = []
    if fitting_policy == "fit_score_v2":
        key_points = points[_fitting_sample_indices(num_u_points, 2)].reshape(-1, 3)
    for precision in EDGE_FITTING_TOLERANCE:
        try:
            approx_edge = fit_edge(u_points_array, precision, use_variational_smoothing)
            if fitting_policy == "legacy":
                error = eval_fitting_edge(approx_edge, points)
                acceptable = error < 1e-2
            else:
                topo_edge = BRepBuilderAPI_MakeEdge(approx_edge).Edge()
                rmse, max_error = _fitting_metrics_v2(topo_edge, key_points)
                error = rmse + max_error
                acceptable = max_error <= 1e-3
            approx_edge_list.append(approx_edge)
            error_list.append(error)
            if acceptable:
                break
        except Exception as e:
            continue

    if len(approx_edge_list) > 0:
        approx_edge = approx_edge_list[np.argmin(error_list)]
    else:
        approx_edge = fit_edge(u_points_array, ROUGH_FITTING_TOLERANCE, use_variational_smoothing=False)

    return approx_edge
