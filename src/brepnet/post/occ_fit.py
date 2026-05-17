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


def create_surface(points, use_variational_smoothing=USE_VARIATIONAL_SMOOTHING):
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
    for precision in FACE_FITTING_TOLERANCE:
        try:
            approx_face = fit_face(uv_points_array, precision, use_variational_smoothing)
            error = eval_fitting_face(approx_face, points)
            approx_face_list.append(approx_face)
            error_list.append(error)
            if error < 1e-2:
                break
        except Exception as e:
            continue

    if len(approx_face_list) > 0:
        approx_face = approx_face_list[np.argmin(error_list)]
    else:
        approx_face = fit_face(uv_points_array, ROUGH_FITTING_TOLERANCE, use_variational_smoothing=False)

    approx_face = set_face_uv_periodic(approx_face, points)
    return approx_face


def create_edge(points, use_variational_smoothing=USE_VARIATIONAL_SMOOTHING):
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
    for precision in EDGE_FITTING_TOLERANCE:
        try:
            approx_edge = fit_edge(u_points_array, precision, use_variational_smoothing)
            error = eval_fitting_edge(approx_edge, points)
            approx_edge_list.append(approx_edge)
            error_list.append(error)
            if error < 1e-2:
                break
        except Exception as e:
            continue

    if len(approx_edge_list) > 0:
        approx_edge = approx_edge_list[np.argmin(error_list)]
    else:
        approx_edge = fit_edge(u_points_array, ROUGH_FITTING_TOLERANCE, use_variational_smoothing=False)

    return approx_edge
