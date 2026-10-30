from OCC.Core.GeomAbs import GeomAbs_C2


EDGE_FITTING_TOLERANCE = [1e-3, 5e-3, 8e-3, 5e-2]
FACE_FITTING_TOLERANCE = [1e-3, 1e-2, 3e-2, 5e-2, 8e-2]

face_fiting_deg_min, face_fiting_deg_max = 3, 8
edge_fitting_deg_min, edge_fitting_deg_max = 0, 8

ROUGH_FITTING_TOLERANCE = 1e-1

FIX_TOLERANCE = 1e-2
FIX_PRECISION = 1e-2
CONNECT_TOLERANCE = [2e-3, 6e-3, 1e-2, 1.5e-2, 2e-2, 2.5e-2, 5e-2, 8e-2]
SEWING_TOLERANCE = 1e-1
REMOVE_EDGE_TOLERANCE = 1e-3
TRANSFER_PRECISION = 1e-6
MAX_DISTANCE_THRESHOLD = 1e-1
USE_VARIATIONAL_SMOOTHING = True
FIX_CLOSE_TOLERANCE = 1
FIX_GAP_TOLERANCE = 1e-1
weight_CurveLength, weight_Curvature, weight_Torsion = 1, 1, 1
IS_VIZ_WIRE, IS_VIZ_FACE, IS_VIZ_SHELL = False, False, False
CONTINUITY = GeomAbs_C2

INTERPOLATION_PRECISION = 0.05
