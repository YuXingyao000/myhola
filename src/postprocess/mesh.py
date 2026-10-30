import numpy as np

from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import topods


def get_separated_surface(trimmed_faces, v_precision1=1e-3, v_precision2=1e-1):
    points = []
    faces = []
    num_points = 0
    for face in trimmed_faces:
        loc = TopLoc_Location()
        mesh = BRepMesh_IncrementalMesh(face, v_precision1, False, v_precision2)
        triangulation = BRep_Tool.Triangulation(face, loc)
        if triangulation is None:
            continue

        v_points = np.zeros((triangulation.NbNodes(), 3), dtype=np.float32)
        f_faces = np.zeros((triangulation.NbTriangles(), 3), dtype=np.int64)
        for i in range(0, triangulation.NbNodes()):
            pnt = triangulation.Node(i + 1)
            v_points[i, 0] = pnt.X()
            v_points[i, 1] = pnt.Y()
            v_points[i, 2] = pnt.Z()
        for i in range(0, triangulation.NbTriangles()):
            tri = triangulation.Triangles().Value(i + 1)
            f_faces[i, 0] = tri.Get()[0] + num_points - 1
            f_faces[i, 1] = tri.Get()[1] + num_points - 1
            f_faces[i, 2] = tri.Get()[2] + num_points - 1
        points.append(v_points)
        faces.append(f_faces)
        num_points += v_points.shape[0]
    if len(points) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    points = np.concatenate(points, axis=0, dtype=np.float32)
    faces = np.concatenate(faces, axis=0, dtype=np.int64)
    return points, faces


def triangulate_face(v_face):
    loc = TopLoc_Location()
    triangulation = BRep_Tool.Triangulation(v_face, loc)

    if triangulation is None:
        mesh = BRepMesh_IncrementalMesh(v_face, 0.01)
        triangulation = BRep_Tool.Triangulation(v_face, loc)
        if triangulation is None:
            return np.zeros((0, 3)), np.zeros((0, 3))

    v_points = np.zeros((triangulation.NbNodes(), 3), dtype=np.float32)
    f_faces = np.zeros((triangulation.NbTriangles(), 3), dtype=np.int64)
    for i in range(0, triangulation.NbNodes()):
        pnt = triangulation.Node(i + 1)
        v_points[i, 0] = pnt.X()
        v_points[i, 1] = pnt.Y()
        v_points[i, 2] = pnt.Z()
    for i in range(0, triangulation.NbTriangles()):
        tri = triangulation.Triangles().Value(i + 1)
        f_faces[i, 0] = tri.Get()[0] - 1
        f_faces[i, 1] = tri.Get()[1] - 1
        f_faces[i, 2] = tri.Get()[2] - 1
    return v_points, f_faces


def triangulate_shape(v_shape):
    exp = TopExp_Explorer(v_shape, TopAbs_FACE)
    points = []
    faces = []
    num_points = 0
    while exp.More():
        face = topods.Face(exp.Current())

        loc = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, loc)

        if triangulation is None:
            mesh = BRepMesh_IncrementalMesh(face, 0.1)
            triangulation = BRep_Tool.Triangulation(face, loc)
            if triangulation is None:
                exp.Next()
                continue

        v_points = np.zeros((triangulation.NbNodes(), 3), dtype=np.float32)
        f_faces = np.zeros((triangulation.NbTriangles(), 3), dtype=np.int64)
        for i in range(0, triangulation.NbNodes()):
            pnt = triangulation.Node(i + 1)
            v_points[i, 0] = pnt.X()
            v_points[i, 1] = pnt.Y()
            v_points[i, 2] = pnt.Z()
        for i in range(0, triangulation.NbTriangles()):
            tri = triangulation.Triangles().Value(i + 1)
            f_faces[i, 0] = tri.Get()[0] + num_points - 1
            f_faces[i, 1] = tri.Get()[1] + num_points - 1
            f_faces[i, 2] = tri.Get()[2] + num_points - 1
        points.append(v_points)
        faces.append(f_faces)
        num_points += v_points.shape[0]
        exp.Next()
    if len(points) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    points = np.concatenate(points, axis=0, dtype=np.float32)
    faces = np.concatenate(faces, axis=0, dtype=np.int64)
    return points, faces
