import random
from itertools import combinations

import numpy as np
import trimesh
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeSolid, BRepBuilderAPI_MakeVertex, BRepBuilderAPI_Sewing
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.Interface import Interface_Static
from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCC.Core.ShapeAnalysis import (
    ShapeAnalysis_FreeBounds,
    ShapeAnalysis_ShapeTolerance,
    ShapeAnalysis_Shell,
    ShapeAnalysis_Wire,
)
from OCC.Core.ShapeFix import ShapeFix_Face, ShapeFix_ShapeTolerance, ShapeFix_Shell, ShapeFix_Solid, ShapeFix_Wire
from OCC.Core.TopAbs import (
    TopAbs_COMPOUND,
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_SHELL,
    TopAbs_SOLID,
    TopAbs_VERTEX,
    TopAbs_WIRE,
)
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopTools import TopTools_HSequenceOfShape
from OCC.Core.TopoDS import topods
from OCC.Core.gp import gp_Pnt

from shared.occ_utils import get_primitives
from src.brepnet.post.constants import CONNECT_TOLERANCE, REMOVE_EDGE_TOLERANCE, TRANSFER_PRECISION
from src.brepnet.post.debug import Colors, viz_shapes
from src.brepnet.post.mesh import get_separated_surface


def get_edge_vertexes(edge):
    vertex_explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
    vertexes = []
    while vertex_explorer.More():
        vertex = topods.Vertex(vertex_explorer.Current())
        vertexes.append(vertex)
        vertex_explorer.Next()
    return vertexes


def explore_edges(shape):
    edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    edges = []
    while edge_explorer.More():
        edge = topods.Edge(edge_explorer.Current())
        edges.append(edge)
        edge_explorer.Next()
    return edges


def get_edge_length(edge, NUM_SEGMENTS=100):
    curve_data = BRep_Tool.Curve(edge)
    if curve_data and len(curve_data) == 3:
        curve_handle, first, last = curve_data
        segment_length = (last - first) / NUM_SEGMENTS
        edge_length = 0
        for i in range(NUM_SEGMENTS):
            u1 = first + segment_length * i
            u2 = first + segment_length * (i + 1)
            edge_length += np.linalg.norm(np.array(curve_handle.Value(u1).Coord()) - np.array(curve_handle.Value(u2).Coord()))
        return edge_length
    else:
        return 0


def check_edges_similarity(edge1, edge2, dis_threshold=1e-1, unit_sample_num=1000):
    def sample_edge(edge):
        sample_num = min(int(unit_sample_num * get_edge_length(edge)), 1000)
        curve_data = BRep_Tool.Curve(edge)
        if curve_data and len(curve_data) == 3:
            curve_handle, first, last = curve_data
            points = []
            for i in range(sample_num):
                param = first + (last - first) * i / (sample_num - 1)
                point = curve_handle.Value(param)
                points.append(point)
            return points
        else:
            return None

    edge1_sample_points = sample_edge(edge1)
    edge2_sample_points = sample_edge(edge2)

    if edge1_sample_points is None or edge2_sample_points is None:
        return False, -1

    dis_list1 = []
    for p1, p2 in zip(edge1_sample_points, edge2_sample_points):
        dis = p1.Distance(p2)
        dis_list1.append(dis)

    dis_list2 = []
    for p1, p2 in zip(edge1_sample_points, edge2_sample_points[::-1]):
        dis = p1.Distance(p2)
        dis_list2.append(dis)

    dis_list = dis_list1 if sum(dis_list1) < sum(dis_list2) else dis_list2

    is_similar = all([dis < dis_threshold for dis in dis_list])
    return is_similar, np.mean(dis_list)


def calculate_wire_bounding_box_length(wire):
    bbox = Bnd_Box()
    brepbnd = brepbndlib()
    brepbndlib.Add(wire, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    length = (xmax - xmin) + (ymax - ymin) + (zmax - zmin)
    return length


def check_edges_in_face(face, edges, dist_tol=CONNECT_TOLERANCE):
    for edge in edges:
        edge_vertexes = get_edge_vertexes(edge)
        for vertex in edge_vertexes:
            dist_shape_shape = BRepExtrema_DistShapeShape(vertex, face)
            min_dist = dist_shape_shape.Value()
            if min_dist > dist_tol:
                return False
    return True


def create_wire_from_unordered_edges(face_edges, connected_tolerance, max_retry_times=3, is_sort_by_length=True):
    wire_array = None
    for i in range(max_retry_times):
        random.shuffle(face_edges)
        edges_seq = TopTools_HSequenceOfShape()
        for edge in face_edges:
            edges_seq.Append(edge)
        wire_array_c = ShapeAnalysis_FreeBounds.ConnectEdgesToWires(edges_seq, connected_tolerance, False)

        all_wire_valid = True
        for i in range(1, wire_array_c.Length() + 1):
            wire_c = wire_array_c.Value(i)
            wire_analyzer = BRepCheck_Analyzer(wire_c)
            if not wire_analyzer.IsValid():
                all_wire_valid = False
                break
        if not all_wire_valid:
            break

        if wire_array is None:
            wire_array = wire_array_c
            continue

        if wire_array_c.Length() < wire_array.Length():
            wire_array = wire_array_c

    if wire_array is None or wire_array.Length() == 0:
        return None

    wire_list = [wire_array.Value(i) for i in range(1, wire_array.Length() + 1)]

    if is_sort_by_length:
        wire_list = sorted(wire_list, key=calculate_wire_bounding_box_length, reverse=True)

    return wire_list


def set_tolerance(v_item, v_precision):
    tolorancer = ShapeFix_ShapeTolerance()
    tolorancer.SetTolerance(v_item, v_precision)
    return v_item


def get_tolerance(v_item, v_type):
    tolorancer = ShapeAnalysis_ShapeTolerance()
    return tolorancer.Tolerance(v_item, v_type)


def create_trimmed_face_from_wire(geom_face, face_edges, wire_list, connected_tolerance):
    topo_face = BRepBuilderAPI_MakeFace(geom_face, 1e-6).Face()
    is_debug = False
    if len(wire_list) == 6:
        pass
    if (geom_face.IsUPeriodic() or geom_face.IsVPeriodic()) and len(get_primitives(topo_face, TopAbs_WIRE)) == 1 and len(face_edges) == 2:
        length_edge1 = get_edge_length(face_edges[0])
        length_edge2 = get_edge_length(face_edges[1])
        if abs(length_edge1 - length_edge2) < 0.01:
            final_wire = get_primitives(topo_face, TopAbs_WIRE)[0]
            final_wire = set_tolerance(final_wire, connected_tolerance)
            wire_list = [final_wire]
            pass
    face_fixer = ShapeFix_Face()
    face_fixer.Init(geom_face, connected_tolerance, True)
    fixed_wire_list = []
    for wire in wire_list:
        wire_fixer = ShapeFix_Wire(wire, topo_face, connected_tolerance)
        wire_fixer.SetModifyTopologyMode(True)
        wire_fixer.SetModifyGeometryMode(True)
        wire_fixer.FixSmall(False, REMOVE_EDGE_TOLERANCE)
        wire_fixer.SetMaxTolerance(connected_tolerance)
        wire_fixer.SetPrecision(connected_tolerance)
        wire_fixer.FixGaps3d()

        if wire_fixer.Wire().NbChildren() == 1 and not wire_fixer.Wire().Closed():
            continue

        fixed_wire = wire_fixer.Wire()

        if not fixed_wire.Closed():
            continue

        fixed_wire = set_tolerance(fixed_wire, connected_tolerance)
        face_fixer.Add(fixed_wire)
        fixed_wire_list.append(fixed_wire)

    if len(fixed_wire_list) == 0:
        return None

    try:
        face_fixer.FixWireTool().SetModifyGeometryMode(True)
        face_fixer.FixWireTool().SetMaxTolerance(connected_tolerance)
        face_fixer.FixWireTool().SetPrecision(connected_tolerance)
        face_fixer.FixWireTool().SetFixShiftedMode(True)
        face_fixer.FixWireTool().SetClosedWireMode(True)

        face_fixer.SetAutoCorrectPrecisionMode(False)
        face_fixer.SetPrecision(connected_tolerance)
        face_fixer.SetMaxTolerance(connected_tolerance)
        face_fixer.SetFixOrientationMode(True)
        face_fixer.SetFixMissingSeamMode(False)
        face_fixer.SetFixWireMode(True)
        face_fixer.SetFixLoopWiresMode(False)
        face_fixer.SetFixIntersectingWiresMode(False)
        face_fixer.SetFixPeriodicDegeneratedMode(False)
        face_fixer.SetFixSmallAreaWireMode(False)
        face_fixer.Perform()

        face_fixer.FixOrientation()
        face_fixer.FixMissingSeam()
        face_fixer.FixIntersectingWires()
        face_fixer.FixOrientation()

    except Exception as e:
        return None

    face_occ = face_fixer.Face()
    if face_occ.IsNull():
        return None
    face_occ = set_tolerance(face_occ, connected_tolerance)

    if is_debug:
        viz_shapes([face_occ])

    face_analyzer = BRepCheck_Analyzer(face_occ)
    if face_analyzer.IsValid():
        return face_occ
    else:
        return None


def drop_edges(face_edges_np, is_edge_closed, drop_edge_num=0, accepted_connected_loss=0.2):
    saved_edge_idx = list(combinations(range(face_edges_np.shape[0]), face_edges_np.shape[0] - drop_edge_num))
    saved_edge_idx += [list(range(face_edges_np.shape[0]))]
    connected_loss = []
    for edge_idx in saved_edge_idx:
        drop_edge_idx = list(set(range(face_edges_np.shape[0])) - set(edge_idx))
        if len(drop_edge_idx) != 0 and is_edge_closed[drop_edge_idx[0]]:
            connected_loss.append(1e6)
            continue
        new_face_edges_np = face_edges_np[list(edge_idx)]
        face_edges_endpoints = np.concatenate([new_face_edges_np[:, 0], new_face_edges_np[:, -1]])
        dist_matrix = np.linalg.norm(face_edges_endpoints[:, np.newaxis] - face_edges_endpoints, axis=2)
        dist_matrix = dist_matrix + np.eye(dist_matrix.shape[0]) * 1e6
        connected_loss_c = dist_matrix.min(axis=0).sum()
        connected_loss.append(connected_loss_c)

    connected_loss = np.array(connected_loss)
    accepted_combination_idx = list(np.where(connected_loss < accepted_connected_loss)[0])
    if len(connected_loss) - 1 in accepted_combination_idx:
        accepted_combination_idx.remove(len(connected_loss) - 1)

    if len(accepted_combination_idx) == 0:
        return None

    optional_drop_edge_idx = []
    for combinations_idx in accepted_combination_idx:
        drop_edge_idx = list(set(range(face_edges_np.shape[0])) - set(saved_edge_idx[combinations_idx]))
        optional_drop_edge_idx.append(drop_edge_idx)

    return optional_drop_edge_idx


def create_trimmed_face1(geom_face, face_edges, connected_tolerance, face_edges_numpy=None, is_edge_closed=None, drop_edge_num=0):
    if drop_edge_num > 0 and face_edges_numpy is not None:
        if len(face_edges) - drop_edge_num < 1:
            return None, None, False
        optional_drop_edge_idx = drop_edges(face_edges_numpy, is_edge_closed, drop_edge_num=drop_edge_num)
        if optional_drop_edge_idx is None:
            return None, None, False
        optional_face_edges = []
        for drop_edge_idx in optional_drop_edge_idx:
            optional_face_edges.append([face_edges[i] for i in range(len(face_edges)) if i not in drop_edge_idx])
    else:
        optional_face_edges = [face_edges]

    for face_edges in optional_face_edges:
        wire_list = create_wire_from_unordered_edges(face_edges, connected_tolerance)
        if wire_list is None:
            continue
        trimmed_face = create_trimmed_face_from_wire(geom_face, face_edges, wire_list, connected_tolerance)
        if trimmed_face is None or trimmed_face.IsNull():
            continue
        shape_tol_setter = ShapeFix_ShapeTolerance()
        shape_tol_setter.SetTolerance(trimmed_face, connected_tolerance)
        face_analyzer = BRepCheck_Analyzer(trimmed_face, False)
        is_face_valid = face_analyzer.IsValid()
        if is_face_valid:
            return wire_list, trimmed_face, is_face_valid
    return None, None, False


def create_trimmed_face2(geom_face, topo_face, face_edges, connected_tolerance):
    topo_face_edges = explore_edges(topo_face)
    replace_dict = {}
    for idx1, checked_edge in enumerate(face_edges):
        optional_edges = {}
        for idx2, replaced_edge in enumerate(topo_face_edges):
            is_similar, mean_dis = check_edges_similarity(checked_edge, replaced_edge)
            if is_similar:
                optional_edges[idx2] = mean_dis
        if len(optional_edges) == 0:
            continue
        optimal_edge_idx = min(optional_edges, key=optional_edges.get)
        replace_dict[idx1] = optimal_edge_idx
        face_edges[idx1] = topo_face_edges[optimal_edge_idx]
    return create_trimmed_face1(geom_face, face_edges, connected_tolerance)


def try_create_trimmed_face(geom_face, topo_face, face_edges, connected_tolerance, face_edges_numpy, is_edge_closed, max_drop_edge_num=2):
    wire_list1, trimmed_face1, is_face_valid1 = create_trimmed_face1(geom_face, face_edges, connected_tolerance)
    if is_face_valid1:
        return wire_list1, trimmed_face1, True

    for drop_edge_num in range(1, max_drop_edge_num + 1):
        wire_list1, trimmed_face1, is_face_valid1 = create_trimmed_face1(geom_face, face_edges, connected_tolerance, face_edges_numpy,
                                                                         is_edge_closed, drop_edge_num)
        if is_face_valid1:
            return wire_list1, trimmed_face1, True

    if trimmed_face1 is None:
        return wire_list1, trimmed_face1, False


def get_solid(trimmed_faces, connected_tolerance):
    try:
        random.shuffle(trimmed_faces)
        sewing = BRepBuilderAPI_Sewing()
        sewing.SetTolerance(connected_tolerance)
        for face in trimmed_faces:
            sewing.Add(face)
        sewing.Perform()
        sewn_shell = sewing.SewedShape()
        if sewn_shell.ShapeType() == TopAbs_COMPOUND:
            return None
            best_shell = None
            best_num = 0
            for shell in get_primitives(sewn_shell, TopAbs_SHELL):
                if best_shell is None or len(get_primitives(shell, TopAbs_FACE)) > best_num:
                    best_shell = shell
                    best_num = len(get_primitives(shell, TopAbs_FACE))
            if best_shell is None:
                return None
            sewn_shell = best_shell
        shape_tol_setter = ShapeFix_ShapeTolerance()
        shape_tol_setter.SetTolerance(sewn_shell, connected_tolerance)

        if not BRepCheck_Analyzer(sewn_shell).IsValid():
            fix_shell = ShapeFix_Shell(sewn_shell)
            fix_shell.SetPrecision(connected_tolerance)
            fix_shell.SetFixFaceMode(True)
            fix_shell.SetFixOrientationMode(True)
            fix_shell.Perform()
            sewn_shell = fix_shell.Shell()
            shape_tol_setter = ShapeFix_ShapeTolerance()
            shape_tol_setter.SetTolerance(sewn_shell, connected_tolerance)

        maker = BRepBuilderAPI_MakeSolid()
        maker.Add(sewn_shell)
        maker.Build()
        solid = maker.Solid()
        shape_tol_setter = ShapeFix_ShapeTolerance()
        shape_tol_setter.SetTolerance(solid, connected_tolerance)

        if not BRepCheck_Analyzer(solid).IsValid() or True:
            fix_solid = ShapeFix_Solid(solid)
            fix_solid.SetPrecision(connected_tolerance)
            fix_solid.SetMaxTolerance(connected_tolerance)
            fix_solid.SetFixShellMode(True)
            fix_solid.SetFixShellOrientationMode(True)
            fix_solid.SetCreateOpenSolidMode(False)
            fix_solid.Perform()
            solid = fix_solid.Solid()

        set_tolerance(solid, connected_tolerance)
        if solid.ShapeType() == TopAbs_SOLID and BRepCheck_Analyzer(solid).IsValid():
            return solid
        else:
            return None
    except Exception as e:
        print(e)
        return None


def get_compound(trimmed_faces, connected_tolerance):
    try:
        random.shuffle(trimmed_faces)
        sewing = BRepBuilderAPI_Sewing()
        sewing.SetTolerance(connected_tolerance)
        for face in trimmed_faces:
            sewing.Add(face)
        sewing.Perform()
        sewn_shell = sewing.SewedShape()
        if sewn_shell is not None:
            return sewn_shell
        else:
            return None
    except Exception as e:
        print(e)
        return None


def construct_brep(v_shape, connected_tolerance, isdebug=False):
    debug_idx = []
    if isdebug:
        print(f"{Colors.GREEN}################################ 1. Fit primitives ################################{Colors.RESET}")

    recon_edge_points = v_shape.recon_edge_points
    recon_geom_faces = v_shape.recon_geom_faces
    recon_topo_faces = v_shape.recon_topo_faces
    recon_curves = v_shape.recon_geom_curves
    recon_edge = v_shape.recon_topo_curves
    FaceEdgeAdj = v_shape.face_edge_adj
    is_edge_closed = v_shape.openness

    if False:
        viz_shapes(recon_geom_faces, transparency=0.5)

    if isdebug:
        print(f"{Colors.GREEN}################################ 2. Trim Face ######################################{Colors.RESET}")
    is_face_success_list = []
    trimmed_faces = []
    for idx, (geom_face, topo_face, face_edge_idx) in enumerate(zip(recon_geom_faces, recon_topo_faces, FaceEdgeAdj)):
        if isdebug:
            print(f"Process Face {idx}")
        if len(face_edge_idx) == 0:
            if isdebug:
                print(f"Face {idx} has no edge")
            is_face_success_list.append(False)
            continue
        face_edges = [recon_edge[edge_idx] for edge_idx in face_edge_idx]
        face_edges_numpy = recon_edge_points[face_edge_idx]
        is_edge_closed_c = [is_edge_closed[edge_idx] for edge_idx in face_edge_idx]
        wire_list, trimmed_face, is_valid = try_create_trimmed_face(geom_face, topo_face, face_edges, connected_tolerance,
                                                                    face_edges_numpy, is_edge_closed_c)
        is_valid = False if trimmed_face is None else is_valid

        if idx in debug_idx:
            pass

        if isdebug and not is_valid:
            print(f"Face {idx} is not valid{Colors.RESET}")

        is_face_success_list.append(is_valid)
        if is_valid:
            trimmed_faces.append(trimmed_face)

    result = [is_face_success_list, None, None]
    if len(trimmed_faces) > int(0.8 * len(recon_geom_faces)):
        v, f = get_separated_surface(trimmed_faces, v_precision1=0.1, v_precision2=0.2)
        separated_surface = trimesh.Trimesh(vertices=v, faces=f)
        result[1] = separated_surface
        result[2] = get_solid(trimmed_faces, connected_tolerance)

    if isdebug:
        print(f"{Colors.GREEN}################################ Construct Done ################################{Colors.RESET}")
    return result


def solid_valid_check(solid, tolerance=0.01):
    face_exp = TopExp_Explorer(solid, TopAbs_FACE)
    while face_exp.More():
        face = topods.Face(face_exp.Current())
        loc = TopLoc_Location()
        mesh = BRepMesh_IncrementalMesh(face, 0.01)
        triangulation = BRep_Tool.Triangulation(face, loc)
        if triangulation is None:
            return False
        wire_exp = TopExp_Explorer(face, TopAbs_WIRE)
        while wire_exp.More():
            wire = topods.Wire(wire_exp.Current())
            wire_analyzer = ShapeAnalysis_Wire(wire, face, tolerance)
            is_wire_ordered = wire_analyzer.CheckOrder(False)
            if not is_wire_ordered:
                return False
            is_no_wire_self_intersection = wire_analyzer.CheckSelfIntersection()
            if not is_no_wire_self_intersection:
                return False
            wire_exp.Next()
        face_exp.Next()

    shell_exp = TopExp_Explorer(solid, TopAbs_SHELL)
    while shell_exp.More():
        shell = topods.Shell(shell_exp.Current())
        shell_analyzer = ShapeAnalysis_Shell()
        shell_analyzer.LoadShells(shell)
        has_bad_edges = shell_analyzer.HasBadEdges()
        if has_bad_edges:
            return False
        shell_exp.Next()

    return True


def can_be_triangularized(shape):
    faces = get_primitives(shape, TopAbs_FACE)
    for face in faces:
        loc = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, loc)
        if triangulation is None:
            return False
    return True


def my_write_stp_file(v_shape, path):
    step_writer = STEPControl_Writer()
    dd = step_writer.WS().TransferWriter().FinderProcess()
    Interface_Static.SetCVal("write.step.schema", "AP203")
    step_writer.Transfer(v_shape, STEPControl_AsIs)
    status = step_writer.Write(path)

    if status != IFSelect_RetDone:
        raise AssertionError("load failed")


def solid_to_faceadj_graph(solid, radius=0.01):
    import numpy as np
    from occwl.solid import Solid
    from occwl.viewer import Viewer
    from occwl.graph import face_adjacency

    solid = Solid(solid)
    g = face_adjacency(solid, self_loops=True)

    print(f"Number of nodes (faces): {len(g.nodes)}")
    print(f"Number of edges: {len(g.edges)}")

    v = Viewer()
    v.display(solid, transparency=0.8)

    face_centers = {}
    for face_idx in g.nodes():
        face = g.nodes[face_idx]["face"]
        parbox = face.uv_bounds()
        umin, vmin = parbox.min_point()
        umax, vmax = parbox.max_point()
        center_uv = (umin + 0.5 * (umax - umin), vmin + 0.5 * (vmax - vmin))
        center = face.point(center_uv)
        v.display(Solid.make_sphere(center=center, radius=radius))
        face_centers[face_idx] = center

    for fi, fj in g.edges():
        pt1 = face_centers[fi]
        pt2 = face_centers[fj]
        up_dir = pt2 - pt1
        height = np.linalg.norm(up_dir)
        if height > 1e-3:
            v.display(
                    Solid.make_cylinder(
                            radius=radius, height=height, base_point=pt1, up_dir=up_dir
                    )
            )

    v.fit()
    v.show()


def solid_to_vertadj_graph(solid, radius=0.01):
    import numpy as np
    from occwl.solid import Solid
    from occwl.viewer import Viewer
    from occwl.graph import vertex_adjacency

    solid = Solid(solid)
    g = vertex_adjacency(solid, self_loops=True)

    print(f"Number of nodes (vertices): {len(g.nodes)}")
    print(f"Number of edges: {len(g.edges)}")

    v = Viewer()
    v.display(solid, transparency=0.5)
    points = {}
    for vert_idx in g.nodes:
        pt = g.nodes[vert_idx]["vertex"].point()
        points[vert_idx] = pt
        v.display(Solid.make_sphere(center=pt, radius=radius))

    for vi, vj in g.edges:
        pt1 = points[vi]
        pt2 = points[vj]
        up_dir = pt2 - pt1
        if np.linalg.norm(up_dir) < 1e-6:
            continue
        v.display(
                Solid.make_cylinder(
                        radius=radius, height=np.linalg.norm(up_dir), base_point=pt1, up_dir=up_dir
                )
        )

    v.fit()
    v.show()
