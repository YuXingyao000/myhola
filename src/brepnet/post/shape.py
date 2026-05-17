from itertools import combinations

import numpy as np
import torch
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_WIRE

from shared.occ_utils import get_primitives
from src.brepnet.post.constants import INTERPOLATION_PRECISION
from src.brepnet.post.distance import ChamferDistance
from src.brepnet.post.occ_topology import check_edges_similarity


_SKIP_FACE_PAIR = object()


def _build_face_pair_edge_candidates(edge_face_connectivity):
    pair_to_edge_ids = {}
    for conec in edge_face_connectivity.astype(np.int64):
        if (conec[1], conec[2]) in pair_to_edge_ids:
            pair_to_edge_ids[(conec[1], conec[2])].append(conec[0])
        elif (conec[2], conec[1]) in pair_to_edge_ids:
            pair_to_edge_ids[(conec[2], conec[1])].append(conec[0])
        else:
            pair_to_edge_ids[(conec[1], conec[2])] = [conec[0]]
    return pair_to_edge_ids


def _closest_endpoint_pair(edges, edge1, edge2):
    def dis(a, b):
        return np.linalg.norm(a - b)

    dis1 = dis(edges[edge1, 0], edges[edge2, 0])
    dis2 = dis(edges[edge1, 0], edges[edge2, -1])
    dis3 = dis(edges[edge1, -1], edges[edge2, 0])
    dis4 = dis(edges[edge1, -1], edges[edge2, -1])
    dises = [dis1, dis2, dis3, dis4]
    return np.argmin(dises), np.min(dises)


def _corner_endpoint_flags(i12_code, i23_code):
    endpoint_map = {
        (0, 0): [False, False, False],
        (0, 1): [False, False, True],
        (1, 2): [False, True, False],
        (1, 3): [False, True, True],
        (2, 0): [True, False, False],
        (2, 1): [True, False, True],
        (3, 2): [True, True, False],
        (3, 3): [True, True, True],
    }
    return endpoint_map.get((i12_code, i23_code))


def interpolation_face_points(face, is_use_cuda=False, precision=INTERPOLATION_PRECISION):
    if type(face) is np.ndarray:
        if is_use_cuda:
            face = torch.from_numpy(face).cuda()
        else:
            face = torch.from_numpy(face)

    res = face.shape[0]
    x_density = (res * torch.linalg.norm(face[0, 0] - face[0, 1], dim=-1) / precision).to(torch.long)
    y_density = (res * torch.linalg.norm(face[0, 0] - face[1, 0], dim=-1) / precision).to(torch.long)
    x_density, y_density = max(x_density, res), max(y_density, res)
    x = torch.linspace(-1., 1., x_density).to(face.device)
    y = torch.linspace(-1, 1, y_density).to(face.device)
    x, y = torch.meshgrid(x, y, indexing='ij')
    coords = torch.stack([x, y], dim=-1).reshape(1, -1, 1, 2)
    interpolation_face = torch.nn.functional.grid_sample(face[None].permute(0, 3, 1, 2),
                                                         coords, align_corners=True)[0, :, :, 0].permute(1, 0)
    assert interpolation_face.shape[0] >= res * res
    return interpolation_face


class Shape:
    """预测 BRep 的中间状态容器。

    该类是单个样本的唯一状态容器。字段被 `construct_brep.py` 和历史数据脚本
    直接读取，因此这里不拆成多个类，也不修改字段名、构造参数和方法副作用。

    生命周期：
    1. 初始化后持有网络预测的 face/edge/connectivity。
    2. 拓扑预处理后生成 openness、face_edge_adj、pair1/is_end_point。
    3. OCC 拟合后由调用方补充 recon_geom_* 和 recon_topo_* 字段。
    4. 周期面处理可能替换 recon_topo_curves 中的部分 edge。
    """

    def __init__(self, v_face_point, v_edge_points, v_connectivity, is_use_cuda=True):
        # 输入预测数据：面点、边点、边-双面连接关系。后续方法会原地更新这些字段。
        self.recon_face_points = v_face_point
        self.recon_edge_points = v_edge_points
        self.edge_face_connectivity = v_connectivity.astype(np.int64)
        self.device = torch.device('cuda') if is_use_cuda else torch.device('cpu')

        # 优化阶段会使用的加密采样面点。
        self.interpolation_face = []
        for face in self.recon_face_points:
            self.interpolation_face.append(interpolation_face_points(face, is_use_cuda=is_use_cuda))

        self.chamferdist = ChamferDistance()

        # 记录从原始输入或后处理过程中删除的边。
        self.remove_edge_idx_src = []
        self.remove_edge_idx_new = []

        self.have_data = True

    def _edge_to_face_distance(self, edge_id, face_id1, face_id2):
        edge = torch.from_numpy(self.recon_edge_points[edge_id]).to(self.device)
        distance1 = torch.sqrt(self.chamferdist(
                edge[None],
                self.interpolation_face[face_id1][None]))
        distance2 = torch.sqrt(self.chamferdist(
                edge[None],
                self.interpolation_face[face_id2][None]))
        return (distance1 + distance2) / 2

    def _select_edge_for_face_pair(self, edge_ids, face_id1, face_id2, edge2face_threshold):
        distance1 = self._edge_to_face_distance(edge_ids[0], face_id1, face_id2)

        if len(edge_ids) == 2:
            distance2 = self._edge_to_face_distance(edge_ids[1], face_id1, face_id2)
            if distance1 > edge2face_threshold and distance2 > edge2face_threshold:
                return None
            if distance1 < edge2face_threshold and distance1 < distance2:
                return edge_ids[0]
            return edge_ids[1]

        if len(edge_ids) == 1:
            if distance1 > edge2face_threshold:
                return None
            return edge_ids[0]

        return _SKIP_FACE_PAIR

    def select_valid_half_edges(self, edge2face_threshold=0.3, is_check_intersection=False, face2face_threshold=0.06):
        pair_to_edge_ids = _build_face_pair_edge_candidates(self.edge_face_connectivity)

        edges = []
        edge_face_connectivity = []

        for key, value in pair_to_edge_ids.items():
            face_id1 = key[0]
            face_id2 = key[1]
            selected_edge_id = self._select_edge_for_face_pair(value, face_id1, face_id2, edge2face_threshold)
            if selected_edge_id is _SKIP_FACE_PAIR:
                continue
            if selected_edge_id is None:
                for item in value:
                    self.remove_edge_idx_src.append(item)
                continue
            edges.append(self.recon_edge_points[selected_edge_id])
            edge_face_connectivity.append([len(edges) - 1, face_id1, face_id2])

        if len(edges) == 0 or len(edge_face_connectivity) == 0:
            self.have_data = False
            return

        self.recon_edge_points = np.stack(edges, axis=0)
        self.edge_face_connectivity = np.stack(edge_face_connectivity, axis=0)
        pass

    def remove_half_edges(self, edge2face_threshold=0.3, is_check_intersection=False, face2face_threshold=0.06):
        return self.select_valid_half_edges(edge2face_threshold, is_check_intersection, face2face_threshold)

    def detect_edge_openness(self, v_threshold=0.95):
        recon_edge = self.recon_edge_points
        dirs = (recon_edge[:, [0, -1]] - np.mean(recon_edge, axis=1, keepdims=True))
        cos_dir = ((dirs[:, 0] * dirs[:, 1]).sum(axis=1) /
                   (np.linalg.norm(dirs[:, 0], axis=1) * np.linalg.norm(dirs[:, 1], axis=1) + 1e-6))
        self.openness = cos_dir > v_threshold

        delta = recon_edge[:, [0, -1]].mean(axis=1)

    def check_openness(self, v_threshold=0.95):
        return self.detect_edge_openness(v_threshold)

    def build_face_edge_adjacency(self):
        self.face_edge_adj = [[] for _ in range(self.recon_face_points.shape[0])]
        for edge_face1_face2 in self.edge_face_connectivity:
            edge, face1, face2 = edge_face1_face2
            if face1 == face2:
                print("Face1 and Face2 should be different")
                continue
            assert edge not in self.face_edge_adj[face1]
            self.face_edge_adj[face1].append(edge)
            self.face_edge_adj[face2].append(edge)

    def build_fe(self):
        return self.build_face_edge_adjacency()

    def _build_inverse_edge_face_connectivity(self):
        inv_edge_face_connectivity = {}
        for edge_idx, face_idx1, face_idx2 in self.edge_face_connectivity:
            inv_edge_face_connectivity[(face_idx1, face_idx2)] = edge_idx
            inv_edge_face_connectivity[(face_idx2, face_idx1)] = edge_idx
        return inv_edge_face_connectivity

    def _build_face_adjacency_matrix(self):
        num_faces = self.recon_face_points.shape[0]
        face_adj = np.zeros((num_faces, num_faces), dtype=bool)
        face_adj[self.edge_face_connectivity[:, 1], self.edge_face_connectivity[:, 2]] = True
        np.fill_diagonal(face_adj, False)
        return np.logical_or(face_adj, face_adj.T)

    def build_corner_constraints(self, v_threshold=1e-1):
        num_faces = self.recon_face_points.shape[0]
        edges = self.recon_edge_points
        inv_edge_face_connectivity = self._build_inverse_edge_face_connectivity()
        face_adj = self._build_face_adjacency_matrix()

        pair1 = []
        is_end_point = []

        for face1 in range(num_faces):
            for face2 in range(num_faces):
                if not face_adj[face1, face2]:
                    continue
                e12 = inv_edge_face_connectivity[(face1, face2)]
                for face3 in range(num_faces):
                    if not face_adj[face1, face3] or not face_adj[face2, face3]:
                        continue
                    e13 = inv_edge_face_connectivity[(face1, face3)]
                    e23 = inv_edge_face_connectivity[(face2, face3)]

                    i12 = _closest_endpoint_pair(edges, e12, e13)
                    i23 = _closest_endpoint_pair(edges, e13, e23)
                    endpoint_flags = _corner_endpoint_flags(i12[0], i23[0])
                    if endpoint_flags is not None:
                        pair1.append([e12, e13, e23])
                        is_end_point.append(endpoint_flags)

        if len(pair1) == 0:
            self.pair1 = None
            self.is_end_point = None
            return
        self.pair1 = np.asarray(pair1).astype(np.int64)
        self.is_end_point = np.asarray(is_end_point).astype(bool)
        idx = np.zeros_like(self.is_end_point).astype(np.int64)
        idx[self.is_end_point] = 15
        vertex_clusters = edges[self.pair1, idx]

        mean_dis = np.linalg.norm(vertex_clusters[:, 0] - vertex_clusters[:, 1], axis=1) + \
                   np.linalg.norm(vertex_clusters[:, 1] - vertex_clusters[:, 2], axis=1) + \
                   np.linalg.norm(vertex_clusters[:, 2] - vertex_clusters[:, 0], axis=1)
        mean_dis /= 3

        flag = np.ones_like(self.is_end_point[:, 0])
        flag[mean_dis > v_threshold] = 0
        self.pair1 = self.pair1[flag]
        self.is_end_point = self.is_end_point[flag]
        pass

    def build_vertices(self, v_threshold=1e-1):
        return self.build_corner_constraints(v_threshold)

    def remove_isolated_edges(self):
        if self.pair1 is None:
            return
        edge_face_connectivity = self.edge_face_connectivity
        is_edge_in_pair = np.zeros(self.recon_edge_points.shape[0], dtype=bool)
        pair_edge_idx = np.unique(self.pair1.flatten())
        is_edge_closed = self.openness
        is_edge_in_pair[pair_edge_idx] = True
        is_edge_isolate = np.logical_and(~is_edge_in_pair, ~is_edge_closed)
        for iso_edge_idx in np.where(is_edge_isolate)[0]:
            edge_face1_face2 = edge_face_connectivity[edge_face_connectivity[:, 0] == iso_edge_idx, 1:]
            if edge_face1_face2.shape[0] == 0:
                continue
            face_idx1, face_idx2 = edge_face1_face2[0]
            if iso_edge_idx in self.face_edge_adj[face_idx1]:
                self.face_edge_adj[face_idx1].remove(iso_edge_idx)
            elif iso_edge_idx in self.face_edge_adj[face_idx2]:
                self.face_edge_adj[face_idx2].remove(iso_edge_idx)
            edge_face_connectivity = edge_face_connectivity[edge_face_connectivity[:, 0] != iso_edge_idx]
            self.remove_edge_idx_new.append(iso_edge_idx)

    def drop_edges(self, max_drop_num=2):
        recon_edge_points = self.recon_edge_points
        remove_edges_idx = [[] for _ in range(self.recon_face_points.shape[0])]
        for face_idx, face_edge_adj_c in enumerate(self.face_edge_adj):
            if len(face_edge_adj_c) == 0:
                continue
            max_combination_num = min(len(face_edge_adj_c) - 1, max_drop_num)
            all_combinations = []
            for combinations_num in range(1, max_combination_num + 1):
                if len(face_edge_adj_c) - combinations_num == 0:
                    continue
                all_combinations += list(combinations(face_edge_adj_c, len(face_edge_adj_c) - combinations_num))
            all_combinations += [face_edge_adj_c]

            connected_loss = []
            for sampled_face_edge_adj in all_combinations:
                dropp_edges = list(set(face_edge_adj_c) - set(sampled_face_edge_adj))
                if len(dropp_edges) != 0 and self.openness[dropp_edges[0]]:
                    connected_loss.append(1e6)
                    continue
                face_edges = recon_edge_points[list(sampled_face_edge_adj)]
                face_edge_endpoint = np.concatenate([face_edges[:, 0], face_edges[:, -1]])
                dist_matrix = np.linalg.norm(face_edge_endpoint[:, np.newaxis] - face_edge_endpoint, axis=2)
                dist_matrix = dist_matrix + np.eye(dist_matrix.shape[0]) * 1e6
                connected_loss_c = dist_matrix.min(axis=0).sum()
                connected_loss.append(connected_loss_c)
            if len(connected_loss) <= 1:
                continue
            best_combination_idx = np.argmin(connected_loss)
            if best_combination_idx < len(all_combinations) - 1:
                for edge_idx in face_edge_adj_c:
                    if edge_idx not in all_combinations[best_combination_idx]:
                        if not self.openness[edge_idx]:
                            remove_edges_idx[face_idx].append(edge_idx)

        remove_edges_idx_real = []
        for face_idx, edge_idx_list in enumerate(remove_edges_idx):
            for edge_idx in edge_idx_list:
                idx = np.where(self.edge_face_connectivity[:, 0] == edge_idx)[0]
                if len(idx) == 0:
                    continue
                face_idx1, face_idx2 = self.edge_face_connectivity[idx[0], 1:]
                another_face_idx = face_idx1 if face_idx1 != face_idx else face_idx2
                if edge_idx not in remove_edges_idx[another_face_idx]:
                    continue
                remove_edges_idx_real.append(edge_idx)
                self.edge_face_connectivity = np.delete(self.edge_face_connectivity, idx, axis=0)
                if edge_idx in self.face_edge_adj[face_idx1]:
                    self.face_edge_adj[face_idx1].remove(edge_idx)
                if edge_idx in self.face_edge_adj[face_idx2]:
                    self.face_edge_adj[face_idx2].remove(edge_idx)
        self.remove_edge_idx_new.extend(np.unique(remove_edges_idx_real))

    def replace_periodic_face_edges(self, is_replace_edge=False, connected_tolerance=0.1):
        self.clinder_face_idx = []
        self.replace_edge_idx = []
        if not is_replace_edge:
            return

        for face_idx, geom_face in enumerate(self.recon_geom_faces):
            face_edges_idx_list = self.face_edge_adj[face_idx]
            topo_face = self.recon_topo_faces[face_idx]
            if ((geom_face.IsUPeriodic() or geom_face.IsVPeriodic()) and len(face_edges_idx_list) == 2
                    and len(get_primitives(topo_face, TopAbs_WIRE)) == 1):
                self.clinder_face_idx.append(face_idx)
                face_edge_from_geom_face = get_primitives(topo_face, TopAbs_EDGE)
                for edge_idx in face_edges_idx_list:
                    recon_topo_edge = self.recon_topo_curves[edge_idx]
                    near_edge_from_geom_face = []
                    for edge_from_geom_face in face_edge_from_geom_face:
                        is_similar, dis = check_edges_similarity(recon_topo_edge, edge_from_geom_face)
                        if is_similar:
                            near_edge_from_geom_face.append((edge_from_geom_face, dis))
                    if len(near_edge_from_geom_face) == 0:
                        continue
                    if len(near_edge_from_geom_face) > 1:
                        near_edge_from_geom_face = sorted(near_edge_from_geom_face, key=lambda x: x[1])
                    self.recon_topo_curves[edge_idx] = near_edge_from_geom_face[0][0]
                    self.replace_edge_idx.append(edge_idx)
        pass

    def build_geom(self, is_replace_edge=False, connected_tolerance=0.1):
        return self.replace_periodic_face_edges(is_replace_edge, connected_tolerance)
