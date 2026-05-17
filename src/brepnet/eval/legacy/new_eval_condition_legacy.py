import os
from pathlib import Path
from itertools import permutations, product
import traceback
import time

import matplotlib.pyplot as plt
import torch
import numpy as np

from tqdm import tqdm
import trimesh
import argparse

from chamferdist import ChamferDistance

from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE
from OCC.Core.BRep import BRep_Tool

import ray

from shared.occ_utils import get_primitives, get_triangulations, get_points_along_edge, get_curve_length
from src.brepnet.eval.metrics.validity import check_step_valid_soild


ROTATION_COMPARE_EPS = 1e-8


def get_octahedral_rotation_matrices():
    identity = np.eye(3, dtype=np.float32)
    rotations = []
    seen = set()
    for perm in permutations(range(3)):
        for signs in product((1, -1), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.float32)
            for row, col in enumerate(perm):
                matrix[row, col] = signs[row]
            if round(np.linalg.det(matrix)) != 1:
                continue
            key = tuple(int(v) for v in matrix.reshape(-1))
            if key in seen:
                continue
            seen.add(key)
            rotations.append(matrix)
    rotations.sort(
        key=lambda matrix: (
            0 if np.array_equal(matrix, identity) else 1,
            tuple(int(v) for v in matrix.reshape(-1)),
        )
    )
    if len(rotations) != 24:
        raise ValueError(f"Expected 24 octahedral rotations, got {len(rotations)}")
    return tuple(rotations)


OCTAHEDRAL_ROTATIONS = get_octahedral_rotation_matrices()


def is_vertex_close(p1, p2, tol=1e-3):
    return np.linalg.norm(np.array(p1) - np.array(p2)) < tol


def compute_statistics(eval_root, v_only_valid, listfile):
    all_folders = [folder for folder in os.listdir(eval_root) if os.path.isdir(os.path.join(eval_root, folder))]
    if listfile != '':
        valid_names = [item.strip() for item in open(listfile, 'r').readlines()]
        all_folders = list(set(all_folders) & set(valid_names))
        all_folders.sort()
    exception_folders = []
    results = {
        "prefix": []
    }
    for folder_name in tqdm(all_folders):
        if not os.path.exists(os.path.join(eval_root, folder_name, 'eval.npz')):
            exception_folders.append(folder_name)
            continue

        item = np.load(os.path.join(eval_root, folder_name, 'eval.npz'), allow_pickle=True)['results'].item()
        if item['num_recon_face'] == 1:
            exception_folders.append(folder_name)
            if v_only_valid:
                continue

        if v_only_valid and not os.path.exists(os.path.join(eval_root, folder_name, 'success.txt')):
            continue

        results["prefix"].append(folder_name)
        for key in item:
            if key not in results:
                results[key] = []
            results[key].append(item[key])

    if len(exception_folders) != 0:
        print(f"Found exception folders: {exception_folders}")

    for key in results:
        results[key] = np.array(results[key])

    results_str = ""
    results_str += "Number\n"
    results_str += f"Vertices: {np.mean(results['num_recon_vertex'])}/{np.mean(results['num_gt_vertex'])}\n"
    results_str += f"Edge: {np.mean(results['num_recon_edge'])}/{np.mean(results['num_gt_edge'])}\n"
    results_str += f"Face: {np.mean(results['num_recon_face'])}/{np.mean(results['num_gt_face'])}\n"

    results_str += "Chamfer\n"
    results_str += f"Vertices: {np.mean(results['vertex_cd'])}\n"
    results_str += f"Edge: {np.mean(results['edge_cd'])}\n"
    results_str += f"Face: {np.mean(results['face_cd'])}\n"

    results_str += "Detection\n"
    results_str += f"Vertices: {np.mean(results['vertex_fscore'])}\n"
    results_str += f"Edge: {np.mean(results['edge_fscore'])}\n"
    results_str += f"Face: {np.mean(results['face_fscore'])}\n"

    results_str += "Topology\n"
    results_str += f"FE: {np.mean(results['fe_fscore'])}\n"
    results_str += f"EV: {np.mean(results['ev_fscore'])}\n"

    results_str += "Accuracy\n"
    results_str += f"Vertices: {np.mean(results['vertex_acc_cd'])}\n"
    results_str += f"Edge: {np.mean(results['edge_acc_cd'])}\n"
    results_str += f"Face: {np.mean(results['face_acc_cd'])}\n"
    results_str += f"FE: {np.mean(results['fe_pre'])}\n"
    results_str += f"EV: {np.mean(results['ev_pre'])}\n"

    results_str += "Completeness\n"
    results_str += f"Vertices: {np.mean(results['vertex_com_cd'])}\n"
    results_str += f"Edge: {np.mean(results['edge_com_cd'])}\n"
    results_str += f"Face: {np.mean(results['face_com_cd'])}\n"
    results_str += f"FE: {np.mean(results['fe_rec'])}\n"
    results_str += f"EV: {np.mean(results['ev_rec'])}\n"
    print(results_str)
    print("{:.4f} {:.4f} {:.4f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(
        np.mean(results['vertex_cd']), np.mean(results['edge_cd']), np.mean(results['face_cd']),
        np.mean(results['vertex_fscore']), np.mean(results['edge_fscore']), np.mean(results['face_fscore']),
        np.mean(results['fe_fscore']), np.mean(results['ev_fscore']),
    ))
    print("{:.4f} {:.4f} {:.4f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}".format(
        np.mean(results['vertex_cd']), np.mean(results['edge_cd']), np.mean(results['face_cd']),
        np.mean(results['vertex_fscore']), np.mean(results['edge_fscore']), np.mean(results['face_fscore']),
        np.mean(results['fe_fscore']), np.mean(results['ev_fscore']),
    ))

    print("\nMean:")
    print(
        "{:.0f}/{:.0f} {:.0f}/{:.0f} {:.0f}/{:.0f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(
            np.mean(results['num_recon_vertex']), np.mean(results['num_gt_vertex']),
            np.mean(results['num_recon_edge']), np.mean(results['num_gt_edge']),
            np.mean(results['num_recon_face']), np.mean(results['num_gt_face']),
            np.mean(results['vertex_acc_cd']), np.mean(results['edge_acc_cd']), np.mean(results['face_acc_cd']),
            np.mean(results['vertex_com_cd']), np.mean(results['edge_com_cd']), np.mean(results['face_com_cd']),
            np.mean(results['vertex_pre']), np.mean(results['edge_pre']), np.mean(results['face_pre']),
            np.mean(results['fe_pre']), np.mean(results['ev_pre']),
            np.mean(results['vertex_rec']), np.mean(results['edge_rec']), np.mean(results['face_rec']),
            np.mean(results['fe_rec']), np.mean(results['ev_rec'])
        ))

    print("\nMedian:")
    print(
        "{:.0f}/{:.0f} {:.0f}/{:.0f} {:.0f}/{:.0f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(
            np.median(results['num_recon_vertex']), np.median(results['num_gt_vertex']),
            np.median(results['num_recon_edge']), np.median(results['num_gt_edge']),
            np.median(results['num_recon_face']), np.median(results['num_gt_face']),
            np.median(results['vertex_acc_cd']), np.median(results['edge_acc_cd']), np.median(results['face_acc_cd']),
            np.median(results['vertex_com_cd']), np.median(results['edge_com_cd']), np.median(results['face_com_cd']),
            np.median(results['vertex_pre']), np.median(results['edge_pre']), np.median(results['face_pre']),
            np.median(results['fe_pre']), np.median(results['ev_pre']),
            np.median(results['vertex_rec']), np.median(results['edge_rec']), np.median(results['face_rec']),
            np.median(results['fe_rec']), np.median(results['ev_rec'])
        ))
    print(f"{results['face_cd'].shape[0]}/{len(all_folders)} are valid")

    def draw():
        face_chamfer = results['face_cd']
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.hist(face_chamfer, bins=50, range=(0, 0.05), density=True, alpha=0.5, color='b', label='Face')
        ax.set_title('Face Chamfer Distance')
        ax.set_xlabel('Chamfer Distance')
        ax.set_ylabel('Density')
        ax.legend()
        plt.savefig(str(eval_root) + "_face_chamfer.png", dpi=600)

    draw()


def get_data(v_shape, v_num_per_m=100):
    faces, face_points, edges, edge_points, vertices, vertex_points = [], [], [], [], [], []
    for face in get_primitives(v_shape, TopAbs_FACE, v_remove_half=True):
        try:
            v, f = get_triangulations(face, 0.1, 0.1)
            if len(f) == 0:
                print("Ignore 0 face")
                continue
        except:
            print("Ignore 1 face")
            continue
        mesh_item = trimesh.Trimesh(vertices=v, faces=f)
        area = mesh_item.area
        num_samples = min(max(int(v_num_per_m * v_num_per_m * area), 5), 10000)
        pc_item, id_face = trimesh.sample.sample_surface(mesh_item, num_samples)
        normals = mesh_item.face_normals[id_face]
        faces.append(face)
        face_points.append(np.concatenate((pc_item, normals), axis=1))
    for edge in get_primitives(v_shape, TopAbs_EDGE, v_remove_half=True):
        length = get_curve_length(edge)
        num_samples = min(max(int(v_num_per_m * length), 5), 10000)
        v = get_points_along_edge(edge, num_samples)
        edges.append(edge)
        edge_points.append(v)
    for vertex in get_primitives(v_shape, TopAbs_VERTEX, v_remove_half=True):
        vertices.append(vertex)
        vertex_points.append(np.asarray([BRep_Tool.Pnt(vertex).Coord()]))
    vertex_points = np.stack(vertex_points, axis=0)
    return faces, face_points, edges, edge_points, vertices, vertex_points


def get_chamfer(v_recon_points, v_gt_points):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    chamfer_distance = ChamferDistance()
    recon_fp = torch.from_numpy(np.concatenate(v_recon_points, axis=0)).float().to(device)[:, :3]
    gt_fp = torch.from_numpy(np.concatenate(v_gt_points, axis=0)).float().to(device)[:, :3]
    fp_acc_cd = chamfer_distance(recon_fp.unsqueeze(0), gt_fp.unsqueeze(0),
                                 bidirectional=False, point_reduction='mean').cpu().item()
    fp_com_cd = chamfer_distance(gt_fp.unsqueeze(0), recon_fp.unsqueeze(0),
                                 bidirectional=False, point_reduction='mean').cpu().item()
    fp_cd = fp_acc_cd + fp_com_cd
    return fp_acc_cd, fp_com_cd, fp_cd


def get_match_ids(v_recon_points, v_gt_points):
    from scipy.optimize import linear_sum_assignment

    cost = np.zeros([len(v_recon_points), len(v_gt_points)])
    for i in range(cost.shape[0]):
        for j in range(cost.shape[1]):
            _, _, cost[i][j] = get_chamfer(
                v_recon_points[i][..., :3][None, ..., :3],
                v_gt_points[j][..., :3][None, ..., :3]
            )

    recon_indices, recon_to_gt = linear_sum_assignment(cost)

    result_recon2gt = -1 * np.ones(len(v_recon_points), dtype=np.int32)
    result_gt2recon = -1 * np.ones(len(v_gt_points), dtype=np.int32)

    result_recon2gt[recon_indices] = recon_to_gt
    result_gt2recon[recon_to_gt] = recon_indices
    return result_recon2gt, result_gt2recon, cost


def get_detection(id_recon_gt, id_gt_recon, cost_matrix, v_threshold=0.1):
    true_positive = 0
    for i in range(len(id_recon_gt)):
        if id_recon_gt[i] != -1 and cost_matrix[i, id_recon_gt[i]] < v_threshold:
            true_positive += 1
    precision = true_positive / (len(id_recon_gt) + 1e-6)
    recall = true_positive / (len(id_gt_recon) + 1e-6)
    return 2 * precision * recall / (precision + recall + 1e-6), precision, recall


def get_topology(faces, edges, vertices):
    recon_face_edge, recon_edge_vertex = {}, {}
    for i_face, face in enumerate(faces):
        face_edge = []
        for edge in get_primitives(face, TopAbs_EDGE):
            face_edge.append(edges.index(edge) if edge in edges else edges.index(edge.Reversed()))
        recon_face_edge[i_face] = list(set(face_edge))

    for i_edge, edge in enumerate(edges):
        edge_vertex = []
        for vertex in get_primitives(edge, TopAbs_VERTEX):
            edge_vertex.append(vertices.index(vertex) if vertex in vertices else vertices.index(vertex.Reversed()))
        recon_edge_vertex[i_edge] = list(set(edge_vertex))
    return recon_face_edge, recon_edge_vertex


def get_topo_detection(recon_face_edge, gt_face_edge, id_recon_gt_face, id_recon_gt_edge):
    positive = 0
    for i_recon_face, edges in recon_face_edge.items():
        if i_recon_face >= len(id_recon_gt_face):
            continue
        i_gt_face = id_recon_gt_face[i_recon_face]
        if i_gt_face == -1:
            continue
        for i_edge in edges:
            if id_recon_gt_edge[i_edge] in gt_face_edge[i_gt_face]:
                positive += 1
    precision = positive / (sum([len(edges) for edges in recon_face_edge.values()]) + 1e-6)
    recall = positive / (sum([len(edges) for edges in gt_face_edge.values()]) + 1e-6)
    return 2 * precision * recall / (precision + recall + 1e-6), precision, recall


def transform_points(points, rotation_matrix):
    transformed = np.array(points, copy=True)
    transformed[..., :3] = transformed[..., :3] @ rotation_matrix.T
    if transformed.shape[-1] >= 6:
        transformed[..., 3:6] = transformed[..., 3:6] @ rotation_matrix.T
    return transformed


def transform_point_sets(point_sets, rotation_matrix):
    return [transform_points(points, rotation_matrix) for points in point_sets]


def write_error_marker(eval_root, folder_name, message):
    sample_dir = eval_root / folder_name
    sample_dir.mkdir(parents=True, exist_ok=True)
    with open(sample_dir / "error.txt", "w", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


def metrics_are_finite(results):
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            if not np.all(np.isfinite(value)):
                return False
        elif isinstance(value, (float, np.floating)):
            if not np.isfinite(value):
                return False
    return True


def build_rotation_summary(rotation_index, rotation_matrix, result=None, error=""):
    summary = {
        "rotation_index": rotation_index,
        "rotation_matrix": rotation_matrix.astype(np.float32),
        "valid": result is not None,
        "error": error,
        "face_fscore": float("-inf"),
        "edge_fscore": float("-inf"),
        "vertex_fscore": float("-inf"),
        "face_cd": float("inf"),
        "edge_cd": float("inf"),
        "vertex_cd": float("inf"),
    }
    if result is not None:
        summary["face_fscore"] = float(result["face_fscore"])
        summary["edge_fscore"] = float(result["edge_fscore"])
        summary["vertex_fscore"] = float(result["vertex_fscore"])
        summary["face_cd"] = float(result["face_cd"])
        summary["edge_cd"] = float(result["edge_cd"])
        summary["vertex_cd"] = float(result["vertex_cd"])
    return summary


def is_better_result(candidate, incumbent):
    if incumbent is None:
        return True
    ordered_keys = [
        ("face_fscore", True),
        ("edge_fscore", True),
        ("vertex_fscore", True),
        ("face_cd", False),
        ("edge_cd", False),
        ("vertex_cd", False),
    ]
    for key, larger_is_better in ordered_keys:
        lhs = float(candidate[key])
        rhs = float(incumbent[key])
        if abs(lhs - rhs) <= ROTATION_COMPARE_EPS:
            continue
        if larger_is_better:
            return lhs > rhs
        return lhs < rhs
    return int(candidate["best_rotation_index"]) < int(incumbent["best_rotation_index"])


def evaluate_with_points(
    recon_face_points, recon_edge_points, recon_vertex_points, recon_face_edge, recon_edge_vertex,
    gt_face_points, gt_edge_points, gt_vertex_points, gt_face_edge, gt_edge_vertex
):
    face_acc_cd, face_com_cd, face_cd = get_chamfer(recon_face_points, gt_face_points)
    edge_acc_cd, edge_com_cd, edge_cd = get_chamfer(recon_edge_points, gt_edge_points)
    vertex_acc_cd, vertex_com_cd, vertex_cd = get_chamfer(recon_vertex_points, gt_vertex_points)

    id_recon_gt_face, id_gt_recon_face, cost_face = get_match_ids(recon_face_points, gt_face_points)
    id_recon_gt_edge, id_gt_recon_edge, cost_edge = get_match_ids(recon_edge_points, gt_edge_points)
    id_recon_gt_vertex, id_gt_recon_vertex, cost_vertices = get_match_ids(recon_vertex_points, gt_vertex_points)

    face_fscore, face_pre, face_rec = get_detection(id_recon_gt_face, id_gt_recon_face, cost_face)
    edge_fscore, edge_pre, edge_rec = get_detection(id_recon_gt_edge, id_gt_recon_edge, cost_edge)
    vertex_fscore, vertex_pre, vertex_rec = get_detection(id_recon_gt_vertex, id_gt_recon_vertex, cost_vertices)

    fe_fscore, fe_pre, fe_rec = get_topo_detection(recon_face_edge, gt_face_edge, id_recon_gt_face, id_recon_gt_edge)
    ev_fscore, ev_pre, ev_rec = get_topo_detection(recon_edge_vertex, gt_edge_vertex, id_recon_gt_edge,
                                                   id_recon_gt_vertex)

    return {
        "face_cd": face_cd,
        "edge_cd": edge_cd,
        "vertex_cd": vertex_cd,

        "face_fscore": face_fscore,
        "edge_fscore": edge_fscore,
        "vertex_fscore": vertex_fscore,
        "fe_fscore": fe_fscore,
        "ev_fscore": ev_fscore,

        "face_acc_cd": face_acc_cd,
        "edge_acc_cd": edge_acc_cd,
        "vertex_acc_cd": vertex_acc_cd,

        "face_com_cd": face_com_cd,
        "edge_com_cd": edge_com_cd,
        "vertex_com_cd": vertex_com_cd,

        "fe_pre": fe_pre,
        "ev_pre": ev_pre,
        "fe_rec": fe_rec,
        "ev_rec": ev_rec,

        "vertex_pre": vertex_pre,
        "edge_pre": edge_pre,
        "face_pre": face_pre,

        "vertex_rec": vertex_rec,
        "edge_rec": edge_rec,
        "face_rec": face_rec,

        "num_recon_face": len(recon_face_points),
        "num_gt_face": len(gt_face_points),
        "num_recon_edge": len(recon_edge_points),
        "num_gt_edge": len(gt_edge_points),
        "num_recon_vertex": len(recon_vertex_points),
        "num_gt_vertex": len(gt_vertex_points),
    }


def eval_one_with_try(eval_root, gt_root, folder_name, is_point2cad=False, is_complexgen=False, is_nvdnet=False, v_num_per_m=100):
    try:
        eval_one(eval_root, gt_root, folder_name, is_point2cad, is_complexgen, is_nvdnet, v_num_per_m)
    except Exception:
        error_message = traceback.format_exc()
        write_error_marker(eval_root, folder_name, error_message)
        print(f"Failed to evaluate {folder_name}\n{error_message}")


def eval_one(eval_root, gt_root, folder_name, is_point2cad=False, is_complexgen=False, is_nvdnet=False, v_num_per_m=100):
    if os.path.exists(eval_root / folder_name / 'error.txt'):
        os.remove(eval_root / folder_name / 'error.txt')
    if os.path.exists(eval_root / folder_name / 'eval.npz'):
        os.remove(eval_root / folder_name / 'eval.npz')

    assert [is_point2cad, is_complexgen, is_nvdnet].count(True) <= 1, \
        "Only one of [is_point2cad, is_complexgen, is_nvdnet] can be True"

    if is_point2cad or is_complexgen or is_nvdnet:
        raise NotImplementedError("new_eval_condition.py only supports the default eval_one else branch.")

    step_name = "recon_brep.step"
    recon_step_path = eval_root / folder_name / step_name
    gt_step_path = gt_root / folder_name / "normalized_shape.step"

    if not recon_step_path.exists():
        raise FileNotFoundError(f"Missing recon STEP: {recon_step_path}")
    if not gt_step_path.exists():
        raise FileNotFoundError(f"Missing GT STEP: {gt_step_path}")

    valid, recon_shape = check_step_valid_soild(recon_step_path, return_shape=True)
    if recon_shape is None :
        raise RuntimeError(f"Invalid recon STEP: {recon_step_path}")
    valid, gt_shape = check_step_valid_soild(gt_step_path, return_shape=True)
    if gt_shape is None:
        raise RuntimeError(f"Invalid GT STEP: {gt_step_path}")

    recon_faces, recon_face_points, recon_edges, recon_edge_points, recon_vertices, recon_vertex_points = get_data(recon_shape, v_num_per_m)
    recon_face_edge, recon_edge_vertex = get_topology(recon_faces, recon_edges, recon_vertices)

    gt_faces, gt_face_points, gt_edges, gt_edge_points, gt_vertices, gt_vertex_points = get_data(gt_shape, v_num_per_m)
    gt_face_edge, gt_edge_vertex = get_topology(gt_faces, gt_edges, gt_vertices)

    best_results = None
    rotation_scores = []
    num_valid_rotations = 0

    for rotation_index, rotation_matrix in enumerate(OCTAHEDRAL_ROTATIONS):
        try:
            rotated_gt_face_points = transform_point_sets(gt_face_points, rotation_matrix)
            rotated_gt_edge_points = transform_point_sets(gt_edge_points, rotation_matrix)
            rotated_gt_vertex_points = transform_points(gt_vertex_points, rotation_matrix)

            result = evaluate_with_points(
                recon_face_points, recon_edge_points, recon_vertex_points, recon_face_edge, recon_edge_vertex,
                rotated_gt_face_points, rotated_gt_edge_points, rotated_gt_vertex_points, gt_face_edge, gt_edge_vertex
            )
            if not metrics_are_finite(result):
                rotation_scores.append(build_rotation_summary(
                    rotation_index, rotation_matrix, result=None, error="NaN/Inf metrics"
                ))
                continue

            result["best_rotation_index"] = rotation_index
            result["best_rotation_matrix"] = rotation_matrix.astype(np.float32)
            rotation_scores.append(build_rotation_summary(rotation_index, rotation_matrix, result=result))
            num_valid_rotations += 1

            if is_better_result(result, best_results):
                best_results = result
        except Exception:
            rotation_scores.append(build_rotation_summary(
                rotation_index, rotation_matrix, result=None, error=traceback.format_exc()
            ))

    if best_results is None:
        raise RuntimeError(f"All 24 rotations failed for sample {folder_name}")

    best_results["rotation_scores"] = np.array(rotation_scores, dtype=object)
    best_results["rotation_search_status"] = "completed" if num_valid_rotations == 24 else "partial_failed"

    if not os.path.exists(eval_root / folder_name):
        os.makedirs(eval_root / folder_name)
    np.savez_compressed(eval_root / folder_name / 'eval.npz', results=best_results, allow_pickle=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate The Generated Brep')
    parser.add_argument('--eval_root', type=str, default=r"E:\data\img2brep\.43\2024_09_22_21_57_44_0921_pure_out2")
    parser.add_argument('--gt_root', type=str, default=r"E:\data\img2brep\deepcad_whole_v5\deepcad_whole_test_v5")
    parser.add_argument('--use_ray', action='store_true')
    parser.add_argument('--num_cpus', type=int, default=16)
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--list', type=str, default='')
    parser.add_argument('--from_scratch', action='store_true')
    parser.add_argument('--is_point2cad', action='store_true')
    parser.add_argument('--is_complexgen', action='store_true')
    parser.add_argument('--is_nvdnet', action='store_true')
    parser.add_argument('--only_valid', action='store_true')
    parser.add_argument('--ray_num_workers', type=int, default=0)
    parser.add_argument('--ray_cpus_per_worker', type=float, default=1.0)
    parser.add_argument('--ray_gpus_per_worker', type=float, default=0.0)
    parser.add_argument('--ray_timeout', type=int, default=60 * 3)
    args = parser.parse_args()
    eval_root = Path(args.eval_root)
    gt_root = Path(args.gt_root)
    is_use_ray = args.use_ray
    num_cpus = args.num_cpus
    listfile = args.list
    from_scratch = args.from_scratch
    is_point2cad = args.is_point2cad
    is_complexgen = args.is_complexgen
    is_nvdenet = args.is_nvdnet
    only_valid = args.only_valid
    ray_num_workers = args.ray_num_workers
    ray_cpus_per_worker = args.ray_cpus_per_worker
    ray_gpus_per_worker = args.ray_gpus_per_worker
    ray_timeout = args.ray_timeout

    if not os.path.exists(eval_root):
        raise ValueError(f"Data root path {eval_root} does not exist.")
    if not os.path.exists(gt_root):
        raise ValueError(f"Output root path {gt_root} does not exist.")

    assert [is_point2cad, is_complexgen, is_nvdenet].count(True) <= 1, \
        "Only one of [is_point2cad, is_complexgen, is_nvdenet] can be True"

    if args.prefix != '':
        eval_one_with_try(eval_root, gt_root, args.prefix, is_point2cad, is_complexgen, is_nvdenet)
        exit()

    all_folders = [folder for folder in os.listdir(eval_root) if os.path.isdir(eval_root / folder)]
    ori_length = len(all_folders)
    if listfile != '':
        valid_names = [item.strip() for item in open(listfile, 'r').readlines()]
        all_folders = valid_names
        all_folders.sort()
    print(f"Total {len(all_folders)}/{ori_length} folders to evaluate")

    if not from_scratch:
        print("Filtering the folders that have eval.npz")
        all_folders = [folder for folder in all_folders if not os.path.exists(eval_root / folder / 'eval.npz')]
        print(f"Total {len(all_folders)} folders to compute after caching")

    if not is_use_ray:
        for i in tqdm(range(len(all_folders))):
            print(f"Processing {all_folders[i]}")
            eval_one_with_try(eval_root, gt_root, all_folders[i],
                              is_point2cad, is_complexgen, is_nvdenet)
    else:
        effective_num_workers = ray_num_workers
        if effective_num_workers <= 0:
            if ray_gpus_per_worker > 0:
                effective_num_workers = max(1, num_cpus)
            else:
                effective_num_workers = max(1, int(num_cpus / max(ray_cpus_per_worker, 1)))
        print(
            f"Ray config: workers={effective_num_workers}, "
            f"cpus/worker={ray_cpus_per_worker}, gpus/worker={ray_gpus_per_worker}, timeout={ray_timeout}s"
        )
        ray_init_kwargs = dict(
            dashboard_host="0.0.0.0",
            dashboard_port=8080,
            num_cpus=num_cpus,
        )
        if ray_gpus_per_worker > 0:
            ray_init_kwargs["num_gpus"] = max(1, int(np.ceil(effective_num_workers * ray_gpus_per_worker)))
        ray.init(
            **ray_init_kwargs,
        )
        eval_one_remote = ray.remote(
            max_retries=0,
            num_cpus=ray_cpus_per_worker,
            num_gpus=ray_gpus_per_worker,
        )(eval_one_with_try)
        pending_tasks = {}
        timeout_cancel_list = []
        submitted = 0
        progress = tqdm(total=len(all_folders), desc="Ray Eval")
        try:
            while submitted < len(all_folders) or pending_tasks:
                while submitted < len(all_folders) and len(pending_tasks) < effective_num_workers:
                    folder_name = all_folders[submitted]
                    task = eval_one_remote.remote(
                        eval_root, gt_root, folder_name, is_point2cad, is_complexgen, is_nvdenet
                    )
                    pending_tasks[task] = {
                        "folder_name": folder_name,
                        "start_time": time.time(),
                    }
                    submitted += 1
                    progress.set_postfix_str(
                        f"submitted={submitted} running={len(pending_tasks)} done={progress.n}"
                    )

                expired = [
                    task for task, info in pending_tasks.items()
                    if time.time() - info["start_time"] > ray_timeout
                ]
                for task in expired:
                    timeout_cancel_list.append(pending_tasks[task]["folder_name"])
                    ray.cancel(task, force=True)
                    pending_tasks.pop(task)
                    progress.update(1)

                if not pending_tasks:
                    continue

                ready, _ = ray.wait(list(pending_tasks.keys()), num_returns=1, timeout=1)
                if not ready:
                    continue

                task = ready[0]
                pending_tasks.pop(task)
                try:
                    ray.get(task)
                except Exception:
                    pass
                progress.update(1)
                progress.set_postfix_str(
                    f"submitted={submitted} running={len(pending_tasks)} done={progress.n}"
                )
        finally:
            progress.close()
        print(f"Cancel for timeout: {timeout_cancel_list}")

    print("Computing statistics...")
    compute_statistics(eval_root, only_valid, listfile)
    print("Done")
