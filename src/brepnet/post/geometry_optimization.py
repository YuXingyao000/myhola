import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from src.brepnet.post.distance import ChamferDistance


def apply_transform_batch(tensor, transform):
    src_shape = tensor.shape
    if len(src_shape) > 3:
        tensor = tensor.reshape(tensor.shape[0], -1, 3)
    scales = transform[:, :3].view(-1, 1, 3)
    offsets = transform[:, 3:].view(-1, 1, 3)
    centers = tensor.mean(dim=1, keepdim=True)
    scaled_tensor = (tensor - centers) * scales + centers + offsets
    if len(src_shape) > 3:
        scaled_tensor = scaled_tensor.reshape(*src_shape)
    return scaled_tensor


def apply_offset_batch(tensor, offsets):
    src_shape = tensor.shape
    if len(src_shape) > 3:
        tensor = tensor.reshape(tensor.shape[0], -1, 3)
    offsets = offsets.view(-1, 1, 3)
    scaled_tensor = tensor + offsets
    if len(src_shape) > 3:
        scaled_tensor = scaled_tensor.reshape(*src_shape)
    return scaled_tensor


def optimize(
        v_interpolation_face, recon_edge_points, recon_face_points,
        edge_face_connectivity, is_end_point, pair1,
        face_edge_adj, v_islog=True, v_max_iter=1000, use_cuda=True):
    device = torch.device('cuda') if torch.cuda.is_available() and use_cuda else torch.device('cpu')
    interpolation_face = []
    for item in v_interpolation_face:
        interpolation_face.append(item.to(device))
    padded_points = pad_sequence(interpolation_face, batch_first=True, padding_value=10)
    edge_points = torch.from_numpy(recon_edge_points.copy()).to(device)
    face_points = torch.from_numpy(recon_face_points.copy()).to(device)
    edge_face_connectivity = torch.from_numpy(edge_face_connectivity.copy()).to(device)
    if pair1 is not None:
        pair1 = torch.from_numpy(pair1.copy()).to(device)
    idx = np.zeros_like(is_end_point).astype(np.int64)
    idx[is_end_point] = 15
    idx = torch.from_numpy(idx).to(device)

    edge_st = nn.Parameter(
            torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.float32, device=device).unsqueeze(0).repeat(edge_points.shape[0], 1))
    face_t = nn.Parameter(
            torch.tensor([0, 0, 0], dtype=torch.float32, device=device).unsqueeze(0).repeat(face_points.shape[0], 1))

    edge_src_st = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.float32, device=device).unsqueeze(0).repeat(edge_points.shape[0], 1)
    face_src_t = torch.tensor([0, 0, 0], dtype=torch.float32, device=device).unsqueeze(0).repeat(face_points.shape[0], 1)

    edge_st.requires_grad = True
    face_t.requires_grad = True
    optimizer = torch.optim.AdamW([edge_st, face_t], lr=1e-3, betas=(0.95, 0.999), eps=1e-08)

    init_max_iter = v_max_iter
    final_max_iter = init_max_iter * 3
    init_loss = float('inf')
    best_loss = float('inf')
    is_optimization_diverged = False
    if v_islog:
        pbar = tqdm(total=v_max_iter, desc='Geom Optimization', unit='iter')

    chamferdist = ChamferDistance()
    iter = 0
    while iter < v_max_iter:
        transformed_edges = apply_transform_batch(edge_points, edge_st)
        transformed_padded_points = apply_offset_batch(padded_points, face_t)
        dis_matrix1 = chamferdist(
                transformed_edges[edge_face_connectivity[:, 0]],
                transformed_padded_points[edge_face_connectivity[:, 1]],
                batch_reduction=None, point_reduction="sum", bidirectional=False)
        dis_matrix2 = chamferdist(
                transformed_edges[edge_face_connectivity[:, 0]],
                transformed_padded_points[edge_face_connectivity[:, 2]],
                batch_reduction=None, point_reduction="sum", bidirectional=False)
        adj_distance_loss = (dis_matrix1 + dis_matrix2 + 1e-2 * (dis_matrix1 - dis_matrix2).abs()).mean()

        if pair1 is None:
            corners_loss = torch.zeros_like(adj_distance_loss)
        else:
            corners = transformed_edges[pair1, idx]
            corners_loss = torch.linalg.norm(corners[:, 0] - corners[:, 1], dim=-1) + \
                           torch.linalg.norm(corners[:, 1] - corners[:, 2], dim=-1) + \
                           torch.linalg.norm(corners[:, 0] - corners[:, 2], dim=-1)
            corners_loss = (corners_loss / 3).mean()

        if len(face_edge_adj) == 0:
            wire_connected_loss = torch.zeros_like(adj_distance_loss)
        else:
            max_length = max(len(face_edge_idx) for face_edge_idx in face_edge_adj)
            all_endpoints = []
            for face_edge_idx in face_edge_adj:
                if len(face_edge_idx) == 0:
                    continue
                face_edge_endpoint = torch.cat((transformed_edges[face_edge_idx, 0, :],
                                                transformed_edges[face_edge_idx, -1, :]), dim=0)
                padded_endpoints = torch.nn.functional.pad(face_edge_endpoint, (0, 0, 0, max_length * 2 - face_edge_endpoint.shape[0]),
                                                           value=1e6)
                all_endpoints.append(padded_endpoints)

            batch_endpoints = torch.stack(all_endpoints)
            dist_matrix = torch.cdist(batch_endpoints, batch_endpoints)
            dist_matrix = dist_matrix + torch.eye(dist_matrix.shape[-1], device=device).unsqueeze(0) * 1e6
            connected_loss = dist_matrix.min(dim=-1)[0]
            wire_connected_loss = connected_loss[connected_loss < 100].mean()

        loss = (adj_distance_loss + corners_loss + wire_connected_loss +
                1e-8 * (edge_st - edge_src_st).norm(p=1, dim=1).sum() + 1e-8 * (face_t - face_src_t).norm(p=1, dim=1).sum())

        if iter == 0:
            init_loss = loss.item()
        if loss.item() < best_loss:
            best_loss = loss.item()

        if iter > 30 and best_loss > init_loss:
            is_optimization_diverged = True
            print(f'Optimization diverged, best loss: {best_loss}, init loss: {init_loss}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iter == v_max_iter - 2 and corners_loss.item() > 0.001:
            if v_max_iter + init_max_iter > final_max_iter:
                v_max_iter = final_max_iter
            else:
                v_max_iter += init_max_iter
            if v_islog:
                pbar.total = v_max_iter
        if v_islog:
            pbar.set_postfix(loss=loss.item(),
                             adj=adj_distance_loss.cpu().item(),
                             corner=corners_loss.cpu().item(),
                             connect=wire_connected_loss.cpu().item())
            pbar.update(1)
        iter += 1

    if v_islog:
        print('Optimization finished!')
        pbar.close()

    if is_optimization_diverged:
        return recon_face_points, recon_edge_points

    transformed_edges = apply_transform_batch(edge_points, edge_st).detach().cpu().numpy()
    transformed_faces = apply_offset_batch(face_points, face_t).detach().cpu().numpy()
    return transformed_faces, transformed_edges


def optimize_ray(interpolation_face, recon_edge_points,
                 edge_face_connectivity, is_end_point, pair1, face_edge_adj, v_max_iter=1000):
    return optimize(interpolation_face, recon_edge_points,
                    edge_face_connectivity, is_end_point, pair1, face_edge_adj, v_islog=False, v_max_iter=v_max_iter)
