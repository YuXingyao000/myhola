"""Native next/pcurve VAE: full HoLa backbone, explicit loop assembly.

GT next conditions geometry only during supervised training. Latent-only
inference predicts candidates and next, and rejects unmatched loops explicitly.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GATv2Conv
from torch_scatter import scatter_mean

from .legacy import AutoEncoder_light
from .pcurve_relations import (
    NativeRelationEncoder,
    NextDecoder,
    PcurveDecoder,
    inverse_next,
    relation_loss,
    assign_next,
)
from .pcurve_geometry import SplineGeometry


class AutoEncoder_pcurve(AutoEncoder_light):
    def __init__(self, cfg):
        if cfg.get("init_hola_checkpoint") and cfg.get("checkpoint"):
            raise ValueError("Choose HoLa initialization or a complete pcurve checkpoint")
        # Load the published model strictly before removing its unused output heads.
        super().__init__({**cfg, "checkpoint": cfg.get("init_hola_checkpoint")})
        del self.edge_points_decoder
        del self.edge_center_scale_decoder
        settings = cfg["pcurve"]
        self.max_slots = int(settings["max_slots"])
        self.pair_chunk_size = int(settings["pair_chunk_size"])
        self.loss_weights = dict(settings["loss_weights"])
        self.drop_probability = float(settings["drop_probability"])
        self.false_ratio = float(settings["false_ratio"])
        shape = dict(width=int(settings["width"]), depth=int(settings["depth"]))
        self.relation_encoder = NativeRelationEncoder(**shape)
        self.next_decoder = NextDecoder(**shape)
        self.pcurve_decoder = PcurveDecoder(**shape, controls=int(settings["controls"]))
        self.splines = SplineGeometry(controls=int(settings["controls"]))
        self.slot_embedding = nn.Embedding(self.max_slots, 64)
        self.seam_side = nn.Embedding(2, 64)
        nn.init.normal_(self.slot_embedding.weight, std=0.02)
        nn.init.normal_(self.seam_side.weight, std=0.02)
        with torch.no_grad():
            self.slot_embedding.weight[0].zero_()
        self._load_checkpoint(cfg.get("checkpoint"))
        self._apply_runtime_mode()

    def encode(self, data, v_test=False):
        face = self.face_coords(data["face_points"].permute(0, 3, 1, 2).contiguous())
        edge = self.edge_coords(data["edge_points"].permute(0, 2, 1).contiguous())
        face = self.relation_encoder(face, edge, data["pcurve_uv"], data["owner"], data["mate"], data["next"])
        enriched = face + self.face_attn_proj_out(self.face_attn(self.face_attn_proj_in(face), data["attn_mask"]))
        x = enriched
        adjacency = data["edge_face_connectivity"][:, 1:].T
        for layer in self.graph_face_edge:
            x = layer(x, adjacency, edge) + x if isinstance(layer, GATv2Conv) else layer(x)
        face = x + enriched
        group = torch.arange(len(data["face_counts"]), device=face.device).repeat_interleave(data["face_counts"])
        global_feature = self.global_feature1(scatter_mean(face, group, dim=0)).repeat_interleave(
            data["face_counts"], dim=0
        )
        return {"face_features": face + self.global_feature2(torch.cat([face, global_feature], 1))}

    def _faces(self, z, mask=None):
        features = self.face_attn_proj_out2(self.face_attn2(self.face_attn_proj_in2(z), mask))
        with torch.autocast(features.device.type, enabled=False):
            local = self.face_points_decoder(features.float())
            bbox = self.face_center_scale_decoder(features.float())
        xyz = local[..., :3].float() * bbox[:, None, None, 3:].float() + bbox[:, None, None, :3].float()
        return features, local, bbox, xyz

    def _candidates(self, face, counts):
        groups = []
        offset = 0
        for count in counts:
            groups.append(torch.triu_indices(count, count, device=face.device).T + offset)
            offset += count
        physical_pairs = torch.cat(groups)
        directed = physical_pairs[:, None, :].expand(-1, 2, -1).clone()
        directed[:, 1] = directed[:, 1].flip(-1)
        tokens = face[directed.flatten(0, 1)]
        parts = []
        for chunk in tokens.split(self.pair_chunk_size):
            if self.training and torch.is_grad_enabled():
                parts.append(checkpoint(self.inter, chunk, use_reentrant=False))
            else:
                parts.append(self.inter(chunk))
        h = torch.cat(parts).reshape(-1, 2, 64)
        h = h[:, None, :, :] + self.slot_embedding.weight[None, :, None, :]
        seam = (physical_pairs[:, 0] == physical_pairs[:, 1])[:, None, None, None]
        h = h + seam * self.seam_side.weight[None, None, :, :]
        h = h.flatten(0, 2)
        owner = directed[:, None, :, 0].expand(-1, self.max_slots, -1).flatten()
        mate = torch.arange(len(h), device=h.device) ^ 1
        # A physical candidate has two uses, including a self-owner seam.
        # One existence decision activates/deactivates both uses together.
        logits = self.classifier(h).flatten().reshape(-1, 2).mean(1)
        return h, owner, mate, logits, physical_pairs

    def _positive_ids(self, data, physical_pairs):
        if int(data["slot"].max()) >= self.max_slots:
            raise ValueError("Configured candidate capacity does not contain all native GT edges")
        lookup = data["owner"].new_full((len(data["face_points"]), len(data["face_points"])), -1)
        lookup[physical_pairs[:, 0], physical_pairs[:, 1]] = torch.arange(len(physical_pairs), device=lookup.device)
        other = data["owner"][data["mate"]]
        a, b = torch.minimum(data["owner"], other), torch.maximum(data["owner"], other)
        return (lookup[a, b] * self.max_slots + data["slot"]) * 2 + data["side"]

    def _relation_objective(self, h, face, owner, mate, positive, data, negative_physical):
        scores, end_null, start_null = self.next_decoder(h[positive], face, data["owner"], data["mate"])
        previous = inverse_next(data["next"])
        clean = relation_loss(scores, end_null, start_null, data["next"], previous)
        accuracy = (torch.cat([scores, end_null[:, None]], 1).argmax(1) == data["next"]).float().mean()
        if not self.training:
            return clean, accuracy
        true_physical = (positive // 2).unique()
        keep = torch.rand(len(true_physical), device=h.device) >= self.drop_probability
        keep[0] = True
        false_count = min(len(negative_physical), max(1, int(len(true_physical) * self.false_ratio)))
        false = negative_physical[torch.randperm(len(negative_physical), device=h.device)[:false_count]]
        selected_physical = torch.cat([true_physical[keep], false])
        selected = (selected_physical[:, None] * 2 + torch.arange(2, device=h.device)).flatten()
        old_gt = torch.full((len(h),), -1, dtype=torch.long, device=h.device)
        old_gt[positive] = torch.arange(len(positive), device=h.device)
        selected_gt = old_gt[selected]
        remap = torch.full_like(old_gt, -1)
        remap[selected] = torch.arange(len(selected), device=h.device)
        known = selected_gt >= 0
        target_next = torch.where(known, remap[positive[data["next"][selected_gt.clamp_min(0)]]], -1)
        target_previous = torch.where(known, remap[positive[previous[selected_gt.clamp_min(0)]]], -1)
        scores, en, sn = self.next_decoder(
            h[selected], face, owner[selected], torch.arange(len(selected), device=h.device) ^ 1
        )
        corrupted = relation_loss(scores, en, sn, target_next, target_previous)
        return 0.5 * (clean + corrupted), accuracy

    def forward(self, data, v_test=False):
        encoding = self.encode(data)
        z, kl, stats = self.sample(encoding["face_features"], v_is_test=not self.training)
        face, local, bbox, grid = self._faces(z, data["attn_mask"])
        h, owner, mate, logits, pairs = self._candidates(face, data["face_counts"].tolist())
        positive = self._positive_ids(data, pairs)
        target = torch.zeros_like(logits)
        positive_physical = (positive // 2).unique()
        target[positive_physical] = 1
        negatives = torch.where(target == 0)[0]
        if self.training:
            negatives_for_loss = negatives[torch.randperm(len(negatives), device=h.device)[: len(positive_physical)]]
        else:
            negatives_for_loss = negatives
        classification_ids = torch.cat([positive_physical, negatives_for_loss])
        if not self.training and len(negatives):
            classification = 0.5 * (
                F.binary_cross_entropy_with_logits(logits[positive_physical], target[positive_physical])
                + F.binary_cross_entropy_with_logits(logits[negatives], target[negatives])
            )
        else:
            classification = F.binary_cross_entropy_with_logits(logits[classification_ids], target[classification_ids])
        relation, accuracy = self._relation_objective(h, face, owner, mate, positive, data, negatives)
        poles = self.pcurve_decoder(h[positive], face, data["owner"], data["mate"], data["next"])
        uv, xyz = self.splines(grid, data["owner"], poles)
        reference = data["edge_points"][..., :3]
        compatibility = (xyz - xyz[data["mate"]].flip(1)).abs().mean()
        losses = {
            "face_local": F.l1_loss(local, data["face_norm"]),
            "face_bbox": F.l1_loss(bbox, data["face_bbox"]),
            "classification": classification,
            "next_relation": relation,
            "pcurve_uv": F.l1_loss(uv, data["pcurve_uv"]),
            "edge_xyz": F.l1_loss(xyz, reference),
            "mate_compatibility": compatibility,
            "kl_loss": kl,
        }
        if self.loss_weights.get("pcurve_gt_surface", 0.0) > 0:
            # Only the predicted UV can move this trace; GT geometry is fixed.
            gt_trace = self.splines.evaluate(data["face_points"][..., :3].detach(), data["owner"], uv)
            losses["pcurve_gt_surface"] = F.l1_loss(gt_trace, reference)
        weighted = {f"weighted_{k}": v * self.loss_weights.get(k, 1.0) for k, v in losses.items()}
        total = sum(weighted.values())
        with torch.no_grad():
            prediction = logits > 0
            correct = (prediction & target.bool()).sum()
            start = xyz[:, 0]
            centers = scatter_mean(start, data["vertex"], dim=0)
            delta = xyz - reference
            metrics = {
                "next_accuracy_gt_candidates": accuracy,
                "precision": correct / prediction.sum().clamp_min(1),
                "recall": correct / target.sum().clamp_min(1),
                "endpoint_mae": delta[:, [0, -1]].abs().mean(),
                "middle_mae": delta[:, 1:-1].abs().mean(),
                "endpoint_l2": torch.linalg.vector_norm(delta[:, [0, -1]], dim=-1).mean(),
                "vertex_rms": (start - centers[data["vertex"]]).square().sum(-1).mean().sqrt(),
                "mate_endpoint_gap": torch.linalg.vector_norm(
                    xyz[:, [0, -1]] - xyz[data["mate"]][:, [-1, 0]], dim=-1
                ).mean(),
                "same_face_uv_gap": torch.linalg.vector_norm(uv[:, -1] - uv[data["next"], 0], dim=-1).max(),
                "uv_out_of_chart": ((uv < 0) | (uv > 1)).any(-1).float().mean(),
            }
        output = {**{k: v.detach() for k, v in losses.items()}, **weighted, **metrics, "total_loss": total}
        recon = {}
        if v_test:
            items = []
            for latent in z.split(data["face_counts"].tolist()):
                items.append(self.decode_latents(latent))
            for item, latent_stats in zip(items, stats.split(data["face_counts"].tolist())):
                item["cached_latent_stats"] = latent_stats.detach()
            recon = {"items": items}
        return output, recon

    @torch.no_grad()
    def decode_latents(self, z):
        face, local, bbox, grid = self._faces(z)
        h, owner, mate, logits, pairs = self._candidates(face, [len(z)])
        physical = torch.where(logits > 0)[0]
        selected = (physical[:, None] * 2 + torch.arange(2, device=z.device)).flatten()
        owner = owner[selected]
        mate = torch.arange(len(selected), device=z.device) ^ 1
        result = {
            "pred_face": grid,
            "pred_edge_confidence": logits[physical].sigmoid().repeat_interleave(2),
            "representation": "native_pcurve_v1",
            "pred_owner": owner,
            "pred_mate": mate,
        }
        adjacency = torch.zeros(len(face), len(face), device=z.device, dtype=torch.bool)
        adjacency[owner, owner[mate]] = True
        result["pred_face_adj"] = adjacency
        if len(selected) == 0:
            result.update(
                pred_edge=grid.new_empty((0, 16, 3)),
                pred_next=owner,
                pred_edge_face_connectivity=owner.new_empty((0, 3)),
                topology_complete=False,
            )
            return result
        scores, en, sn = self.next_decoder(h[selected], face, owner, mate)
        nxt = assign_next(scores, en, sn, owner)
        result["pred_next"] = nxt
        result["topology_complete"] = bool((nxt >= 0).all())
        result["pred_edge_face_connectivity"] = torch.stack(
            [torch.arange(len(owner), device=z.device), owner, owner[mate]], 1
        )
        if not result["topology_complete"]:
            result["pred_candidate_edge_face_connectivity"] = result["pred_edge_face_connectivity"]
            result["pred_edge_face_connectivity"] = owner.new_empty((0, 3))
            result["pred_edge"] = grid.new_empty((0, 16, 3))
            return result
        poles = self.pcurve_decoder(h[selected], face, owner, mate, nxt)
        uv, xyz = self.splines(grid, owner, poles)
        result.update(pred_pcurve_controls=poles, pred_pcurve_uv=uv, pred_edge=xyz)
        return result
