"""Real validation mini-batch checks; no optimizer step or quality claim."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.brepnet.dataset import AutoEncoder_joint_dataset
from src.brepnet.models.vae import AutoEncoder_intersection, _map_directed_targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--uv-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with initialize_config_dir(config_dir=str(ROOT / 'configs'), version_base=None):
        cfg = compose(config_name='train_vae_shared_curve_uv', overrides=[
            'model.init_shared_curve_checkpoint=null', 'dataset.edge_uv_root=null',
        ])
    cfg.model.init_shared_curve_checkpoint = str(args.checkpoint.resolve())
    cfg.dataset.raw_data_root = str(args.data_root.resolve())
    cfg.dataset.edge_uv_root = str(args.uv_root.resolve())
    sample_ids = ROOT / 'checkpoints/shared_curve_uv/validation/sample_ids.txt'
    for name in ('train_dataset', 'val_dataset', 'test_dataset'):
        cfg.dataset[name] = str(sample_ids)
    dataset = AutoEncoder_joint_dataset('validation', cfg.dataset)
    samples = [dataset[i] for i in range(len(dataset.data_folders))]
    batch = dataset.collate_fn(samples[:2])

    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to(args.device)
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, list):
            return [move(item) for item in value]
        return value

    batch = move(batch)
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    previous_cfg = {**model_cfg, 'init_shared_curve_checkpoint': None,
                    'shared_curve': {'enabled': False}, 'checkpoint': str(args.checkpoint)}
    previous = AutoEncoder_intersection(previous_cfg).to(args.device).eval()
    model = AutoEncoder_intersection(model_cfg).to(args.device).eval()
    mapped = set(model.initialization_report['mapped_uv_keys'])
    retained = [key for key in previous.state_dict() if key not in mapped]
    assert all(torch.equal(previous.state_dict()[key], model.state_dict()[key]) for key in retained)
    assert len(model.decoder.inter.layers) == 8
    assert len(model.decoder.uv_boundary.layers) == 2
    with torch.no_grad():
        old_encoding, encoding = previous.encode(batch), model.encode(batch)
        repeated_encoding = previous.encode(batch)
        encoder_difference = float((old_encoding['face_features'] - encoding['face_features']).abs().max())
        encoder_repeat_difference = float((old_encoding['face_features'] - repeated_encoding['face_features']).abs().max())
        # CUDA graph scatter/reductions may differ by a few ulps even on repeat.
        # Retained weights are checked bitwise above; use one common z below.
        torch.testing.assert_close(old_encoding['face_features'], encoding['face_features'], rtol=1e-5, atol=1e-5)
        z = model.sample(encoding['face_features'], True)[0]
        old, new = previous.decode_state(z, batch['face_counts']), model.decode_state(z, batch['face_counts'])
        torch.testing.assert_close(old.coarse.edge_geometry.points, new.coarse.edge_geometry.points, rtol=0, atol=0)
        face_difference = float((old.refined.face_geometry.points - new.refined.face_geometry.points).abs().max())
        torch.testing.assert_close(old.refined.face_geometry.points, new.refined.face_geometry.points, rtol=1e-5, atol=1e-6)
        canonical = new.pairs.indices[:, 0] < new.pairs.indices[:, 1]
        canonical_difference = float((old.refined.edge_geometry.points[canonical]
                                      - new.refined.edge_geometry.points[canonical]).abs().max())
        torch.testing.assert_close(old.refined.edge_geometry.points[canonical],
                                   new.refined.edge_geometry.points[canonical], rtol=1e-4, atol=1e-5)
        targets = _map_directed_targets(new, batch)
        gt = batch['edge_points'][targets.edge_ids, :, :3]
        def edge_cd(output):
            pred = output.refined.edge_geometry.points[targets.candidate_rows]
            distances = torch.cdist(pred.float(), gt.float()).square()
            return float((distances.min(1).values.mean() + distances.min(2).values.mean()).cpu())
        initial_cd = {'previous': edge_cd(old), 'shared_curve_initial': edge_cd(new)}
        exported = model.export_items(new, z)
        for item in exported:
            assert item['pred_edge_uv'].shape == (len(item['pred_edge']), 2, 16, 2)
    counts = {'previous': sum(p.numel() for p in previous.parameters()),
              'candidate': sum(p.numel() for p in model.parameters())}
    del previous, old, new
    torch.cuda.empty_cache()
    records = []
    for precision in ('float32', 'bf16-mixed'):
        model.train()
        model.zero_grad(set_to_none=True)
        start = time.monotonic()
        torch.cuda.reset_peak_memory_stats(args.device)
        context = torch.autocast('cuda', dtype=torch.bfloat16) if precision == 'bf16-mixed' else nullcontext()
        with context:
            losses, _ = model(batch)
        assert all(torch.isfinite(value).all() for value in losses.values())
        losses['total_loss'].backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert all(torch.isfinite(value).all() for value in gradients)
        assert all(p.grad is None for p in model.encoder.parameters())
        selected = {
            'fusion_output': model.decoder.shared_curve.residual[-1].weight,
            'curve_embedding': model.decoder.uv_boundary.curve_embedding[0].weight,
            'uv_context': model.decoder.uv_boundary.context[0].weight,
            'refiner': model.decoder.refiner.edge_delta.weight,
        }
        grad_norms = {key: float(p.grad.abs().sum()) for key, p in selected.items()}
        assert all(value > 0 for value in grad_norms.values())
        records.append({'precision': precision, 'seconds': time.monotonic()-start,
                        'losses': {key: float(value.detach()) for key, value in losses.items()},
                        'nonempty_gradients': len(gradients), 'selected_gradient_l1': grad_norms,
                        'peak_memory_bytes': torch.cuda.max_memory_allocated(args.device)})
    model.eval()
    model.zero_grad(set_to_none=True)
    prediction = model.decode_state(z, batch['face_counts'])
    geometry = prediction.refined.edge_geometry
    reverse = prediction.pairs.reverse
    torch.testing.assert_close(geometry.points, geometry.points[reverse].flip(1), rtol=0, atol=0)
    torch.testing.assert_close(prediction.edge_uv, prediction.edge_uv[reverse].flip((1, 2)), rtol=0, atol=0)
    geometry.points.retain_grad()
    targets = _map_directed_targets(prediction, batch)
    aligned = model.criterion._shared_curve_batch(prediction, batch, targets)
    uv_loss = model.criterion._uv_boundary(prediction, aligned, targets)['uv']
    uv_loss.backward()
    assert geometry.points.grad is not None and geometry.points.grad.abs().sum() > 0
    report = {
        'purpose': 'implementation_only_no_optimizer_steps',
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'git_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()),
        'checkpoint': str(args.checkpoint), 'initialization': model.initialization_report,
        'parameters': {**counts, 'added': counts['candidate']-counts['previous']},
        'validation_ids': dataset.data_folders, 'loaded_samples': len(samples),
        'face_counts': batch['face_counts'].tolist(), 'connection_rows': len(batch['edge_face_connectivity']),
        'initial_edge_cd_squared_two_objects_not_quality_evaluation': initial_cd,
        'checks': records, 'curve_xyz_gradient_from_uv_loss_l1': float(geometry.points.grad.abs().sum()),
        'coarse_geometry_bitwise_unchanged': True,
        'refined_face_max_abs_difference': face_difference,
        'canonical_curve_initial_max_abs_difference': canonical_difference,
        'encoder_max_abs_difference': encoder_difference,
        'same_encoder_repeat_max_abs_difference': encoder_repeat_difference,
        'reciprocal_xyz_and_uv_exact': True, 'retained_weights_exact': len(retained),
        'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(args.device),
    }
    files = ['src/brepnet/models/vae/checkpoints.py', 'src/brepnet/models/vae/intersection.py',
             'src/brepnet/models/vae/decoders/intersection.py', 'src/brepnet/models/vae/decoders/primitives.py',
             'src/brepnet/models/vae/losses/intersection.py', 'configs/train_vae_shared_curve_uv.yaml']
    report['source_sha256'] = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps({key: report[key] for key in ('parameters', 'face_counts', 'connection_rows',
        'initial_edge_cd_squared_two_objects_not_quality_evaluation', 'curve_xyz_gradient_from_uv_loss_l1')}, indent=2))


if __name__ == '__main__':
    main()
