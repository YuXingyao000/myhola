"""CPU 审核固定十件的实际分层产物；未增强和缺失项保留在分母中。"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, '/mnt/d/model/Flux1_ControlNet_Depth/tools/python')
import OpenEXR


LAYERS = ('with_part', 'empty', 'foreground', 'interaction', 'shadow_only')


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def exr_channels(path):
    with OpenEXR.File(str(path), separate_channels=True) as image:
        return {name: channel.pixels.copy() for name, channel in image.channels().items()}


def rgba(channels):
    return np.stack([channels[name] for name in 'RGBA'], axis=-1).astype(np.float64)


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return dict(count=0)
    return dict(count=int(values.size), mean=float(values.mean()), p95=float(np.quantile(values, .95)),
                p99=float(np.quantile(values, .99)), maximum=float(values.max()))


def difference(left, right, region=None):
    delta = np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))
    if region is not None:
        delta = delta[region]
    result = stats(delta)
    result['changed_values'] = int(np.count_nonzero(delta))
    return result


def image_array(path):
    with Image.open(path) as image:
        return np.asarray(image).copy()


def load_composer(output):
    spec = importlib.util.spec_from_file_location('frozen_compose_for_audit', output / 'compose_layers.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decompose(images):
    original, foreground, interaction = [images[name] for name in ('with_part', 'foreground', 'interaction')]
    alpha = foreground[..., 3]
    predicted = foreground[..., :3] + (1 - alpha[..., None]) * interaction[..., :3]
    regions = dict(all=np.ones(alpha.shape, bool), opaque=alpha >= 1 - 1e-6,
                   partial=(alpha > 1e-6) & (alpha < 1 - 1e-6), transparent=alpha <= 1e-6)
    return dict(formula='I_rgb ≈ F_premultiplied_rgb + (1 − F_alpha) G_rgb',
                unit='scene-linear radiance; diagnostic, no identity hard gate',
                regions={name: dict(pixels=int(region.sum()), **difference(original[..., :3], predicted, region))
                         for name, region in regions.items()},
                erroneous_double_alpha_mae=float(np.abs(original[..., :3] -
                    (alpha[..., None] * foreground[..., :3] + (1-alpha[..., None]) * interaction[..., :3])).mean()))


def composite_audit(directory, png, mask, composer):
    if not (directory / 'enhanced_empty.png').exists():
        return dict(status='not_enhanced', note='Only rendered passes audited; no final output yet.')
    if not (directory / 'final.png').exists():
        return dict(status='not_composited')
    enhanced = Image.open(directory / 'enhanced_empty.png').convert('RGB')
    main, reference, metrics, diagnostics = composer.compose_layers(
        Image.fromarray(png['foreground']), Image.fromarray(png['empty']),
        Image.fromarray(png['interaction']), Image.fromarray(png['shadow_only']), enhanced)
    final = image_array(directory / 'final.png')
    alpha = png['foreground'][..., 3] / 255.
    background = np.asarray(composer.encode(diagnostics['background']))
    ring = (~mask) & (distance_transform_edt(~mask) <= 2)
    clear_ring = ring & (alpha == 0)
    old_path = directory / 'old_hard_ring.png'
    checks = dict(final_equals_recomputed=np.array_equal(final, np.asarray(main)),
                  opaque_equals_foreground=np.array_equal(final[alpha == 1], png['foreground'][..., :3][alpha == 1]),
                  transparent_equals_new_background=np.array_equal(final[alpha == 0], background[alpha == 0]))
    outputs = dict(final=difference(final, np.asarray(main)))
    for name, expected in [('rgb_transfer_reference.png', np.asarray(reference)),
                           ('alpha.png', png['foreground'][..., 3]),
                           ('background_with_interactions.png', background)]:
        path = directory / name
        checks[name + '_equals_recomputed'] = path.exists() and np.array_equal(image_array(path), expected)
        outputs[name] = difference(image_array(path), expected) if path.exists() else dict(status='missing')
    lighting_path = directory / 'lighting.npz'
    if lighting_path.exists():
        with np.load(lighting_path) as saved:
            for name in ('shadow_gain', 'indirect_gain', 'total_gain'):
                checks[name + '_equals_recomputed_float32'] = np.array_equal(saved[name], diagnostics[name].astype(np.float32))
    else:
        checks['lighting.npz_exists'] = False
    ring_report = dict(old_binary_mask_outer_2px_pixels=int(ring.sum()),
                       alpha_zero_in_old_ring_pixels=int(clear_ring.sum()),
                       alpha_zero_old_ring_equals_new_background=bool(np.array_equal(final[clear_ring], background[clear_ring])),
                       alpha_zero_old_ring_changed_from_original_pixels=int(np.any(final[clear_ring] != png['with_part'][clear_ring], axis=-1).sum()),
                       note='Exact re-composition proves no forced old ring; coincident RGB values alone prove neither copying nor its absence.')
    if old_path.exists():
        old = image_array(old_path)
        ring_report['saved_old_ring_equals_original'] = bool(np.array_equal(old[ring], png['with_part'][ring]))
    return dict(status='pass' if all(checks.values()) else 'failed', checks=checks, output_differences=outputs,
                metrics_recomputed=metrics, old_ring_support=ring_report,
                note='Display-linear approximation after baked AgX; neutral gain deliberately does not enforce RGB identity.')


def audit_case(output, pair, composer, request_hash, input_hashes):
    directory = output / pair['model_id']
    metadata_path = directory / 'metadata.json'
    if not metadata_path.exists():
        return dict(status='missing', missing=['metadata.json'])
    metadata = read_json(metadata_path)
    required = [Path(metadata.get(layer, {}).get(kind, directory / layer / filename))
                for layer in LAYERS for kind, filename in
                [('rgb', 'rgb.png'), ('mask', 'mask.png'), ('depth', 'depth.exr'), ('radiance', 'radiance.exr')]]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        return dict(status='render_incomplete', renderer_status=metadata['status'], missing=missing)
    png, masks, linear, noisy, depth, inventory = {}, {}, {}, {}, {}, {}
    for layer in LAYERS:
        paths = metadata[layer]
        png[layer] = image_array(paths['rgb'])
        masks[layer] = image_array(paths['mask'])
        channels = exr_channels(paths['radiance'])
        linear[layer] = rgba(channels)
        depth[layer] = exr_channels(paths['depth'])['V']
        record = dict(png_shape=list(png[layer].shape), png_dtype=str(png[layer].dtype),
                      mask_unique=np.unique(masks[layer]).tolist(), mask_nonzero_pixels=int(np.count_nonzero(masks[layer])),
                      exr_channels=list(channels), exr_shape=list(linear[layer].shape),
                      exr_finite=bool(np.isfinite(linear[layer]).all()),
                      exr_rgb_min=float(linear[layer][..., :3].min()), exr_rgb_max=float(linear[layer][..., :3].max()),
                      exr_alpha_min=float(linear[layer][..., 3].min()), exr_alpha_max=float(linear[layer][..., 3].max()),
                      depth_finite=bool(np.isfinite(depth[layer]).all()))
        noisy_path = Path(paths['noisy']) if paths.get('noisy') else None
        if noisy_path is not None and noisy_path.exists():
            noisy[layer] = rgba(exr_channels(noisy_path))
            record['noisy_finite'] = bool(np.isfinite(noisy[layer]).all())
        else:
            record['noisy_status'] = 'unavailable'
        inventory[layer] = record
    alpha = png['foreground'][..., 3] / 255.
    exr_alpha = linear['foreground'][..., 3]
    mask = masks['with_part'] > 0
    support = alpha > 0
    source_mask = image_array(pair['inputs']['mask']) > 0
    outside = support & ~mask
    uncovered = mask & ~support
    alpha_report = dict(partial_pixels=int(((alpha > 0) & (alpha < 1)).sum()), opaque_pixels=int((alpha == 1).sum()),
                        nonzero_pixels=int(support.sum()), png_vs_exr=difference(alpha, exr_alpha),
                        png_vs_rounded_exr_differing_pixels=int(np.count_nonzero(png['foreground'][..., 3] !=
                            np.rint(np.clip(exr_alpha, 0, 1) * 255).astype(np.uint8))),
                        support_outside_binary_mask_pixels=int(outside.sum()),
                        support_outside_binary_mask_distance_px=stats(distance_transform_edt(~mask)[outside]),
                        support_outside_binary_mask_gt2px=int((distance_transform_edt(~mask)[outside] > 2).sum()),
                        binary_mask_without_alpha_pixels=int(uncovered.sum()),
                        binary_mask_without_alpha_distance_px=stats(distance_transform_edt(~support)[uncovered]),
                        note='AA filter support can cross the binary IndexOB boundary; distance is an observation, not a geometry proof.')
    checks = dict(request_hash_matches=metadata.get('request_sha256') == request_hash,
                  genuine_partial_alpha=alpha_report['partial_pixels'] > 0,
                  has_opaque_alpha=alpha_report['opaque_pixels'] > 0,
                  png_alpha_matches_quantized_exr=alpha_report['png_vs_rounded_exr_differing_pixels'] == 0,
                  source_mask_equal=bool(np.array_equal(mask, source_mask)),
                  foreground_mask_equals_with_part=bool(np.array_equal(masks['foreground'], masks['with_part'])),
                  foreground_depth_on_target_matches_with_part=bool(np.array_equal(depth['foreground'][mask], depth['with_part'][mask])),
                  background_layer_masks_empty=all(not np.any(masks[name]) for name in ('empty', 'interaction', 'shadow_only')),
                  all_radiance_finite=all(record['exr_finite'] for record in inventory.values()),
                  available_noisy_finite=all(record.get('noisy_finite', True) for record in inventory.values()),
                  all_depth_finite=all(record['depth_finite'] for record in inventory.values()))
    hash_mismatches = [name for name, digest in metadata['output_sha256'].items()
                       if not (directory / name).exists() or sha256(directory / name) != digest]
    checks['render_output_hashes_match'] = not hash_mismatches
    source_inputs = {}
    for name in ('rgb', 'mask', 'metadata', 'request', 'stl'):
        path = pair['inputs'][name]
        actual = sha256(path)
        source_inputs[name] = dict(path=path, sha256=actual, matches_frozen_request=actual == input_hashes[path])
    checks['source_input_hashes_match'] = all(item['matches_frozen_request'] for item in source_inputs.values())
    result = dict(status='passes_audited', renderer_status=metadata['status'], checks=checks,
                  inventory=inventory, output_hash_mismatches=hash_mismatches,
                  source_input_files=source_inputs,
                  alpha=alpha_report, source_mask_differing_pixels=int((mask != source_mask).sum()),
                  foreground_vs_original_png_255={name: dict(pixels=int(region.sum()), **difference(
                      png['foreground'][..., :3], png['with_part'], region))
                      for name, region in [('opaque', alpha == 1), ('partial', (alpha > 0) & (alpha < 1))]},
                  radiance_decomposition=decompose(linear),
                  noisy_decomposition=decompose(noisy) if all(name in noisy for name in ('with_part', 'foreground', 'interaction')) else dict(status='unavailable'),
                  renderer_reported_not_independently_recomputed={
                      key: metadata.get(key) for key in ['geometry_before', 'geometry_after', 'camera_matrix_max_abs_error',
                      'model_matrix_max_abs_error', 'contact', 'comparison_to_source', 'layer_settings', 'noisy_output']})
    source_depth = Path(pair['source_metadata']['images']['depth'])
    if source_depth.exists():
        source = exr_channels(source_depth)['V']
        delta = np.abs(depth['with_part'].astype(np.float64) - source)
        y, x = np.unravel_index(delta.argmax(), delta.shape)
        local = source[max(y-1, 0):y+2, max(x-1, 0):x+2]
        result['source_depth_difference_m'] = dict(all=difference(depth['with_part'], source),
            target_mask=difference(depth['with_part'], source, mask),
            opaque=difference(depth['with_part'], source, alpha == 1), byte_exact=bool(np.array_equal(depth['with_part'], source)),
            above_10um_pixels=int((delta > 1e-5).sum()), above_10um_inside_target=int(((delta > 1e-5) & mask).sum()),
            max_at_row_col=[int(y), int(x)], max_at_target=bool(mask[y, x]),
            max_location_source_3x3_depth_range_m=float(local.max()-local.min()),
            note='Different sampling/device history can change depth numerics; reported without a byte-equality hard gate.')
    else:
        result['source_depth_difference_m'] = dict(status='source_missing', path=str(source_depth))
    if 'cached_background' in pair:
        cached = pair['cached_background']
        cached_empty = image_array(cached['empty'])
        delta = np.abs(png['empty'].astype(int) - cached_empty.astype(int))
        result['cached_background'] = dict(empty_difference_255=difference(png['empty'], cached_empty),
                                          changed_pixels=int(np.any(delta != 0, axis=-1).sum()),
                                          byte_exact=bool(not delta.any()), max_le_one=bool(delta.max() <= 1),
                                          note='Actual deviation reported separately; the frozen SPEC determines whether reuse is authorized.')
        previous_depth = Path(cached['empty']).parent.parent / 'with_part/depth.exr'
        if previous_depth.exists():
            previous = exr_channels(previous_depth)['V']
            result['cached_background']['previous_pair_depth'] = dict(path=str(previous_depth),
                byte_exact=bool(np.array_equal(depth['with_part'], previous)), difference_m=difference(depth['with_part'], previous))
        enhanced_path = directory / 'enhanced_empty.png'
        if enhanced_path.exists():
            result['cached_background']['enhanced_pixels_equal_cached'] = bool(np.array_equal(image_array(enhanced_path), image_array(cached['enhanced'])))
            checks['cached_enhanced_unchanged'] = result['cached_background']['enhanced_pixels_equal_cached']
    result['composition'] = composite_audit(directory, png, mask, composer)
    enhancement_path = directory / 'enhancement.json'
    if enhancement_path.exists():
        enhancement = read_json(enhancement_path)
        result['enhancement_reported_status'] = enhancement['status']
        result['enhancement_output_hash_mismatches'] = [name for name, digest in enhancement.get('outputs', {}).items()
            if not (directory / name).exists() or sha256(directory / name) != digest]
        checks['enhancement_output_hashes_match'] = not result['enhancement_output_hash_mismatches']
    if not all(checks.values()) or result['composition']['status'] == 'failed':
        result['status'] = 'check_failed'
    elif result['composition']['status'] == 'pass':
        result['status'] = 'passes_and_composition_audited'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--ids', nargs='+')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    request = read_json(output / 'request.json')
    all_ids = [pair['model_id'] for pair in request['pairs']]
    assert len(all_ids) == 10 and len(set(all_ids)) == 10, all_ids
    selected = set(args.ids or all_ids)
    assert selected <= set(all_ids), selected - set(all_ids)
    composer = load_composer(output)
    request_hash = sha256(output / 'request.json')
    report = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), output_dir=str(output),
                  planned=10, selected=len(selected), request_sha256=request_hash,
                  audit_script=str(Path(__file__).resolve()), audit_script_sha256=sha256(__file__),
                  openexr_version=OpenEXR.__version__, compute='CPU only; no renderer or image generator invoked',
                  frozen_composer_sha256=sha256(output / 'compose_layers.py'),
                  frozen_composer_matches_request=sha256(output / 'compose_layers.py') == request['script_sha256']['compose_layers.py'],
                  evidence_scope='Pixels/EXR/output hashes independently read. Mesh/camera checks are renderer reports, not independently recomputed.',
                  interpretation='No numerical identity threshold for independent Monte Carlo, denoising or filtered alpha boundaries. Visual review remains separate.',
                  samples=[])
    report['frozen_source_code_hash_mismatches'] = [name for name, digest in request['source_sha256'].items()
                                                   if sha256(output / 'source' / name) != digest]
    render_states, enhancement_states = Counter(), Counter()
    for pair in request['pairs']:
        model_id = pair['model_id']
        directory = output / model_id
        observed = {}
        for name, filename, counts in [('render', 'metadata.json', render_states), ('enhancement', 'enhancement.json', enhancement_states)]:
            state = read_json(directory / filename).get('status', 'unknown') if (directory / filename).exists() else 'missing'
            observed[name + '_state'] = state
            counts[state] += 1
        if model_id in selected:
            try:
                result = audit_case(output, pair, composer, request_hash, request['input_sha256'])
            except Exception as error:
                result = dict(status='audit_error', error=f'{type(error).__name__}: {error}')
        else:
            result = dict(status='not_selected')
        report['samples'].append(dict(model_id=model_id, **observed, **result))
        print(f"{model_id}: {result['status']}", flush=True)
    report['coverage'] = dict(denominator=10, render_states=dict(render_states), enhancement_states=dict(enhancement_states),
                              audit_states=dict(Counter(row['status'] for row in report['samples'])))
    filename = 'audit_results' + ('_' + '_'.join(args.ids) if args.ids else '') + '.json'
    destination = args.report or output / filename
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(destination, flush=True)
    return int(not report['frozen_composer_matches_request'] or bool(report['frozen_source_code_hash_mismatches']) or
               any(row['status'] in ('check_failed', 'audit_error') for row in report['samples']))


if __name__ == '__main__':
    raise SystemExit(main())
