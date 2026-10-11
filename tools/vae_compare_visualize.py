"""Render preselected, paired CAD reconstructions with shared camera and bounds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np


def grid_triangles(faces):
    faces = np.asarray(faces)[..., :3]
    a, b = faces[:, :-1, :-1], faces[:, 1:, :-1]
    c, d = faces[:, 1:, 1:], faces[:, :-1, 1:]
    return np.concatenate((np.stack((a, b, c), -2).reshape(-1, 3, 3),
                           np.stack((a, c, d), -2).reshape(-1, 3, 3)))


def raw_geometry(path, truth=False):
    with np.load(path, allow_pickle=False) as data:
        faces = data['sample_points_faces' if truth else 'pred_face'][..., :3]
        curves = data['sample_points_lines' if truth else 'pred_edge'][..., :3]
        conn = data['edge_face_connectivity' if truth else 'pred_edge_face_connectivity']
        if truth:
            selected = {}
            for edge, first, second in conn.astype(int):
                if first != second:
                    selected.setdefault(tuple(sorted((first, second))), edge)
            curves = curves[list(selected.values())]
        kind = ('shared boundaries' if truth else
                'boundary queries' if str(data.get('representation', '')) == 'shared_boundary_v1' else
                'half-edges')
        return grid_triangles(faces), curves, f'{len(faces)} faces, {len(curves)} {kind}'


def step_geometry(path):
    from OCC.Extend.DataExchange import read_step_file
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
    from shared.occ_utils import get_primitives, get_triangulations, get_points_along_edge
    shape = read_step_file(str(path), as_compound=False, verbosity=False)
    vertices, triangles = get_triangulations(shape, 0.02, 0.15)
    faces = get_primitives(shape, TopAbs_FACE, v_remove_half=True)
    edges = get_primitives(shape, TopAbs_EDGE, v_remove_half=True)
    curves = [get_points_along_edge(edge, 64)[:, :3] for edge in edges]
    return np.asarray(vertices)[np.asarray(triangles)], curves, f'{len(faces)} faces, {len(edges)} edges'


def draw_row(axes, geometries, names, *, xray=False):
    points = [mesh.reshape(-1, 3) for mesh, _, _ in geometries if mesh is not None]
    finite = np.concatenate(points)
    finite = finite[np.isfinite(finite).all(1)]
    center = (finite.max(0) + finite.min(0)) / 2
    radius = max(float(np.ptp(finite, axis=0).max()) * 0.55, 0.01)
    for ax, (triangles, curves, detail), label in zip(axes, geometries, names):
        ax.computed_zorder = not xray
        ax.set_title(label + '\n' + detail, fontsize=10, pad=2)
        if triangles is not None:
            ax.add_collection3d(Poly3DCollection(triangles, facecolor='#92adc9',
                                                edgecolor='none', alpha=0.38 if xray else 0.82,
                                                zsort='average', zorder=1))
            if len(curves):
                ax.add_collection3d(Line3DCollection(curves, colors='#173759', linewidths=0.65, alpha=0.85, zorder=2))
        for set_lim, coordinate in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
            set_lim(coordinate - radius, coordinate + radius)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=24, azim=-58)
        ax.set_axis_off()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--mode', choices=['posterior_mean', 'archived_latents'], default='posterior_mean')
    parser.add_argument('--post-mode', choices=['fit_only', 'optimized'])
    args = parser.parse_args()
    selected = json.loads((args.comparison_root / 'selected_examples.json').read_text())['examples']
    suffix = args.post_mode or 'raw'
    out = args.comparison_root / 'visualizations' / args.mode / suffix
    out.mkdir(parents=True, exist_ok=True)
    contact, contact_axes = plt.subplots(len(selected), 3, figsize=(12, 3.5 * len(selected)),
                                         subplot_kw={'projection': '3d'}, squeeze=False)
    for index, item in enumerate(selected):
        name = item['id']
        ground = args.data_root / name
        geometries = [step_geometry(ground / 'normalized_shape.step') if args.post_mode else
                      raw_geometry(ground / 'data.npz', truth=True)]
        for method in ('legacy', 'joint'):
            try:
                if args.post_mode:
                    path = args.comparison_root / 'post' / args.post_mode / method / name / 'recon_brep.step'
                    geom = step_geometry(path)
                else:
                    path = args.comparison_root / 'raw_predictions' / args.mode / method / name / 'data.npz'
                    geom = raw_geometry(path)
                geometries.append(geom)
            except Exception as exc:
                geometries.append((None, [], 'No readable output: ' + type(exc).__name__))
        labels = [name + ' | GT', 'Legacy', 'Joint']
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), subplot_kw={'projection': '3d'})
        draw_row(axes, geometries, labels, xray=args.post_mode is None)
        caption = f'{args.mode} | {suffix} | ' + ('STEP surfaces; check validity report separately' if args.post_mode else
                                                'Untrimmed UV grids; all exported curves shown through surfaces')
        fig.text(0.5, 0.01, caption, ha='center', fontsize=9)
        fig.subplots_adjust(left=0, right=1, top=0.89, bottom=0.06, wspace=0)
        fig.savefig(out / (name + '.png'), dpi=160, facecolor='white')
        plt.close(fig)
        draw_row(contact_axes[index], geometries, labels, xray=args.post_mode is None)
    contact.suptitle(f'Preselected examples | {args.mode} | {suffix}', fontsize=14, y=0.998)
    contact.subplots_adjust(left=0, right=1, bottom=0.005, top=0.978, hspace=0.10, wspace=0)
    contact.savefig(out / 'contact_sheet.png', dpi=120, facecolor='white')
    plt.close(contact)
    print(out / 'contact_sheet.png')


if __name__ == '__main__':
    main()
