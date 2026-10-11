"""Render preselected, paired CAD reconstructions with shared camera and bounds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
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
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX, TopAbs_REVERSED
    from OCC.Core.TopLoc import TopLoc_Location
    from shared.occ_utils import get_primitives
    from tools.vae_compare_post import read_step_at_precision, native_validity
    shape = read_step_at_precision(path)
    valid = native_validity(shape)['native_valid_solid']
    faces = get_primitives(shape, TopAbs_FACE, v_remove_half=True)
    edges = get_primitives(shape, TopAbs_EDGE, v_remove_half=True)
    meshes, curves, missing = [], [], 0
    for face in faces:
        BRepMesh_IncrementalMesh(face, 0.02, False, 0.15)
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, location)
        if triangulation is None or not triangulation.NbTriangles():
            missing += 1
            continue
        transform = location.Transformation()
        vertices = np.asarray([triangulation.Node(i).Transformed(transform).Coord()
                               for i in range(1, triangulation.NbNodes() + 1)])
        triangles = np.asarray([triangulation.Triangle(i).Get()
                                for i in range(1, triangulation.NbTriangles() + 1)]) - 1
        if face.Orientation() == TopAbs_REVERSED:
            triangles = triangles[:, ::-1]
        meshes.append(vertices[triangles])
    for edge in edges:
        if BRep_Tool.Degenerated(edge):
            vertices = get_primitives(edge, TopAbs_VERTEX, v_remove_half=True)
            if vertices:
                curves.append(np.repeat(np.asarray(BRep_Tool.Pnt(vertices[0]).Coord())[None], 64, axis=0))
        else:
            curve = BRepAdaptor_Curve(edge)
            curves.append(np.asarray([curve.Value(float(u)).Coord()
                                      for u in np.linspace(curve.FirstParameter(), curve.LastParameter(), 64)]))
    if not meshes:
        raise ValueError('No renderable face meshes')
    detail = f'{len(faces)} faces, {len(edges)} edges | valid solid: {"yes" if valid else "no"}'
    if missing:
        detail += f'\n{missing} faces have no renderable mesh'
    return np.concatenate(meshes), curves, detail


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
            surface = Poly3DCollection(triangles, facecolors='#92adc9',
                                       alpha=0.38 if xray else 0.82,
                                       zsort='average', zorder=1, shade=True,
                                       lightsource=LightSource(azdeg=315, altdeg=40))
            surface.set_edgecolor('none')
            ax.add_collection3d(surface)
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
                    status_path = (args.comparison_root / 'metadata' / 'post' / args.post_mode /
                                   'samples' / method / (name + '.json'))
                    status = json.loads(status_path.read_text())
                    if status.get('geometry_status') != 'ok':
                        detail = ('STEP valid solid: ' + ('yes' if status.get('native_valid_solid') else 'no') +
                                  '\nGeometry unavailable: ' + status.get('status', 'unknown'))
                        geometries.append((None, [], detail))
                        continue
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
