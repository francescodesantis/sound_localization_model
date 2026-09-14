#!/usr/bin/env python3
"""
Where each modelled nucleus sits inside the 4-sphere head model.

Two views of the same geometry:

--view panels   one figure: sagittal, coronal, axial and a 3-D rendering, with
                every requested nucleus on each
--view coronal  one coronal figure per nucleus, annotated with its coordinates

Positions come from recon_core.head_geometry, the parameter file the pipeline
itself reads, so these figures cannot drift from the simulation. They are
atlas-derived; see RESULTS/atlas_validation/ for the provenance.

Head-centred coordinates (mm):
  x  left (-) / right (+)          mediolateral
  y  posterior (-) / anterior (+)  anteroposterior
  z  inferior (-) / superior (+)   inferosuperior
  origin = centre of the head sphere; Cz (vertex) = [0, 0, +90]

Usage:
  python ABR_reconstruction/plots/positions.py
  python ABR_reconstruction/plots/positions.py --view coronal --nuclei LSO GBC
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D    # noqa: F401  (registers the 3d projection)

from ABR_reconstruction.plots import common

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'figures')

STYLE = {
    'MSO': dict(colour='tab:blue', ellipse_mm=10, display='MSO'),
    'LSO': dict(colour='tab:red', ellipse_mm=10, display='LSO'),
    'GBC': dict(colour='tab:orange', ellipse_mm=8, display='AVCN GBC'),
    'SBC': dict(colour='tab:purple', ellipse_mm=8, display='AVCN SBC'),
    'MNTB': dict(colour='tab:brown', ellipse_mm=8, display='MNTB'),
}
SIDE_MARKER = {'R': 'o', 'L': 's'}
AXIS_LABEL = {
    0: 'x — right (+) / left (−)  [mm]',
    1: 'y — anterior (+) / posterior (−)  [mm]',
    2: 'z — superior (+) / inferior (−)  [mm]',
}


def _draw_shells(ax, filled=True):
    """The four concentric head shells, as a cross-section."""
    theta = np.linspace(0, 2 * np.pi, 360)
    for radius, label, colour, alpha in zip(
            common.SHELL_RADII_MM[::-1], common.SHELL_LABELS[::-1],
            common.SHELL_COLOURS[::-1], common.SHELL_ALPHAS[::-1]):
        if filled:
            ax.fill(radius * np.cos(theta), radius * np.sin(theta),
                    color=colour, alpha=alpha,
                    label=label if label == 'brain' else None)
        ax.plot(radius * np.cos(theta), radius * np.sin(theta), 'k-',
                lw=0.5, alpha=0.5)
    for radius, label in zip(common.SHELL_RADII_MM, common.SHELL_LABELS):
        ax.text(radius * 0.707 + 1, radius * 0.707 + 1, label,
                fontsize=6, color='gray', rotation=45)


def _frame(ax, i, j):
    ax.set_xlabel(AXIS_LABEL[i])
    ax.set_ylabel(AXIS_LABEL[j])
    ax.set_aspect('equal')
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.axvline(0, color='gray', lw=0.5, ls=':')
    ax.set_xlim(-95, 95)
    ax.set_ylim(-95, 95)


def _electrodes_2d(ax, i, j, ms=7):
    for name, pos in common.ELECTRODE_POS_MM.items():
        ax.plot(pos[i], pos[j], 'g^', ms=ms)
        ax.text(pos[i] + 1.5, pos[j] + 1.5, name, fontsize=8, color='darkgreen')


def _legend_handles(nuclei):
    handles = [Line2D([0], [0], color=STYLE[n]['colour'], marker='o',
                      linestyle='none', markersize=7, label=n) for n in nuclei]
    handles += [
        Line2D([0], [0], color='k', marker='o', linestyle='none', markersize=7,
               markerfacecolor='none', label='R (circle)'),
        Line2D([0], [0], color='k', marker='s', linestyle='none', markersize=7,
               markerfacecolor='none', label='L (square)'),
    ]
    return handles


def _projection(ax, nuclei, i, j, title):
    _draw_shells(ax)
    for name in nuclei:
        cfg, pos = STYLE[name], common.NUCLEUS_POS_MM[name]
        for side, point in pos.items():
            u, v = point[i], point[j]
            ax.plot(u, v, cfg['colour'], marker=SIDE_MARKER[side], ms=7,
                    linestyle='none')
            ax.add_patch(Ellipse((u, v), width=cfg['ellipse_mm'],
                                 height=cfg['ellipse_mm'], edgecolor=cfg['colour'],
                                 facecolor='none', linestyle='--', linewidth=0.7,
                                 alpha=0.5))
    _electrodes_2d(ax, i, j)
    _frame(ax, i, j)
    ax.set_title(title)
    ax.legend(handles=_legend_handles(nuclei), fontsize=6.5, loc='upper left')


def _three_d(ax, nuclei):
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))

    for radius, colour, alpha in ((common.SHELL_RADII_MM[0], 'steelblue', 0.08),
                                  (common.SHELL_RADII_MM[3], 'burlywood', 0.04)):
        ax.plot_surface(radius * xs, radius * ys, radius * zs, color=colour,
                        alpha=alpha, linewidth=0)
        ax.plot_wireframe(radius * xs, radius * ys, radius * zs, color='gray',
                          linewidth=0.2, alpha=0.1)

    for name in nuclei:
        cfg, pos = STYLE[name], common.NUCLEUS_POS_MM[name]
        for side, point in pos.items():
            ax.scatter(*point, color=cfg['colour'], marker=SIDE_MARKER[side], s=60)
        right = pos['R']
        ax.text(right[0], right[1], right[2] + 3, name, fontsize=7,
                color=cfg['colour'])

    colours = {'Cz': 'lime', 'A1': 'orange', 'A2': 'orange'}
    for name, pos in common.ELECTRODE_POS_MM.items():
        ax.scatter(*pos, color=colours[name], marker='^', s=60, zorder=5)
        ax.text(pos[0] + 2, pos[1] + 2, pos[2] + 2, name, fontsize=8,
                color='darkgreen')

    ax.set_xlabel('x [mm]', fontsize=8)
    ax.set_ylabel('y [mm]', fontsize=8)
    ax.set_zlabel('z [mm]', fontsize=8)
    ax.set_title('3-D head model')
    ax.set_xlim(-92, 92)
    ax.set_ylim(-92, 92)
    ax.set_zlim(-92, 92)
    ax.legend(handles=_legend_handles(nuclei), fontsize=6, loc='upper left')
    ax.view_init(elev=25, azim=-60)


def plot_panels(nuclei, out_png):
    """Sagittal, coronal, axial and 3-D, with every nucleus on each."""
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Brainstem nuclei dipole positions in the 4-sphere head model\n'
                 f'{", ".join(nuclei)} — head-centred coordinates (mm)',
                 fontsize=9, y=0.99)

    _projection(fig.add_subplot(2, 2, 1), nuclei, 1, 2, 'Sagittal (x = 0)')
    _projection(fig.add_subplot(2, 2, 2), nuclei, 0, 2, 'Coronal')
    _projection(fig.add_subplot(2, 2, 3), nuclei, 0, 1, 'Axial')
    _three_d(fig.add_subplot(2, 2, 4, projection='3d'), nuclei)

    lines = ['Positions (mm, head-centred; R shown — L mirrors x):']
    for name in nuclei:
        p = common.NUCLEUS_POS_MM[name]['R']
        lines.append(f'  {name:4s} R=[{p[0]:+.1f},{p[1]:+.1f},{p[2]:+.1f}]  '
                     f'|r|={np.linalg.norm(p):.1f} mm')
    lines.append(f'All |r| < {common.SHELL_RADII_MM[0]:.0f} mm brain sphere. '
                 'Source: recon_core/head_geometry.py')
    fig.text(0.01, 0.01, '\n'.join(lines), fontsize=7.5, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout(rect=[0, 0.14, 1, 0.93])
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'figure saved to {out_png}')


def plot_coronal(name, out_png):
    """One nucleus, coronal, with both sides' coordinates annotated."""
    cfg, pos = STYLE[name], common.NUCLEUS_POS_MM[name]
    fig, ax = plt.subplots(figsize=(8, 8))
    _draw_shells(ax)

    for side, marker, colour, dx in (('R', 'o', 'tab:blue', 1),
                                     ('L', 's', 'tab:orange', -1)):
        x, z = pos[side][0], pos[side][2]
        ax.plot(x, z, color=colour, marker=marker, ms=10, linestyle='none',
                label=f'{name} {side}')
        ax.add_patch(Ellipse((x, z), width=cfg['ellipse_mm'],
                             height=cfg['ellipse_mm'], edgecolor=colour,
                             facecolor='none', linestyle='--', linewidth=0.8,
                             alpha=0.6))
        ax.annotate(f'[{pos[side][0]:+.1f}, {pos[side][1]:.1f}, '
                    f'{pos[side][2]:.2f}] mm',
                    xy=(x, z),
                    xytext=(x + dx * 25, z - 10 if side == 'R' else z + 12),
                    fontsize=9, color=colour,
                    arrowprops=dict(arrowstyle='->', color=colour, lw=1.0))

    _electrodes_2d(ax, 0, 2, ms=9)
    _frame(ax, 0, 2)
    ax.set_title(f'Coronal cross-section  (y ≈ {pos["R"][1]:.1f} mm)\n'
                 f'{cfg["display"]} dipole in the 4-sphere head model')
    ax.legend(fontsize=10, loc='upper left')

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'figure saved to {out_png}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--view', choices=['panels', 'coronal'], default='panels')
    ap.add_argument('--nuclei', nargs='+', default=list(STYLE),
                    choices=list(STYLE), help='which nuclei to draw')
    ap.add_argument('--out', default=None,
                    help='output file (panels) or directory (coronal)')
    args = ap.parse_args()

    if args.view == 'panels':
        plot_panels(args.nuclei,
                    args.out or os.path.join(FIGURES_DIR,
                                             'all_nuclei_head_position.png'))
        for name in args.nuclei:
            for side, p in common.NUCLEUS_POS_MM[name].items():
                print(f'{name} {side}: head = {p} mm  |r| = {np.linalg.norm(p):.2f} mm')
    else:
        out_dir = args.out or FIGURES_DIR
        for name in args.nuclei:
            plot_coronal(name, os.path.join(out_dir, f'{name.lower()}_coronal.png'))


if __name__ == '__main__':
    main()
