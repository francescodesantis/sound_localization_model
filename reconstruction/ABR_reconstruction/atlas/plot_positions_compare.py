#!/usr/bin/env python3
"""Figure: the current nucleus positions against the two atlas candidate sets.

Three views through the 4-sphere head model (sagittal, coronal, axial) plus an
MNI152 sagittal overlay, with the pipeline's live positions, the route-A
(Sitek) positions and the A+B hybrid drawn together.

    python ABR_reconstruction/atlas/plot_positions_compare.py
writes RESULTS/atlas_validation/figures/positions_compare.png
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
ABR_DIR = os.path.dirname(_HERE)
PACKAGE_ROOT = os.path.dirname(ABR_DIR)
sys.path.insert(0, PACKAGE_ROOT)

# reconstruction/ is a couple of levels up; put it on sys.path so the atlas
# scripts share the pipeline's own repository root.
from recon_core.paths import REPO_ROOT                            # noqa: E402
sys.path.insert(0, _HERE)
sys.path.insert(0, REPO_ROOT)

from recon_core import head_geometry as hg                                        # noqa: E402
from mni_head_transform import head_centre_mni_mm                 # noqa: E402

CAND_STYLE = {
    'current':   dict(marker='o', ms=8,  mfc='none', mew=2.0, label='current (pipeline)'),
    'A_sitek':   dict(marker='s', ms=7,  mew=1.2, label='A: Sitek MNI atlas'),
    'AB_hybrid': dict(marker='^', ms=8,  mew=1.2, label='A+B: Sitek anchor + ANCHOR offsets'),
}
GEN_COLOUR = {'MSO': 'tab:blue', 'LSO': 'tab:red', 'MNTB': 'tab:brown',
              'GBC': 'tab:orange', 'SBC': 'tab:purple'}


def _shells(ax, radii_mm):
    th = np.linspace(0, 2 * np.pi, 361)
    for r, lab in zip(radii_mm, ['brain', 'CSF', 'skull', 'scalp']):
        ax.plot(r * np.cos(th), r * np.sin(th), 'k-', lw=0.5, alpha=0.4)
    ax.plot(radii_mm[0] * np.cos(th), radii_mm[0] * np.sin(th), 'k-', lw=1.0, alpha=0.7)


def _draw_view(ax, table, ia, ib, title, xl, yl, radii, zoom=None):
    if zoom is None:
        _shells(ax, radii)
        for e in ('Cz', 'M1', 'M2'):
            p = hg.ELECTRODE_POS[e] * 1e-3
            ax.plot(p[ia], p[ib], 'kv', ms=6)
            ax.annotate(e, (p[ia], p[ib]), fontsize=7,
                        textcoords='offset points', xytext=(4, 4))
        ax.set_xlim(-95, 95)
        ax.set_ylim(-95, 95)
    for gen in GEN_COLOUR:
        cur = [r for r in table if r['generator'] == gen and r['side'] == 'R'
               and r['candidate'] == 'current']
        for r in table:
            if (r['generator'] == gen and r['side'] == 'R'
                    and r['candidate'] != 'current' and cur):
                a, b = np.asarray(cur[0]['head_mm']), np.asarray(r['head_mm'])
                ax.plot([a[ia], b[ia]], [a[ib], b[ib]], '-',
                        color=GEN_COLOUR[gen], lw=0.9, alpha=0.5, zorder=1)
    for r in table:
        if r['side'] != 'R':
            continue
        st = dict(CAND_STYLE[r['candidate']])
        st.pop('label')
        p = np.asarray(r['head_mm'])
        ax.plot(p[ia], p[ib], color=GEN_COLOUR[r['generator']],
                linestyle='none', zorder=3, **st)
    if zoom is not None:
        pts = np.array([r['head_mm'] for r in table if r['side'] == 'R'])
        c0, c1 = pts[:, ia].mean(), pts[:, ib].mean()
        h = max(pts[:, ia].ptp(), pts[:, ib].ptp()) * 0.75 + 6
        ax.set_xlim(c0 - h, c0 + h)
        ax.set_ylim(c1 - h, c1 + h)
        for r in table:
            if r['side'] == 'R' and r['candidate'] == 'current':
                p = np.asarray(r['head_mm'])
                ax.annotate(r['generator'], (p[ia], p[ib]), fontsize=8,
                            color=GEN_COLOUR[r['generator']],
                            textcoords='offset points', xytext=(7, 5))
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xl, fontsize=9)
    ax.set_ylabel(yl, fontsize=9)
    ax.grid(alpha=0.2)


def _mni_overlay(ax, table):
    try:
        import nibabel as nib                                      # noqa: WPS433
        from nilearn.datasets import load_mni152_template          # noqa: WPS433
        img = load_mni152_template()
        data = np.asarray(img.get_fdata())
        inv = np.linalg.inv(img.affine)
        i0 = int(round((inv @ np.array([7.0, -36.0, -42.0, 1.0]))[0]))
        sl = data[i0, :, :]
        corners = []
        for a in (0, sl.shape[0] - 1):
            for b in (0, sl.shape[1] - 1):
                corners.append(nib.affines.apply_affine(img.affine, [i0, a, b]))
        corners = np.array(corners)
        ax.imshow(sl.T, cmap='gray', origin='lower', aspect='equal',
                  extent=[corners[:, 1].min(), corners[:, 1].max(),
                          corners[:, 2].min(), corners[:, 2].max()])
    except Exception as exc:                                       # pragma: no cover
        ax.text(0.5, 0.5, 'MNI template unavailable\n%s' % exc,
                ha='center', transform=ax.transAxes)
    for r in table:
        if r['side'] != 'R':
            continue
        st = dict(CAND_STYLE[r['candidate']])
        st.pop('label')
        p = np.asarray(r['mni_mm_fitted'])
        ax.plot(p[1], p[2], color=GEN_COLOUR[r['generator']], linestyle='none',
                zorder=3, **st)
    ax.set_xlabel('MNI y (mm)', fontsize=9)
    ax.set_ylabel('MNI z (mm)', fontsize=9)
    ax.set_xlim(-75, 15)
    ax.set_ylim(-70, 20)


def _delta_panel(ax, table):
    ax.axis('off')
    lines = ['displacement from the current position (right side, head frame)', '']
    lines.append('%-6s %-11s %8s %8s %8s %7s' % ('', 'candidate', 'dx', 'dy', 'dz', '|d|'))
    for gen in GEN_COLOUR:
        cur = [r for r in table if r['generator'] == gen and r['side'] == 'R'
               and r['candidate'] == 'current']
        if not cur:
            continue
        for cand in ('A_sitek', 'AB_hybrid'):
            m = [r for r in table if r['generator'] == gen and r['side'] == 'R'
                 and r['candidate'] == cand]
            if not m:
                continue
            d = np.asarray(m[0]['head_mm']) - np.asarray(cur[0]['head_mm'])
            lines.append('%-6s %-11s %+8.2f %+8.2f %+8.2f %7.2f'
                         % (gen, cand.replace('_', ' '), d[0], d[1], d[2],
                            np.linalg.norm(d)))
        lines.append('')
    lines.append('mm.  Every nucleus is 25-30 mm from where the')
    lines.append('atlases place it, dominated by dz (the model sits')
    lines.append('~21-26 mm too superior).')
    ax.text(0.0, 1.0, '\n'.join(lines), family='monospace', fontsize=8.5,
            va='top', ha='left', transform=ax.transAxes)


def build_figure(table):
    radii = [r * 1e-3 for r in hg.FOUR_SPHERE_RADII]
    fig, axes = plt.subplots(2, 4, figsize=(21, 11.4))

    views = [(1, 2, 'sagittal (y-z)', 'y  posterior -> anterior (mm)',
              'z  inferior -> superior (mm)'),
             (0, 2, 'coronal (x-z)', 'x  left -> right (mm)',
              'z  inferior -> superior (mm)'),
             (0, 1, 'axial (x-y)', 'x  left -> right (mm)',
              'y  posterior -> anterior (mm)')]

    for k, (ia, ib, title, xl, yl) in enumerate(views):
        _draw_view(axes[0, k], table, ia, ib, title + '  -  whole head', xl, yl, radii)
        _draw_view(axes[1, k], table, ia, ib, title + '  -  zoom', xl, yl, radii,
                   zoom=True)

    _mni_overlay(axes[0, 3], table)
    axes[0, 3].set_title('MNI152 sagittal at x = +7 mm\n(current rows back-mapped with '
                         'the FITTED head origin)', fontsize=10)
    _delta_panel(axes[1, 3], table)

    gh = [plt.Line2D([], [], color=c, marker='o', ls='none', label=g)
          for g, c in GEN_COLOUR.items()]
    ch = [plt.Line2D([], [], color='k', ls='none', **CAND_STYLE[c]) for c in CAND_STYLE]
    fig.legend(handles=gh + ch, loc='lower center', ncol=8, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.005))

    c = head_centre_mni_mm('fitted')
    fig.suptitle('Nucleus positions in the ABR head model: current vs human atlases   '
                 '(right side shown; left mirrors x)\n'
                 'head-frame origin, fitted = MNI [%.1f, %.1f, %.1f] mm      '
                 'NO pipeline constant was changed' % tuple(c), fontsize=13)
    fig.tight_layout(rect=[0, 0.045, 1, 0.945], h_pad=3.4, w_pad=2.0)
    return fig


def main():
    from compare_positions import build, rows                      # noqa: WPS433
    d = build()
    fig = build_figure(rows(d))
    out_dir = os.path.join(REPO_ROOT, 'RESULTS', 'atlas_validation', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'positions_compare.png')
    fig.savefig(path, dpi=140)
    print('written: %s' % path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
