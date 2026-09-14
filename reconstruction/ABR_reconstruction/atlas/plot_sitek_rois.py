#!/usr/bin/env python3
"""Overlay the Sitek 2019 ROIs on the MNI152 template (route A sanity figure).

    python ABR_reconstruction/atlas/plot_sitek_rois.py
writes RESULTS/atlas_validation/figures/sitek_rois.png
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, PACKAGE_ROOT)

# reconstruction/ is a couple of levels up; put it on sys.path so the atlas
# scripts share the pipeline's own repository root.
from recon_core.paths import REPO_ROOT                            # noqa: E402
sys.path.insert(0, _HERE)

from fetch_sitek import cached_path, fetch                     # noqa: E402
from positions_sitek import LABEL_MAP, analyse_all, consensus  # noqa: E402

COLOURS = {'CN': '#d95f02', 'SOC': '#1b9e77', 'IC': '#7570b3', 'MGB': '#e7298a'}


def _template_slices():
    """MNI152 T1 as (data, affine) for background slices."""
    from nilearn.datasets import load_mni152_template            # noqa: WPS433
    img = load_mni152_template()
    return np.asarray(img.get_fdata()), img.affine


def _world_to_vox(aff, xyz):
    inv = np.linalg.inv(aff)
    return (inv @ np.append(np.asarray(xyz, float), 1.0))[:3]


def main():
    import nibabel as nib                                        # noqa: WPS433

    fetch(verbose=False)
    per_mod = analyse_all()
    cons = consensus(per_mod)

    img = nib.load(cached_path('bigbrain'))
    data = np.asarray(img.get_fdata())
    ijk = np.argwhere(data > 0)
    labs = data[ijk[:, 0], ijk[:, 1], ijk[:, 2]].astype(int)
    xyz = nib.affines.apply_affine(img.affine, ijk.astype(float))

    tpl, tpl_aff = _template_slices()

    # Slice through the SOC (the structure the model cares about most).
    soc = cons[('SOC', 'R')]['centroid_mni_mm']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))

    views = [
        ('axial  (z = %.0f mm)' % soc[2], 2, soc[2], 0, 1, 'x (mm)', 'y (mm)'),
        ('coronal (y = %.0f mm)' % soc[1], 1, soc[1], 0, 2, 'x (mm)', 'z (mm)'),
        ('sagittal (x = %.0f mm)' % soc[0], 0, soc[0], 1, 2, 'y (mm)', 'z (mm)'),
    ]

    for ax, (title, axis, level, ia, ib, xl, yl) in zip(axes, views):
        vox = _world_to_vox(tpl_aff, [soc[0], soc[1], soc[2]])
        k = int(round(vox[axis]))
        sl = [slice(None)] * 3
        sl[axis] = k
        bg = tpl[tuple(sl)]

        # extents of the background slice in MNI mm
        corners = []
        for a in (0, bg.shape[0] - 1):
            for b in (0, bg.shape[1] - 1):
                v = [0, 0, 0]
                v[axis] = k
                rem = [i for i in range(3) if i != axis]
                v[rem[0]], v[rem[1]] = a, b
                corners.append(nib.affines.apply_affine(tpl_aff, v))
        corners = np.array(corners)
        ext = [corners[:, ia].min(), corners[:, ia].max(),
               corners[:, ib].min(), corners[:, ib].max()]

        ax.imshow(bg.T if ia < ib else bg, cmap='gray', origin='lower',
                  extent=ext, aspect='equal')

        # ROI voxels within +-2 mm of the slice
        near = np.abs(xyz[:, axis] - level) <= 2.0
        for lab, (struct, side) in LABEL_MAP.items():
            sel = near & (labs == lab)
            if not sel.any():
                continue
            ax.scatter(xyz[sel, ia], xyz[sel, ib], s=0.6, alpha=0.35,
                       color=COLOURS[struct], linewidths=0,
                       label=struct if side == 'R' else None)
        for (struct, side), r in cons.items():
            if struct not in COLOURS:
                continue
            c = r['centroid_mni_mm']
            if abs(c[axis] - level) > 12:
                continue
            ax.plot(c[ia], c[ib], marker='+', ms=9, mew=1.8, color='w')
            ax.plot(c[ia], c[ib], marker='+', ms=7, mew=1.0, color=COLOURS[struct])

        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        pad = 45
        ax.set_xlim(soc[ia] - pad, soc[ia] + pad)
        ax.set_ylim(soc[ib] - pad, soc[ib] + pad)

    axes[0].legend(loc='upper right', fontsize=8, markerscale=8, framealpha=0.85)
    fig.suptitle('Sitek et al. 2019 auditory ROIs (bigbrain histology) on the MNI152 template\n'
                 'crosses = consensus centroids over the 3 modalities', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    out_dir = os.path.join(REPO_ROOT, 'RESULTS', 'atlas_validation', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'sitek_rois.png')
    fig.savefig(path, dpi=150)
    print('written: %s' % path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
