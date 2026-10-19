#!/usr/bin/env python3
"""ANCHOR route-B sanity figure: the SOC as annotated on the adult brainstem.

Left panels: the annotated polygons of a few sections, drawn in each section's
own derived anatomical frame (raphe at the origin, dorsal upward), with the SOC
subnuclei highlighted. Right panel: the resulting 3-D layout, lateral distance
from the midline against rostrocaudal position.

    python ABR_reconstruction/atlas/plot_anchor_sections.py
writes RESULTS/atlas_validation/figures/anchor_sections.png
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

import fetch_anchor as fa                                          # noqa: E402
import positions_anchor as pan                                     # noqa: E402
from positions_sitek import analyse_all, consensus                 # noqa: E402

HILITE = {'SO': ('#111111', 1.6), 'MSO': ('#1b9e77', 2.2), 'LSO': ('#d95f02', 2.2),
          'SpOn': ('#7570b3', 2.0), 'tz': ('#e7298a', 1.6)}


def main():
    pairs = fa.fetch_all_annotations(fa.ADULT, verbose=False)
    secs = [(r, g) for r, g in sorted(pairs, key=lambda p: p[0]['mm'])
            if r['stain'] == 'N' and 10.0 <= r['mm'] <= 17.5]

    meas, _ = pan.collect()
    cents = pan.structure_centroids(pan.structure_clouds(meas))
    cons = consensus(analyse_all())
    cal = pan.calibrate(cents, meas, cons[('SOC', 'R')]['centroid_mni_mm'])
    mm_per_px = float(np.mean([c['mm_per_px'] for c in cal.values()]))

    n = len(secs)
    fig = plt.figure(figsize=(4.1 * n, 8.6))
    gs = fig.add_gridspec(2, n, height_ratios=[1.35, 1.0], hspace=0.22)

    for i, (rec, g) in enumerate(secs):
        ax = fig.add_subplot(gs[0, i])
        byac = pan._features_by_acronym(g)
        frame = pan.section_frame(byac)
        if frame is None:
            continue
        o, e_l, e_d = frame
        for ac, items in byac.items():
            col, lw = HILITE.get(ac, ('#c8c8c8', 0.5))
            for it in items:
                if it['ring'] is None:
                    continue
                v = (it['ring'] - o) * mm_per_px
                ax.plot(v @ e_l, v @ e_d, color=col, lw=lw,
                        zorder=3 if ac in HILITE else 1)
        for ac in HILITE:
            for it in byac.get(ac, []):
                v = (it['centroid'] - o) * mm_per_px
                ax.plot(v @ e_l, v @ e_d, 'o', ms=4, color=HILITE[ac][0],
                        mec='w', mew=0.6, zorder=4)
        ax.axvline(0, color='k', lw=0.6, ls=':')
        ax.axhline(0, color='k', lw=0.6, ls=':')
        ax.set_aspect('equal')
        ax.set_title('%.2f mm  (secID %s)' % (rec['mm'], rec['secID']), fontsize=10)
        ax.set_xlabel('mediolateral (mm)')
        if i == 0:
            ax.set_ylabel('dorsal (+) / ventral (-)  (mm)')

    handles = [plt.Line2D([], [], color=c, lw=lw, label=a) for a, (c, lw) in HILITE.items()]
    handles.append(plt.Line2D([], [], color='#c8c8c8', lw=0.8, label='other structures'))
    fig.legend(handles=handles, loc='upper center', ncol=6, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.985))

    # --- 3-D layout summary
    ax = fig.add_subplot(gs[1, :])
    clouds = pan.structure_clouds(meas)
    for ac, (col, _lw) in HILITE.items():
        if ac not in clouds:
            continue
        d = clouds[ac]
        ax.scatter(d['mm'], d['lat_abs'] * mm_per_px, s=34, color=col,
                   label='%s (%s)' % (ac, pan.STRUCTURES.get(ac, ac)),
                   edgecolor='w', linewidth=0.5, zorder=3)
        c = cents[ac]
        ax.plot([c['mm']], [c['lat_abs_px'] * mm_per_px], marker='*', ms=17,
                color=col, mec='k', mew=0.7, zorder=4)
    ax.set_xlabel('rostrocaudal position (mm, ANCHOR section coordinate - metric, '
                  'needs no calibration)')
    ax.set_ylabel('distance from the midline (mm)')
    ax.set_title('SOC subnuclei in the adult human brainstem (stars = area-weighted '
                 'centroids); scale %.6f mm/px' % mm_per_px, fontsize=10)
    ax.legend(fontsize=9, ncol=5, loc='upper left')
    ax.grid(alpha=0.3)

    fig.suptitle('ANCHOR adult human brainstem (54 y, brain %s) - route B'
                 % fa.SPECIMENS[fa.ADULT][0], fontsize=13, y=0.999)

    out_dir = os.path.join(REPO_ROOT, 'RESULTS', 'atlas_validation', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'anchor_sections.png')
    fig.savefig(path, dpi=140, bbox_inches='tight')
    print('written: %s' % path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
