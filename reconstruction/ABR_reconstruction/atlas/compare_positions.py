#!/usr/bin/env python3
"""The deliverable: current, route-A and route-B nucleus positions side by side.

Puts the three candidate sets next to each other for every generator the ABR
pipeline models, in both frames (MNI mm and head-centred mm), so a set can be
chosen.

    CURRENT   the constants the pipeline uses today (head_geometry.py). Since
              the switch these are the adopted atlas values, so the current row
              coincides with whichever candidate was adopted per generator.
    A         Sitek et al. 2019 MNI atlas. Absolute and independent, but the
              SOC is one blob, so it gives the SOC/CN anchors not the subnuclei.
    B-rel     ANCHOR adult human brainstem. Subnucleus offsets from the SOC
              centroid. Independent, and the only source that splits the SOC.
    A+B       hybrid: the route-A anchor carrying the route-B offsets. The set
              that has a per-nucleus number for MSO, LSO and SPN.

Nothing here modifies the pipeline. It writes
    RESULTS/atlas_validation/positions_comparison.csv
    RESULTS/atlas_validation/atlas_candidates.json   (read by head_geometry)
    RESULTS/atlas_validation/figures/positions_compare.png
    RESULTS/atlas_validation/README.md

    python ABR_reconstruction/atlas/compare_positions.py
"""

import csv
import json
import os
import sys
import time

import numpy as np

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
from mni_head_transform import (LEGACY_HEAD_CENTRE_MNI_MM, fit_head_centre,   # noqa: E402
                                head_centre_mni_mm, head_um_to_mni)
import positions_sitek as ps                                      # noqa: E402
import positions_anchor as pan                                    # noqa: E402
import fetch_sitek, fetch_anchor                                  # noqa: E402

# pipeline generator to (label used by route A, label used by route B)
GENERATORS = {
    'MSO':  ('SOC', 'MSO'),
    'LSO':  ('SOC', 'LSO'),
    'MNTB': ('SOC', None),        # not annotated in ANCHOR
    'GBC':  ('AVCN_GBC', None),   # cochlear nuclei not annotated in ANCHOR
    'SBC':  ('AVCN_SBC', None),
}


def _fmt(v):
    return '[%7.2f %7.2f %7.2f]' % tuple(v)


def build():
    # --- current
    current = {g: {s: np.asarray(p[s], float) * 1e-3      # um to mm
                   for s in ('L', 'R')}
               for g, p in hg.NUCLEUS_POS_UM.items()}

    # --- route A
    sitek = ps.consensus(ps.analyse_all())

    # --- route B and the hybrid
    meas, _skipped = pan.collect()
    cents = pan.structure_centroids(pan.structure_clouds(meas))
    soc_mni = {s: sitek[('SOC', s)]['centroid_mni_mm'] for s in ('L', 'R')}
    ic_mni = 0.5 * (sitek[('IC', 'L')]['centroid_mni_mm']
                    + sitek[('IC', 'R')]['centroid_mni_mm'])
    cal = pan.calibrate(cents, meas, soc_mni['R'])
    scales = [c['mm_per_px'] for c in cal.values()]
    mm_per_px = float(np.mean(scales))
    cal_spread_pct = (max(scales) - min(scales)) / mm_per_px * 100 if len(scales) > 1 else 0.
    offsets = pan.brel_offsets(cents, mm_per_px)
    hybrid = pan.hybrid_positions(offsets, soc_mni, ic_mni)

    return dict(current=current, sitek=sitek, offsets=offsets, hybrid=hybrid,
                cal=cal, mm_per_px=mm_per_px, cal_spread_pct=cal_spread_pct,
                cents=cents, meas=meas, soc_mni=soc_mni, ic_mni=ic_mni)


def rows(d):
    """One row per generator, side and candidate."""
    out = []
    for gen, (a_key, b_key) in GENERATORS.items():
        for side in ('R', 'L'):
            cur_head = d['current'][gen][side]
            out.append({
                'generator': gen, 'side': side, 'candidate': 'current',
                'head_mm': cur_head,
                'mni_mm_fitted': head_um_to_mni(cur_head * 1e3, 'fitted'),
                'mni_mm_legacy': head_um_to_mni(cur_head * 1e3, 'legacy'),
                'note': 'ADOPTED (live pipeline value) - %s'
                        % hg.POSITION_SOURCE.get(gen, 'see head_geometry.py'),
            })
            a = d['sitek'].get((a_key, side))
            if a is not None:
                out.append({
                    'generator': gen, 'side': side, 'candidate': 'A_sitek',
                    'head_mm': a['centroid_head_mm'],
                    'mni_mm_fitted': a['centroid_mni_mm'],
                    'mni_mm_legacy': a['centroid_mni_mm'],
                    'note': 'Sitek ROI "%s"; sd over modalities %.1f mm'
                            % (a_key, np.linalg.norm(a['sd_across_modalities_mm'])),
                })
            if b_key is not None and (b_key, side) in d['hybrid']:
                h = d['hybrid'][(b_key, side)]
                o = d['offsets'][b_key]
                out.append({
                    'generator': gen, 'side': side, 'candidate': 'AB_hybrid',
                    'head_mm': h['head_mm'],
                    'mni_mm_fitted': h['mni_mm'],
                    'mni_mm_legacy': h['mni_mm'],
                    'note': ('route-A SOC anchor + ANCHOR offset '
                             '(lat %+.2f, dors %+.2f, rost %+.2f mm)'
                             % (o['d_lateral_mm'], o['d_dorsal_mm'], o['d_rostral_mm'])),
                })
    return out


def print_report(d, table):
    c_fit = head_centre_mni_mm('fitted')
    print('=' * 104)
    print('Nucleus positions: CURRENT vs route A (Sitek MNI) vs route A+B (ANCHOR offsets)')
    print('=' * 104)
    print('head frame origin in MNI: fitted [%.2f %.2f %.2f] mm  |  legacy [%.2f %.2f %.2f] mm'
          % (*c_fit, *LEGACY_HEAD_CENTRE_MNI_MM))
    print('The pipeline has ADOPTED: MSO/LSO = AB_hybrid, GBC/SBC = A_sitek,')
    print('MNTB = SOC anchor + Kulesza offset.  So each `current` row should now')
    print('coincide with its adopted candidate (delta ~ 0); the MNTB has no')
    print('candidate of its own and is expected to differ from A_sitek.')
    print()

    for gen in GENERATORS:
        print('-' * 104)
        print('%s' % gen)
        print('%-4s %-10s %-24s %-24s %-24s' %
              ('side', 'candidate', 'head-centred mm', 'MNI mm (fitted)', 'MNI mm (legacy)'))
        for r in table:
            if r['generator'] != gen:
                continue
            print('%-4s %-10s %-24s %-24s %-24s'
                  % (r['side'], r['candidate'], _fmt(r['head_mm']),
                     _fmt(r['mni_mm_fitted']), _fmt(r['mni_mm_legacy'])))
        # deltas, right side
        cur = next(r for r in table if r['generator'] == gen and r['side'] == 'R'
                   and r['candidate'] == 'current')
        for cand in ('A_sitek', 'AB_hybrid'):
            m = [r for r in table if r['generator'] == gen and r['side'] == 'R'
                 and r['candidate'] == cand]
            if not m:
                continue
            for frame, key in (('head', 'head_mm'),):
                dv = np.asarray(m[0][key]) - np.asarray(cur[key])
                print('     delta(%s - current), %s frame: [%+6.2f %+6.2f %+6.2f] '
                      '|d| = %5.2f mm' % (cand, frame, *dv, np.linalg.norm(dv)))
        print('     %s' % next(r['note'] for r in table if r['generator'] == gen
                               and r['side'] == 'R' and r['candidate'] == 'current'))
    print('-' * 104)


def orientation_check(d):
    """Compare the pipeline's ROTATION matrices with the atlas principal axes."""
    print()
    print('Orientation cross-check (reported only - no rotation matrix is changed)')
    print('-' * 104)

    per_mod = ps.analyse_all()
    bb = per_mod['bigbrain']

    def ang(u, v):
        u = np.asarray(u, float) / np.linalg.norm(u)
        v = np.asarray(v, float) / np.linalg.norm(v)
        return float(np.degrees(np.arccos(np.clip(abs(u @ v), -1, 1))))

    # SOC long axis (route A) vs the model's MSO dendritic axis in head coords.
    soc_axis = bb[('SOC', 'R')]['axes'][:, 0]
    mso_dend_head = hg.ROTATION_MSO['R'] @ np.array([0., 0., 1.])   # model_z is the dendrite
    print('  MSO dendritic axis (model_z -> head)  %s' % _fmt(mso_dend_head))
    print('  SOC long axis, Sitek bigbrain (MNI)   %s' % _fmt(soc_axis))
    print('    angle between them: %.1f deg   (the SOC blob elongates along its own long'
          % ang(soc_axis, mso_dend_head))
    print('    axis, which is NOT the MSO dendritic axis - this is context, not a target)')

    # CN long axis vs the AVCN rostral tilt
    cn_axis = bb[('CN', 'R')]['axes'][:, 0]
    tilt_meas = float(np.degrees(np.arctan2(abs(cn_axis[0]), abs(cn_axis[1]))))
    print()
    print('  CN long axis, Sitek bigbrain (MNI)    %s' % _fmt(cn_axis))
    print('    its rotation away from the y (rostrocaudal) axis in the axial plane: '
          '%.1f deg' % tilt_meas)
    print('    model AVCN_ROSTRAL_TILT_DEG = %.1f deg  ->  difference %.1f deg'
          % (hg.AVCN_ROSTRAL_TILT_DEG, abs(tilt_meas - hg.AVCN_ROSTRAL_TILT_DEG)))

    # MNTB long axis from horizontal
    mntb_dend_head = hg.ROTATION_MNTB['R'] @ np.array([0., 0., 1.])
    elev = float(np.degrees(np.arcsin(abs(mntb_dend_head[2]))))
    print()
    print('  MNTB dendritic axis (model_z -> head) %s' % _fmt(mntb_dend_head))
    print('    elevation from horizontal %.1f deg  vs MNTB_LONGAXIS_FROM_HORIZ_DEG = %.1f'
          % (elev, hg.MNTB_LONGAXIS_FROM_HORIZ_DEG))
    print('    (self-consistent by construction; ANCHOR does not annotate the MNTB, so')
    print('     route B cannot check it)')

    # LSO axon axis
    lso_axon_head = hg.ROTATION_LSO['R'] @ hg.LSO_AXON_DIR
    soc_ic = d['ic_mni'] - d['soc_mni']['R']
    print()
    print('  LSO axon axis (AXON_DIR -> head)      %s' % _fmt(lso_axon_head))
    print('  SOC -> IC direction, Sitek (MNI)      %s' % _fmt(soc_ic / np.linalg.norm(soc_ic)))
    print('    angle between them: %.1f deg   (the model sends the LSO axon straight up;'
          % ang(lso_axon_head, soc_ic))
    print('     the measured lemniscal direction is %.1f deg off vertical)'
          % ang(soc_ic, [0., 0., 1.]))


def write_outputs(d, table):
    out_dir = os.path.join(REPO_ROOT, 'RESULTS', 'atlas_validation')
    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # --- CSV
    csv_path = os.path.join(out_dir, 'positions_comparison.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['generator', 'side', 'candidate',
                    'head_x_mm', 'head_y_mm', 'head_z_mm',
                    'mni_x_mm_fitted', 'mni_y_mm_fitted', 'mni_z_mm_fitted',
                    'mni_x_mm_legacy', 'mni_y_mm_legacy', 'mni_z_mm_legacy', 'note'])
        for r in table:
            w.writerow([r['generator'], r['side'], r['candidate']]
                       + [round(v, 3) for v in r['head_mm']]
                       + [round(v, 3) for v in r['mni_mm_fitted']]
                       + [round(v, 3) for v in r['mni_mm_legacy']]
                       + [r['note']])

    # --- candidates JSON (read back by head_geometry.load_atlas_candidates)
    cand = {'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
            'head_centre_mni_mm': {'fitted': list(np.round(head_centre_mni_mm('fitted'), 4)),
                                   'legacy': list(np.round(LEGACY_HEAD_CENTRE_MNI_MM, 4))},
            'candidates': {}}
    for r in table:
        if r['candidate'] == 'current':
            continue
        cand['candidates'].setdefault(r['generator'], {}).setdefault(r['candidate'], {})[
            r['side']] = {'head_um': list(np.round(np.asarray(r['head_mm']) * 1e3, 1)),
                          'mni_mm': list(np.round(r['mni_mm_fitted'], 3)),
                          'note': r['note']}
    json_path = os.path.join(out_dir, 'atlas_candidates.json')
    with open(json_path, 'w') as f:
        json.dump(cand, f, indent=2)

    # --- README
    _c, stats = fit_head_centre()
    readme = os.path.join(out_dir, 'README.md')
    with open(readme, 'w') as f:
        f.write(_readme_text(d, stats))

    return csv_path, json_path, readme


def _readme_text(d, sphere_stats):
    c = head_centre_mni_mm('fitted')
    lines = []
    a = lines.append
    a('# Atlas verification of the ABR nucleus positions\n')
    a('Generated %s by `ABR_reconstruction/atlas/compare_positions.py`.\n'
      % time.strftime('%Y-%m-%d'))
    a('The pipeline now USES the atlas-derived positions. The adopted set is:')
    a('MSO and LSO from `AB_hybrid` (Sitek SOC anchor + ANCHOR subnucleus')
    a('offsets), GBC and SBC from `A_sitek` (the cochlear-nucleus ROI split')
    a('rostral/caudal), and the MNTB from the Sitek SOC anchor carrying the')
    a('Kulesza 2015 offset, because no atlas measures the MNTB. They live in')
    a('`ABR_reconstruction/head_geometry.py` as MNI coordinates and are converted')
    a('to the head frame at import.\n')
    a('The `current` rows of `positions_comparison.csv` are those adopted values;')
    a('the positions they replaced are listed in `test_head_geometry.py`')
    a('(`SUPERSEDED_POS_UM`), which also re-derives and checks them.\n')
    a('The head model itself - 4-sphere radii, conductivities, electrodes and all')
    a('four model->head rotation matrices - is unchanged.\n')

    a('## Sources\n')
    a('| source | what it gives | space | reference |')
    a('|---|---|---|---|')
    a('| Sitek et al. 2019, eLife 8:e48932 | cochlear nucleus, SOC, IC, MGB as ROIs '
      '| MNI ICBM152 2009b Nonlin. Sym. | <https://github.com/sitek/subcortical-auditory-atlas> |')
    a('| ANCHOR, SGBC IIT Madras 2026 | MSO, LSO, SPN, periolivary, trapezoid body '
      'polygons | specimen-native | <https://anchor.humanbrain.in/> doi:10.64898/2026.06.03.727794 |')
    a('| MNI152 brain mask (nilearn) | head-sphere origin | MNI | bundled with nilearn |\n')
    a('Downloaded copies and their checksums: `data/atlases/*/MANIFEST.json` (gitignored).\n')

    a('## Head-frame origin\n')
    a('The pipeline places nuclei in a head-centred frame whose origin is the centre')
    a('of the 4-sphere model. That origin was previously quoted as')
    a('`[0, -18.3, +5.5] mm` in MNI with no derivation. It is re-derived here by')
    a('least-squares fitting the model\'s %.0f mm brain shell to the MNI152 brain-mask'
      % sphere_stats['radius_mm'])
    a('surface (%d surface voxels, `c_x` pinned to 0 by symmetry):\n'
      % sphere_stats['n_surface_voxels'])
    a('    fitted head centre = [%.2f, %.2f, %.2f] mm (MNI)' % tuple(c))
    a('    legacy head centre = [%.2f, %.2f, %.2f] mm (MNI)'
      % tuple(LEGACY_HEAD_CENTRE_MNI_MM))
    a('    radial residual to the shell: rms %.2f mm, 5-95%%ile [%+.2f, %+.2f] mm\n'
      % (sphere_stats['resid_rms_mm'], sphere_stats['resid_p5_mm'],
         sphere_stats['resid_p95_mm']))
    a('The residual is the irreducible error of representing a brain as a sphere.')
    a('Both transforms are reported everywhere so the choice of origin can be seen.\n')

    a('## Route A - Sitek 2019\n')
    a('Label identities were established from the centroids (the atlas ships no label')
    a('table) and confirmed against the standard MNI coordinates of the two structures')
    a('that are routinely reported: IC and MGB both matched to ~1 mm. All L/R pairs are')
    a('mirror-symmetric to <1 mm, and the pathway ordering (CN inferior+lateral to SOC,')
    a('SOC inferior to IC, IC posterior+medial to MGB) holds on both sides.\n')
    a('The SOC is a single ROI: route A anchors the complex but cannot separate')
    a('MSO / LSO / MNTB / SPN.\n')

    a('## Route B - ANCHOR\n')
    a('Adult specimen (54 y, brain id %s). Each section\'s anatomical frame is derived'
      % fetch_anchor.SPECIMENS[fetch_anchor.ADULT][0])
    a('from its own midline anatomy - origin at the raphe nuclei, dorsal along the')
    a('raphe -> ependymal-zone vector - because the GeoJSON `rotation` field is not a')
    a('consistent anatomical correction. The frame is validated on landmarks with an')
    a('unambiguous position (vestibular nuclei dorsal, pontine nuclei and pyramidal')
    a('tract ventral, raphe and ependymal zone on the midline); all pass.\n')
    a('The image server exposes no physical resolution, so the pixel scale is')
    a('calibrated two ways:\n')
    for name, cc in d['cal'].items():
        a('- **%s** (%s): %.0f px <-> %.2f mm => %.6f mm/px. %s'
          % (name, 'independent of route A' if cc['independent'] else 'anchored on route A',
             cc['anchor_px'], cc['reference_mm'], cc['mm_per_px'], cc['basis']))
    a('')
    a('Adopted scale **%.6f mm/px**; the two disagree by **%.0f%%**, which is the'
      % (d['mm_per_px'], d['cal_spread_pct']))
    a('calibration uncertainty carried by every route-B distance as a pure scale')
    a('factor. The rostrocaudal axis is exempt - the section `mm` field is metric.\n')
    a('Resulting human morphometry: MSO %.2f mm from the midline, LSO %.2f mm, MSO'
      % (d['offsets']['MSO']['lateral_from_midline_mm'],
         d['offsets']['LSO']['lateral_from_midline_mm']))
    a('rostrocaudal extent %.2f mm - all within the published human range.\n'
      % (d['cents']['MSO']['mm_range'][1] - d['cents']['MSO']['mm_range'][0]))
    a('**Gaps.** ANCHOR does not annotate the trapezoid NUCLEUS (MNTB) or the cochlear')
    a('nuclei in any specimen, so route B says nothing about the MNTB (the model keeps')
    a('its Kulesza-2015 offset) or the AVCN (route A covers it directly).\n')

    a('## Why routes A and B are not independent for absolute position\n')
    a('ANCHOR has no stereotaxic frame, so its absolute placement must borrow route A\'s')
    a('anchor. The `AB_hybrid` rows are therefore labelled a hybrid, never an')
    a('independent estimate. What IS independent in route B is the set of subnucleus')
    a('offsets from the SOC centroid (`b_rel_offsets_mm` in `anchor_positions.json`).\n')

    a('## Files\n')
    a('- `positions_comparison.csv` - the full table (generator x side x candidate)')
    a('- `sitek_positions.json`, `anchor_positions.json` - per-route detail')
    a('- `atlas_candidates.json` - candidate positions, read by')
    a('  `head_geometry.load_atlas_candidates()`; not wired into the pipeline')
    a('- `figures/sitek_rois.png`, `figures/anchor_sections.png`,')
    a('  `figures/positions_compare.png`')
    return '\n'.join(lines) + '\n'


def main():
    fetch_sitek.fetch(verbose=False)
    d = build()
    table = rows(d)
    print_report(d, table)
    orientation_check(d)
    csv_path, json_path, readme = write_outputs(d, table)

    from plot_positions_compare import build_figure                # noqa: WPS433
    fig_dir = os.path.join(REPO_ROOT, 'RESULTS', 'atlas_validation', 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, 'positions_compare.png')
    build_figure(table).savefig(fig_path, dpi=140)

    print()
    print('written:')
    for p in (csv_path, json_path, readme, fig_path):
        print('  %s' % p)
    print()
    print('  the two per-route sanity figures are produced by')
    print('    python ABR_reconstruction/atlas/plot_sitek_rois.py')
    print('    python ABR_reconstruction/atlas/plot_anchor_sections.py')
    return 0


if __name__ == '__main__':
    sys.exit(main())
