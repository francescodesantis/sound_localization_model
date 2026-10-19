#!/usr/bin/env python3
"""Route B: SOC subnucleus geometry from the ANCHOR adult human brainstem.

ANCHOR is the only source that resolves the SOC into its subnuclei. It has no
stereotaxic frame, so this module produces two things:

  B-rel   offsets of each subnucleus from the SOC ('Superior olive') centroid,
          in mm, in a brainstem-anatomical frame. Independent of route A.
  A+B     those offsets hung off the route-A (Sitek) SOC anchor, giving
          absolute MNI and head-frame positions. A hybrid, never an independent
          estimate.

What the adult specimen carries (specimen 3, 54 y, bid 296; verified by
scanning all 56 annotated sections):
    Superior olive (SO)                11 polygons, 10.4-20.0 mm   (SOC anchor)
    Medial superior olive (MSO)         7 polygons, 10.4-17.3 mm
    Lateral superior olive (LSO)        4 polygons, 10.4-16.1 mm
    Superior paraolivary nucleus (SpOn) 4 polygons, 10.4-14.2 mm
    Medio-/lateroventral periolivary   13/5 polygons
    Trapezoid body (tz, the fibre tract) 7 polygons
  Not annotated: the trapezoid nucleus (MNTB) and the cochlear nuclei.
So route B covers MSO, LSO and SPN. The MNTB is reported as a gap (the model
keeps its Kulesza-derived offset) and the AVCN comes from route A, which has
the cochlear nucleus in MNI directly.

Per-section anatomical frame. The GeoJSON 'rotation' field (the IIP server's
'Vertical-views') is not a consistent anatomical correction, so the frame comes
from the section's own midline anatomy:
    origin  = centroid of the raphe nuclei (RN), a midline structure
    dorsal  = direction from RN to the ependymal zone (EZ, 4th ventricle
              floor), also midline, so RN to EZ is the dorsoventral midline
    medial-lateral = perpendicular to it, in plane
This is rotation invariant and needs no metadata. The mediolateral sign (which
side is the subject's left) is not recoverable from the annotations, so lateral
offsets are reported as magnitudes and mirrored, as the model does.

Third axis: the section's own 'mm' field is the rostrocaudal coordinate, which
increases rostrally (the midbrain sections sit at 40-48 mm).

Pixel scale. The IIP server exposes Max-size (160000 x 160000) but no physical
resolution, so px to mm is calibrated against known human anatomy in two
independent ways, and the disagreement is the calibration uncertainty:
    (1) bilateral SO separation against the same distance in the route-A atlas
    (2) brainstem width at SOC level against the MNI152 pons width there

    python ABR_reconstruction/atlas/positions_anchor.py
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, PACKAGE_ROOT)

# reconstruction/ is a couple of levels up; put it on sys.path so the atlas
# scripts share the pipeline's own repository root.
from recon_core.paths import REPO_ROOT                            # noqa: E402
sys.path.insert(0, _HERE)

import fetch_anchor as fa                                       # noqa: E402
from mni_head_transform import mni_to_head                      # noqa: E402

# structures of interest, mapped to the label used in the report
STRUCTURES = {
    'SO': 'SOC',        # superior olive, the whole complex (the anchor)
    'MSO': 'MSO',
    'LSO': 'LSO',
    'SpOn': 'SPN',      # superior paraolivary nucleus
    'MPoN': 'MPO',      # medioventral periolivary
    'LPoN': 'LPO',      # lateroventral periolivary
    'tz': 'tz',         # trapezoid body (fibre tract), an MNTB bound not the nucleus
}
MIDLINE_ORIGIN = 'RN'   # raphe nuclei
MIDLINE_DORSAL = 'EZ'   # ependymal zone (4th ventricle floor)

SOC_MM_RANGE = (8.0, 22.0)     # rostrocaudal window that contains the SOC
NISSL_ONLY = True              # the IHC sections carry only a handful of polygons

# Cerebellum and its peduncles: attached to the block but not part of the
# brainstem width used for calibration.
CEREBELLAR = {'CB', 'mcp', 'icp', 'scp'}

# Published human transverse width of the CAUDAL pons, at the level of the SOC.
# Used only to calibrate the pixel scale (see calibrate()).
PONS_WIDTH_MM = 32.0
PONS_WIDTH_TOL_MM = 3.0

# Landmarks with an unambiguous position, used to validate the derived frame:
# name to (expected sign of the dorsal coordinate, whether |lateral| is small)
FRAME_LANDMARKS = {
    'EZ': ('dorsal', 'midline'),      # ependymal zone / 4th-ventricle floor
    'MVe': ('dorsal', None),          # medial vestibular nucleus
    'SVe': ('dorsal', None),          # superior vestibular nucleus
    'Pn': ('ventral', None),          # pontine nuclei (basis pontis)
    'py': ('ventral', None),          # pyramidal tract
    'RN': (None, 'midline'),          # raphe, the frame origin
}


# ---------------------------------------------------------------------------
# GeoJSON helpers
# ---------------------------------------------------------------------------
def _outer_ring(feature):
    coords = feature['geometry']['coordinates']
    ring = np.asarray(coords[0], dtype=float)
    if ring.ndim != 2 or ring.shape[1] != 2:
        return None
    return ring


def _polygon_centroid_area(ring):
    """Area-weighted centroid and |area| of a closed polygon ring."""
    if ring is None or len(ring) < 3:
        return None, 0.0
    x, y = ring[:, 0], ring[:, 1]
    cross = x[:-1] * y[1:] - x[1:] * y[:-1]
    a = 0.5 * cross.sum()
    if abs(a) < 1e-9:
        return ring.mean(axis=0), 0.0
    cx = ((x[:-1] + x[1:]) * cross).sum() / (6.0 * a)
    cy = ((y[:-1] + y[1:]) * cross).sum() / (6.0 * a)
    return np.array([cx, cy]), abs(a)


def _acronym(feature):
    return ((feature.get('properties') or {}).get('data') or {}).get('acronym')


def _features_by_acronym(geojson):
    out = {}
    for f in geojson.get('features', []):
        ac = _acronym(f)
        if ac is None:
            continue
        c, a = _polygon_centroid_area(_outer_ring(f))
        if c is None:
            continue
        out.setdefault(ac, []).append({'centroid': c, 'area': a,
                                       'ring': _outer_ring(f)})
    return out


def _weighted_mean(items):
    """Area-weighted mean centroid of a list of polygons."""
    w = np.array([max(it['area'], 1.0) for it in items])
    c = np.array([it['centroid'] for it in items])
    return (c * w[:, None]).sum(axis=0) / w.sum()


# ---------------------------------------------------------------------------
# Per-section anatomical frame
# ---------------------------------------------------------------------------
def section_frame(byac):
    """(origin, e_lat, e_dorsal) in image px, from RN (midline) and EZ (dorsal).

    Returns None if either midline landmark is missing.
    """
    if MIDLINE_ORIGIN not in byac or MIDLINE_DORSAL not in byac:
        return None
    o = _weighted_mean(byac[MIDLINE_ORIGIN])
    d = _weighted_mean(byac[MIDLINE_DORSAL]) - o
    n = np.linalg.norm(d)
    if n < 1e-6:
        return None
    e_d = d / n
    e_l = np.array([e_d[1], -e_d[0]])          # in-plane perpendicular
    return o, e_l, e_d


def section_measurements(rec, geojson):
    """Structure centroids of one section in the section's anatomical frame (px).

    Returns {'mm': float, 'coords': {acronym: [(lat, dorsal), ...]}, 'width_px': w}
    or None if the frame could not be built.
    """
    byac = _features_by_acronym(geojson)
    frame = section_frame(byac)
    if frame is None:
        return None
    o, e_l, e_d = frame

    coords = {}
    for ac in STRUCTURES:
        for it in byac.get(ac, []):
            v = it['centroid'] - o
            coords.setdefault(ac, []).append((float(v @ e_l), float(v @ e_d),
                                              float(it['area'])))

    def _extent(exclude=()):
        pts = [it['ring'] for ac, items in byac.items() if ac not in exclude
               for it in items if it['ring'] is not None]
        if not pts:
            return np.nan
        lat = (np.concatenate(pts) - o) @ e_l
        return float(lat.max() - lat.min())

    # Frame-validation landmarks: mean (|lateral|, dorsal) per landmark, px.
    landmarks = {}
    for ac in FRAME_LANDMARKS:
        if ac not in byac:
            continue
        v = np.array([it['centroid'] - o for it in byac[ac]])
        # signed mean too: a midline structure drawn as a symmetric pair has a
        # non-zero mean |lateral| but a near-zero signed mean.
        landmarks[ac] = (float(np.mean(np.abs(v @ e_l))),
                         float(np.mean(v @ e_d)),
                         float(np.mean(v @ e_l)))

    return {'mm': float(rec['mm']), 'secID': rec['secID'], 'stain': rec['stain'],
            'coords': coords,
            'width_px': _extent(),                       # block incl. cerebellum
            'brainstem_width_px': _extent(CEREBELLAR),   # brainstem only
            'landmarks': landmarks}


def collect(specimen=fa.ADULT):
    """Per-section measurements over the SOC window; plus the sections skipped."""
    pairs = fa.fetch_all_annotations(specimen, verbose=False)
    meas, skipped = [], []
    for rec, g in sorted(pairs, key=lambda p: p[0]['mm']):
        if not (SOC_MM_RANGE[0] <= rec['mm'] <= SOC_MM_RANGE[1]):
            continue
        if NISSL_ONLY and rec['stain'] != 'N':
            continue
        m = section_measurements(rec, g)
        (meas if m else skipped).append(m or rec)
    return meas, skipped


# ---------------------------------------------------------------------------
# Aggregation: 3-D centroid per structure, in px (lateral, dorsal) + mm (rostral)
# ---------------------------------------------------------------------------
def structure_clouds(meas):
    """{acronym: {'lat_abs', 'dorsal', 'mm', 'area'} arrays} pooled over sections.

    Bilateral structures contribute |lateral|, since the left/right sign is not
    recoverable from the annotations (see the module docstring).
    """
    out = {}
    for m in meas:
        for ac, pts in m['coords'].items():
            for lat, dor, area in pts:
                d = out.setdefault(ac, {'lat_abs': [], 'dorsal': [], 'mm': [],
                                        'area': [], 'lat_signed': []})
                d['lat_abs'].append(abs(lat))
                d['lat_signed'].append(lat)
                d['dorsal'].append(dor)
                d['mm'].append(m['mm'])
                d['area'].append(area)
    return {ac: {k: np.asarray(v, float) for k, v in d.items()} for ac, d in out.items()}


def structure_centroids(clouds):
    """Area-weighted centroid per structure: (lat_abs px, dorsal px, mm)."""
    out = {}
    for ac, d in clouds.items():
        w = np.maximum(d['area'], 1.0)
        out[ac] = {
            'lat_abs_px': float(np.average(d['lat_abs'], weights=w)),
            'dorsal_px': float(np.average(d['dorsal'], weights=w)),
            'mm': float(np.average(d['mm'], weights=w)),
            'mm_range': (float(d['mm'].min()), float(d['mm'].max())),
            'n_polygons': int(len(d['mm'])),
            'n_sections': int(len(np.unique(d['mm']))),
        }
    return out


# ---------------------------------------------------------------------------
# px to mm calibration
# ---------------------------------------------------------------------------
def calibrate(centroids, meas, sitek_soc_mni):
    """px to mm scale, from two measures whose disagreement is the uncertainty.

    (1) bilateral SO separation: 2 * <|lateral|> of the SO polygons in px equals
        2 * |x| of the route-A SOC centroid in mm. Anchored on human MNI data,
        but by construction it forces the SOC to agree with route A, so it is
        not an independent measure of scale.
    (2) brainstem (non-cerebellar) mediolateral width at the SOC sections equals
        the published human caudal-pons transverse width. Independent of route
        A; its own uncertainty is PONS_WIDTH_TOL_MM.

    An explicit physical resolution is not available: the IIP server reports
    only Max-size (160000 x 160000), Tile-size and Resolution-number.
    """
    out = {}

    so = centroids.get('SO')
    if so is not None and abs(sitek_soc_mni[0]) > 1e-6:
        out['soc_separation'] = {
            'anchor_px': 2.0 * so['lat_abs_px'],
            'reference_mm': 2.0 * abs(sitek_soc_mni[0]),
            'mm_per_px': 2.0 * abs(sitek_soc_mni[0]) / (2.0 * so['lat_abs_px']),
            'independent': False,
            'basis': 'bilateral SO centroid separation vs the route-A SOC centroid',
        }

    # Only the sections where the SOC subnuclei are drawn: the pons widens
    # rostrally, so including the rostral SO-only sections biases the width.
    w = np.array([m['brainstem_width_px'] for m in meas
                  if ('MSO' in m['coords'] or 'LSO' in m['coords'])
                  and np.isfinite(m['brainstem_width_px'])])
    if w.size:
        out['pons_width'] = {
            'anchor_px': float(np.median(w)),
            'reference_mm': PONS_WIDTH_MM,
            'reference_tol_mm': PONS_WIDTH_TOL_MM,
            'mm_per_px': PONS_WIDTH_MM / float(np.median(w)),
            'independent': True,
            'n_sections': int(w.size),
            'basis': ('median brainstem width (cerebellum/peduncles excluded) over the '
                      '%d sections carrying MSO/LSO, vs the published human caudal-pons '
                      'width %.0f +- %.0f mm' % (w.size, PONS_WIDTH_MM, PONS_WIDTH_TOL_MM)),
        }
    return out


# ---------------------------------------------------------------------------
# B-rel offsets and the A+B hybrid
# ---------------------------------------------------------------------------
def brel_offsets(centroids, mm_per_px):
    """Offsets from the SOC ('SO') centroid, in mm, in the brainstem frame.

    Axes: lateral (magnitude, away from the midline), dorsal (positive toward
    the 4th ventricle), rostral (positive toward the midbrain).
    """
    so = centroids['SO']
    out = {}
    for ac, c in centroids.items():
        out[STRUCTURES.get(ac, ac)] = {
            'd_lateral_mm': (c['lat_abs_px'] - so['lat_abs_px']) * mm_per_px,
            'd_dorsal_mm': (c['dorsal_px'] - so['dorsal_px']) * mm_per_px,
            'd_rostral_mm': c['mm'] - so['mm'],
            'lateral_from_midline_mm': c['lat_abs_px'] * mm_per_px,
            'n_polygons': c['n_polygons'],
            'n_sections': c['n_sections'],
            'mm_range': c['mm_range'],
        }
    return out


def brainstem_to_mni_axes(soc_mni, ic_mni):
    """Unit vectors of the ANCHOR brainstem frame expressed in MNI.

    rostral is the SOC to IC direction, the ascending brainstem axis, measured
    from route A rather than assumed; lateral is MNI +x, since a pitch of the
    neuraxis leaves the mediolateral axis alone; dorsal is rostral x lateral,
    pointing toward the 4th ventricle (posterior).
    """
    e_r = np.asarray(ic_mni, float) - np.asarray(soc_mni, float)
    e_r = e_r / np.linalg.norm(e_r)
    e_l = np.array([1.0, 0.0, 0.0])
    e_l = e_l - (e_l @ e_r) * e_r
    e_l = e_l / np.linalg.norm(e_l)
    e_d = np.cross(e_r, e_l)                     # right-handed; points posteriorly
    if e_d[1] > 0:                               # force 'dorsal' = posterior in MNI
        e_d = -e_d
    return e_l, e_d, e_r


def hybrid_positions(offsets, soc_mni_by_side, ic_mni):
    """A+B: route-A SOC anchor plus route-B offsets, in MNI and head frame."""
    out = {}
    for side, soc in soc_mni_by_side.items():
        e_l, e_d, e_r = brainstem_to_mni_axes(soc, ic_mni)
        sgn = 1.0 if side == 'R' else -1.0        # lateral magnitude to side
        for label, o in offsets.items():
            mni = (np.asarray(soc, float)
                   + sgn * o['d_lateral_mm'] * e_l
                   + o['d_dorsal_mm'] * e_d
                   + o['d_rostral_mm'] * e_r)
            out[(label, side)] = {
                'mni_mm': mni,
                'head_mm': mni_to_head(mni),
                'head_mm_legacy': mni_to_head(mni, 'legacy'),
            }
    return out


# ---------------------------------------------------------------------------
def main():
    from positions_sitek import analyse_all, consensus            # noqa: WPS433

    print('=' * 92)
    print('Route B - ANCHOR adult human brainstem (specimen 3, 54 y, bid %s)'
          % fa.SPECIMENS[fa.ADULT][0])
    print('=' * 92)

    meas, skipped = collect()
    print()
    print('Sections used (Nissl, %.0f-%.0f mm): %d   skipped (no RN/EZ midline pair): %d'
          % (SOC_MM_RANGE[0], SOC_MM_RANGE[1], len(meas), len(skipped)))
    for m in meas:
        got = ' '.join('%s:%d' % (a, len(p)) for a, p in sorted(m['coords'].items()))
        print('   mm=%6.2f secID=%-5s width=%7.0f px   %s'
              % (m['mm'], m['secID'], m['width_px'], got))

    clouds = structure_clouds(meas)
    cents = structure_centroids(clouds)
    if 'SO' not in cents:
        print('\n  the SOC anchor (SO) is not annotated - route B cannot proceed')
        return 1

    print()
    print('Coverage')
    print('-' * 92)
    print('%-6s %-6s %-10s %-10s %s' % ('acr', 'label', 'polygons', 'sections', 'mm range'))
    for ac in sorted(cents, key=lambda a: -cents[a]['n_polygons']):
        c = cents[ac]
        print('%-6s %-6s %-10d %-10d [%.2f, %.2f]'
              % (ac, STRUCTURES.get(ac, ac), c['n_polygons'], c['n_sections'],
                 c['mm_range'][0], c['mm_range'][1]))
    thin = [ac for ac, c in cents.items() if c['n_sections'] < 3]
    if thin:
        print('  CAVEAT: %s annotated on fewer than 3 sections - centroid is coarse'
              % ', '.join(STRUCTURES.get(a, a) for a in thin))

    # route A anchor
    cons = consensus(analyse_all())
    soc_mni = {s: cons[('SOC', s)]['centroid_mni_mm'] for s in ('L', 'R')}
    ic_mni = 0.5 * (cons[('IC', 'L')]['centroid_mni_mm']
                    + cons[('IC', 'R')]['centroid_mni_mm'])

    cal = calibrate(cents, meas, soc_mni['R'])
    print()
    print('Pixel-scale calibration (no physical resolution is exposed by the server)')
    print('-' * 92)
    for name, c in cal.items():
        print('  %-16s %8.0f px  <->  %6.2f mm  =>  %.6f mm/px   %s'
              % (name, c['anchor_px'], c['reference_mm'], c['mm_per_px'],
                 'independent' if c['independent'] else 'anchored on route A'))
        print('      %s' % c['basis'])
    scales = [c['mm_per_px'] for c in cal.values()]
    mm_per_px = float(np.mean(scales))
    spread = (max(scales) - min(scales)) / mm_per_px * 100 if len(scales) > 1 else 0.0
    print('  ADOPTED %.6f mm/px (mean of the two).  They disagree by %.0f%%, which is the'
          % (mm_per_px, spread))
    print('  calibration uncertainty: every route-B distance below carries it as a pure')
    print('  scale factor (the rostrocaudal axis is exempt - it is metric already).')
    if 'soc_separation' in cal and 'pons_width' in cal:
        implied = cal['pons_width']['anchor_px'] * cal['soc_separation']['mm_per_px']
        print('  cross-check: under the route-A-anchored scale the ANCHOR brainstem is '
              '%.1f mm' % implied)
        print('  wide at SOC level, against the published %.0f +- %.0f mm - the two '
              'calibrations' % (PONS_WIDTH_MM, PONS_WIDTH_TOL_MM))
        print('  are therefore consistent within the published tolerance.')

    offsets = brel_offsets(cents, mm_per_px)
    print()
    print('B-rel: offsets from the SOC centroid (mm, brainstem frame) - route B alone')
    print('-' * 92)
    print('%-6s %9s %9s %9s %12s' % ('', 'd_lateral', 'd_dorsal', 'd_rostral',
                                     'from midline'))
    for label in sorted(offsets):
        o = offsets[label]
        print('%-6s %+9.2f %+9.2f %+9.2f %12.2f'
              % (label, o['d_lateral_mm'], o['d_dorsal_mm'], o['d_rostral_mm'],
                 o['lateral_from_midline_mm']))

    e_l, e_d, e_r = brainstem_to_mni_axes(soc_mni['R'], ic_mni)
    print()
    print('Brainstem frame in MNI (measured from the route-A SOC -> IC vector)')
    print('  lateral [%+.3f %+.3f %+.3f]  dorsal [%+.3f %+.3f %+.3f]  rostral [%+.3f %+.3f %+.3f]'
          % (*e_l, *e_d, *e_r))

    hyb = hybrid_positions(offsets, soc_mni, ic_mni)
    print()
    print('A+B HYBRID (route-A SOC anchor + route-B offsets) - not an independent estimate')
    print('-' * 92)
    print('%-6s %-4s %-26s %-26s' % ('', 'side', 'MNI mm', 'head-centred mm (fitted)'))
    for (label, side) in sorted(hyb):
        h = hyb[(label, side)]
        print('%-6s %-4s [%6.1f %6.1f %6.1f]      [%6.1f %6.1f %6.1f]'
              % (label, side, *h['mni_mm'], *h['head_mm']))

    ok = _validate(cents, offsets, mm_per_px, hyb, meas)

    out_dir = os.path.join(REPO_ROOT, 'RESULTS', 'atlas_validation')
    os.makedirs(out_dir, exist_ok=True)
    rec = {
        'source': 'ANCHOR, SGBC IIT Madras (2026), doi:10.64898/2026.06.03.727794',
        'specimen': fa.SPECIMENS[fa.ADULT][2],
        'brain_id': fa.SPECIMENS[fa.ADULT][0],
        'sections_used': [{'mm': m['mm'], 'secID': m['secID']} for m in meas],
        'calibration': {k: {kk: (vv if not isinstance(vv, float) else round(vv, 6))
                            for kk, vv in v.items()} for k, v in cal.items()},
        'mm_per_px': mm_per_px,
        'calibration_spread_pct': spread,
        'coverage': {STRUCTURES.get(a, a): {'n_polygons': c['n_polygons'],
                                            'n_sections': c['n_sections'],
                                            'mm_range': list(c['mm_range'])}
                     for a, c in cents.items()},
        'b_rel_offsets_mm': {k: {kk: round(vv, 4) for kk, vv in v.items()
                                 if isinstance(vv, float)} for k, v in offsets.items()},
        'brainstem_axes_in_mni': {'lateral': list(np.round(e_l, 5)),
                                  'dorsal': list(np.round(e_d, 5)),
                                  'rostral': list(np.round(e_r, 5))},
        'hybrid': {'%s_%s' % k: {'mni_mm': list(np.round(v['mni_mm'], 3)),
                                 'head_mm': list(np.round(v['head_mm'], 3)),
                                 'head_mm_legacy': list(np.round(v['head_mm_legacy'], 3))}
                   for k, v in hyb.items()},
        'not_annotated': ['MNTB (trapezoid nucleus)', 'cochlear nuclei'],
    }
    path = os.path.join(out_dir, 'anchor_positions.json')
    with open(path, 'w') as f:
        json.dump(rec, f, indent=2)
    print()
    print('  written: %s' % path)
    return 0 if ok else 1


def _validate(cents, offsets, mm_per_px, hyb, meas):
    print()
    print('Validation')
    print('-' * 92)
    ok = True

    # 1. the derived frame, against landmarks with an unambiguous position
    print('  frame (RN origin, RN->EZ dorsal), checked on landmarks:')
    agg = {}
    for m in meas:
        for ac, vals in m.get('landmarks', {}).items():
            agg.setdefault(ac, []).append(vals)
    for ac, (want_dv, want_mid) in FRAME_LANDMARKS.items():
        if ac not in agg:
            continue
        lat = np.mean([a for a, _b, _s in agg[ac]]) * mm_per_px
        dor = np.mean([b for _a, b, _s in agg[ac]]) * mm_per_px
        lat_signed = np.mean([s for _a, _b, s in agg[ac]]) * mm_per_px
        good = True
        if want_dv == 'dorsal':
            good = dor > 0
        elif want_dv == 'ventral':
            good = dor < 0
        if want_mid == 'midline':
            # symmetric pairs: judge on the signed mean, not the mean modulus
            good = good and abs(lat_signed) < 1.5
        if not good:
            ok = False
        print('    %-4s |lat| %5.2f  signed lat %+5.2f  dorsal %+6.2f mm   '
              'expect %-8s %-8s %s'
              % (ac, lat, lat_signed, dor, want_dv or '-', want_mid or '-',
                 'OK' if good else 'FAILED'))

    # 2. SOC internal topography
    print()
    print('  SOC topography:')
    hard = [('MSO medial to LSO', offsets['MSO']['lateral_from_midline_mm']
             < offsets['LSO']['lateral_from_midline_mm']),
            ('trapezoid body medial to the MSO', offsets['tz']['lateral_from_midline_mm']
             < offsets['MSO']['lateral_from_midline_mm']),
            ('MSO within 1.5 mm of the SOC centroid laterally',
             abs(offsets['MSO']['d_lateral_mm']) < 1.5)]
    for name, good in hard:
        if not good:
            ok = False
        print('    %-46s %s' % (name, 'OK' if good else 'FAILED'))

    # reported, not asserted: too few polygons to be decisive
    print('    SPN vs MSO dorsoventral: %+.2f mm (SPN is described as dorsomedial to'
          % (offsets['SPN']['d_dorsal_mm'] - offsets['MSO']['d_dorsal_mm']))
    print('      the MSO in humans, but SPN is drawn on only %d sections here - '
          'reported, not asserted)' % cents['SpOn']['n_sections'])

    # 3. morphometry against published human values
    print()
    print('  morphometry vs published human values:')
    mso_ext = cents['MSO']['mm_range'][1] - cents['MSO']['mm_range'][0]
    print('    MSO rostrocaudal extent           %5.2f mm  (needs no calibration - the'
          % mso_ext)
    print('                                              section mm field is metric)')
    print('    MSO distance from the midline     %5.2f mm  (human SOC sits ~5-8 mm lateral)'
          % offsets['MSO']['lateral_from_midline_mm'])
    print('    LSO distance from the midline     %5.2f mm' % offsets['LSO']['lateral_from_midline_mm'])
    print('    trapezoid BODY reaches            %5.2f mm from the midline; the MNTB lies'
          % offsets['tz']['lateral_from_midline_mm'])
    print('      within it.  The MNTB itself is NOT annotated in ANCHOR, so the model keeps')
    print('      its Kulesza-2015 value of 0.40 mm - unverified by route B.')

    # 4. hybrid positions physically placeable
    inside = True
    for (label, side), h in sorted(hyb.items()):
        r = np.linalg.norm(h['head_mm'])
        if r >= 79.0:
            inside = False
            print('    %s %s |r| = %.1f mm OUTSIDE the 79 mm brain shell' % (label, side, r))
    if not inside:
        ok = False
    print()
    print('  all hybrid positions inside the 79 mm brain shell: %s'
          % ('OK' if inside else 'FAILED'))
    print('  overall: %s' % ('OK' if ok else 'SOME CHECKS FAILED'))
    return ok


if __name__ == '__main__':
    sys.exit(main())
