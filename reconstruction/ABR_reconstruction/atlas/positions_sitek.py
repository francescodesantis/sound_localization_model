#!/usr/bin/env python3
"""Route A: nucleus positions from the Sitek et al. (2019) MNI atlas.

Reports, per structure and side, the MNI centroid, the principal axes (PCA) and
the volume for each of the three modalities (bigbrain histology, post mortem 7T
MRI, in vivo 7T fMRI). The spread across modalities is the uncertainty.

Label map. The three volumes share one affine (0.1 mm isotropic) and one
8-label scheme, odd left and even right. The atlas ships no label table, so the
mapping below was established from the centroids themselves; the four target
structures are tens of mm apart and cannot be confused:

  1/2  z ~ -44.5, |x| ~ 13   most inferior and lateral   cochlear nucleus
  3/4  z ~ -41.2, |x| ~  7   ventral pons, medial to CN   superior olivary complex
  5/6  z ~ -11.2, |x| ~  5   midbrain tectum              inferior colliculus
  7/8  z ~  -5.6, |x| ~ 16   thalamic, most anterior      medial geniculate body

cross-checked against the standard human MNI coordinates for the two
structures that are routinely reported: IC ~ (+-6, -34, -11) and
MGB ~ (+-16, -26, -5). Both match labels 5/6 and 7/8 to about 1 mm.

The SOC is a single blob, so this route anchors the complex in MNI but does not
separate MSO, LSO, MNTB and SPN; that split comes from route B (ANCHOR/Allen).
The CN blob is split rostrocaudally here to approximate the AVCN and its SBC
(rostral) and GBC (caudal) subfields.

    python ABR_reconstruction/atlas/positions_sitek.py
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

from fetch_sitek import MNI_SPACE, cached_path, fetch  # noqa: E402
from mni_head_transform import mni_to_head                              # noqa: E402

# label to (structure, side). Odd is left, even is right (see the docstring).
LABEL_MAP = {
    1: ('CN', 'L'), 2: ('CN', 'R'),
    3: ('SOC', 'L'), 4: ('SOC', 'R'),
    5: ('IC', 'L'), 6: ('IC', 'R'),
    7: ('MGB', 'L'), 8: ('MGB', 'R'),
}

MODALITIES = ('bigbrain', 'postmortem', 'invivo')

# Fraction of the CN blob's rostrocaudal extent taken as the AVCN. In the human
# CN the ventral division occupies the rostral ~2/3 of the nucleus and the AVCN
# proper is its rostral half. Within the AVCN the spherical-cell field is the
# rostral pole and the globular field lies caudal to it, the anatomy encoded as
# SBC_ROSTRAL_OFFSET_UM in main_abr_avcn.py.
# Splits are taken on quantiles of the voxel distribution rather than on the
# bounding extent: the in vivo ROIs are thresholded fMRI maps with a few
# rostral straggler voxels, and an extent-based cut put 1 voxel on one side.
AVCN_ROSTRAL_FRACTION = 0.50     # rostral 50% of the CN voxels = AVCN
SBC_GBC_SPLIT = 0.50             # AVCN split at its median y: rostral = SBC, caudal = GBC
_MIN_VOXELS = 20                 # below this a sub-blob is not reported


def _load(modality):
    import nibabel as nib                                     # noqa: WPS433
    path = cached_path(modality)
    if not os.path.exists(path):
        fetch(verbose=False)
    img = nib.load(path)
    return img, np.asarray(img.get_fdata())


def _voxel_coords_mni(img, data):
    """(labels, xyz) for every non-zero voxel, xyz in MNI mm."""
    import nibabel as nib                                     # noqa: WPS433
    ijk = np.argwhere(data > 0)
    labels = data[ijk[:, 0], ijk[:, 1], ijk[:, 2]].astype(int)
    xyz = nib.affines.apply_affine(img.affine, ijk.astype(float))
    return labels, xyz


def _shape_stats(xyz, voxel_mm3):
    """Centroid, PCA axes/extents and volume of a voxel cloud (MNI mm)."""
    if xyz.shape[0] < 3:
        raise ValueError('voxel cloud too small for PCA (n=%d)' % xyz.shape[0])
    c = xyz.mean(axis=0)
    d = xyz - c
    # PCA on the voxel cloud; eigenvectors = principal anatomical axes.
    cov = np.cov(d.T)
    w, v = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    w, v = w[order], v[:, order]
    # Sign convention: make each axis point +z-ward (or +y, then +x, if flat) so
    # left/right axes are comparable rather than arbitrarily flipped.
    for k in range(3):
        ref = np.argmax(np.abs(v[:, k]))
        if v[ref, k] < 0:
            v[:, k] *= -1
    proj = d @ v
    return {
        'centroid_mni_mm': c,
        'axes': v,                              # columns = principal axes
        'sd_mm': np.sqrt(np.maximum(w, 0.0)),   # sd along each axis
        'extent_mm': proj.max(axis=0) - proj.min(axis=0),
        'bbox_min_mni_mm': xyz.min(axis=0),
        'bbox_max_mni_mm': xyz.max(axis=0),
        'n_voxels': int(xyz.shape[0]),
        'volume_mm3': float(xyz.shape[0] * voxel_mm3),
    }


def analyse_modality(modality):
    """{(structure, side): stats} for one modality."""
    img, data = _load(modality)
    voxel_mm3 = float(np.prod(img.header.get_zooms()[:3]))
    labels, xyz = _voxel_coords_mni(img, data)

    out = {}
    for lab, (struct, side) in LABEL_MAP.items():
        sel = labels == lab
        if not sel.any():
            continue
        pts = xyz[sel]
        # A handful of stray voxels leak across the midline in the in vivo
        # volume (label 1 reaches x = +10.5). Keep only the dominant hemisphere
        # so the centroid is not dragged toward the midline.
        want = -1.0 if side == 'L' else 1.0
        keep = np.sign(pts[:, 0]) == want
        n_drop = int((~keep).sum())
        pts = pts[keep]
        st = _shape_stats(pts, voxel_mm3)
        st['n_stray_voxels_dropped'] = n_drop
        out[(struct, side)] = st

        # AVCN / SBC / GBC subdivision of the CN blob (rostral = +y).
        if struct == 'CN':
            y = pts[:, 1]
            avcn = pts[y >= np.quantile(y, 1.0 - AVCN_ROSTRAL_FRACTION)]
            if avcn.shape[0] < _MIN_VOXELS:
                continue
            out[('AVCN', side)] = _shape_stats(avcn, voxel_mm3)
            ay = avcn[:, 1]
            cut2 = np.quantile(ay, 1.0 - SBC_GBC_SPLIT)
            sbc, gbc = avcn[ay >= cut2], avcn[ay < cut2]
            if min(sbc.shape[0], gbc.shape[0]) >= _MIN_VOXELS:
                out[('AVCN_SBC', side)] = _shape_stats(sbc, voxel_mm3)
                out[('AVCN_GBC', side)] = _shape_stats(gbc, voxel_mm3)
    return out


def analyse_all(modalities=MODALITIES):
    return {m: analyse_modality(m) for m in modalities}


def consensus(per_modality):
    """Mean and spread of the centroid across modalities, in MNI and head mm."""
    keys = sorted({k for m in per_modality.values() for k in m},
                  key=lambda k: (k[0], k[1]))
    out = {}
    for key in keys:
        cs = np.array([per_modality[m][key]['centroid_mni_mm']
                       for m in per_modality if key in per_modality[m]])
        mni = cs.mean(axis=0)
        out[key] = {
            'centroid_mni_mm': mni,
            'sd_across_modalities_mm': cs.std(axis=0, ddof=0),
            'range_mm': cs.max(axis=0) - cs.min(axis=0),
            'centroid_head_mm': mni_to_head(mni),                 # fitted transform
            'centroid_head_mm_legacy': mni_to_head(mni, 'legacy'),
            'n_modalities': int(cs.shape[0]),
            'per_modality_mni_mm': {m: per_modality[m][key]['centroid_mni_mm']
                                    for m in per_modality if key in per_modality[m]},
        }
    return out


# ---------------------------------------------------------------------------
def _print_report(per_modality, cons):
    print('=' * 96)
    print('Route A - Sitek et al. 2019 subcortical auditory atlas')
    print('space: %s   |   3 modalities: %s' % (MNI_SPACE, ', '.join(MODALITIES)))
    print('=' * 96)

    print()
    print('Per-modality centroids (MNI mm)')
    print('-' * 96)
    hdr = '%-10s %-4s' % ('structure', 'side')
    for m in MODALITIES:
        hdr += ' %-24s' % m
    print(hdr + ' vol mm3 (mean)')
    for key in sorted(cons, key=lambda k: (k[0], k[1])):
        struct, side = key
        row = '%-10s %-4s' % (struct, side)
        vols = []
        for m in MODALITIES:
            st = per_modality[m].get(key)
            if st is None:
                row += ' %-24s' % '-'
                continue
            c = st['centroid_mni_mm']
            row += ' [%6.1f %6.1f %6.1f]     ' % (c[0], c[1], c[2])
            vols.append(st['volume_mm3'])
        row += ' %8.1f' % (np.mean(vols) if vols else float('nan'))
        print(row)

    print()
    print('Consensus (mean over modalities) -> MNI and head-centred frames')
    print('-' * 96)
    print('%-10s %-4s %-24s %-9s %-24s' %
          ('structure', 'side', 'MNI mm', 'sd mm', 'head-centred mm'))
    for key in sorted(cons, key=lambda k: (k[0], k[1])):
        struct, side = key
        r = cons[key]
        print('%-10s %-4s [%6.1f %6.1f %6.1f]     %-9s [%6.1f %6.1f %6.1f]'
              % (struct, side,
                 r['centroid_mni_mm'][0], r['centroid_mni_mm'][1], r['centroid_mni_mm'][2],
                 '%.1f' % np.linalg.norm(r['sd_across_modalities_mm']),
                 r['centroid_head_mm'][0], r['centroid_head_mm'][1], r['centroid_head_mm'][2]))

    print()
    print('Principal axes (bigbrain modality; unit vectors in MNI, longest first)')
    print('-' * 96)
    for key in sorted(per_modality['bigbrain'], key=lambda k: (k[0], k[1])):
        st = per_modality['bigbrain'][key]
        v, ext = st['axes'], st['extent_mm']
        print('%-10s %-4s  a1 [%5.2f %5.2f %5.2f] len %5.1f | a2 [%5.2f %5.2f %5.2f] len %5.1f'
              % (key[0], key[1], v[0, 0], v[1, 0], v[2, 0], ext[0],
                 v[0, 1], v[1, 1], v[2, 1], ext[1]))


def _validate(per_modality, cons):
    print()
    print('Validation')
    print('-' * 96)
    ok = True

    # mirror symmetry
    for struct in sorted({k[0] for k in cons}):
        L, R = cons.get((struct, 'L')), cons.get((struct, 'R'))
        if L is None or R is None:
            continue
        d = np.abs(L['centroid_mni_mm'] * np.array([-1, 1, 1]) - R['centroid_mni_mm'])
        flag = 'OK' if d.max() < 3.0 else 'CHECK'
        if d.max() >= 3.0:
            ok = False
        print('  L/R mirror %-9s |dx| %.2f  |dy| %.2f  |dz| %.2f mm   %s'
              % (struct, d[0], d[1], d[2], flag))

    # cross-modality spread
    print()
    worst = max(cons.items(), key=lambda kv: np.linalg.norm(kv[1]['range_mm']))
    print('  largest cross-modality range: %s %s = %.1f mm'
          % (worst[0][0], worst[0][1], np.linalg.norm(worst[1]['range_mm'])))

    # pathway order: CN caudal and inferior to SOC, SOC inferior to IC, IC posterior to MGB
    for side in ('L', 'R'):
        cn, soc, ic, mgb = (cons[(s, side)]['centroid_mni_mm']
                            for s in ('CN', 'SOC', 'IC', 'MGB'))
        checks = [
            ('CN inferior to SOC', cn[2] < soc[2]),
            ('CN lateral to SOC', abs(cn[0]) > abs(soc[0])),
            ('SOC inferior to IC', soc[2] < ic[2]),
            ('IC posterior to MGB', ic[1] < mgb[1]),
            ('IC medial to MGB', abs(ic[0]) < abs(mgb[0])),
        ]
        for name, good in checks:
            if not good:
                ok = False
            print('  %-4s %-22s %s' % (side, name, 'OK' if good else 'FAILED'))

    # AVCN split must sit rostral to the CN centroid
    for side in ('L', 'R'):
        cn, avcn = cons[('CN', side)], cons[('AVCN', side)]
        sbc, gbc = cons[('AVCN_SBC', side)], cons[('AVCN_GBC', side)]
        good = avcn['centroid_mni_mm'][1] > cn['centroid_mni_mm'][1]
        rostral = sbc['centroid_mni_mm'][1] > gbc['centroid_mni_mm'][1]
        d_sbc_gbc = sbc['centroid_mni_mm'][1] - gbc['centroid_mni_mm'][1]
        if not (good and rostral):
            ok = False
        print('  %-4s AVCN rostral to CN     %s   | SBC rostral to GBC %s (dy = %+.2f mm; '
              'model uses +1.50)' % (side, 'OK' if good else 'FAILED',
                                     'OK' if rostral else 'FAILED', d_sbc_gbc))
    print()
    print('  overall: %s' % ('OK' if ok else 'SOME CHECKS FAILED'))
    return ok


def _save(cons, per_modality, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rec = {
        'source': 'Sitek et al. 2019 eLife 8:e48932',
        'space': MNI_SPACE,
        'label_map': {str(k): list(v) for k, v in LABEL_MAP.items()},
        'head_frame_note': ('centroid_head_mm uses the fitted head centre '
                            '(mni_head_transform.fit_head_centre); '
                            'centroid_head_mm_legacy uses [0,-18.3,+5.5]'),
        'avcn_rostral_fraction': AVCN_ROSTRAL_FRACTION,
        'sbc_gbc_split': SBC_GBC_SPLIT,
        'consensus': {
            '%s_%s' % k: {
                'centroid_mni_mm': list(np.round(v['centroid_mni_mm'], 3)),
                'centroid_head_mm': list(np.round(v['centroid_head_mm'], 3)),
                'centroid_head_mm_legacy': list(np.round(v['centroid_head_mm_legacy'], 3)),
                'sd_across_modalities_mm': list(np.round(v['sd_across_modalities_mm'], 3)),
                'per_modality_mni_mm': {m: list(np.round(c, 3))
                                        for m, c in v['per_modality_mni_mm'].items()},
            } for k, v in cons.items()
        },
        'volumes_mm3': {
            '%s_%s' % k: {m: round(per_modality[m][k]['volume_mm3'], 2)
                          for m in per_modality if k in per_modality[m]}
            for k in cons
        },
    }
    path = os.path.join(out_dir, 'sitek_positions.json')
    with open(path, 'w') as f:
        json.dump(rec, f, indent=2)
    return path


def main():
    out_dir = os.path.join(REPO_ROOT, 'RESULTS', 'atlas_validation')
    fetch(verbose=False)
    per_modality = analyse_all()
    cons = consensus(per_modality)
    _print_report(per_modality, cons)
    ok = _validate(per_modality, cons)
    path = _save(cons, per_modality, out_dir)
    print()
    print('  written: %s' % path)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
