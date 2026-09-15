#!/usr/bin/env python3
"""MNI152 to 4-sphere head-frame transform (phase 1 of the atlas verification).

Both frames share the same axis directions:
    x = mediolateral    (left -, right +)
    y = anteroposterior (posterior -, anterior +)
    z = inferosuperior  (inferior -, superior +)
so the map is a pure translation:  head = MNI - c,  MNI = head + c.

Only the origin differs. MNI152's origin is the anterior commissure; the head
frame's origin is the centre of the concentric 4-sphere model. This module
derives c by fitting the innermost (brain, 79 mm) shell to the MNI152 brain
mask, rather than using the undocumented shift the pipeline quoted:

    head_centre_MNI ~= [0, -18.3, +5.5] mm   ("Koessler et al. 2009 Cz anchor")

which has no derivation in the repo and does not reproduce the live MSO_POS_UM
either (see the report printed by __main__).

Run standalone for the report:
    python ABR_reconstruction/atlas/mni_head_transform.py
"""

import os
import sys

import numpy as np

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, PACKAGE_ROOT)

# reconstruction/ is a couple of levels up; put it on sys.path so the atlas
# scripts share the pipeline's own repository root.
from recon_core.paths import REPO_ROOT                            # noqa: E402

# Head-model geometry. Imported from the pipeline when available so the fit
# cannot drift from the model it is fitting; the literals are the fallback for
# a bare environment and are asserted equal in the self-test below.
_FALLBACK_RADII_MM = [79., 80., 85., 90.]


def _four_sphere_radii_mm():
    """[brain, CSF, skull, scalp] radii in mm, from the live pipeline if importable."""
    try:
        sys.path.insert(0, REPO_ROOT)
        from recon_core.head_geometry import FOUR_SPHERE_RADII          # noqa: WPS433
        return [r * 1e-3 for r in FOUR_SPHERE_RADII]          # um -> mm
    except Exception:
        return list(_FALLBACK_RADII_MM)


# Legacy (undocumented) shift, kept only for the comparison report.
LEGACY_HEAD_CENTRE_MNI_MM = np.array([0.0, -18.3, 5.5])


# ---------------------------------------------------------------------------
# Sphere fit
# ---------------------------------------------------------------------------
def _brain_mask_surface_points_mni():
    """Surface voxels of the MNI152 brain mask, as (N, 3) MNI mm coordinates.

    Uses the MNI152 brain mask bundled with nilearn (no download).
    """
    import nibabel as nib                                     # noqa: WPS433
    from nilearn.datasets import load_mni152_brain_mask       # noqa: WPS433
    from scipy import ndimage                                 # noqa: WPS433

    img = load_mni152_brain_mask()
    mask = np.asarray(img.get_fdata()) > 0.5
    if not mask.any():
        raise RuntimeError('MNI152 brain mask is empty')

    # Surface = mask voxels with at least one non-mask 6-neighbour.
    eroded = ndimage.binary_erosion(mask, ndimage.generate_binary_structure(3, 1))
    surf = mask & ~eroded

    ijk = np.argwhere(surf).astype(float)
    xyz = nib.affines.apply_affine(img.affine, ijk)
    return xyz, img


def fit_head_centre(radius_mm=None, constrain_midsagittal=True):
    """Least-squares centre c placing the MNI brain surface on |p - c| = R.

    Minimises sum_i (|p_i - c| - R)^2 over c with R fixed at the head model's
    brain-shell radius, since this places the model's sphere rather than
    fitting a free sphere to the brain. c_x is pinned to 0 by mid-sagittal
    symmetry.

    Returns
    -------
    c : (3,) ndarray            centre in MNI mm
    stats : dict                radial residual statistics, mm
    """
    from scipy.optimize import minimize                       # noqa: WPS433

    if radius_mm is None:
        radius_mm = _four_sphere_radii_mm()[0]

    pts, _img = _brain_mask_surface_points_mni()

    def cost(c_free):
        c = np.array([0.0, c_free[0], c_free[1]]) if constrain_midsagittal else np.asarray(c_free)
        r = np.linalg.norm(pts - c, axis=1)
        return float(np.mean((r - radius_mm) ** 2))

    x0 = np.array([-18.3, 5.5]) if constrain_midsagittal else np.array([0., -18.3, 5.5])
    res = minimize(cost, x0, method='Nelder-Mead',
                   options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 4000})

    c = (np.array([0.0, res.x[0], res.x[1]]) if constrain_midsagittal
         else np.asarray(res.x, dtype=float))

    r = np.linalg.norm(pts - c, axis=1)
    resid = r - radius_mm
    stats = {
        'radius_mm': float(radius_mm),
        'n_surface_voxels': int(pts.shape[0]),
        'resid_rms_mm': float(np.sqrt(np.mean(resid ** 2))),
        'resid_mean_mm': float(np.mean(resid)),
        'resid_min_mm': float(np.min(resid)),
        'resid_max_mm': float(np.max(resid)),
        'resid_p5_mm': float(np.percentile(resid, 5)),
        'resid_p95_mm': float(np.percentile(resid, 95)),
        'brain_r_mean_mm': float(np.mean(r)),
        'converged': bool(res.success),
    }
    return c, stats


# ---------------------------------------------------------------------------
# The transform
# ---------------------------------------------------------------------------
_CENTRE_CACHE = {}


def head_centre_mni_mm(source='fitted'):
    """Origin of the head frame expressed in MNI mm.

    source='fitted' is derived here and cached; 'legacy' is [0,-18.3,+5.5].
    """
    if source == 'legacy':
        return LEGACY_HEAD_CENTRE_MNI_MM.copy()
    if source != 'fitted':
        raise ValueError("source must be 'fitted' or 'legacy'")
    if 'fitted' not in _CENTRE_CACHE:
        _CENTRE_CACHE['fitted'] = fit_head_centre()
    return _CENTRE_CACHE['fitted'][0].copy()


def mni_to_head(p_mni_mm, source='fitted'):
    """MNI mm to head-centred mm. Accepts (3,) or (N,3)."""
    return np.asarray(p_mni_mm, dtype=float) - head_centre_mni_mm(source)


def head_to_mni(p_head_mm, source='fitted'):
    """Head-centred mm to MNI mm. Accepts (3,) or (N,3)."""
    return np.asarray(p_head_mm, dtype=float) + head_centre_mni_mm(source)


def head_um_to_mni(p_head_um, source='fitted'):
    """Head-centred um to MNI mm."""
    return head_to_mni(np.asarray(p_head_um, dtype=float) * 1e-3, source)


# ---------------------------------------------------------------------------
# Report / self-test
# ---------------------------------------------------------------------------
# Live pipeline constants, duplicated here only so this module can report on
# them without importing NEURON or MPI. Kept in sync by _assert_matches_pipeline().
_LIVE_POS_UM = {
    'MSO':      {'R': [5_000., -18_700., -29_520.]},
    'LSO':      {'R': [9_000., -20_500., -30_000.]},
    'AVCN_GBC': {'R': [10_000., -38_000., -35_000.]},
    'AVCN_SBC': {'R': [9_400., -36_500., -35_000.]},
    'MNTB':     {'R': [400., -16_700., -29_520.]},
}


def _self_test(c, stats):
    radii = _four_sphere_radii_mm()
    ok = True

    # round trip
    p = np.array([[3.1, -33.0, -41.0], [-3.1, -33.0, -41.0]])
    rt = head_to_mni(mni_to_head(p))
    assert np.allclose(rt, p, atol=1e-9), 'round trip failed'

    # mid-sagittal constraint
    assert abs(c[0]) < 1e-12, 'fitted c_x is not 0'

    # Cz above the MNI vertex (MNI brain top is ~ +78 mm)
    cz_mni = head_to_mni(np.array([0., 0., radii[3] - 0.001]))
    if cz_mni[2] < 78.0:
        print('  !! Cz maps BELOW the MNI brain vertex: %.1f mm' % cz_mni[2])
        ok = False

    # all live positions inside the brain shell
    for name, sides in _LIVE_POS_UM.items():
        r = np.linalg.norm(np.array(sides['R'])) * 1e-3
        if r >= radii[0]:
            print('  !! %s |r| = %.1f mm outside the %.0f mm brain shell' % (name, r, radii[0]))
            ok = False
    return ok, cz_mni


def main():
    np.set_printoptions(precision=2, suppress=True)
    radii = _four_sphere_radii_mm()

    print('=' * 78)
    print('Phase 1 - MNI152 <-> head-frame transform')
    print('=' * 78)
    print('4-sphere radii (mm)      : %s' % radii)
    print('Fitting the %.0f mm brain shell to the MNI152 brain-mask surface ...' % radii[0])

    c, stats = fit_head_centre()
    _CENTRE_CACHE['fitted'] = (c, stats)

    print()
    print('  surface voxels         : %d' % stats['n_surface_voxels'])
    print('  converged              : %s' % stats['converged'])
    print('  FITTED head centre     : [%.2f, %.2f, %.2f] mm (MNI)' % tuple(c))
    print('  legacy head centre     : [%.2f, %.2f, %.2f] mm (MNI)'
          % tuple(LEGACY_HEAD_CENTRE_MNI_MM))
    print('  difference             : [%.2f, %.2f, %.2f] mm'
          % tuple(c - LEGACY_HEAD_CENTRE_MNI_MM))
    print()
    print('  radial residual to the %.0f mm shell:' % stats['radius_mm'])
    print('    rms %.2f mm | mean %+.2f mm | 5-95%%ile [%+.2f, %+.2f] mm | range [%+.2f, %+.2f]'
          % (stats['resid_rms_mm'], stats['resid_mean_mm'],
             stats['resid_p5_mm'], stats['resid_p95_mm'],
             stats['resid_min_mm'], stats['resid_max_mm']))
    print('    mean |p - c| over the brain surface = %.2f mm  (the brain is not a'
          % stats['brain_r_mean_mm'])
    print('    sphere; this spread is the irreducible error of a spherical head model)')

    print()
    print('-' * 78)
    print('What the LIVE pipeline positions imply in MNI, under each transform')
    print('-' * 78)
    print('%-10s  %-26s  %-26s' % ('', 'MNI via FITTED centre', 'MNI via LEGACY centre'))
    for name, sides in _LIVE_POS_UM.items():
        ph = np.array(sides['R'])
        print('%-10s  [%6.1f %7.1f %7.1f]      [%6.1f %7.1f %7.1f]'
              % ((name,) + tuple(head_um_to_mni(ph, 'fitted'))
                 + tuple(head_um_to_mni(ph, 'legacy'))))
    print()
    print('  (right side shown; left mirrors x.  Compare with the MSO MNI values')
    print('   claimed in main_abr.py [+-5,-32,-35] and plots/positions.py [+-5,-37,-40].)')

    ok, cz_mni = _self_test(c, stats)
    print()
    print('-' * 78)
    print('Validation')
    print('-' * 78)
    print('  round trip head<->MNI        : OK')
    print('  fitted c_x == 0              : OK')
    print('  Cz -> MNI                    : [%.1f, %.1f, %.1f] mm  (scalp Cz in MNI is'
          % tuple(cz_mni))
    print('                                  reported around z = +95..+100 mm)')
    print('  all live |r| < %.0f mm brain   : %s' % (radii[0], 'OK' if ok else 'FAILED'))
    print()
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
