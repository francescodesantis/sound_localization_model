#!/usr/bin/env python3
"""Check head_geometry.py: unchanged head model, correctly derived positions.

Two separate things are asserted.

1. The head model is unchanged. The 4-sphere radii and conductivities, the
   electrode positions and all four model to head ROTATION matrices must still
   equal the pre-refactor literals, transcribed here from
       main_abr.py:85-131,133-159   main_abr_lso.py:61-114
       main_abr_avcn.py:65-145      main_abr_mntb.py:60-125
       plots/electrodes.py
   Only the nucleus positions were meant to move.

2. The positions are derived, not typed. Each head-frame position must be
   exactly its atlas MNI coordinate mapped through the head-frame origin, the
   L/R pair must be mirror-symmetric, the MNTB must satisfy the Kulesza
   constraints it is built from, and everything must sit inside the brain shell.

The superseded positions are printed alongside, with the displacement.

    python ABR_reconstruction/test_head_geometry.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recon_core import head_geometry as hg           # noqa: E402


# Positions the pipeline used before the atlas verification (head frame, um,
# right side). None was atlas-derived. Reported, not asserted.
SUPERSEDED_POS_UM = {
    'MSO':  [5_000., -18_700., -29_520.],
    'LSO':  [9_000., -20_500., -30_000.],
    'GBC':  [10_000., -38_000., -35_000.],
    'SBC':  [9_400., -36_500., -35_000.],
    'MNTB': [400., -16_700., -29_520.],
}

FAILURES = []


def _on_scalp(v, r):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v) * r


def check(name, got, want, atol=0.0):
    got, want = np.asarray(got, float), np.asarray(want, float)
    ok = np.allclose(got, want, rtol=0.0, atol=atol)
    print('  %-38s %s' % (name, 'OK' if ok else 'MISMATCH'))
    if not ok:
        FAILURES.append((name, got, want))
    return ok


def _rx(deg):
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])


def _rz(deg):
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def check_head_model():
    print('1. head model vs the pre-refactor literals')
    print('-' * 66)

    check('FOUR_SPHERE_RADII', hg.FOUR_SPHERE_RADII, [79_000., 80_000., 85_000., 90_000.])
    check('FOUR_SPHERE_SIGMAS', hg.FOUR_SPHERE_SIGMAS, [0.33, 1.79, 0.008, 0.3])
    check('SCALP_R', hg.SCALP_R, 89_999.)
    check('_SCALP_R alias', hg._SCALP_R, 89_999.)

    r = 89_999.
    check('ELECTRODE_POS[Cz]', hg.ELECTRODE_POS['Cz'], [0., 0., r])
    check('ELECTRODE_POS[M1]', hg.ELECTRODE_POS['M1'],
          _on_scalp([-74_300., -42_200., -28_100.], r))
    check('ELECTRODE_POS[M2]', hg.ELECTRODE_POS['M2'],
          _on_scalp([74_300., -42_200., -28_100.], r))
    check('ELECTRODE_POS[T7]', hg.ELECTRODE_POS['T7'], _on_scalp([-90_000., 0., 0.], r))
    check('ELECTRODE_POS[T8]', hg.ELECTRODE_POS['T8'], _on_scalp([90_000., 0., 0.], r))
    check('A1 aliases M1', hg.ELECTRODE_POS['A1'], hg.ELECTRODE_POS['M1'])
    check('A2 aliases M2', hg.ELECTRODE_POS['A2'], hg.ELECTRODE_POS['M2'])

    check('ROTATION_MSO[R]', hg.ROTATION_MSO['R'],
          [[0., 0., -1.], [1., 0., 0.], [0., -1., 0.]])
    check('ROTATION_MSO[L]', hg.ROTATION_MSO['L'],
          [[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])

    check('AVCN_ROSTRAL_TILT_DEG', hg.AVCN_ROSTRAL_TILT_DEG, 32.5)
    base = {'R': np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]),
            'L': np.array([[-1., 0., 0.], [0., 0., -1.], [0., -1., 0.]])}
    check('ROTATION_AVCN[R]', hg.ROTATION_AVCN['R'], _rz(-32.5) @ base['R'], atol=1e-15)
    check('ROTATION_AVCN[L]', hg.ROTATION_AVCN['L'], _rz(+32.5) @ base['L'], atol=1e-15)

    check('MNTB_LONGAXIS_FROM_HORIZ_DEG', hg.MNTB_LONGAXIS_FROM_HORIZ_DEG, 76.0)
    mbase = {'R': np.eye(3), 'L': np.diag([-1., 1., -1.])}
    tilt = -(90.0 - 76.0)
    check('ROTATION_MNTB[R]', hg.ROTATION_MNTB['R'], _rx(tilt) @ mbase['R'], atol=1e-15)
    check('ROTATION_MNTB[L]', hg.ROTATION_MNTB['L'], _rx(tilt) @ mbase['L'], atol=1e-15)

    from recon_core import paths
    sys.path.insert(0, paths.MSO_MODELS_DIR)
    from build_lso_axon import AXON_DIR                              # noqa: WPS433

    def lso_rot_right(a):
        ez = np.asarray(a, float)
        ez = ez / np.linalg.norm(ez)
        ex = np.array([0., 0., 1.]) - np.array([0., 0., 1.]).dot(ez) * ez
        if np.linalg.norm(ex) < 1e-6:
            ex = np.array([1., 0., 0.]) - np.array([1., 0., 0.]).dot(ez) * ez
        ex = -ex / np.linalg.norm(ex)
        return np.vstack([ex, np.cross(ez, ex), ez])

    # The left cell uses a mirrored morphology, so its rotation is the sagittal
    # mirror of the right one, R_L = M R_R M, not a per-side axis flip. That is
    # what keeps the lemniscal axon on head +z on both sides while tonotopy and
    # the dendritic tilt stay mirror-symmetric.
    mirror = np.diag([-1., 1., 1.])
    rot_r = lso_rot_right(AXON_DIR)
    check('LSO_AXON_DIR', hg.LSO_AXON_DIR, AXON_DIR)
    check('ROTATION_LSO[R]', hg.ROTATION_LSO['R'], rot_r, atol=1e-15)
    check('ROTATION_LSO[L]', hg.ROTATION_LSO['L'], mirror @ rot_r @ mirror,
          atol=1e-15)

    for nm, rot in (('MSO', hg.ROTATION_MSO), ('AVCN', hg.ROTATION_AVCN),
                    ('MNTB', hg.ROTATION_MNTB), ('LSO', hg.ROTATION_LSO)):
        for side in ('R', 'L'):
            check('det(ROTATION_%s[%s]) == +1' % (nm, side),
                  np.linalg.det(rot[side]), 1.0, atol=1e-12)


def check_positions():
    print()
    print('2. positions derived from the atlas MNI values')
    print('-' * 66)

    for name, mni in hg.NUCLEUS_MNI_MM.items():
        for side in ('R', 'L'):
            want = (np.asarray(mni[side], float) - hg.HEAD_CENTRE_MNI_MM) * 1e3
            check('%s %s derived from its MNI value' % (name, side),
                  hg.NUCLEUS_POS_UM[name][side], want, atol=1e-9)

    for name, pos in hg.NUCLEUS_POS_UM.items():
        check('%s L mirrors R' % name, pos['L'],
              pos['R'] * np.array([-1., 1., 1.]), atol=1e-9)

    mntb = hg.NUCLEUS_MNI_MM['MNTB']
    check('MNTB |x| == MNTB_MIDLINE_DIST_MM',
          abs(mntb['R'][0]), hg.MNTB_MIDLINE_DIST_MM, atol=1e-9)
    step = np.asarray(mntb['R'], float) - np.asarray(hg.SOC_MNI_MM['R'], float)
    check('MNTB rostral step along the neuraxis', step[1:],
          (hg.MNTB_ROSTRAL_OFFSET_MM * hg.BRAINSTEM_ROSTRAL_MNI)[1:], atol=1e-9)

    d = hg.SBC_POS_UM['R'] - hg.AVCN_POS_UM['R']
    check('SBC_ROSTRAL_OFFSET_UM', hg.SBC_ROSTRAL_OFFSET_UM, [0., d[1], d[2]], atol=1e-9)
    check('SBC_MEDIAL_OFFSET_UM', hg.SBC_MEDIAL_OFFSET_UM, -d[0], atol=1e-9)

    print()
    r_brain = hg.FOUR_SPHERE_RADII[0]
    for name, pos in hg.NUCLEUS_POS_UM.items():
        rr = float(np.linalg.norm(pos['R']))
        ok = rr < r_brain
        print('  %-38s %s  (|r| = %.1f mm < %.0f)'
              % ('%s inside the brain shell' % name, 'OK' if ok else 'FAILED',
                 rr * 1e-3, r_brain * 1e-3))
        if not ok:
            FAILURES.append((name, rr, r_brain))

    P = hg.NUCLEUS_POS_UM
    for label, good in (
            ('MSO medial to LSO', abs(P['MSO']['R'][0]) < abs(P['LSO']['R'][0])),
            ('MNTB most medial', abs(P['MNTB']['R'][0]) < abs(P['MSO']['R'][0])),
            ('AVCN lateral to the SOC', abs(P['GBC']['R'][0]) > abs(P['MSO']['R'][0])),
            ('AVCN inferior to the SOC', P['GBC']['R'][2] < P['MSO']['R'][2]),
            ('SBC rostral to GBC', P['SBC']['R'][1] > P['GBC']['R'][1])):
        print('  %-38s %s' % (label, 'OK' if good else 'FAILED'))
        if not good:
            FAILURES.append((label, None, None))


def report_moves():
    print()
    print('superseded -> current (right side, head frame, mm)')
    print('-' * 66)
    for name, old in SUPERSEDED_POS_UM.items():
        new = hg.NUCLEUS_POS_UM[name]['R'] * 1e-3
        old = np.asarray(old, float) * 1e-3
        d = new - old
        print('  %-5s [%6.2f %7.2f %7.2f] -> [%6.2f %7.2f %7.2f]  |d| = %5.2f mm'
              % (name, old[0], old[1], old[2], new[0], new[1], new[2],
                 np.linalg.norm(d)))
        print('        %s' % hg.POSITION_SOURCE[name])


def main():
    check_head_model()
    check_positions()
    report_moves()
    print()
    print('-' * 66)
    if FAILURES:
        print('%d FAILURE(S):' % len(FAILURES))
        for name, got, want in FAILURES:
            print('  %s\n    got  %s\n    want %s' % (name, got, want))
        return 1
    print('head model unchanged; positions correctly derived from the atlas values')
    return 0


if __name__ == '__main__':
    sys.exit(main())
