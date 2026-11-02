#!/usr/bin/env python3
"""Head model geometry: 4-sphere shells, electrodes, nucleus positions, the
auditory-nerve path and the model -> head rotations.

Coordinate frame (head centred, µm):
    x   mediolateral    left (-) / right (+)
    y   anteroposterior posterior (-) / anterior (+)
    z   inferosuperior  inferior (-) / superior (+)
    origin = centre of the 4-sphere model, Cz = [0, 0, +90 mm]

Nucleus positions are stored as MNI152 coordinates and converted to the head
frame at import:
    MSO, LSO   Sitek et al. 2019 SOC anchor + ANCHOR adult-brainstem offsets
    GBC, SBC   Sitek et al. 2019 cochlear-nucleus ROI (caudal / rostral AVCN)
    MNTB       SOC anchor + Kulesza 2015 morphometry (no atlas resolves it)

    python reconstruction/recon_core/config/head_geometry.py   # print the geometry
"""

import os
import sys

# Runnable as a script: put reconstruction/ on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np


# ===========================================================================
# 4-sphere head model
# ===========================================================================
FOUR_SPHERE_RADII  = [79_000., 80_000., 85_000., 90_000.]  # um: brain,CSF,skull,scalp
FOUR_SPHERE_SIGMAS = [0.33, 1.79, 0.008, 0.3]              # S/m

# Electrodes sit 1 µm inside the scalp (FourSphereVolumeConductor needs r < r_scalp).
_ELECTRODE_INSET_UM = 1.0
SCALP_R = FOUR_SPHERE_RADII[3] - _ELECTRODE_INSET_UM       # 89_999. um

_SCALP_R = SCALP_R


def _on_scalp(v):
    """Project a vector radially onto the electrode shell, just inside the scalp."""
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v) * SCALP_R


# Scalp electrodes (10-20 system): Cz vertex, M1/M2 left/right mastoid
# (A1/A2 aliases), T7/T8 temporal, FPz anterior midline (72 deg from the
# vertex), Fz (36 deg), EL1/EL2 earlobes (azimuth -12 deg, elevation -25 deg).
_MASTOID_R = np.array([74_300., -42_200., -28_100.])

_FPZ_FROM_CZ_DEG = 72.0
_FPZ_DIR = np.array([0.0,
                     np.sin(np.radians(_FPZ_FROM_CZ_DEG)),   # anterior
                     np.cos(np.radians(_FPZ_FROM_CZ_DEG))])  # superior

ELECTRODE_POS = {
    'Cz': np.array([0., 0., SCALP_R]),
    'M1': _on_scalp(_MASTOID_R * np.array([-1., 1., 1.])),   # left mastoid
    'M2': _on_scalp(_MASTOID_R),                             # right mastoid
    'T7': _on_scalp(np.array([-90_000., 0., 0.])),           # left temporal
    'T8': _on_scalp(np.array([ 90_000., 0., 0.])),           # right temporal
    'FPz': _on_scalp(_FPZ_DIR),                              # anterior midline
}
ELECTRODE_POS['A1'] = ELECTRODE_POS['M1']                    # earlobe alias
ELECTRODE_POS['A2'] = ELECTRODE_POS['M2']

_FZ_FROM_CZ_DEG = 36.0
_EARLOBE_AZ_DEG, _EARLOBE_EL_DEG = -12.0, -25.0
_EARLOBE_R = np.array([np.cos(np.radians(_EARLOBE_EL_DEG)) * np.cos(np.radians(_EARLOBE_AZ_DEG)),
                       np.cos(np.radians(_EARLOBE_EL_DEG)) * np.sin(np.radians(_EARLOBE_AZ_DEG)),
                       np.sin(np.radians(_EARLOBE_EL_DEG))])
ELECTRODE_POS['Fz'] = _on_scalp([0.0, np.sin(np.radians(_FZ_FROM_CZ_DEG)),
                                 np.cos(np.radians(_FZ_FROM_CZ_DEG))])
ELECTRODE_POS['EL1'] = _on_scalp(_EARLOBE_R * np.array([-1., 1., 1.]))  # left earlobe
ELECTRODE_POS['EL2'] = _on_scalp(_EARLOBE_R)                             # right earlobe

# Electrodes the ABR pipeline records from.
ABR_ELECTRODES = ['Cz', 'M1', 'M2']


# ===========================================================================
# Nucleus positions
# ===========================================================================

#: Origin of the head frame in MNI mm: least-squares fit of the 79 mm brain
#: shell to the MNI152 brain-mask surface.
HEAD_CENTRE_MNI_MM = np.array([0.0, -22.5289, 12.0007])

#: Former head-frame origin, kept for comparison.
LEGACY_HEAD_CENTRE_MNI_MM = np.array([0.0, -18.3, 5.5])


def mni_mm_to_head_um(p_mni_mm, centre=None):
    """MNI152 mm to head-centred micrometres, the pipeline's native unit."""
    c = HEAD_CENTRE_MNI_MM if centre is None else np.asarray(centre, float)
    return (np.asarray(p_mni_mm, dtype=float) - c) * 1e3


# ---------------------------------------------------------------------------
# Brainstem anatomical axes in MNI, from the SOC -> IC vector.
# ---------------------------------------------------------------------------
BRAINSTEM_LATERAL_MNI = np.array([0.97576, 0.00217, 0.21875])
BRAINSTEM_DORSAL_MNI = np.array([0.0, -0.99995, 0.00987])
BRAINSTEM_ROSTRAL_MNI = np.array([-0.21875, 0.00949, 0.97575])

# ---------------------------------------------------------------------------
# Atlas coordinates per side (MNI mm). The L/R difference (<= 0.6 mm) is within
# the atlases' cross-modality spread; the pipeline uses the symmetrised values.
# ---------------------------------------------------------------------------
ATLAS_MNI_MM_RAW = {
    'MSO':  {'R': [6.683, -35.027, -43.160], 'L': [-6.815, -35.250, -43.156]},
    'LSO':  {'R': [7.157, -35.752, -43.147], 'L': [-7.289, -35.974, -43.132]},
    'GBC':  {'R': [13.644, -39.506, -44.234], 'L': [-13.341, -39.769, -44.059]},
    'SBC':  {'R': [13.356, -38.512, -44.323], 'L': [-13.087, -38.630, -44.344]},
    'SOC':  {'R': [7.082, -35.835, -41.648], 'L': [-7.196, -36.048, -41.634]},
}


def _symmetrise(pair):
    """Mirror-symmetric pair from a raw L/R measurement: |x| and y,z averaged."""
    r, l = np.asarray(pair['R'], float), np.asarray(pair['L'], float)
    x = 0.5 * (abs(r[0]) + abs(l[0]))
    yz = 0.5 * (r[1:] + l[1:])
    return {'R': np.array([x, yz[0], yz[1]]),
            'L': np.array([-x, yz[0], yz[1]])}


NUCLEUS_MNI_MM = {k: _symmetrise(v) for k, v in ATLAS_MNI_MM_RAW.items()}
SOC_MNI_MM = NUCLEUS_MNI_MM.pop('SOC')

# ---------------------------------------------------------------------------
# MNTB: Kulesza (2015) morphometry on the SOC position: MNTB_MIDLINE_DIST_MM
# from the midline, 2 mm rostral to the SOC centroid along the neuraxis.
# ---------------------------------------------------------------------------
MNTB_MIDLINE_DIST_MM = 3.900
MNTB_ROSTRAL_OFFSET_MM = 2.000


def _mntb_mni():
    out = {}
    for side in ('R', 'L'):
        p = np.asarray(SOC_MNI_MM[side], float) + \
            MNTB_ROSTRAL_OFFSET_MM * BRAINSTEM_ROSTRAL_MNI
        p[0] = np.sign(SOC_MNI_MM[side][0]) * MNTB_MIDLINE_DIST_MM
        out[side] = p
    return out


NUCLEUS_MNI_MM['MNTB'] = _mntb_mni()

# ---------------------------------------------------------------------------
# Head-frame positions (µm).
# ---------------------------------------------------------------------------
NUCLEUS_POS_UM = {name: {s: mni_mm_to_head_um(p[s]) for s in ('R', 'L')}
                  for name, p in NUCLEUS_MNI_MM.items()}

MSO_POS_UM = NUCLEUS_POS_UM['MSO']
LSO_POS_UM = NUCLEUS_POS_UM['LSO']
AVCN_POS_UM = NUCLEUS_POS_UM['GBC']       # the GBC field is the AVCN reference
SBC_POS_UM = NUCLEUS_POS_UM['SBC']
MNTB_POS_UM = NUCLEUS_POS_UM['MNTB']

# Displacement of the SBC field from the GBC field.
_SBC_DELTA_UM = SBC_POS_UM['R'] - AVCN_POS_UM['R']
SBC_ROSTRAL_OFFSET_UM = np.array([0., _SBC_DELTA_UM[1], _SBC_DELTA_UM[2]])
SBC_MEDIAL_OFFSET_UM = float(-_SBC_DELTA_UM[0])       # positive is toward the midline

#: Where each position came from, for reports and figure captions.
POSITION_SOURCE = {
    'MSO': 'AB_hybrid: Sitek SOC anchor + ANCHOR MSO offset',
    'LSO': 'AB_hybrid: Sitek SOC anchor + ANCHOR LSO offset',
    'GBC': 'A_sitek: cochlear-nucleus ROI, caudal half of the AVCN',
    'SBC': 'A_sitek: cochlear-nucleus ROI, rostral half of the AVCN',
    'MNTB': 'Sitek SOC anchor + Kulesza 2015 (3.90 mm from midline, 2 mm rostral)',
}


# ===========================================================================
# Auditory nerve
# ===========================================================================
# The modiolus is placed by stepping laterally from the cochlear-nucleus
# position along the internal auditory canal (IAC) axis, so the total fibre
# path equals the cochlear-nerve length.
from recon_core.config import anf_geometry as _ag              # noqa: E402

#: Human cochlear-nerve length, spiral ganglion to cochlear-nucleus root.
#: Gives terminal -> brainstem entry 22.0 mm and a 12.1 mm cisternal segment
#: (IAC ~9.84 mm, Marques et al. 2012; cisternal 11.5 +- 2.0 mm, Gupta et al. 2023).
COCHLEAR_NERVE_LEN_UM = 24_000.0

#: IAC axis, right side, head frame: lateral, slightly anterior and superior.
IAC_AXIS_HEAD_R = np.array([0.94, 0.28, 0.19])
IAC_AXIS_HEAD_R = IAC_AXIS_HEAD_R / np.linalg.norm(IAC_AXIS_HEAD_R)


def _cochlea_to_head_rotation(side):
    """Cochlear frame -> head frame, per side.

    The cochlear +z is the modiolar axis, pointing laterally along the IAC. The
    left transform is the sagittal reflection of the right (det = -1): the two
    cochleae are mirror-image spirals.
    """
    ez = IAC_AXIS_HEAD_R                                  # modiolar axis
    up = np.array([0., 0., 1.])
    ex = up - up.dot(ez) * ez
    ex = ex / np.linalg.norm(ex)
    ey = np.cross(ez, ex)
    R = np.stack([ex, ey, ez], axis=1)                    # columns = frame axes
    if side == 'R':
        return R
    return np.diag([-1., 1., 1.]) @ R                     # sagittal reflection


COCHLEA_TO_HEAD = {s: _cochlea_to_head_rotation(s) for s in ('R', 'L')}


def _iac_axis(side):
    """IAC axis in the head frame, mirrored per side."""
    axis = IAC_AXIS_HEAD_R.copy()
    if side == 'L':
        axis = axis * np.array([-1., 1., 1.])
    return axis


def _modiolus_pos(side, dist_um):
    """Modiolus origin: CN root + `dist_um` along the IAC axis."""
    return NUCLEUS_POS_UM['GBC'][side] + dist_um * _iac_axis(side)


def _path_from_anchor(side, dist_um):
    """Fibre path for a given anchor distance -> (vertices, total arc length)."""
    coch = _ag.representative_trajectory()
    head = (COCHLEA_TO_HEAD[side] @ coch.T).T + _modiolus_pos(side, dist_um)

    # Central run: porus (last intracochlear vertex) -> CN root.
    target = NUCLEUS_POS_UM['GBC'][side]
    n_central = 120
    tail = np.linspace(head[-1], target, n_central + 1)[1:]
    path = np.vstack([head, tail])
    total = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
    return path, total


def _solve_anchor(side, tol_um=1.0, max_iter=50):
    """Anchor distance whose total path length is COCHLEAR_NERVE_LEN_UM.

    The intracochlear trajectory is part of the nerve, so the anchor is solved
    for (secant iteration) rather than set to the full nerve length.
    """
    lo = 0.5 * COCHLEAR_NERVE_LEN_UM
    hi = COCHLEAR_NERVE_LEN_UM
    f_lo = _path_from_anchor(side, lo)[1] - COCHLEAR_NERVE_LEN_UM
    f_hi = _path_from_anchor(side, hi)[1] - COCHLEAR_NERVE_LEN_UM
    for _ in range(max_iter):
        if abs(f_hi - f_lo) < 1e-9:
            break
        mid = hi - f_hi * (hi - lo) / (f_hi - f_lo)
        f_mid = _path_from_anchor(side, mid)[1] - COCHLEAR_NERVE_LEN_UM
        lo, f_lo, hi, f_hi = hi, f_hi, mid, f_mid
        if abs(f_mid) < tol_um:
            break
    if abs(f_hi) >= tol_um:
        raise RuntimeError(
            f'_solve_anchor({side!r}) did not converge: residual '
            f'{abs(f_hi):.3f} um >= tol {tol_um:.3f} um after {max_iter} '
            f'iterations (anchor={hi:.3f} um, target total length='
            f'{COCHLEAR_NERVE_LEN_UM:.3f} um).')
    return hi


ANCHOR_DIST_UM = {s: _solve_anchor(s) for s in ('R', 'L')}
MODIOLUS_POS_UM = {s: _modiolus_pos(s, ANCHOR_DIST_UM[s]) for s in ('R', 'L')}
ANF_PATH_UM = {s: _path_from_anchor(s, ANCHOR_DIST_UM[s])[0] for s in ('R', 'L')}


# ---------------------------------------------------------------------------
# Conductivity junctions along the nerve
# ---------------------------------------------------------------------------
# Stationary potentials arise where the propagating volley crosses a
# conductivity boundary (Martin, Pratt & Schwegler 1995). Junctions are arc
# positions along ANF_PATH_UM, measured from the spiral ganglion (soma centre).

#: Spiral ganglion -> modiolar aperture (wave I boundary source with
#: --wave1-source modiolus).
MODIOLUS_FROM_GANGLION_UM = 0.0

#: Spiral ganglion -> porus acusticus (adult IAC ~9.84 mm, Marques et al. 2012).
PORUS_FROM_GANGLION_UM = 9_000.0

#: Nerve root entry zone -> cochlear-nucleus centroid. ANF_PATH_UM ends at the
#: cochlear nucleus, so the brainstem junction is this far before the path end
#: and follows NUCLEUS_POS_UM['GBC'].
CN_ENTRY_TO_GBC_UM = 2_000.0


def _soma_centre_arc_um():
    """Arc position of the spiral-ganglion soma centre along ANF_PATH_UM, µm,
    matching the layout of models/anf/anf_model.build_anf.

    params is imported lazily: it imports this module.
    """
    from recon_core.config import params as _P                # noqa: WPS433
    traj = _ag.representative_trajectory()
    seg = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    b = _ag.segment_bounds(len(traj))
    return (seg[b['peripheral'][0]:b['peripheral'][1]].sum()
            + 0.5 * _P.ANF_SOMA_DIAM)


def junction_arcs_um(side='R'):
    """{name: arc position um} of each conductivity junction along the nerve."""
    path = ANF_PATH_UM[side]
    total = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
    return {
        # Entry aperture of the IAC; used by --wave1-source modiolus.
        'modiolus': _soma_centre_arc_um() + MODIOLUS_FROM_GANGLION_UM,
        'porus': _soma_centre_arc_um() + PORUS_FROM_GANGLION_UM,
        'brainstem': total - CN_ENTRY_TO_GBC_UM,
    }


def junction_frame(name, side='R'):
    """(position µm (3,), unit tangent (3,)) of one junction on ANF_PATH_UM.

    The tangent is the local nerve direction, taken as the interface normal.
    """
    arc = junction_arcs_um(side)[name]
    path = np.asarray(ANF_PATH_UM[side], dtype=float)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if not (0.0 < arc < s[-1]):
        raise ValueError(f'junction {name!r} arc {arc:.1f} um outside the '
                         f'path (0, {s[-1]:.1f})')
    pos = np.array([np.interp(arc, s, path[:, k]) for k in range(3)])
        # Central difference over +-1 µm of arc.
    h = 1.0
    a = np.array([np.interp(max(arc - h, 0.0), s, path[:, k]) for k in range(3)])
    b = np.array([np.interp(min(arc + h, s[-1]), s, path[:, k]) for k in range(3)])
    t = b - a
    return pos, t / np.linalg.norm(t)


def anf_path_binned(n_bins, side):
    """Resample ANF_PATH_UM to n_bins equal-arc-length bins -> (n_bins+1, 3)."""
    p = ANF_PATH_UM[side]
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    s_new = np.linspace(0.0, s[-1], n_bins + 1)
    return np.stack([np.interp(s_new, s, p[:, k]) for k in range(3)], axis=1)


POSITION_SOURCE['ANF'] = (
    'Verbist 2010 cochlear frame + Potrusil 2020 micro-CT pathway, anchored by '
    f'{COCHLEAR_NERVE_LEN_UM * 1e-3:.0f} mm along the IAC axis from the '
    'Sitek cochlear-nucleus position')


# ===========================================================================
# Rotation helpers
# ===========================================================================
def _rx(deg):
    """Proper rotation about the mediolateral (head_x) axis, deg degrees."""
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[1., 0., 0.],
                     [0., c, -s],
                     [0., s,  c]])


def _rz(deg):
    """Proper rotation about the head vertical (head_z) axis, deg degrees."""
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.],
                     [s,  c, 0.],
                     [0., 0., 1.]])


# ---------------------------------------------------------------------------
# MSO: medial dendrite (+model_z) points at the midline; model_x (CF) runs
# anterior (+head_y) on both sides; head_z by the right-hand rule.
# ---------------------------------------------------------------------------
ROTATION_MSO = {
    'R': np.array([[ 0., 0.,-1.],
                   [ 1., 0., 0.],
                   [ 0.,-1., 0.]]),
    'L': np.array([[ 0., 0., 1.],
                   [ 1., 0., 0.],
                   [ 0., 1., 0.]]),
}

# ---------------------------------------------------------------------------
# AVCN: mirrored across the mid-sagittal plane (both proper rotations).
#   R:  head_x=+model_x, head_y=-model_z, head_z=+model_y
#   L:  head_x=-model_x, head_y=-model_z, head_z=-model_y
# plus the outward rostral rotation of the human cochlear nucleus (Moore/Osen),
# a rotation about head_z by -/+AVCN_ROSTRAL_TILT_DEG.
# ---------------------------------------------------------------------------
AVCN_ROSTRAL_TILT_DEG = 32.5

_AVCN_BASE = {
    'R': np.array([[ 1., 0.,  0.],
                   [ 0., 0., -1.],
                   [ 0., 1.,  0.]]),
    'L': np.array([[-1., 0.,  0.],
                   [ 0., 0., -1.],
                   [ 0., -1., 0.]]),
}


def build_avcn_rotation(tilt_deg):
    """Side-specific AVCN model to head rotation with an outward rostral tilt."""
    return {'R': _rz(-tilt_deg) @ _AVCN_BASE['R'],
            'L': _rz(+tilt_deg) @ _AVCN_BASE['L']}


ROTATION_AVCN = build_avcn_rotation(AVCN_ROSTRAL_TILT_DEG)

# ---------------------------------------------------------------------------
# GBC axon direction: the chord from each side's GBC to the contralateral MNTB
# (medial, rostral, dorsal). Kept per side because ROTATION_AVCN['L'] mirrors
# both x and z. The model-frame target LFPy aligns each axon to is derived from it.
# ---------------------------------------------------------------------------
_CONTRA = {'R': 'L', 'L': 'R'}


def _gbc_to_contra_mntb(side):
    """Unit head-frame chord from the GBC of `side` to the contralateral MNTB."""
    d = NUCLEUS_POS_UM['MNTB'][_CONTRA[side]] - NUCLEUS_POS_UM['GBC'][side]
    return d / np.linalg.norm(d)


AVCN_AXON_HEAD_DIR = {s: _gbc_to_contra_mntb(s) for s in ('R', 'L')}
AVCN_AXON_TARGET = {}
for _s in ('R', 'L'):
    _t = ROTATION_AVCN[_s].T @ AVCN_AXON_HEAD_DIR[_s]
    AVCN_AXON_TARGET[_s] = _t / np.linalg.norm(_t)

#: Straight-line GBC -> contralateral MNTB distance, µm.
GBC_MNTB_CHORD_UM = float(np.linalg.norm(
    NUCLEUS_POS_UM['MNTB']['L'] - NUCLEUS_POS_UM['GBC']['R']))

#: Axon path length / chord, for a gently curving central tract (1.15-1.25).
GBC_MNTB_TORTUOSITY = 1.15

#: Modelled GBC axon length, µm.
GBC_AXON_LENGTH_UM = GBC_MNTB_CHORD_UM * GBC_MNTB_TORTUOSITY

# ---------------------------------------------------------------------------
# MNTB (Kulesza 2014/2015). Model frame:
#   model_x = pre-calyx trapezoid-body fibre (mediolateral)
#   model_y = principal-cell axon, rostrocaudal
#   model_z = principal dendritic long axis
# Right: identity; left: sagittal mirror diag(-1,1,-1). The dendritic long axis
# is ~76 deg from horizontal (mean of transverse 73 and coronal 79 deg),
# applied as a tilt about head_x.
# ---------------------------------------------------------------------------
MNTB_LONGAXIS_FROM_HORIZ_DEG = 76.0

_MNTB_BASE = {
    'R': np.eye(3),
    'L': np.diag([-1., 1., -1.]),
}
_MNTB_DEND_TILT_FROM_VERTICAL = -(90.0 - MNTB_LONGAXIS_FROM_HORIZ_DEG)   # ~ -14 deg
ROTATION_MNTB = {s: _rx(_MNTB_DEND_TILT_FROM_VERTICAL) @ _MNTB_BASE[s]
                 for s in _MNTB_BASE}

# ---------------------------------------------------------------------------
# LSO: model axes mapped onto the brainstem axes (head and MNI axes coincide):
#   model x  dorsal                   -> BRAINSTEM_DORSAL_MNI
#   model y  rostral, the dendrites   -> BRAINSTEM_ROSTRAL_MNI
#   model z  medial, the tonotopy     -> -BRAINSTEM_LATERAL_MNI  (right side)
# ---------------------------------------------------------------------------
def _nearest_rotation(m):
    """Nearest proper rotation to m (polar decomposition); the BRAINSTEM_*
    vectors are orthonormal only to ~1e-5."""
    u, _s, vt = np.linalg.svd(np.asarray(m, float))
    r = u @ vt
    if np.linalg.det(r) < 0:
        raise ValueError('brainstem axes are left-handed; check their signs')
    return r


# The left LSO uses a mirrored morphology (model x negated), so its rotation is
# the sagittal mirror of the right one.
_SAGITTAL_MIRROR = np.diag([-1., 1., 1.])
ROTATION_LSO = {'R': _nearest_rotation(np.column_stack(
    [BRAINSTEM_DORSAL_MNI, BRAINSTEM_ROSTRAL_MNI, -BRAINSTEM_LATERAL_MNI]))}
ROTATION_LSO['L'] = _SAGITTAL_MIRROR @ ROTATION_LSO['R'] @ _SAGITTAL_MIRROR


# ===========================================================================
if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True)
    print('4-sphere radii (um) : %s' % FOUR_SPHERE_RADII)
    print('sigmas (S/m)        : %s' % FOUR_SPHERE_SIGMAS)
    print('electrode shell (um): %.1f' % SCALP_R)
    print()
    for k in ('Cz', 'M1', 'M2', 'T7', 'T8'):
        print('  %-3s %s  |r| = %.1f' % (k, ELECTRODE_POS[k],
                                         np.linalg.norm(ELECTRODE_POS[k])))
    print()
    print('%-6s %-26s %-26s %s' % ('', 'R (um)', 'L (um)', '|r| mm'))
    for name, pos in NUCLEUS_POS_UM.items():
        print('%-6s %-26s %-26s %.1f'
              % (name, pos['R'], pos['L'], np.linalg.norm(pos['R']) * 1e-3))
    print()
    for name, rot in (('MSO', ROTATION_MSO), ('AVCN', ROTATION_AVCN),
                      ('MNTB', ROTATION_MNTB), ('LSO', ROTATION_LSO)):
        print('%-5s det R=%+.3f  det L=%+.3f'
              % (name, np.linalg.det(rot['R']), np.linalg.det(rot['L'])))
