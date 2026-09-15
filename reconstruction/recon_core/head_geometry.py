#!/usr/bin/env python3
"""Single source of truth for the ABR head model: geometry, electrodes, nucleus
positions and model to head rotations.

Every ABR entry point (main_abr*.py) and every figure script imports from here.
Nothing outside this module should declare these constants again.

Coordinate frame (head centred, micrometres, the lfpykit native unit):
    x   mediolateral    left (-) / right (+)
    y   anteroposterior posterior (-) / anterior (+)
    z   inferosuperior  inferior (-) / superior (+)
    origin = centre of the concentric 4-sphere head model
    Cz (vertex) = [0, 0, +90 mm]

The five nucleus positions come from human atlases (ABR_reconstruction/atlas/,
RESULTS/atlas_validation/):

    MSO   Sitek 2019 SOC anchor (MNI152) + ANCHOR adult-brainstem MSO offset
    LSO   Sitek 2019 SOC anchor (MNI152) + ANCHOR adult-brainstem LSO offset
    GBC   Sitek 2019 cochlear-nucleus ROI, caudal half of the AVCN
    SBC   Sitek 2019 cochlear-nucleus ROI, rostral half of the AVCN
    MNTB  no atlas measures it (ANCHOR annotates the trapezoid body, not the
          nucleus), so the Kulesza 2015 morphometry is re-anchored onto the
          atlas-derived SOC position

They are stored as MNI coordinates and converted to the head frame at import,
so editing an atlas value or HEAD_CENTRE_MNI_MM propagates everywhere.

ATLAS_CANDIDATES loads the full comparison table for reporting only; it is not
what the pipeline reads.
"""

import json
import os
import sys

# This module doubles as a script (python recon_core/head_geometry.py prints the
# live geometry), and running a file inside the package puts recon_core/ on
# sys.path rather than reconstruction/, so point at the package root first.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from recon_core.bootstrap import PACKAGE_ROOT
from recon_core import paths

# The atlas derivation and the LSO axon builder live outside this package and
# are imported lazily below, so they need their own directories on sys.path.
_ATLAS_DIR = os.path.join(PACKAGE_ROOT, 'ABR_reconstruction', 'atlas')


# ===========================================================================
# 4-sphere head model
# ===========================================================================
FOUR_SPHERE_RADII  = [79_000., 80_000., 85_000., 90_000.]  # um: brain,CSF,skull,scalp
FOUR_SPHERE_SIGMAS = [0.33, 1.79, 0.008, 0.3]              # S/m

# Electrodes must lie strictly inside the scalp (FourSphereVolumeConductor
# requires r < r_scalp), so they sit 1 um under it. Derived from the radii.
_ELECTRODE_INSET_UM = 1.0
SCALP_R = FOUR_SPHERE_RADII[3] - _ELECTRODE_INSET_UM       # 89_999. um

# Alias kept because main_abr.py exported this name.
_SCALP_R = SCALP_R


def _on_scalp(v):
    """Project a vector radially onto the electrode shell, just inside the scalp."""
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v) * SCALP_R


# Scalp electrode positions (um). Clinical 10-20 system: Cz = vertex,
# M1 = left mastoid, M2 = right mastoid. Standard ABR derivations are Cz-M1
# (ipsilateral left) and Cz-M2 (ipsilateral right), vertex positive upward.
# The ground electrode (Fz) is not modelled; it has no effect in a 4-sphere
# volume conductor.
# A1/A2 are the earlobe labels for the same two sites, provided so the
# simulation and the figure scripts can use either name. T7/T8 are extra
# lateral sites used only by plots/electrodes.py.
_MASTOID_R = np.array([74_300., -42_200., -28_100.])

ELECTRODE_POS = {
    'Cz': np.array([0., 0., SCALP_R]),
    'M1': _on_scalp(_MASTOID_R * np.array([-1., 1., 1.])),   # left mastoid
    'M2': _on_scalp(_MASTOID_R),                             # right mastoid
    'T7': _on_scalp(np.array([-90_000., 0., 0.])),           # left temporal
    'T8': _on_scalp(np.array([ 90_000., 0., 0.])),           # right temporal
}
ELECTRODE_POS['A1'] = ELECTRODE_POS['M1']                    # earlobe alias
ELECTRODE_POS['A2'] = ELECTRODE_POS['M2']

# The three electrodes the ABR pipeline actually records from.
ABR_ELECTRODES = ['Cz', 'M1', 'M2']


# ===========================================================================
# Nucleus positions
# ===========================================================================
# The primary quantity is the atlas-measured MNI152 coordinate of each
# generator; the head-frame position the ABR pipeline consumes is derived from
# it here, so changing an atlas value or the head-frame origin propagates.
#
# Source per generator (see RESULTS/atlas_validation/ and atlas/):
#   MSO, LSO   ANCHOR subnucleus offsets carried on the Sitek SOC anchor
#              ("AB_hybrid"). The only source that separates MSO from LSO.
#   GBC, SBC   Sitek cochlear-nucleus ROI, rostral half = AVCN, split
#              rostral/caudal into the spherical and globular cell fields
#              ("A_sitek"). ANCHOR does not annotate the cochlear nuclei.
#   MNTB       No atlas answer exists, since ANCHOR annotates the trapezoid
#              body but not the trapezoid nucleus. Derived below from the Sitek
#              SOC anchor plus the Kulesza (2015) morphometry.
# ---------------------------------------------------------------------------

#: Origin of the head frame, in MNI mm. Least-squares fit of the model's 79 mm
#: brain shell to the MNI152 brain-mask surface; regenerate with
#: python ABR_reconstruction/atlas/mni_head_transform.py.
HEAD_CENTRE_MNI_MM = np.array([0.0, -22.5289, 12.0007])

#: The undocumented shift the code used to quote. Kept for comparison only.
LEGACY_HEAD_CENTRE_MNI_MM = np.array([0.0, -18.3, 5.5])


def mni_mm_to_head_um(p_mni_mm, centre=None):
    """MNI152 mm to head-centred micrometres, the pipeline's native unit."""
    c = HEAD_CENTRE_MNI_MM if centre is None else np.asarray(centre, float)
    return (np.asarray(p_mni_mm, dtype=float) - c) * 1e3


# ---------------------------------------------------------------------------
# Brainstem anatomical axes in MNI, measured from the route-A SOC to IC vector.
# Used to apply the MNTB's literature offset along the true neuraxis rather
# than along a head axis.
# ---------------------------------------------------------------------------
BRAINSTEM_LATERAL_MNI = np.array([0.97576, 0.00217, 0.21875])
BRAINSTEM_DORSAL_MNI = np.array([0.0, -0.99995, 0.00987])
BRAINSTEM_ROSTRAL_MNI = np.array([-0.21875, 0.00949, 0.97575])

# ---------------------------------------------------------------------------
# Atlas measurements, exactly as reported per side (MNI mm). The atlases show a
# real left/right difference of <= 0.6 mm, well inside their own 1.3-1.8 mm
# cross-modality spread. Everything downstream assumes mirror symmetry, so the
# values used are the symmetrised ones below; the raw pair is kept so the
# asymmetry stays visible.
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
# MNTB: no atlas measurement exists, so the model's Kulesza (2015) morphometry
# is re-anchored onto the new SOC position: ~400 um from the midline, ~2 mm
# rostral to the SOC centroid. The rostral step follows the measured neuraxis
# (BRAINSTEM_ROSTRAL_MNI), which is what "rostral" anatomically means.
# Anchoring on the MSO instead of the SOC centroid would move it <1.5 mm.
# ---------------------------------------------------------------------------
MNTB_MIDLINE_DIST_MM = 0.400
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
# Head-frame positions consumed by the ABR pipeline (um). Derived, not typed.
# ---------------------------------------------------------------------------
NUCLEUS_POS_UM = {name: {s: mni_mm_to_head_um(p[s]) for s in ('R', 'L')}
                  for name, p in NUCLEUS_MNI_MM.items()}

MSO_POS_UM = NUCLEUS_POS_UM['MSO']
LSO_POS_UM = NUCLEUS_POS_UM['LSO']
AVCN_POS_UM = NUCLEUS_POS_UM['GBC']       # the GBC field is the AVCN reference
SBC_POS_UM = NUCLEUS_POS_UM['SBC']
MNTB_POS_UM = NUCLEUS_POS_UM['MNTB']

# The SBC field's displacement from the GBC field. Both fields are measured
# separately, so these are diagnostics only (still exported because
# main_abr_avcn.py imports the names).
_SBC_DELTA_UM = SBC_POS_UM['R'] - AVCN_POS_UM['R']
SBC_ROSTRAL_OFFSET_UM = np.array([0., _SBC_DELTA_UM[1], _SBC_DELTA_UM[2]])
SBC_MEDIAL_OFFSET_UM = float(-_SBC_DELTA_UM[0])       # positive is toward the midline

#: Where each position came from, for reports and figure captions.
POSITION_SOURCE = {
    'MSO': 'AB_hybrid: Sitek SOC anchor + ANCHOR MSO offset',
    'LSO': 'AB_hybrid: Sitek SOC anchor + ANCHOR LSO offset',
    'GBC': 'A_sitek: cochlear-nucleus ROI, caudal half of the AVCN',
    'SBC': 'A_sitek: cochlear-nucleus ROI, rostral half of the AVCN',
    'MNTB': 'Sitek SOC anchor + Kulesza 2015 (0.40 mm from midline, 2 mm rostral)',
}


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
# MSO: model axes to head axes.
#
# MSO HOC (from LAYER_BOUNDARIES): medial dend z>0, lateral dend z<0.
#
# head_x: mediolateral
#   Right MSO: medial dend (+model_z) points at the midline, so -head_x
#   Left  MSO: medial dend (+model_z) points at the midline, so +head_x
# head_y: anteroposterior (tonotopic: low CF posterior, high CF anterior).
#   model_x increases = higher CF = more anterior, so head_y = +model_x on both
#   sides (tonotopy runs the same anatomical direction bilaterally).
# head_z: fixed by the right-hand rule (det = +1).
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
# AVCN: side-specific (mirrored across the mid-sagittal plane) so the GBC axon,
# aligned ventromedially in the model frame by set_rotations
# (AXON_TARGET = [-1,0,-1]), points at the contralateral MNTB and so crosses
# the midline on both sides. Follows the MSO convention: flip head_x and head_z
# between sides, keep head_y (both proper rotations, det = +1).
#
#   R:  head_x=+model_x, head_y=-model_z, head_z=+model_y   (Rx(+90 deg))
#   L:  head_x=-model_x, head_y=-model_z, head_z=-model_y   (sagittal mirror)
#
# Axon crossing check (AXON_TARGET=[-1,0,-1] in head frame, before tilt):
#   R gives [-1,+1,0], head_x=-1, toward the midline from the right AVCN
#   L gives [+1,+1,0], head_x=+1, toward the midline from the left AVCN
#
# Anatomical refinement (Moore/Osen): the human CN rostral pole is rotated
# outward ~30-35 deg from the neuraxis. A rotation about head_z by
# -/+AVCN_ROSTRAL_TILT_DEG swings the rostral end (+head_y) toward +head_x on
# the right and -head_x on the left. The midline crossing survives any tilt;
# set the angle to 0 (or --avcn-tilt-deg 0) to recover the untilted mapping.
# ---------------------------------------------------------------------------
AVCN_ROSTRAL_TILT_DEG = 32.5   # human CN outward rostral rotation; 0 disables

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
# GBC axon direction. The globular bushy cell's axon crosses the midline to the
# contralateral MNTB calyx, so in head coordinates it runs mediolaterally
# toward the midline, the same trapezoid-body direction as the MNTB pre-calyx
# fibre below, which keeps the two nuclei's crossing volleys consistent.
#
# The model-frame target LFPy aligns each cell's axon to is derived from that
# head-frame requirement rather than typed in, so it cannot drift from the
# rotation. One vector serves both sides: the side-specific rotation already
# mirrors it, which is why laterality is carried by the input population
# (X_pops) and not by flipping the geometry.
# ---------------------------------------------------------------------------
AVCN_AXON_HEAD_DIR = np.array([-1., 0., 0.])          # right side; left mirrors
AVCN_AXON_TARGET = ROTATION_AVCN['R'].T @ AVCN_AXON_HEAD_DIR
AVCN_AXON_TARGET = AVCN_AXON_TARGET / np.linalg.norm(AVCN_AXON_TARGET)

# ---------------------------------------------------------------------------
# MNTB: side-specific, built from human MNTB morphometry (Kulesza 2014/2015).
#
# MNTB model frame (mntb_model_active.hoc / calyx_model.hoc):
#   model_x = pre-calyx axon, the crossing trapezoid-body fibre (the dominant
#             generator, a giant mediolateral axial current)
#   model_y = principal-cell axon (efferent to MSO/LSO/SPN), rostrocaudal
#   model_z = principal dendritic long axis (bipolar tufts)
#
# Base map:
#   RIGHT: identity, calyx model_x to +head_x, dendrite model_z to head_z.
#   LEFT : sagittal mirror diag(-1,1,-1), calyx to -head_x.
# So the calyx crossing dipole is mediolateral and mirrored across sides, sums
# coherently bilaterally, and matches the AVCN GBC axon orientation.
#
# The MNTB principal long (dendritic) axis is ~73+-5 deg (transverse) / 79+-7
# deg (coronal) from horizontal, dendrites perpendicular to the mediolateral
# trapezoid-body fibres, and the whole SOC has a slight superior-rostral tilt.
# All three come from a single tilt about head_x, which leaves the dominant
# mediolateral calyx dipole invariant.
# ---------------------------------------------------------------------------
MNTB_LONGAXIS_FROM_HORIZ_DEG = 76.0    # mean of transverse 73 + coronal 79 (Kulesza);
                                       # subsumes the SOC superior-rostral tilt

_MNTB_BASE = {
    'R': np.eye(3),
    'L': np.diag([-1., 1., -1.]),
}
_MNTB_DEND_TILT_FROM_VERTICAL = -(90.0 - MNTB_LONGAXIS_FROM_HORIZ_DEG)   # ~ -14 deg
ROTATION_MNTB = {s: _rx(_MNTB_DEND_TILT_FROM_VERTICAL) @ _MNTB_BASE[s]
                 for s in _MNTB_BASE}

# ---------------------------------------------------------------------------
# LSO: the extended LSO axon runs along AXON_DIR (rostro-dorsal in the model
# frame, build_lso_axon.py) and the ascending lateral lemniscus climbs toward
# the IC at +head_z. Both sides map AXON_DIR to +head_z (axons ascend
# bilaterally). The mediolateral axis is mirrored per side so that "medial"
# (+model_z) always points at the midline:
#   Right LSO (at +head_x): medial to -head_x   (s = -1)
#   Left  LSO (at -head_x): medial to +head_x   (s = +1)
# So head_z (axon toward IC) is identical on both sides while head_x and head_y
# mirror. This differs from the MSO (which mirrors head_x and head_z and keeps
# head_y) because here head_z is the axon axis and must not flip. It keeps
# tonotopy anatomically consistent bilaterally (lateral low CF, medial high).
# ---------------------------------------------------------------------------
def lso_rotation(a, side):
    """Proper rotation (det=+1): R @ a = +head_z; R @ model_z = -/+head_x for R/L.

    ey = ez x ex with orthonormal {ex, ez} is right-handed, so det = +1 by
    construction.
    """
    ez = np.asarray(a, float)
    ez = ez / np.linalg.norm(ez)
    s = -1.0 if side == 'R' else 1.0                      # mirror mediolateral per side
    ex = np.array([0., 0., 1.]) - np.array([0., 0., 1.]).dot(ez) * ez   # model z perp ez
    if np.linalg.norm(ex) < 1e-6:                         # fallback when a is parallel to z
        ex = np.array([1., 0., 0.]) - np.array([1., 0., 0.]).dot(ez) * ez
    ex = s * ex / np.linalg.norm(ex)
    ey = np.cross(ez, ex)
    return np.vstack([ex, ey, ez])


def _lso_axon_dir():
    """Model-frame LSO axon unit vector, from models/mso/build_lso_axon.py."""
    sys.path.insert(0, paths.MSO_MODELS_DIR)
    from build_lso_axon import AXON_DIR       # noqa: WPS433
    return AXON_DIR


LSO_AXON_DIR = _lso_axon_dir()

# The left LSO uses a mirrored morphology (models/mso/lso_model_active*_left.hoc),
# so its rotation is the exact sagittal mirror of the right one: R_L = M R_R M.
# That is a proper rotation (det = +1) and keeps all three requirements at once:
# the lemniscal axon on head +z, tonotopy on head -/+x, and a mirror-symmetric
# dendritic tilt.
_SAGITTAL_MIRROR = np.diag([-1., 1., 1.])
ROTATION_LSO = {'R': lso_rotation(LSO_AXON_DIR, 'R')}
ROTATION_LSO['L'] = _SAGITTAL_MIRROR @ ROTATION_LSO['R'] @ _SAGITTAL_MIRROR


# ===========================================================================
# MNI <-> head frame, and the atlas-derived candidate positions
# ===========================================================================
def mni_to_head(p_mni_mm, source='fitted'):
    """MNI152 mm to head-centred mm. See atlas/mni_head_transform.py."""
    sys.path.insert(0, _ATLAS_DIR)
    from mni_head_transform import mni_to_head as _f    # noqa: WPS433
    return _f(p_mni_mm, source)


def head_to_mni(p_head_mm, source='fitted'):
    """Head-centred mm to MNI152 mm. See atlas/mni_head_transform.py."""
    sys.path.insert(0, _ATLAS_DIR)
    from mni_head_transform import head_to_mni as _f    # noqa: WPS433
    return _f(p_head_mm, source)


_CANDIDATES_JSON = os.path.join(paths.RESULTS_DIR, 'atlas_validation',
                                'atlas_candidates.json')


def load_atlas_candidates():
    """Atlas-derived candidate positions, or {} if the verification has not run.

    Produced by ABR_reconstruction/atlas/compare_positions.py. Not used by the
    pipeline, reported only, so adopting a candidate set stays a deliberate
    one-line edit to the constants above.
    """
    if not os.path.exists(_CANDIDATES_JSON):
        return {}
    with open(_CANDIDATES_JSON) as f:
        return json.load(f)


ATLAS_CANDIDATES = load_atlas_candidates()


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
