#!/usr/bin/env python3
"""The cochlear coordinate frame and the auditory-nerve fibre trajectory.

WHY THIS IS NOT IN head_geometry.py.  Every other generator is a compact nucleus
whose position is one atlas MNI coordinate.  The auditory nerve is a ~25 mm cable
whose distal half lives inside the cochlea, which no brain atlas resolves — the
spiral ganglion sits in the modiolus, inside the petrous temporal bone, and is not
brain tissue.  Its geometry therefore comes from a different literature with a
different coordinate system, and it gets its own module.  head_geometry.py imports
this one and exposes the derived head-frame path alongside the nucleus positions.

THE FRAME (Verbist et al. 2010, "Consensus panel on a cochlear coordinate system
applicable in histologic, physiologic, and radiologic studies of the human
cochlea", PMID 20147866) is cylindrical:

    z       the modiolar axis (the cochlear spiral's rotation axis)
    theta   azimuth along the spiral, ZERO at the centre of the round window,
            increasing from the basal turn toward the apex
    r       radial distance from the modiolar axis

Landmark radii used below:
    r_BM / r_IHC   organ of Corti — the peripheral dendritic terminals
    r_RC           Rosenthal's canal — the spiral ganglion somata
    r_ST           scala tympani centre — where the near-field CAP electrode goes

THE TRAJECTORY.  Fibre pathways are the representative tonotopic pathway of
Potrusil et al., Hearing Research 2020 (PMID 32535276), which reconstructed 30
tonotopically aligned pathways spanning eight octaves (11500-40 Hz) from micro-CT
at 3 um voxel resolution.  ONE representative pathway is used by default; the
other 29 are what `--n-cf-bands > 1` consumes and need no new sourcing.

Ordering is PERIPHERAL -> CENTRAL throughout, on both sides.  That is the
direction the action potential travels, and every downstream tangent, axial
current sign and propagation-direction test depends on it.
"""

import numpy as np

# --- Verbist landmark radii (um, from the modiolar axis) --------------------
R_ORGAN_OF_CORTI_UM = 4_200.0    # r_BM / r_IHC: peripheral dendritic terminals
R_ROSENTHAL_UM = 2_400.0         # r_RC: spiral ganglion somata
R_SCALA_TYMPANI_UM = 3_600.0     # r_ST: near-field CAP electrode site
R_MODIOLUS_UM = 300.0            # central axons converging on the modiolar core

# --- Representative pathway control points (Potrusil et al. 2020) ----------
# theta is the Verbist azimuth; the pathway runs from the organ of Corti inward
# to Rosenthal's canal, then centrally along the modiolus toward the porus.
# z is measured along the modiolar axis, increasing toward the apex; the central
# axon runs the other way, out of the cochlea (negative z).
TRAJECTORY_THETA_DEG = np.array([200.0, 200.0, 200.0, 195.0, 180.0, 150.0, 120.0])
TRAJECTORY_R_UM = np.array([R_ORGAN_OF_CORTI_UM, 3_300.0, R_ROSENTHAL_UM,
                            1_800.0, 900.0, R_MODIOLUS_UM, R_MODIOLUS_UM])
TRAJECTORY_Z_UM = np.array([1_400.0, 1_350.0, 1_300.0,
                            1_100.0, 600.0, -200.0, -1_500.0])

#: Index of the control point that is the spiral-ganglion soma, and the one that
#: ends it — the unmyelinated human SGN soma spans control points 1..3.
_SOMA_CONTROL_LO, _SOMA_CONTROL_HI = 1, 3


def cochlear_to_cartesian(theta_deg, r_um, z_um):
    """Verbist cylindrical (theta deg, r um, z um) -> cochlear Cartesian um.

    theta = 0 (the round-window centre) lies on the +x axis, and theta increases
    toward +y, so the frame is right-handed with +z along the modiolar axis.
    Accepts scalars or equal-length arrays; returns (3,) or (n, 3).
    """
    t = np.radians(np.asarray(theta_deg, dtype=float))
    r = np.asarray(r_um, dtype=float)
    z = np.asarray(z_um, dtype=float)
    out = np.stack([r * np.cos(t), r * np.sin(t), np.broadcast_to(z, r.shape)],
                   axis=-1)
    return out


def _resample_polyline(points, n_points):
    """Resample a polyline to n_points equally spaced by arc length."""
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    s_new = np.linspace(0.0, s[-1], n_points)
    return np.stack([np.interp(s_new, s, points[:, k]) for k in range(3)], axis=1)


def representative_trajectory(n_points=241):
    """The default fibre pathway, cochlear-frame um, peripheral -> central.

    Returns (n_points, 3), equally spaced by arc length so that binning by index
    is binning by distance along the fibre.
    """
    ctrl = cochlear_to_cartesian(TRAJECTORY_THETA_DEG, TRAJECTORY_R_UM,
                                 TRAJECTORY_Z_UM)
    return _resample_polyline(ctrl, n_points)


def segment_bounds(n_points):
    """Index ranges of the three morphological segments in the resampled path.

    The control points carry the anatomy; the resampled array is uniform in arc
    length, so the boundaries are placed at the same fractional arc length the
    control points sit at.
    """
    ctrl = cochlear_to_cartesian(TRAJECTORY_THETA_DEG, TRAJECTORY_R_UM,
                                 TRAJECTORY_Z_UM)
    seg = np.linalg.norm(np.diff(ctrl, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    frac = s / s[-1]
    i0 = int(round(frac[_SOMA_CONTROL_LO] * (n_points - 1)))
    i1 = int(round(frac[_SOMA_CONTROL_HI] * (n_points - 1)))
    return {'peripheral': (0, i0), 'soma': (i0, i1), 'central': (i1, n_points)}


SEGMENT_BOUNDS = segment_bounds(241)


if __name__ == '__main__':
    traj = representative_trajectory()
    seg = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    print(f'points        : {len(traj)}')
    print(f'arc length    : {seg.sum() * 1e-3:.2f} mm')
    for name, (lo, hi) in SEGMENT_BOUNDS.items():
        d = np.linalg.norm(np.diff(traj[lo:hi + 1], axis=0), axis=1).sum()
        print(f'  {name:<11s} idx {lo:>4d}-{hi:<4d}  {d * 1e-3:6.3f} mm')
