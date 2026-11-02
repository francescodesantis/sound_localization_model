#!/usr/bin/env python3
"""Extended sources: a cable's currents as a chain of dipoles.

The 4-sphere model projects point dipoles. That is accurate when a source is
small compared with its distance to the electrodes, but the auditory nerve is
~24 mm long with its distal end ~20 mm from the mastoid, where the point-dipole
expansion does not converge.

With per-segment transmembrane current i_k (sum_k i_k = 0), S_j = sum_{k<=j} i_k
and axial current I_j = -S_j, Abel summation gives

    p = sum_k r_k i_k = sum_j I_j (r_{j+1} - r_j)

Each term is a dipole on one internode (zero net charge, origin independent).
Binning them keeps the source distributed, and the bins sum to the whole-cell
dipole exactly (n_bins = 1 reproduces it). Grouping segments instead is not
valid: sub-groups carry net current. For branched cells see tree_dipole.py.
"""

import numpy as np

#: Absolute tolerance on sum_k i_k, in the same units as imem (nA).
CHARGE_TOL = 1e-6


def axial_currents(imem):
    """Per-segment transmembrane current (n_seg, T) -> axial current (n_seg-1, T).

    Raises ValueError if the cell does not conserve charge, because every
    identity below depends on it and a silent violation would show up much later
    as an unexplained monopole.
    """
    imem = np.asarray(imem, dtype=float)
    resid = np.abs(imem.sum(axis=0)).max()
    if resid > CHARGE_TOL:
        raise ValueError(
            f'imem does not conserve charge: max |sum_k i_k| = {resid:.3e} '
            f'(tol {CHARGE_TOL:.0e}). A well-defined dipole requires sum(i)=0.')
    return -np.cumsum(imem, axis=0)[:-1]


def whole_cell_dipole(imem, pos_um):
    """Lumped current dipole moment (3, T) nA.um — what LFPy's probe computes."""
    imem = np.asarray(imem, dtype=float)
    pos = np.asarray(pos_um, dtype=float)
    return pos.T @ imem


def bin_to_dipoles(imem, pos_um, n_bins):
    """Cable currents -> n_bins point dipoles along the cable.

    Returns (p, r): p is (n_bins, 3, T) nA.um, r is (n_bins, 3) um.  Bins are
    contiguous groups of internodes; each bin's position is the mean of the
    internode midpoints it contains, weighted by internode length so a bin that
    straddles a geometry change is not biased toward the finely sampled end.

    p.sum(axis=0) equals whole_cell_dipole(imem, pos_um) to float precision.
    """
    imem = np.asarray(imem, dtype=float)
    pos = np.asarray(pos_um, dtype=float)
    n_seg = len(pos)
    if not 1 <= n_bins <= n_seg - 1:
        raise ValueError(f'n_bins must be in 1..{n_seg - 1}, got {n_bins}')

    I = axial_currents(imem)                       # (n_seg-1, T)
    dr = np.diff(pos, axis=0)                      # (n_seg-1, 3)
    mid = 0.5 * (pos[:-1] + pos[1:])               # (n_seg-1, 3)
    length = np.linalg.norm(dr, axis=1)            # (n_seg-1,)

    edges = np.linspace(0, n_seg - 1, n_bins + 1).astype(int)
    p = np.zeros((n_bins, 3, imem.shape[1]))
    r = np.zeros((n_bins, 3))
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        # p_bin = sum_j I_j * dr_j  -> (3, T)
        p[b] = np.einsum('jc,jt->ct', dr[lo:hi], I[lo:hi])
        w = length[lo:hi]
        w_tot = w.sum()
        # Guard against zero-length bins (coincident consecutive positions).
        # All those midpoints coincide anyway, so unweighted mean is correct.
        if w_tot > 0:
            r[b] = (mid[lo:hi] * w[:, None]).sum(axis=0) / w_tot
        else:
            r[b] = mid[lo:hi].mean(axis=0)
    return p, r
