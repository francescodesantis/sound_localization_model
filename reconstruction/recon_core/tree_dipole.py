#!/usr/bin/env python3
"""Branching cells: group a cell's currents into dipoles you may place apart.

WHY.  `line_dipole.py` solves this for an UNBRANCHED cable (the auditory nerve).
A bushy cell is a tree -- VCN_c09 has 443 sections, dendrites branching up to
5-fold -- so "sum over k <= j" has no meaning and Abel summation does not apply.
This is the tree generalisation.

WHAT DOES NOT WORK, AND WHAT DOES.  The distinction is SEGMENTS vs EDGES, and it
is the whole content of this module:

  * Grouping SEGMENTS and placing each group at its own position is invalid.
    p = sum_k r_k i_k is origin-independent only when sum_k i_k = 0, and a
    sub-group of segments carries large net current -- ABR_reconstruction/
    main_abr_avcn.py measured 3.2 nA over the axon alone against 3.0 nA for the
    whole cell.  Displacing such groups drops a monopole term and produces
    spurious anti-phase cancellation.

  * Grouping EDGES is valid.  For a tree with parent map par(j), charge
    conservation gives the axial current entering segment j from its parent as
    the subtree sum

        I_j = sum_{k in subtree(j)} i_k                                     (1)

    and then

        p = sum_j I_j (r_j - r_par(j))                                      (2)

    Every term is a current times a DISPLACEMENT, so every term is individually
    origin-independent, and therefore so is any subset sum.  That is what makes
    a group placeable at its own position.

(2) is the Naess et al. 2021 multi-dipole formulation (NeuroImage 225:117467),
computed from `imem` on the segment tree rather than from `vmem` via `h.ri`.
LFPy ships the `vmem` route as `Cell.get_multi_current_dipole_moments()`; it is
kept in the tests as an independent check and deliberately NOT used here:
measured on the extended GBC it is 30x slower, materialises 41 MB per cell, and
differs from `pos.T @ imem` by 20%, so "sum of groups == the stored whole-cell
record" could not be used as a test.  This route agrees to 1.6e-15.

    sum over all groups  ==  pos.T @ imem,  identically

so reproducing the lumped whole-cell dipole is a TEST, not a claim.

⚠ The tree root is NOT necessarily the soma.  In VCN_c09_extended_axon.hoc the
soma is section 180 and its parent is section 148, a dendrite; section index
order is not topological order either.  Both are handled here; neither may be
assumed by callers.
"""

import numpy as np

#: Absolute tolerance on sum_k i_k, in the same units as imem (nA).
#: Matches line_dipole.CHARGE_TOL.
CHARGE_TOL = 1e-6


def segment_parents(cell):
    """LFPy.Cell -> (n_seg,) parent segment index, -1 at the root.

    The only NEURON-aware function here; everything below is pure numpy so it
    can be tested on synthetic trees.

    Segment order is `cell.allseclist` iteration order, which is what LFPy's own
    `imem`, `x`, `y`, `z` use.  Within a section the parent is the previous
    segment; for a section's first segment it is the segment of the parent
    section containing the connection point.
    """
    index, g = {}, 0
    for sec in cell.allseclist:
        for i in range(sec.nseg):
            index[(sec.name(), i)] = g
            g += 1

    parents = np.full(g, -1, dtype=np.int64)
    g = 0
    for sec in cell.allseclist:
        for i in range(sec.nseg):
            if i > 0:
                parents[g] = g - 1
            else:
                pseg = sec.parentseg()
                if pseg is not None:
                    psec = pseg.sec
                    # which segment of the parent section holds the connection
                    j = min(int(pseg.x * psec.nseg), psec.nseg - 1)
                    parents[g] = index[(psec.name(), j)]
            g += 1
    return parents


def topological_order(parents):
    """Segment indices ordered deepest-first (children before parents).

    NOT `arange`: section index order is hoc definition order, and a child may
    precede its parent (VCN_c09's soma is index 180, its parent 148).  Feeding a
    non-topological order corrupts the subtree sums and therefore BOTH the total
    and the split -- verified in test_tree_dipole -- so the failure is loud, not
    silent.  Callers should let `subtree_currents` derive the order rather than
    pass `arange`.
    """
    parents = np.asarray(parents)
    n = parents.size
    depth = np.zeros(n, dtype=np.int64)
    for i in range(n):
        d, p, guard = 0, parents[i], 0
        while p >= 0:
            d += 1
            p = parents[p]
            guard += 1
            if guard > n:
                raise ValueError('parent map contains a cycle')
        depth[i] = d
    return np.argsort(-depth, kind='stable')


def subtree_currents(imem, parents, order=None):
    """Axial current entering each segment from its parent -- equation (1).

    imem : (n_seg, T) transmembrane current, nA
    returns (n_seg, T); the root entry is the whole-cell sum, ~0 by charge
    conservation, and is unused by `group_dipoles` (the root has no edge).
    """
    imem = np.asarray(imem, dtype=np.float64)
    parents = np.asarray(parents)
    if order is None:
        order = topological_order(parents)
    I = imem.copy()
    for j in order:
        p = parents[j]
        if p >= 0:
            I[p] += I[j]
    return I


def check_charge(imem, tol=CHARGE_TOL):
    """Raise unless sum_k i_k ~ 0, the premise (2) rests on.

    Returns the largest absolute residual so callers can report it.  NOTE the
    NEURON initialisation transient at t=0 violates this by ~0.4 nA on the GBC;
    blank it (main_abr_avcn.SETTLE_MS) before calling, or pass a slice.
    """
    resid = float(np.abs(np.asarray(imem, dtype=np.float64).sum(axis=0)).max())
    if resid > tol:
        raise ValueError(
            f'sum of transmembrane currents is {resid:.3e} nA, above {tol:.1e}. '
            f'Equation (2) assumes charge conservation; a non-zero sum usually '
            f'means the t=0 initialisation transient is still present.')
    return resid


def edge_displacements(pos_um, parents):
    """(n_seg, 3) positions -> (n_seg, 3) r_j - r_par(j); zero at the root."""
    pos_um = np.asarray(pos_um, dtype=np.float64)
    parents = np.asarray(parents)
    d = np.zeros_like(pos_um)
    has = parents >= 0
    d[has] = pos_um[has] - pos_um[parents[has]]
    return d


def group_dipoles(imem, pos_um, parents, group_of_seg, groups, order=None):
    """Equation (2), summed within each group of EDGES.

    group_of_seg : (n_seg,) label per segment; an edge is assigned to its CHILD
                   segment's label, so the root (no edge) never contributes.
    groups       : {name: tuple-of-labels}

    Returns (moments, centroids, weights):
      moments   {name: (3, T)} nA.um
      centroids {name: (3,)} um, RMS-amplitude weighted (see below)
      weights   {name: float} the summed RMS weight

    The centroid is amplitude-weighted, NOT length-weighted: the GBC has 378
    distal-dendrite edges against 1 AIS edge, so a length- or count-weighted
    centroid for a spike group would sit in the dendrites.  Because d_j is
    constant in time, the per-edge RMS moment factorises exactly as
    |d_j| * rms_t(I_j), so no per-edge (3, T) array is ever materialised.
    """
    imem = np.asarray(imem, dtype=np.float64)
    pos_um = np.asarray(pos_um, dtype=np.float64)
    parents = np.asarray(parents)
    group_of_seg = np.asarray(group_of_seg)

    I = subtree_currents(imem, parents, order)
    d = edge_displacements(pos_um, parents)
    has_edge = parents >= 0
    w_all = np.linalg.norm(d, axis=1) * np.sqrt(np.mean(I ** 2, axis=1))

    moments, centroids, weights = {}, {}, {}
    for name, labels in groups.items():
        sel = np.flatnonzero(has_edge & np.isin(group_of_seg, labels))
        if sel.size == 0:
            moments[name] = np.zeros((3, imem.shape[1]))
            centroids[name] = np.zeros(3)
            weights[name] = 0.0
            continue
        moments[name] = d[sel].T @ I[sel]
        w = w_all[sel]
        tot = float(w.sum())
        # midpoint of each edge, weighted by how much moment it carries
        mids = 0.5 * (pos_um[sel] + pos_um[parents[sel]])
        centroids[name] = ((mids * w[:, None]).sum(axis=0) / tot if tot > 0
                           else mids.mean(axis=0))
        weights[name] = tot
    return moments, centroids, weights


def whole_cell_dipole(imem, pos_um, parents, order=None):
    """Equation (2) over every edge. Equals `pos_um.T @ imem` when sum i = 0."""
    imem = np.asarray(imem, dtype=np.float64)
    parents = np.asarray(parents)
    I = subtree_currents(imem, parents, order)
    d = edge_displacements(pos_um, parents)
    sel = np.flatnonzero(parents >= 0)
    return d[sel].T @ I[sel]
