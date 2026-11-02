#!/usr/bin/env python3
"""Branched cells: group a cell's currents into dipoles that can be placed apart.

line_dipole.py handles an unbranched cable; this is the tree generalisation
(bushy cells branch). For a tree with parent map par(j), the axial current
entering segment j from its parent is the subtree sum

    I_j = sum_{k in subtree(j)} i_k                                     (1)

and the dipole moment is

    p = sum_j I_j (r_j - r_par(j))                                      (2)

Each term is a current times a displacement, so any subset of edges is origin
independent and can be placed at its own centroid (Naess et al. 2021,
NeuroImage 225:117467). Grouping segments instead is not valid: a sub-group
of segments carries net current. The sum over all groups equals pos.T @ imem.

The tree root is not necessarily the soma, and section index order is not
topological order; both are handled here.
"""

import numpy as np

#: Absolute tolerance on sum_k i_k, nA.
CHARGE_TOL = 1e-6


def segment_parents(cell):
    """LFPy.Cell -> (n_seg,) parent segment index, -1 at the root.

    Segment order is `cell.allseclist` iteration order, as in LFPy's `imem` and
    coordinates. Within a section the parent is the previous segment; a section's
    first segment's parent is the parent-section segment holding the connection.
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

    Section index order is not topological (a child may precede its parent), so
    subtree_currents derives this order itself.
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
    """Raise unless sum_k i_k ~ 0, the premise of (2); return the largest residual.

    The NEURON initialisation transient at t=0 violates it; blank it first.
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
    """Equation (2), summed within each group of edges.

    group_of_seg : (n_seg,) label per segment; an edge belongs to its child
                   segment's label, so the root (no edge) never contributes.
    groups       : {name: tuple-of-labels}

    Returns (moments, centroids, weights):
      moments   {name: (3, T)} nA·µm
      centroids {name: (3,)} µm, weighted by each edge's RMS moment |d_j| rms_t(I_j)
      weights   {name: float} the summed RMS weight
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
