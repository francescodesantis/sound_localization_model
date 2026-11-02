#!/usr/bin/env python3
"""The human auditory-nerve fibre: peripheral process, spiral-ganglion soma,
central axon.

    terminal             organ of Corti                        ~10 µm   unmyelinated
    peripheral dendrite  -> spiral ganglion           rest of ~0.9 mm  myelinated
    pre-somatic          Rosenthal's canal                     ~100 µm   unmyelinated
    SGN soma             modiolus                               ~22 µm   unmyelinated (human)
    post-somatic         Rosenthal's canal                       ~5 µm   unmyelinated
    central axon         modiolus -> IAC -> CN entry            ~23 mm   node/internode

The cable is straight along +x and carries only arc length; the anatomical
trajectory (recon_core/config/anf_geometry.py) is applied at projection.
Section names follow the SectionList convention, so anf_biophysics decorates
by class.
"""

import os
import sys
from collections import defaultdict

import numpy as np
from neuron import h

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

from recon_core.config import anf_geometry as ag               # noqa: E402
from recon_core.config import params as P                      # noqa: E402

ANF_HOC = os.path.join(_HERE, 'morphology', 'anf_fiber.hoc')

SL_PERIPH = 'Peripheral_Process'
SL_SOMA = 'soma'
SL_NODE = 'Node_of_Ranvier'
SL_INTER = 'Internode'

CLASS_TO_SECTIONLIST = {
    'peripheral': SL_PERIPH,
    'soma': SL_SOMA,
    'node': SL_NODE,
    'internode': SL_INTER,
}


def _ensure_sectionlists():
    for nm in (SL_PERIPH, SL_SOMA, SL_NODE, SL_INTER):
        if getattr(h, nm, None) is None:
            h('objref %s' % nm)
            h('%s = new SectionList()' % nm)


def _append(sl_name, sec):
    sec.push()
    getattr(h, sl_name).append()
    h.pop_section()


def _add(sec, x0, length, diam, sl):
    h.pt3dadd(x0, 0.0, 0.0, diam, sec=sec)
    h.pt3dadd(x0 + length, 0.0, 0.0, diam, sec=sec)
    _append(sl, sec)
    return x0 + length


def build_anf(n_points=241, fiber_diam=None):
    """Build one straight ANF along +x. Returns the sections grouped by class.

    Segment lengths come from the anf_geometry trajectory (peripheral and soma
    portions) and from COCHLEAR_NERVE_LEN_UM for the central axon, so the cable's
    arc length matches the path it will be projected onto.
    """
    # fiber_diam sets the central axon only (node scaled by P.ANF_NODE_DIAM_RATIO);
    # the peripheral dendrite keeps its own diameter.
    fiber_diam = P.ANF_FIBER_DIAM if fiber_diam is None else float(fiber_diam)
    node_diam = fiber_diam * P.ANF_NODE_DIAM_RATIO

    _ensure_sectionlists()
    traj = ag.representative_trajectory(n_points)
    seg = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    bounds = ag.segment_bounds(n_points)

    # The cable length is derived from the path, so projected and measured
    # conduction velocities agree.
    from recon_core.config.head_geometry import ANF_PATH_UM
    path_len = np.linalg.norm(np.diff(ANF_PATH_UM['R'], axis=0), axis=1).sum()

    # The trajectory's soma region (~1.5 mm) contains the 22 µm soma; the rest of
    # it is myelinated axon in Rosenthal's canal and goes to the central axon.
    len_periph_raw = seg[bounds['peripheral'][0]:bounds['peripheral'][1]].sum()
    len_soma_region = seg[bounds['soma'][0]:bounds['soma'][1]].sum()
    len_soma = P.ANF_SOMA_DIAM
    excess = max(0.0, len_soma_region - len_soma)
    len_central = path_len - len_periph_raw - len_soma

    # The peripheral side is one unmyelinated terminal, a myelinated
    # node/internode dendrite, and one unmyelinated pre-somatic segment.
    len_p_myelin = len_periph_raw - P.ANF_TERMINAL_LEN - P.ANF_PRESOMA_LEN
    if len_p_myelin <= 0.0:
        raise RuntimeError(
            f'peripheral region ({len_periph_raw:.1f} um) is shorter than its '
            f'unmyelinated ends ({P.ANF_TERMINAL_LEN + P.ANF_PRESOMA_LEN:.1f} um)')

    parts = defaultdict(list)
    x = 0.0

    # Named 'peripheral' and created first: callers find the drive site with
    # name().startswith('peripheral').
    periph = h.Section(name='peripheral')
    x = _add(periph, x, P.ANF_TERMINAL_LEN, P.ANF_PERIPH_DIAM, SL_PERIPH)
    parts['peripheral'].append(periph)

    # Myelinated peripheral dendrite: P.ANF_PERIPH_N_NODES nodes, the internodes
    # absorb the residual length.
    n_p = int(P.ANF_PERIPH_N_NODES)
    p_step = len_p_myelin / n_p
    p_internode = p_step - P.ANF_NODE_LEN
    if p_internode <= 0.0:
        raise RuntimeError(
            f'{n_p} peripheral nodes do not fit in {len_p_myelin:.1f} um')
    prev = periph
    for i in range(n_p):
        # 'pnode'/'pinternode', so selections by the bare 'node' prefix take the
        # central axon only.
        inter = h.Section(name='pinternode_%d' % i)
        x = _add(inter, x, p_internode, P.ANF_PERIPH_DIAM, SL_INTER)
        inter.connect(prev(1.0), 0)
        parts['p_internodes'].append(inter)

        node = h.Section(name='pnode_%d' % i)
        x = _add(node, x, P.ANF_NODE_LEN, P.ANF_PERIPH_NODE_DIAM, SL_NODE)
        node.connect(inter(1.0), 0)
        parts['p_nodes'].append(node)
        prev = node

    presoma = h.Section(name='presoma')
    x = _add(presoma, x, P.ANF_PRESOMA_LEN, P.ANF_PERIPH_DIAM, SL_PERIPH)
    presoma.connect(prev(1.0), 0)
    parts['peripheral'].append(presoma)

    soma = h.Section(name='soma')
    x = _add(soma, x, len_soma, P.ANF_SOMA_DIAM, SL_SOMA)
    soma.connect(presoma(1.0), 0)
    parts['soma'].append(soma)

    postsoma = h.Section(name='postsoma')
    x = _add(postsoma, x, P.ANF_POSTSOMA_LEN, P.ANF_PERIPH_DIAM, SL_PERIPH)
    postsoma.connect(soma(1.0), 0)
    parts['peripheral'].append(postsoma)

    # n_int by rounding, then the internode length is adjusted so the chain
    # lands exactly on len_central_myelin (node length is kept fixed).
    assert excess >= 0.0
    len_central_myelin = len_central - P.ANF_POSTSOMA_LEN
    step = P.ANF_INTERNODE_LEN + P.ANF_NODE_LEN
    n_int = int(round(len_central_myelin / step))
    internode_eff = len_central_myelin / n_int - P.ANF_NODE_LEN
    prev = postsoma
    for i in range(n_int):
        inter = h.Section(name='internode_%d' % i)
        x = _add(inter, x, internode_eff, fiber_diam, SL_INTER)
        inter.connect(prev(1.0), 0)
        parts['internodes'].append(inter)

        node = h.Section(name='node_%d' % i)
        x = _add(node, x, P.ANF_NODE_LEN, node_diam, SL_NODE)
        node.connect(inter(1.0), 0)
        parts['nodes'].append(node)
        prev = node

    return dict(parts)


def write_anf_hoc(out_path=None):
    """Bake the live cell to a self-contained hoc LFPy can load.

    LFPy.Cell(morphology=file) rebuilds the cell from the file and only tracks
    sections found there, so programmatically created sections must be written
    out before the population pipeline can see their imem.  Same convention as
    models/avcn/axon_builder.write_cell_hoc.
    """
    out_path = out_path or ANF_HOC
    secs = list(h.allsec())
    parent_of, children = {}, defaultdict(list)
    for s in secs:
        ps = s.parentseg()
        if ps is not None:
            parent_of[s.name()] = (ps.sec.name(), float(ps.x))
            children[ps.sec.name()].append(s)

    order, queue = [], [s for s in secs if s.name() not in parent_of]
    while queue:
        cur = queue.pop(0)
        order.append(cur)
        queue.extend(children[cur.name()])
    idx_of = {s.name(): i for i, s in enumerate(order)}

    def sl_of(sec):
        # Order matters: 'pinternode_*' must not be caught by a 'node' test,
        # and the three unmyelinated stretches ('peripheral', 'presoma',
        # 'postsoma') all belong to the peripheral SectionList.
        n = sec.name()
        if n.startswith(('peripheral', 'presoma', 'postsoma')):
            return SL_PERIPH
        if n.startswith('soma'):
            return SL_SOMA
        if n.startswith('internode') or n.startswith('pinternode'):
            return SL_INTER
        if n.startswith('node') or n.startswith('pnode'):
            return SL_NODE
        return SL_INTER

    L = ['//', '//  Generated by models/anf/anf_model.write_anf_hoc',
         '//    Straight human ANF: unmyelinated peripheral process + SGN soma',
         '//    + myelinated node/internode central axon.',
         '//    Geometry only; biophysics via anf_biophysics.decorate_anf', '//', '']
    for sl in (SL_PERIPH, SL_SOMA, SL_NODE, SL_INTER):
        L.append(f'objref {sl}')
        L.append(f'{sl} = new SectionList()')
    L.append(f'create sections[{len(order)}]')
    L.append('')
    for i, sec in enumerate(order):
        L.append(f'access sections[{i}]')
        L.append(f'{sl_of(sec)}.append()')
        if sec.name() in parent_of:
            pname, ploc = parent_of[sec.name()]
            loc = '0' if ploc == 0.0 else '1' if ploc == 1.0 else f'{ploc:g}'
            L.append(f'connect sections[{i}](0), sections[{idx_of[pname]}]({loc})')
        L.append(f'sections[{i}] {{')
        for k in range(int(h.n3d(sec=sec))):
            L.append('    pt3dadd(%.6f, %.6f, %.6f, %.6f)' % (
                h.x3d(k, sec=sec), h.y3d(k, sec=sec),
                h.z3d(k, sec=sec), h.diam3d(k, sec=sec)))
        L.append('}')
        L.append('')
    L.append('access sections[0]')
    L.append('')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('\n'.join(L))
    return len(order)


if __name__ == '__main__':
    parts = build_anf()
    n = write_anf_hoc()
    total = sum(s.L for s in h.allsec())
    print(f'peripheral(unmyel) {len(parts["peripheral"])}  '
          f'soma {len(parts["soma"])}  '
          f'peripheral nodes {len(parts["p_nodes"])}  '
          f'central nodes {len(parts["nodes"])}  '
          f'internodes {len(parts["p_internodes"]) + len(parts["internodes"])}')
    print(f'total length {total * 1e-3:.2f} mm -> {n} sections -> {ANF_HOC}')
