"""Append a synthetic myelinated node/internode axon to a loaded bushy cell.

The EM globular bushy cell carries only ~18 µm of myelinated axon, too short
to propagate an action potential. The axon is extended to the contralateral
MNTB with a node/internode cable, appended to the SectionLists Node_of_Ranvier
and Internode, which gbc_biophysics maps to the 'node' and 'internode'
classes. Geometry and topology only; biophysics in gbc_biophysics.

    python models/avcn/axon_builder.py      # bake VCN_c09_extended_axon.hoc
"""

import os
from collections import defaultdict

import numpy as np
from neuron import h

# --- calibre of the crossing GBC axon ----------------------------------------
# Outer (axon + myelin) diameter 8 µm, within the 6-10 µm quoted for the
# largest mammalian GBC fibres (cat midline: 2.1-2.2 µm, Beckius et al. 1999).
# NEURON `diam` is the conducting core, G_RATIO x outer; myelin enters through
# the internode capacitance (gbc_biophysics.CM_MYELIN).
G_RATIO           = 0.65     #     inner/outer, within the 0.60-0.76 measured
                             #     for large auditory-brainstem CNS fibre
FIBER_OUTER_DIAM  = 8.0      # µm  axon + myelin, the quoted fibre diameter
FIBER_DIAM    = G_RATIO * FIBER_OUTER_DIAM        # µm  conducting core (NEURON diam)
NODE_LEN      = 1.0                               # µm

# --- internodes ---------------------------------------------------------------
# GBC trapezoid-body axons have L/d ~ 60 (sheath lengths stay fixed while the
# axon thickens after hearing onset); measured internodes 200-250 µm.
INTERNODE_LEN_TRACT    = 250.0   # µm  steady state, far from the terminal
INTERNODE_LEN_TERMINAL = 100.0   # µm  final internodes before the heminode

# --- nodes ---------------------------------------------------------------------
# ~1.9 µm in the tract, widening toward the terminal; the larger nodal area
# keeps the safety factor against the calyx's capacitive load.
NODE_DIAM_TRACT    = 1.9     # µm
NODE_DIAM_TERMINAL = 3.0     # µm  at the heminode

# --- distal taper --------------------------------------------------------------
# Internodes shorten and nodes widen together over the last 1.2 mm.
TAPER_LEN = 1200.0   # µm  graded, not stepped

# --- length --------------------------------------------------------------------
# head_geometry.GBC_AXON_LENGTH_UM: atlas chord to the contralateral MNTB x
# tortuosity.
def _default_axon_length():
    import os as _os, sys as _sys
    _root = _os.path.dirname(_os.path.dirname(_os.path.dirname(
        _os.path.abspath(__file__))))
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
    from recon_core.config.head_geometry import GBC_AXON_LENGTH_UM
    return float(GBC_AXON_LENGTH_UM)


AXON_LENGTH = _default_axon_length()   # µm  toward the contralateral MNTB

# The trunk ends at the heminode. The post-heminode neck and the calyx arbor
# belong to the MNTB calyx generator (models/mntb/calyx_model.hoc).


def internode_len_at(dist_from_start, total_length, taper_len=TAPER_LEN):
    """Internode length µm at `dist_from_start`, graded across the taper."""
    f = (dist_from_start - (total_length - taper_len)) / taper_len
    f = min(1.0, max(0.0, f))
    return INTERNODE_LEN_TRACT + f * (INTERNODE_LEN_TERMINAL - INTERNODE_LEN_TRACT)


def node_diam_at(dist_from_start, total_length, taper_len=TAPER_LEN):
    """Node diameter µm at `dist_from_start`, graded across the taper."""
    f = (dist_from_start - (total_length - taper_len)) / taper_len
    f = min(1.0, max(0.0, f))
    return NODE_DIAM_TRACT + f * (NODE_DIAM_TERMINAL - NODE_DIAM_TRACT)


SL_NODE  = 'Node_of_Ranvier'
SL_INTER = 'Internode'


def _ensure_sectionlists():
    for nm in (SL_NODE, SL_INTER):
        existing = getattr(h, nm, None)
        if existing is None:
            h('objref %s' % nm)
            h('%s = new SectionList()' % nm)


def _append(sl_name, sec):
    """Append a Python-created Section to the named hoc SectionList."""
    sec.push()
    getattr(h, sl_name).append()
    h.pop_section()


def _distal_axon_end(direction=None):
    """Return (terminal_section, end_point_xyz, unit_direction) of the axon.

    Prefers the myelinated axon, falling back to the AIS then the hillock. The
    terminal is the axon-list section with no child; the direction comes from
    its last two 3-D points unless overridden.
    """
    axon_secs = []
    for slname in ('Myelinated_Axon', 'Axon_Initial_Segment', 'Axon_Hillock'):
        sl = getattr(h, slname, None)
        if sl is not None:
            axon_secs = list(sl)
            if axon_secs:
                break
    if not axon_secs:
        raise RuntimeError('no axon (Myelinated_Axon/AIS/hillock) to extend')

    parents = set()
    for s in h.allsec():
        ps = s.parentseg()
        if ps is not None:
            parents.add(ps.sec.name())
    tips = [s for s in axon_secs if s.name() not in parents]
    term = tips[-1] if tips else axon_secs[-1]

    n = int(h.n3d(sec=term))
    p_last = np.array([h.x3d(n - 1, sec=term), h.y3d(n - 1, sec=term),
                       h.z3d(n - 1, sec=term)])
    if direction is not None:
        d = np.asarray(direction, float)
    else:
        p_prev = np.array([h.x3d(n - 2, sec=term), h.y3d(n - 2, sec=term),
                           h.z3d(n - 2, sec=term)])
        d = p_last - p_prev
    d = d / np.linalg.norm(d)
    return term, p_last, d


def build_extended_axon(length=AXON_LENGTH, node_len=NODE_LEN,
                        fiber_diam=FIBER_DIAM, taper_len=TAPER_LEN,
                        direction=None, verbose=False):
    """Append a graded node/internode cable, ending at the heminode.

    Returns (nodes, internodes), the created Sections in order. The cable
    begins at the existing axon's distal tip and runs STRAIGHT along `direction`
    (the atlas chord to the contralateral MNTB, supplied by the caller), or
    along the existing axon's exit direction if none is given.

    Internode length and node diameter are constant over the first
    length - taper_len, then graded to the terminal values across the taper
    (see internode_len_at / node_diam_at). The last section created is a node:
    the HEMINODE, where saltatory conduction ends. Nothing distal to it is built
    here -- the post-heminode neck and the calyx arbor belong to the MNTB calyx
    generator.

    Geometry only; call gbc_biophysics.decorate_gbc afterwards to make the nodes
    active.
    """
    _ensure_sectionlists()
    term, pos, d = _distal_axon_end(direction)

    nodes, internodes = [], []
    prev, prev_loc = term, 1.0
    pos = pos.astype(float).copy()
    dist, i = 0.0, 0

    while True:
        inter_len = internode_len_at(dist, length, taper_len)
        # stop before overrunning: the heminode has to fit too
        if dist + inter_len + node_len > length:
            break

        inter = h.Section(name='internode_%d' % i)
        p0 = pos.copy()
        p1 = pos + d * inter_len
        h.pt3dadd(p0[0], p0[1], p0[2], fiber_diam, sec=inter)
        h.pt3dadd(p1[0], p1[1], p1[2], fiber_diam, sec=inter)
        inter.connect(prev(prev_loc), 0)
        _append(SL_INTER, inter)
        internodes.append(inter)
        pos, dist = p1, dist + inter_len
        prev, prev_loc = inter, 1.0

        nd = node_diam_at(dist, length, taper_len)
        node = h.Section(name='node_%d' % i)
        p1n = pos + d * node_len
        h.pt3dadd(pos[0], pos[1], pos[2], nd, sec=node)
        h.pt3dadd(p1n[0], p1n[1], p1n[2], nd, sec=node)
        node.connect(prev(1.0), 0)
        _append(SL_NODE, node)
        nodes.append(node)
        pos, dist = p1n, dist + node_len
        prev, prev_loc = node, 1.0
        i += 1

    # The heminode: the terminal node, widened, where the trunk stops.
    hemi = h.Section(name='heminode')
    p1n = pos + d * node_len
    h.pt3dadd(pos[0], pos[1], pos[2], NODE_DIAM_TERMINAL, sec=hemi)
    h.pt3dadd(p1n[0], p1n[1], p1n[2], NODE_DIAM_TERMINAL, sec=hemi)
    hemi.connect(prev(1.0), 0)
    _append(SL_NODE, hemi)
    nodes.append(hemi)
    dist += node_len

    if verbose:
        print(f'[axon_builder] appended {len(internodes)} internodes + '
              f'{len(nodes)} nodes (last = heminode) = {dist:.0f} \u00b5m of a '
              f'requested {length:.0f} \u00b5m from {term.name()} '
              f'along {np.round(d, 3)}')
        print(f'[axon_builder]   internode {internodes[0].L:.0f} -> '
              f'{internodes[-1].L:.0f} \u00b5m, node diam '
              f'{nodes[0].diam:.2f} -> {nodes[-1].diam:.2f} \u00b5m')
    return nodes, internodes


# ---------------------------------------------------------------------------
# Bake the current NEURON cell (original morphology + extended axon) to .hoc
# ---------------------------------------------------------------------------
# LFPy.Cell(morphology=file) rebuilds the cell from the file and only tracks
# the sections it finds there, so sections added programmatically afterwards
# are invisible to its imem and dipole bookkeeping. For the population pipeline
# the extended axon therefore has to live in the morphology file itself. This
# dumps the live cell to a self-contained syGlass-style hoc, the same
# convention as morphology/dryad/*.hoc and SBC_S113.hoc, which LFPy loads.
CLASS_TO_SECTIONLIST = {
    'soma':              'soma',
    'hillock':           'Axon_Hillock',
    'initialsegment':    'Axon_Initial_Segment',
    'myelinatedaxon':    'Myelinated_Axon',
    'unmyelinatedaxon':  'unmyelinatedaxon',
    'primarydendrite':   'Proximal_Dendrite',
    'secondarydendrite': 'Distal_Dendrite',
    'node':              SL_NODE,
    'internode':         SL_INTER,
}


def write_cell_hoc(out_path, source_note=''):
    """Dump every currently instantiated section to a self-contained hoc file.

    Section identity is preserved by re-emitting each section into the
    SectionList its compartment class maps to (gbc_biophysics.COMPARTMENT_OF),
    so the written file decorates identically to the live cell.
    Geometry only (pt3dadd), no mechanisms.
    """
    import gbc_biophysics as gb
    comp = gb._classify_sections()
    secs = list(h.allsec())

    parent_of, children = {}, defaultdict(list)
    for s in secs:
        ps = s.parentseg()
        if ps is not None:
            parent_of[s.name()] = (ps.sec.name(), float(ps.x))
            children[ps.sec.name()].append(s)

    # topological order: roots first, then breadth-first
    order, queue = [], [s for s in secs if s.name() not in parent_of]
    while queue:
        cur = queue.pop(0)
        order.append(cur)
        queue.extend(children[cur.name()])
    idx_of = {s.name(): i for i, s in enumerate(order)}

    used = sorted({CLASS_TO_SECTIONLIST[comp.get(s.name(), 'soma')] for s in order})
    L = ['//', '//  Generated by models/avcn/axon_builder.write_cell_hoc',
         f'//    {source_note}',
         '//    Original morphology + synthetic node/internode active axon.',
         '//    Geometry only; biophysics via gbc_biophysics.decorate_gbc', '//', '']
    for sl in used:
        L.append(f'objref {sl}')
        L.append(f'{sl} = new SectionList()')
    L.append(f'create sections[{len(order)}]')
    L.append('')

    for i, sec in enumerate(order):
        sl = CLASS_TO_SECTIONLIST[comp.get(sec.name(), 'soma')]
        L.append(f'access sections[{i}]')
        L.append(f'{sl}.append()')
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


def make_extended_morphology(src_hoc, out_hoc, **kwargs):
    """Load a bushy morphology, append the active axon, bake it to out_hoc."""
    h.load_file('stdlib.hoc')
    h.load_file('import3d.hoc')
    h.load_file(src_hoc)
    nodes, inters = build_extended_axon(verbose=True, **kwargs)
    n = write_cell_hoc(out_hoc, source_note=f'source = {os.path.basename(src_hoc)}')
    print(f'Wrote {n} sections ({len(nodes)} nodes + {len(inters)} internodes) '
          f'-> {out_hoc}')
    return n


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Bake an extended active axon into a hoc')
    HERE = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument('--src', default=os.path.join(
        HERE, 'morphology', 'dryad', 'VCN_c09_Full_MeshInflate.hoc'))
    ap.add_argument('--out', default=os.path.join(
        HERE, 'morphology', 'extended', 'VCN_c09_extended_axon.hoc'))
    ap.add_argument('--length', type=float, default=AXON_LENGTH,
                    help='total axon length in um (default: the atlas chord to '
                         'the contralateral MNTB x GBC_MNTB_TORTUOSITY)')
    ap.add_argument('--outer-diam', type=float, default=FIBER_OUTER_DIAM,
                    dest='outer_diam',
                    help='quoted fibre diameter (axon + myelin) in um; the '
                         'conducting core is G_RATIO x this. Internode length '
                         'is NOT derived from it -- see INTERNODE_LEN_TRACT.')
    ap.add_argument('--taper-len', type=float, default=TAPER_LEN,
                    dest='taper_len',
                    help='length of the distal taper in um')
    a = ap.parse_args()
    make_extended_morphology(
        a.src, a.out, length=a.length, taper_len=a.taper_len,
        fiber_diam=G_RATIO * a.outer_diam)
