"""
Append a synthetic node/internode active myelinated axon to a loaded bushy cell.

The EM globular-bushy-cell reconstructions carry only ~18 µm of myelinated
axon, far too short to support a propagating action potential. For the GBC
axonal travelling-wave dipole (the GBC to contralateral MNTB volley, an ABR
generator) the axon is extended with a synthetic node/internode cable of
realistic length and then decorated by gbc_biophysics (nodes active, internodes
passive myelin), so a saltatory AP can propagate and feed CurrentDipoleMoment.

Sections are created in NEURON and appended to two SectionLists,
Node_of_Ranvier and Internode, the names gbc_biophysics.COMPARTMENT_OF maps to
the 'node' and 'internode' compartment classes, so decoration is automatic.

This is a morphology construction step, geometry and topology only; the
biophysics stays in gbc_biophysics.decorate_gbc.
"""

import os
from collections import defaultdict

import numpy as np
from neuron import h

# --- geometry of the crossing GBC axon -------------------------------------
# Calibre is measured, not a placeholder: Beckius, Batra & Oliver (1999)
# reconstructed AVCN axons in cat in 3-D and measured the fibre at the midline,
# exactly this crossing trapezoid-body segment, at 2.09 +/- 0.33 and
# 2.21 +/- 0.44 um. The globular bushy cell's axon is one of the thickest in
# the brainstem, about 3x the spherical bushy cell's, and the calibre is what
# buys the conduction speed the calyx of Held needs.
#
# These numbers are cat. No human GBC axon morphometry exists. Human axons are
# longer and conduction time is broadly conserved across species by scaling
# calibre, so 2.1 um is more likely a floor than a target for human.
FIBER_DIAM    = 2.1     # µm  internode (myelinated) diameter, Beckius 1999, cat
NODE_DIAM     = 1.4     # µm  node constricted to the same 0.67 ratio as before
NODE_LEN      = 1.0     # µm
INTERNODE_LEN = 210.0   # µm  (~100x fibre diameter, the standard scaling)

# Truncation. The real path from the cochlear nucleus to the contralateral MNTB
# is ~15 mm (13.9 mm of it mediolateral, see head_geometry.NUCLEUS_POS_UM), so
# this 4 mm cable is a deliberate stump. It does not shrink the peak dipole: an
# action potential occupies only ~2.2 mm at this conduction velocity, so it fits
# inside the modelled cable. What it truncates is the volley's duration, ~0.8 ms
# modelled against ~3 ms for the full path, making the reconstructed wave III
# sharper than the real one rather than smaller.
AXON_LENGTH   = 4000.0  # µm  extension toward the contralateral MNTB

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


def build_extended_axon(length=AXON_LENGTH, internode_len=INTERNODE_LEN,
                        node_len=NODE_LEN, fiber_diam=FIBER_DIAM,
                        node_diam=NODE_DIAM, direction=None, verbose=False):
    """Append a straight node/internode cable to the cell's distal axon.

    Returns (nodes, internodes), the created Sections in order. The cable
    begins at the existing axon's distal tip and runs straight along its exit
    direction, or along direction if given. Geometry only; call
    gbc_biophysics.decorate_gbc afterwards to make the nodes active.
    """
    _ensure_sectionlists()
    term, pos, d = _distal_axon_end(direction)

    n_internodes = int(round(length / (internode_len + node_len)))
    nodes, internodes = [], []
    prev, prev_loc = term, 1.0
    pos = pos.astype(float).copy()

    for i in range(n_internodes):
        inter = h.Section(name='internode_%d' % i)
        p0 = pos.copy()
        p1 = pos + d * internode_len
        h.pt3dadd(p0[0], p0[1], p0[2], fiber_diam, sec=inter)
        h.pt3dadd(p1[0], p1[1], p1[2], fiber_diam, sec=inter)
        inter.connect(prev(prev_loc), 0)
        _append(SL_INTER, inter)
        internodes.append(inter)
        pos = p1
        prev, prev_loc = inter, 1.0

        node = h.Section(name='node_%d' % i)
        p1n = pos + d * node_len
        h.pt3dadd(pos[0], pos[1], pos[2], node_diam, sec=node)
        h.pt3dadd(p1n[0], p1n[1], p1n[2], node_diam, sec=node)
        node.connect(prev(1.0), 0)
        _append(SL_NODE, node)
        nodes.append(node)
        pos = p1n
        prev, prev_loc = node, 1.0

    if verbose:
        total = n_internodes * (internode_len + node_len)
        print(f'[axon_builder] appended {len(internodes)} internodes + '
              f'{len(nodes)} nodes = {total:.0f} µm from {term.name()} '
              f'along {np.round(d, 2)}')
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
    ap.add_argument('--length', type=float, default=AXON_LENGTH)
    a = ap.parse_args()
    make_extended_morphology(a.src, a.out, length=a.length)
