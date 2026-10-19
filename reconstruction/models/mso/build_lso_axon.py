"""
Generate lso_model_active_axon.hoc: the LSO principal cell with an ascending
lateral lemniscus (LL) active axon replacing the 150 um silent stub of
lso_model_active.hoc.

The Tolnai BIC generator is the LSO projection neuron's spiking output, carried
up the LL as a travelling-wave current dipole. That needs the cell to fire an AP
and a millimetre-scale active axon for the AP to propagate over, since the
150 um stub behaves as a near-stationary point source. This script keeps the
soma and dendrites and appends an AIS plus a node/internode myelinated cable
(active nodes, passive myelin), with all biophysics inline so
LFPy.Cell(morphology=file) loads a fully decorated, self-contained cell with no
custom_fun and tracks every section for the CurrentDipoleMoment.

Anatomy (model axes: x dorsoventral, y rostrocaudal, z mediolateral):
  Primary dendrites lie in the parasagittal (y-x) plane, predominantly
  rostrocaudal, so they are drawn along +-y (dend_A/B, 400 um). The cell is
  strictly bipolar, two opposed primaries and nothing else, so its dendritic
  dipole has no built-in asymmetry. Tonotopy is mediolateral (z), perpendicular
  to the dendritic sheet.
  The axon exits rostro-dorsally (+y rostral, +x dorsal) and ascends the LL
  toward the IC; the ABR rotation maps AXON_DIR onto the head inferosuperior
  axis.

Usage:
  python models/mso/build_lso_axon.py            # writes lso_model_active_axon.hoc
"""

import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# One morphology per side. See axon_dir() for why the left cell is mirrored.
AXON_HOC = {'R': 'lso_model_active_axon.hoc', 'L': 'lso_model_active_axon_left.hoc'}
STUB_HOC = {'R': 'lso_model_active.hoc', 'L': 'lso_model_active_left.hoc'}

# --- orientation (model unit vectors: x dorsoventral, y rostrocaudal, z mediolateral)
DEND_A_DIR = np.array([0., 1., 0.])              # +y rostral
DEND_B_DIR = np.array([0., -1., 0.])             # -y caudal
AXON_DIR   = np.array([1., 1., 0.]); AXON_DIR /= np.linalg.norm(AXON_DIR)  # rostro-dorsal


def axon_dir(side='R'):
    """Model-frame axon direction for one side.

    The left cell is the mirror image of the right one (model x negated), not
    the same cell rotated. That is what lets both sides send their lemniscal
    axon to head +z and keep tonotopy and the dendritic tilt mirror-symmetric:
    with a single un-mirrored morphology only two of the three can hold at
    once, since a proper rotation cannot reproduce a reflection. Everything
    else in the cell lies on y or is radial, so mirroring x moves only the axon.
    """
    d = AXON_DIR.copy()
    if side == 'L':
        d[0] = -d[0]
    return d

SOMA_HALF  = 13.5      # um  soma half-length, elongated along y (dendritic axis)
SOMA_DIA   = 12.0
DEND_LEN   = 400.0

# --- axon geometry (µm), a literature placeholder mammalian myelinated fibre ---
AIS_LEN       = 25.0
AIS_DIAM      = 1.5
FIBER_DIAM    = 1.5     # internode (myelinated) diameter
NODE_DIAM     = 1.0     # node slightly constricted
NODE_LEN      = 1.0
INTERNODE_LEN = 150.0
AXON_LENGTH   = 4000.0  # total extension toward the IC along the LL

# --- biophysics (S/cm2): active node/AIS at LSO axon densities, passive myelin --
RA   = 150.0
ENA  = 55.0
EK   = -77.0
EPAS = -63.0

ACTIVE = dict(gpas=2.0e-3, cm=1.0, nax=0.50, klt=0.010, kht=0.015, ih=0.0005)
MYELIN = dict(gpas=1.0e-5, cm=0.01)   # low capacitance, high Rm, so fast saltation


def _pt3d(p0, p1, d0, d1):
    return [f'    pt3dadd({p0[0]:.4f}, {p0[1]:.4f}, {p0[2]:.4f}, {d0:g})',
            f'    pt3dadd({p1[0]:.4f}, {p1[1]:.4f}, {p1[2]:.4f}, {d1:g})']


def _active_block(indent='    '):
    a = ACTIVE
    return '\n'.join(indent + s for s in [
        f'Ra = {RA}', f'cm = {a["cm"]}',
        f'insert pas   g_pas = {a["gpas"]}   e_pas = {EPAS}',
        f'insert nax   gbar_nax = {a["nax"]}   ena = {ENA}',
        f'insert klt   gbar_klt = {a["klt"]}',
        f'insert kht   gbar_kht = {a["kht"]}',
        f'insert ih    gbar_ih = {a["ih"]}',
        f'ek = {EK}',
    ])


def _myelin_block(indent='    '):
    m = MYELIN
    return '\n'.join(indent + s for s in [
        f'Ra = {RA}', f'cm = {m["cm"]}',
        f'insert pas   g_pas = {m["gpas"]}   e_pas = {EPAS}',
    ])


def _dend_block(name, direction, length, d0, d1, nseg, connect_stmt, start=None):
    p0 = np.array([0., 0., 0.]) if start is None else np.asarray(start, float)
    p1 = p0 + np.asarray(direction, float) * length
    L = [f'{name} {{', f'    nseg = {nseg}',
         '    Ra = 150   cm = 1.0',
         f'    insert pas   g_pas = 2.0e-3   e_pas = {EPAS}',
         '    insert klt   gbar_klt = 0.002',
         '    insert ih    gbar_ih = 0.0001',
         f'    ek = {EK}']
    L += _pt3d(p0, p1, d0, d1)
    L.append('}')
    return L, connect_stmt


def build(out_path, length=AXON_LENGTH, internode_len=INTERNODE_LEN,
          node_len=NODE_LEN, verbose=True, side='R'):
    n = int(round(length / (internode_len + node_len)))
    axis = axon_dir(side)
    L = []
    L.append('// LSO principal cell + ascending lateral-lemniscus ACTIVE axon.')
    L.append('// Generated by models/mso/build_lso_axon.py. Do not edit by hand.')
    L.append('// Dendrites: parasagittal (y-x) plane, primaries along +-y (rostrocaudal).')
    L.append(f'// Axon ({side} side): axis={np.round(axis,3).tolist()}'
             f' (+y rostral, +x dorsal); AIS + {n} x (internode {internode_len:g} um'
             f' + node {node_len:g} um) ~= {AIS_LEN + n*(internode_len+node_len):.0f} um.')
    L.append('// Active nodes/AIS (nax); passive low-cm myelin internodes.')
    L.append('')
    L.append('create soma, dend_A, dend_B, ais')
    L.append(f'create node[{n}], internode[{n}]')
    L.append('objref axon_nodes, axon_internodes')
    L.append('axon_nodes      = new SectionList()')
    L.append('axon_internodes = new SectionList()')
    L.append('')
    L.append('access soma')
    L.append('')

    # soma, elongated along y (dendritic axis)
    sp0 = np.array([0., -SOMA_HALF, 0.]); sp1 = np.array([0., SOMA_HALF, 0.])
    L += ['// soma (L = 27 um, dia = 12 um), elongated along y', 'soma {',
          '    nseg = 1', '    Ra   = 150', '    cm   = 1.0',
          f'    insert pas   g_pas = 2.0e-3   e_pas = {EPAS}',
          '    insert klt   gbar_klt = 0.010',
          '    insert kht   gbar_kht = 0.010',
          '    insert ih    gbar_ih = 0.0005',
          f'    insert nax   gbar_nax = 0.10   ena = {ENA}',
          f'    ek = {EK}',
          '    insert lso_ahp   gbar_lso_ahp = 0.002']
    L += _pt3d(sp0, sp1, SOMA_DIA, SOMA_DIA)
    L += ['}', '']

    # dendrites: two opposed primaries, +-y from the soma poles
    dblocks = [
        _dend_block('dend_A', DEND_A_DIR, DEND_LEN, 3.5, 1.5, 29,
                    'connect dend_A(0), soma(1)', start=sp1),
        _dend_block('dend_B', DEND_B_DIR, DEND_LEN, 3.5, 1.5, 29,
                    'connect dend_B(0), soma(0)', start=sp0),
    ]
    for block, _ in dblocks:
        L += block + ['']

    # AIS, emerging rostro-dorsally from the soma centre along axis
    a_start = axis * SOMA_HALF
    a_end   = a_start + axis * AIS_LEN
    L += ['// Axon initial segment (active, high Na for AP initiation)', 'ais {',
          '    nseg = 5', _active_block()]
    L += _pt3d(a_start, a_end, AIS_DIAM, AIS_DIAM)
    L += ['}', '']

    # node/internode cable along axis: internode_i then node_i
    pos = a_end.copy()
    for i in range(n):
        p_in0, p_in1 = pos.copy(), pos + axis * internode_len
        L += [f'internode[{i}] {{', '    nseg = 5', _myelin_block()]
        L += _pt3d(p_in0, p_in1, FIBER_DIAM, FIBER_DIAM)
        L += ['    axon_internodes.append()', '}']
        pos = p_in1
        p_nd0, p_nd1 = pos.copy(), pos + axis * node_len
        L += [f'node[{i}] {{', '    nseg = 1', _active_block()]
        L += _pt3d(p_nd0, p_nd1, NODE_DIAM, NODE_DIAM)
        L += ['    axon_nodes.append()', '}']
        pos = p_nd1
    L.append('')

    # topology
    for _, stmt in dblocks:
        L.append(stmt)
    L.append('connect ais(0),    soma(0.5)')
    L.append('connect internode[0](0), ais(1)')
    L.append('connect node[0](0),      internode[0](1)')
    L.append(f'for i = 1, {n-1} {{')
    L.append('    connect internode[i](0), node[i-1](1)')
    L.append('    connect node[i](0),      internode[i](1)')
    L.append('}')
    L.append('')
    L.append('access soma')
    L.append('')

    with open(out_path, 'w') as f:
        f.write('\n'.join(L))
    if verbose:
        total = AIS_LEN + n * (internode_len + node_len)
        print(f'Wrote {out_path}: soma+2 dend (bipolar, dends || y) + AIS + {n} nodes + {n} '
              f'internodes (~{total:.0f} um axon along {np.round(axis,3).tolist()})')
    return out_path


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Bake the extended-axon LSO hoc')
    ap.add_argument('--out', default=None,
                    help='output hoc (default: the per-side name below)')
    ap.add_argument('--side', default='both', choices=['R', 'L', 'both'])
    ap.add_argument('--length', type=float, default=AXON_LENGTH)
    ap.add_argument('--internode-len', type=float, default=INTERNODE_LEN,
                    dest='internode_len')
    a = ap.parse_args()
    for side in (['R', 'L'] if a.side == 'both' else [a.side]):
        out = a.out or os.path.join(HERE, AXON_HOC[side])
        build(out, length=a.length, internode_len=a.internode_len, side=side)
