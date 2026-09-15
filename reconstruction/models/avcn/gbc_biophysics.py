"""
Biophysical decoration for globular bushy cell (GBC) morphologies.

Ports the cnmodel XM13_nacncoop (mouse, Type II) channel densities onto a
NEURON morphology using the cnmodel SectionList naming convention (soma,
primarydendrite, secondarydendrite, hillock, unmyelinatedaxon, myelinatedaxon).
Works both standalone, after h.load_file('.../bushy_stick.hoc') for validation,
and inside LFPy, passed as LFPy.Cell(custom_fun=[decorate_gbc]).

Reference conductances are total nS (cnmodel data table XM13nacncoop_channels
and ..._compartments in external/cnmodel/cnmodel/data/ionchannels.py).
Following cnmodel they are converted to a somatic density (S/cm^2) using the
actual soma surface area of the loaded morphology, then scaled per compartment,
so the same function is valid for the stick stand-in and the real EM
reconstruction.

Channels (compiled in models/avcn/x86_64): klt, kht, ihvcn, leak, nacncoop.
"""

from neuron import h

# --- XM13_nacncoop reference (soma) conductances, mouse, in nS ---------------
#     source: ionchannels.py add_table_data('XM13nacncoop_channels', ...)
# klt/kht are renamed kltbc/khtbc (unique SUFFIX) so this mechanism set coexists
# with the MSO/LSO klt/kht that NEURON auto-loads from the repo-root x86_64.
# leak/ihvcn/nacncoop names are already unique.
#
# Two model types, differing only in nacncoop and kltbc (ka_gbar = 0 in both,
# so no extra mechanism):
#   II    globular bushy cell (GBC), the phasic Type II profile.
#   II-I  spherical bushy cell (SBC): less KLT and less Na, so higher input
#         resistance, longer tau, shallower Ih sag and more spikes (Jing et al.
#         2025, Atoh7+ vs Hhip+).
REF_NS_II = {
    'nacncoop': 3000.0,
    'khtbc':      58.0,
    'kltbc':      80.0,
    'ihvcn':      30.0,
    'leak':        2.0,
}
REF_NS_II_I = {
    'nacncoop': 1000.0,   # II-I: 3000 -> 1000
    'khtbc':      58.0,
    'kltbc':      20.0,   # II-I: 80 -> 20  (less low-voltage K)
    'ihvcn':      30.0,
    'leak':        2.0,
}
# Back-compatible default: unqualified REF_NS is the GBC (Type II) set.
REF_NS = REF_NS_II

# --- Per-compartment scale factors relative to soma density --------------------
#     source: add_table_data('XM13nacncoop_channels_compartments', ...)
#     Column order maps to cnmodel SectionList names below.
#
# 'node' and 'internode' are for the synthetic extended active axon (the GBC
# travelling wave, models/avcn/axon_builder.py), not from the cnmodel table:
#   node       active node of Ranvier: very high Na to regenerate the AP plus
#              fast Kv3 (khtbc) repolarisation, giving saltatory conduction.
#   internode  myelinated and passive: no active channels, a tiny leak (high
#              Rm) and the reduced capacitance CM_MYELIN (see below).
SCALE = {
    #             soma  hillock  initialsegment  unmyel  myel   primdend secdend  node internode
    'nacncoop': {'soma': 1.0, 'hillock': 5.0, 'initialsegment': 5.0, 'unmyelinatedaxon': 3.0,
                 'myelinatedaxon': 0.0, 'primarydendrite': 0.50, 'secondarydendrite': 0.25,
                 'node': 10.0, 'internode': 0.0},
    'khtbc':    {'soma': 1.0, 'hillock': 2.0, 'initialsegment': 2.0, 'unmyelinatedaxon': 2.0,
                 'myelinatedaxon': 0.01, 'primarydendrite': 0.5, 'secondarydendrite': 0.25,
                 'node': 3.0, 'internode': 0.0},
    'kltbc':    {'soma': 1.0, 'hillock': 1.0, 'initialsegment': 1.0, 'unmyelinatedaxon': 1.0,
                 'myelinatedaxon': 0.01, 'primarydendrite': 0.5, 'secondarydendrite': 0.25,
                 'node': 0.0, 'internode': 0.0},
    'ihvcn':    {'soma': 1.0, 'hillock': 0.0, 'initialsegment': 0.5, 'unmyelinatedaxon': 0.0,
                 'myelinatedaxon': 0.0, 'primarydendrite': 0.5, 'secondarydendrite': 0.5,
                 'node': 0.0, 'internode': 0.0},
    'leak':     {'soma': 1.0, 'hillock': 1.0, 'initialsegment': 1.0, 'unmyelinatedaxon': 0.25,
                 'myelinatedaxon': 0.25e-3, 'primarydendrite': 0.5, 'secondarydendrite': 0.5,
                 'node': 1.0, 'internode': 1e-3},
}

# --- Reversal potentials / passive (XM13_nacncoop table, RM03 kinetics) ---------
E_NA   = 50.0     # mV
E_K    = -84.0    # mV
E_H    = -43.0    # mV  (ihvcn.eh)
E_LEAK = -65.0    # mV  (leak.erev)
RA     = 150.0    # ohm*cm
CM     = 0.9      # uF/cm^2  (cnmodel membrane cap for mouse bushy)
CM_MYELIN = 0.02  # uF/cm^2  (myelinated internode: ~1/45 of unmyelinated membrane)

# Map every known hoc SectionList name to a cnmodel compartment class.
# Covers both the cnmodel stick (names equal classes) and the Dryad EM
# reconstructions (syGlass SectionList names such as Proximal_Dendrite).
COMPARTMENT_OF = {
    # cnmodel stick (identity)
    'soma': 'soma', 'hillock': 'hillock', 'unmyelinatedaxon': 'unmyelinatedaxon',
    'myelinatedaxon': 'myelinatedaxon', 'primarydendrite': 'primarydendrite',
    'secondarydendrite': 'secondarydendrite',
    # Dryad EM (syGlass) SectionLists
    'Axon_Hillock': 'hillock',
    'Axon_Initial_Segment': 'initialsegment',
    'Myelinated_Axon': 'myelinatedaxon',
    'Proximal_Dendrite': 'primarydendrite',
    'Dendritic_Hub': 'primarydendrite',
    'Distal_Dendrite': 'secondarydendrite',
    'Dendritic_Swelling': 'secondarydendrite',
    # Synthetic extended active axon (axon_builder.py)
    'Node_of_Ranvier': 'node',
    'Internode': 'internode',
}

# Fractional weights for realistic endbulb placement: 70% soma (large endbulbs
# engulf the cell body), 20% proximal dendrite and hubs, 10% hillock and AIS
# (at least one endbulb extends onto the initial segment).
ENDBULB_COMPARTMENT_WEIGHTS = {
    'soma':            0.70,
    'primarydendrite': 0.20,
    'hillock':         0.05,
    'initialsegment':  0.05,
}


def weighted_endbulb_idx(cell, n, weights=None):
    """Return n segment indices for endbulb placement, weighted by compartment.

    weights : dict or None
        Compartment class to fraction. None uses the GBC default
        (ENDBULB_COMPARTMENT_WEIGHTS, 70/20/5/5). Pass an override for the SBC,
        which has few large axosomatic endbulbs, e.g.
        {'soma': 0.85, 'primarydendrite': 0.15}.
    """
    import numpy as np
    weights = dict(weights if weights is not None else ENDBULB_COMPARTMENT_WEIGHTS)
    segs = {cls: seg_idx_for_classes(cell, (cls,)) for cls in weights}
    # missing compartments: redistribute the fraction to soma
    for cls in list(weights):
        if len(segs.get(cls, [])) == 0 and cls != 'soma':
            weights['soma'] += weights.pop(cls)
    # integer counts, fix any rounding residual on soma
    counts = {cls: int(round(n * w)) for cls, w in weights.items() if len(segs.get(cls, []))}
    counts['soma'] = counts.get('soma', 0) + (n - sum(counts.values()))
    # sample per compartment class, then shuffle
    parts = [np.random.choice(segs[cls], size=cnt, replace=True).astype('int32')
             for cls, cnt in counts.items() if cnt > 0]
    idx = np.concatenate(parts) if parts else np.arange(n, dtype='int32')
    np.random.shuffle(idx)
    return idx


def _section_area_um2(sec):
    return sum(seg.area() for seg in sec)


def _classify_sections():
    """Map each Section to its cnmodel compartment class via the hoc SectionLists.

    Returns {sec_hoc_name: compartment_class}. Sections not in any list default
    to 'soma', which is safe since it gets full density.
    """
    cls = {}
    for name, comp in COMPARTMENT_OF.items():
        sl = getattr(h, name, None)
        if sl is None:
            continue
        for sec in sl:
            cls[sec.name()] = comp
    return cls


def seg_idx_for_classes(cell, classes):
    """LFPy segment indices whose section is in one of the compartment classes."""
    import numpy as np
    compartment_of = _classify_sections()
    idx = []
    for sec in h.allsec():
        if compartment_of.get(sec.name(), 'soma') in classes:
            seg = cell.get_idx(section=sec.name())
            if len(seg):
                idx.append(seg)
    return (np.concatenate(idx).astype('int32') if idx
            else np.array([], dtype='int32'))


# Axon compartment classes in priority order. 'internode' first: when a
# synthetic extended axon is present (axon_builder) that mm-scale tract is the
# physically meaningful long axis and gives a far more robust direction than the
# ~18 um EM stub. Otherwise the myelinated axon, then AIS, then hillock, for
# cells lacking a myelinated segment such as VCN_c02 or the truncated SBC.
_AXON_CLASS_PRIORITY = ('internode', 'myelinatedaxon', 'initialsegment', 'hillock')


def native_axon_direction(cell):
    """Unit vector from soma centroid to axon centroid, in LFPy coordinates.

    Uses the highest-priority axon compartment class present. Returns None if
    the morphology has no axon compartment at all.
    """
    import numpy as np
    soma_idx = seg_idx_for_classes(cell, ('soma',))
    if len(soma_idx) == 0:
        soma_idx = cell.get_idx('soma')
    soma_c = np.array([cell.x[soma_idx].mean(), cell.y[soma_idx].mean(),
                       cell.z[soma_idx].mean()])
    for cls in _AXON_CLASS_PRIORITY:
        ax_idx = seg_idx_for_classes(cell, (cls,))
        if len(ax_idx):
            ax_c = np.array([cell.x[ax_idx].mean(), cell.y[ax_idx].mean(),
                             cell.z[ax_idx].mean()])
            v = ax_c - soma_c
            n = np.linalg.norm(v)
            if n > 0:
                return v / n
    return None


def lfpy_align_angles(v_native, v_target):
    """LFPy set_rotation angles {'x','y','z'} rotating v_native onto v_target.

    LFPy set_rotation(order='xyz') transforms row-vector coords as pos.Rx.Ry.Rz
    with each elementary angle negated (LFPy/cell.py). So for column vectors the
    applied matrix is M = (Rx.Ry.Rz)^T and we need M.v_native = v_target. M is
    built with Rodrigues (aligning v_native to v_target), then (Rx.Ry.Rz) = M^T
    is decomposed with scipy 'XYZ' intrinsic Euler and negated to give LFPy's
    (x, y, z).
    """
    import numpy as np
    from scipy.spatial.transform import Rotation
    a = np.asarray(v_native, float); a /= np.linalg.norm(a)
    b = np.asarray(v_target, float); b /= np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-12:            # parallel or anti-parallel
        if c > 0:
            M = np.eye(3)
        else:                                 # 180 deg: rotate about any perpendicular axis
            perp = np.array([1., 0., 0.]) if abs(a[0]) < 0.9 else np.array([0., 1., 0.])
            axis = np.cross(a, perp); axis /= np.linalg.norm(axis)
            M = Rotation.from_rotvec(np.pi * axis).as_matrix()
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        M = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))
    # M applies to column vectors: M.a = b. LFPy applies (Rx.Ry.Rz) = M^T.
    ax, ay, az = Rotation.from_matrix(M.T).as_euler('XYZ')
    return {'x': -ax, 'y': -ay, 'z': -az}


def decorate_gbc(cell=None, set_nseg=True, verbose=False, ref_ns=None):
    """Insert cnmodel bushy-cell channels with XM13_nacncoop densities.

    Parameters
    ----------
    cell : LFPy.Cell or None
        Unused placeholder so this can be passed as an LFPy custom_fun, which
        LFPy calls with the cell instance. Decoration acts on all NEURON
        sections currently instantiated.
    set_nseg : bool
        If True, set an odd nseg per section via the d_lambda rule (100 Hz).
    ref_ns : dict or None
        Reference somatic conductances (nS). None gives REF_NS_II (GBC, Type
        II); pass REF_NS_II_I for the spherical bushy cell (Type II-I).
    """
    if ref_ns is None:
        ref_ns = REF_NS_II
    compartment_of = _classify_sections()

    # 1) somatic densities (S/cm^2) from reference nS and actual soma area
    soma_area = sum(_section_area_um2(sec)
                    for sec in getattr(h, 'soma', []))
    if soma_area <= 0:
        raise RuntimeError('decorate_gbc: soma SectionList empty or zero area')
    # nS / um^2  ->  S/cm^2  is  * 0.1
    soma_density = {mech: ref_ns[mech] * 0.1 / soma_area for mech in ref_ns}

    # 2) decorate every section
    for sec in h.allsec():
        comp = compartment_of.get(sec.name(), 'soma')

        sec.Ra = RA
        sec.cm = CM_MYELIN if comp == 'internode' else CM
        if set_nseg:
            sec.nseg = int((sec.L / (0.1 * h.lambda_f(100, sec=sec)) + 0.9) / 2) * 2 + 1

        for mech in ('leak', 'kltbc', 'khtbc', 'ihvcn', 'nacncoop'):
            gbar = soma_density[mech] * SCALE[mech][comp]
            if gbar <= 0.0:
                continue
            sec.insert(mech)
            for seg in sec:
                setattr(seg, 'gbar_%s' % mech, gbar)

        # reversals (only meaningful where the ion is used)
        for seg in sec:
            if h.ismembrane('na_ion', sec=sec):
                seg.ena = E_NA
            if h.ismembrane('k_ion', sec=sec):
                seg.ek = E_K
            if hasattr(seg, 'eh_ihvcn'):
                seg.eh_ihvcn = E_H
            if hasattr(seg, 'erev_leak'):
                seg.erev_leak = E_LEAK

    if verbose:
        n_sec = len(list(h.allsec()))
        tot_area = sum(_section_area_um2(s) for s in h.allsec())
        print(f'[decorate_gbc] sections={n_sec}  soma_area={soma_area:.1f} um^2  '
              f'total_area={tot_area:.1f} um^2')
        print('[decorate_gbc] soma densities (S/cm^2): ' +
              ', '.join(f'{m}={soma_density[m]:.3e}' for m in ref_ns))
    return soma_density
