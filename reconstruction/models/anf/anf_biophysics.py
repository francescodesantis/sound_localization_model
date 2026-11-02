#!/usr/bin/env python3
"""Channel decoration of the human auditory-nerve fibre.

Four compartment classes, set by SectionList membership:

    peripheral   unmyelinated terminal (organ of Corti), pre- and post-somatic
                 segments: active, thin, slow
    soma         unmyelinated in humans: active, large capacitance
    node         node of Ranvier (peripheral dendrite and central axon)
    internode    myelin: passive, low capacitance (peripheral and central)

Fibre diameter is the sourced quantity (human temporal-bone morphometry);
conduction velocity follows from it and is measured (measure_cv), not tuned.

Mechanisms: sodium `jsrna` (models/anf/jsrnaf.mod, Rothman, Young & Manis
1993, whose steep voltage-dependent recovery from inactivation gives an
absolute refractory period of ~0.7 ms, as in cat single-fibre recordings);
potassium and leak from models/avcn (khtbc, kltbc, leak).
"""

import os
import sys

import numpy as np
from neuron import h

_HERE = os.path.dirname(os.path.abspath(__file__))
_AVCN = os.path.join(os.path.dirname(_HERE), 'avcn')
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

from recon_core.config import params as P                      # noqa: E402

COMPARTMENT_OF = {
    'Peripheral_Process': 'peripheral',
    'soma': 'soma',
    'Node_of_Ranvier': 'node',
    'Internode': 'internode',
}

E_LEAK = -65.0        # mV
CM_MEMBRANE = 0.9     # uF/cm^2  unmyelinated membrane
# Axial resistivity: 70 ohm*cm on the central axon (large myelinated fibre,
# McIntyre-Richardson-Grill), 150 on the peripheral dendrite, soma and its
# nodes. With the per-fibre myelin below the central axon follows the human
# fibre law v ~ 6.7 x d_axon (Rattay et al. 2013).
RA = 150.0            # ohm.cm   peripheral dendrite, soma, nodes there
RA_CENTRAL = 70.0     # ohm.cm   central axon (internode_*/node_* only)

# --- myelin capacitance, per compartment -------------------------------------
# Peripheral dendrite (~1.5 µm, ~20-30 wraps): 0.03 µF/cm², mid-range of 0.02-0.04.
# Central axon: from each fibre's own sheath. Diameters are axon diameters
# (Rattay et al. 2013), outer = d / G_RATIO_CENTRAL, myelin (D - d)/2 at the
# 15.6 nm lamella period, two membranes per lamella in series.
CM_MYELIN_PERIPH = 0.03    # uF/cm^2  peripheral dendrite internodes
MYELIN_LAMELLA_PERIOD_UM = 0.0156
G_RATIO_CENTRAL = 0.7


def myelin_cm(outer_diam_um, g_ratio=G_RATIO_CENTRAL, cm=CM_MEMBRANE,
              period_um=MYELIN_LAMELLA_PERIOD_UM):
    """Internode capacitance uF/cm^2 from the fibre's own myelin thickness."""
    n = (outer_diam_um * (1.0 - g_ratio) / 2.0) / period_um
    return cm / (2.0 * n)


def central_myelin_cm(axon_diam_um):
    """Central internode capacitance for an AXON of this diameter (outer = d/g)."""
    return myelin_cm(axon_diam_um / G_RATIO_CENTRAL)


#: Central myelin capacitance of the default fibre (decorate_anf uses
#: central_myelin_cm(sec.diam) per internode).
CM_MYELIN_CENTRAL = central_myelin_cm(P.ANF_FIBER_DIAM)

#: Conductance densities, S/cm²: sodium on the nodes, passive internodes.
GBAR = {
    'node':       {'jsrna': 3.0,  'khtbc': 0.020, 'kltbc': 0.010, 'leak': 0.0005},
    'peripheral': {'jsrna': 0.30, 'khtbc': 0.010, 'kltbc': 0.005, 'leak': 0.0005},
    'soma':       {'jsrna': 0.15, 'khtbc': 0.010, 'kltbc': 0.005, 'leak': 0.0005},
    'internode':  {'leak': 0.00001},
}


#: Segment-wise timings from the last measure_cv() call (peripheral CV,
#: terminal-to-soma delay).
LAST_CV_DETAIL = {}


def _load_mechanisms():
    """Load the fibre's mechanisms if not already loaded: `jsrna` (models/anf)
    and `khtbc`/`kltbc`/`leak` (models/avcn), via mpi_utils.load_mechanisms.
    """
    from recon_core.utils.mpi_utils import load_mechanisms
    if not hasattr(h, 'jsrna'):
        if not os.path.isdir(os.path.join(_HERE, 'x86_64')):
            raise RuntimeError(
                'NEURON mechanisms not built. Build them once with:\n'
                f'    cd {_HERE} && nrnivmodl')
        load_mechanisms(_HERE)
    if not hasattr(h, 'khtbc'):
        if not os.path.isdir(os.path.join(_AVCN, 'x86_64')):
            raise RuntimeError(
                'NEURON mechanisms not built. Build them once with:\n'
                f'    cd {_AVCN} && nrnivmodl')
        load_mechanisms(_AVCN)


def _anf_sections():
    """{hoc section name: Section} for every section in one of the ANF's own
    SectionLists (COMPARTMENT_OF).  This is the ONLY way decorate_anf finds
    sections to touch -- see the coexistence note below.
    """
    secs = {}
    for sl_name in COMPARTMENT_OF:
        sl = getattr(h, sl_name, None)
        if sl is None:
            continue
        for sec in sl:
            secs[sec.name()] = sec
    return secs


def _classify_sections():
    """Map each ANF section to its compartment class by SectionList membership.

    Only sections in the ANF's own SectionLists are returned, so co-resident
    cells (other models in the same process) are never decorated.
    """
    cls = {}
    for sl_name, comp in COMPARTMENT_OF.items():
        sl = getattr(h, sl_name, None)
        if sl is None:
            continue
        for sec in sl:
            cls[sec.name()] = comp
    return cls


def decorate_anf(verbose=False):
    """Insert mechanisms on every section of the live ANF cell.

    Iterates ONLY sections registered in the ANF's own SectionLists (via
    _anf_sections()/_classify_sections()), never unfiltered h.allsec() -- see
    _classify_sections()'s docstring for why that distinction matters the
    moment another cell is resident in the same process.
    """
    _load_mechanisms()
    anf_secs = _anf_sections()
    compartment_of = _classify_sections()
    counts = {}
    for name, sec in anf_secs.items():
        # 'soma' as a default only; every section is in compartment_of.
        cls = compartment_of.get(name, 'soma')
        counts[cls] = counts.get(cls, 0) + 1
        # RA_CENTRAL on the central axon; peripheral sections carry a 'p' prefix.
        sec.Ra = RA if name.startswith('p') or name in ('soma',) else RA_CENTRAL
        # Peripheral and central internodes share a class and differ in myelin,
        # so they are told apart by name (anf_secs holds only this fibre).
        if cls == 'internode':
            sec.cm = (CM_MYELIN_PERIPH if name.startswith('pinternode')
                      else central_myelin_cm(sec.diam))
        else:
            sec.cm = CM_MEMBRANE
        sec.nseg = max(1, int(sec.L // 10.0) * 2 + 1)
        for mech, gbar in GBAR[cls].items():
            sec.insert(mech)
            for seg in sec:
                if mech == 'leak':
                    setattr(seg, 'gbar_leak', gbar)
                    setattr(seg, 'erev_leak', E_LEAK)
                else:
                    setattr(seg, 'gbar_%s' % mech, gbar)
        for seg in sec:
            if h.ismembrane('na_ion', sec=sec):
                seg.ena = 50.0
            if h.ismembrane('k_ion', sec=sec):
                seg.ek = -84.0
    if verbose:
        print('[anf_biophysics] decorated ' +
              ', '.join(f'{v} {k}' for k, v in sorted(counts.items())))


def measure_cv(stim_amp=2.0, threshold_mv=0.0, return_traces=False):
    """Conduction velocity along the central axon, m/s, from node_1 to the last
    node (node_0 sits at the soma junction, where the AP is still settling).

    Builds the only IClamp and runs the only simulation; callers must not add
    their own. return_traces=True also returns the (n_nodes, T) node voltages.
    """
    # Central nodes only (the peripheral dendrite's nodes, 'pnode_*', have
    # their own CV).
    nodes = [s for s in h.allsec() if s.name().startswith('node')]
    if len(nodes) < 3:
        raise RuntimeError(f'need >=3 nodes to measure CV, found {len(nodes)}')
    nodes.sort(key=lambda s: int(s.name().split('_')[1]))

    pnodes = [s for s in h.allsec() if s.name().startswith('pnode')]
    pnodes.sort(key=lambda s: int(s.name().split('_')[1]))
    soma = [s for s in h.allsec() if s.name() == 'soma']

    periph = [s for s in h.allsec() if s.name().startswith('peripheral')][0]
    stim = h.IClamp(periph(0.05))
    stim.delay, stim.dur, stim.amp = 1.0, 0.1, stim_amp

    recs = [h.Vector().record(n(0.5)._ref_v) for n in nodes]
    # Extra probes for the peripheral segment. Recorded in the SAME run -- the
    # fibre must not be stimulated twice (see this function's docstring).
    p_recs = [h.Vector().record(n(0.5)._ref_v) for n in pnodes]
    term_rec = h.Vector().record(periph(0.5)._ref_v)
    soma_rec = h.Vector().record(soma[0](0.5)._ref_v) if soma else None
    tvec = h.Vector().record(h._ref_t)
    h.load_file('stdrun.hoc')          # defines tstop/run(); not loaded by anf_model
    h.celsius = P.BODY_TEMPERATURE_C   # the pipeline temperature
    # steps_per_ms sets the fixed step; h.dt alone would revert to 0.025 ms.
    h.steps_per_ms = 1.0 / P.DT
    h.dt = P.DT
    h.tstop = 10.0
    h.finitialize(P.ANF_V_INIT)
    h.run()

    # Check the step and the recorded sample spacing.
    if abs(h.dt - P.DT) > 1e-12:
        raise RuntimeError(
            f'h.dt did not stick: asked for {P.DT}, NEURON is at {h.dt}')
    t = np.asarray(tvec)
    sample_dt = np.diff(t)
    if not np.allclose(sample_dt, P.DT, atol=1e-9):
        raise RuntimeError(
            f'recorded sample interval does not match P.DT={P.DT}: '
            f'got min={sample_dt.min():.6f} max={sample_dt.max():.6f} ms')

    def cross(rec):
        v = np.asarray(rec)
        idx = np.flatnonzero(v > threshold_mv)
        return t[idx[0]] if len(idx) else np.nan

    t_first, t_last = cross(recs[1]), cross(recs[-1])
    if not np.isfinite(t_first) or not np.isfinite(t_last):
        raise RuntimeError('AP did not reach both ends; cannot measure CV')

    def xmid(sec):
        return 0.5 * (h.x3d(0, sec=sec) + h.x3d(int(h.n3d(sec=sec)) - 1, sec=sec))

    dist_um = xmid(nodes[-1]) - xmid(nodes[1])
    dt_ms = t_last - t_first
    cv = (dist_um * 1e-6) / (dt_ms * 1e-3)              # m/s

    # Peripheral dendrite, from the same run, into LAST_CV_DETAIL.
    LAST_CV_DETAIL.clear()
    LAST_CV_DETAIL['central_cv'] = cv
    if len(pnodes) >= 2:
        tp0, tp1 = cross(p_recs[0]), cross(p_recs[-1])
        if np.isfinite(tp0) and np.isfinite(tp1) and tp1 > tp0:
            LAST_CV_DETAIL['periph_cv'] = (
                (xmid(pnodes[-1]) - xmid(pnodes[0])) * 1e-6 / ((tp1 - tp0) * 1e-3))
    t_term = cross(term_rec)
    t_soma = cross(soma_rec) if soma_rec is not None else np.nan
    if np.isfinite(t_term) and np.isfinite(t_soma):
        LAST_CV_DETAIL['terminal_to_soma_ms'] = t_soma - t_term

    if return_traces:
        return cv, np.array([np.asarray(r) for r in recs])
    return cv
