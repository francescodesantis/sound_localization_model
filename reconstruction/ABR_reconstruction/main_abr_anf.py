#!/usr/bin/env python3
"""ANF ABR: waves I-II from the auditory nerve's own action potential.

The nerve fibre is a ~24 mm NEURON cable (models/anf). Its transmembrane
currents are the presynaptic generator; the AVCN generators are the
postsynaptic bushy cells, and the endbulb terminal is not modelled.

All fibres share morphology and trajectory and differ only in spike times, so
NEURON runs once per fibre diameter and the population response is the
template convolved with the spike histogram.

The nerve is binned into K dipoles along its path (recon_core/physics/
line_dipole.py), and the boundary reading adds stationary sources at the
modiolus, porus and brainstem entry (recon_core/physics/boundary_dipole.py).
"""

import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from recon_core.physics import boundary_dipole as bd, head_model, line_dipole as ld
from recon_core.config import head_geometry as hg, params as P
from recon_core.utils import paths
from recon_core.utils.io_utils import save_dipole_record
from recon_core.utils.signal_utils import derive

_MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'models', 'anf')

#: Template window, ms: twice the measured settling time of the AP-attributable
#: dipole (17.602 ms), an integer number of DT samples.
TEMPLATE_MS = 35.204

#: Drive: one suprathreshold Exp2Syn EPSC at the peripheral terminal. A synaptic
#: conductance is a membrane current, so the cell conserves charge; an IClamp
#: would inject a monopole the dipole model cannot represent. 0.30 µS is >10x
#: threshold and gives exactly one AP at every node.
ANF_SYN_TAU1 = 0.1                       # ms  fast rise
ANF_SYN_TAU2 = 0.2                       # ms  fast decay, exactly one AP
ANF_SYN_E = 0                            # mV  excitatory reversal
ANF_SYN_WEIGHT = 0.30                    # uS
ANF_STIM_START_MS = 0.5                  # ms, NetStim event time

#: AP initiation threshold at the peripheral terminal, mV (same level as the
#: node crossings in measure_cv).
AP_INIT_THRESHOLD_MV = 0.0


def ap_init_samples(v_terminal):
    """Sample index at which the template's AP initiates at the terminal.

    A cochlear-model spike time is the AP initiation at the terminal, while the
    template's sample 0 precedes the synaptic event by ANF_STIM_START_MS plus the
    time to threshold. The population convolution registers the template on
    this sample.
    """
    idx = np.flatnonzero(np.asarray(v_terminal) > AP_INIT_THRESHOLD_MV)
    if len(idx) == 0:
        raise RuntimeError(
            'template fibre never crossed '
            f'{AP_INIT_THRESHOLD_MV} mV at the terminal: the drive is not '
            'suprathreshold, so the template has no AP to register.')
    return int(idx[0])

#: Version of the cached template content. Bumped whenever build_template's
#: processing or the channel mechanisms change in a way the hashed inputs do not
#: capture (channel densities and mechanism identity are not hashed).
TEMPLATE_CONTENT_VERSION = 'jsrna_junction_iax_v7_axonmyelin_ra70'

#: Tolerance of the bin-sum / whole-cell dipole identity, nA·µm.
IDENTITY_TEST_TOL_NA_UM = 1e-6

#: Largest |head-frame position| of a segment: the fibre's central end.
_R_LAST_UM_MAX = max(float(np.linalg.norm(hg.ANF_PATH_UM[s][-1])) for s in ('R', 'L'))

#: Margin of CHARGE_RESIDUAL_TOL below the identity-test limit (set from the
#: measured multi-spike residual population, max 2.8e-12 nA).
_CHARGE_TOL_MARGIN = 3.0

#: Charge-conservation tolerance of the synaptic drive, nA. The bin-sum identity
#: error equals -r_last * Q(t), with Q(t) = sum_k i_k(t) and r_last the position
#: of the central end, so the identity holds to IDENTITY_TEST_TOL_NA_UM while
#: max|Q| < IDENTITY_TEST_TOL_NA_UM / |r_last|; this sits _CHARGE_TOL_MARGIN below.
CHARGE_RESIDUAL_TOL = IDENTITY_TEST_TOL_NA_UM / (_CHARGE_TOL_MARGIN * _R_LAST_UM_MAX)


def _assert_bin_identity(d, p_whole, label):
    """Raise unless d.sum(axis=0) == p_whole within IDENTITY_TEST_TOL_NA_UM."""
    err = float(np.abs(d.sum(axis=0) - p_whole).max())
    if err >= IDENTITY_TEST_TOL_NA_UM:
        raise RuntimeError(
            f'{label}: bin-sum/whole-cell dipole identity violated: '
            f'max|d.sum(axis=0) - p_whole| = {err:.3e} nA.um >= tol '
            f'{IDENTITY_TEST_TOL_NA_UM:.0e} nA.um. The charge-residual proxy '
            f'(CHARGE_RESIDUAL_TOL={CHARGE_RESIDUAL_TOL:.3e} nA) either was '
            'not checked upstream of this array or is no longer a reliable '
            'predictor for this input -- do not paper over this by loosening '
            'the tolerance; find why the identity itself broke.')


_LAST_WHOLE = {}
_LAST_BIN_ARC_UM = {}


def last_whole_cell_dipole():
    """The whole-cell dipole from the most recent build_template call (3, T)."""
    return _LAST_WHOLE['p']


def last_bin_arc_lengths():
    """Each bin's span along the fibre's own arc length, µm, (n_bins,), for the
    most recent template.
    """
    return _LAST_BIN_ARC_UM['v']


#: Sample index of the AP initiation in the most recent template.
_LAST_AP_INIT = {'n': None}


def ap_init_sample_of_last_template():
    """Registration offset, in samples, for the last template built/loaded."""
    if _LAST_AP_INIT['n'] is None:
        raise RuntimeError('no template built or loaded yet')
    return _LAST_AP_INIT['n']


#: Junction axial currents of the most recent template:
#: {'names': [...], 'arcs_um': (J,), 'i_ax': (J, T)}.
_LAST_JUNCTION = {'v': None}


def junction_currents_of_last_template():
    """Axial current at each conductivity junction, for the last template."""
    if _LAST_JUNCTION['v'] is None:
        raise RuntimeError('no template built or loaded yet')
    return _LAST_JUNCTION['v']


def junction_axial_currents(imem, xs, side):
    """Axial current at each junction -> (names, arcs_um, i_ax (J, T)).

    Gap j between consecutive segments sits at the midpoint of xs[j], xs[j+1];
    each junction takes the gap nearest its arc.
    """
    i_ax = ld.axial_currents(imem)                     # (n_seg-1, T)
    mid = 0.5 * (np.asarray(xs)[:-1] + np.asarray(xs)[1:])
    arcs = hg.junction_arcs_um(side)
    names = sorted(arcs)
    idx = [int(np.argmin(np.abs(mid - arcs[n]))) for n in names]
    return names, np.array([arcs[n] for n in names]), i_ax[idx]


def _geometry_hash(n_bins, side, fiber_diam=None):
    """Hash of every input that changes the template, so a change forces a rebuild."""
    sys.path.insert(0, _MODELS)                      # same as _run_fibre
    import anf_biophysics as _ab                     # noqa: WPS433
    parts = [str(n_bins), side,
             str(P.ANF_FIBER_DIAM if fiber_diam is None else fiber_diam),
             str(P.ANF_PERIPH_DIAM),
             str(P.ANF_NODE_DIAM), str(P.ANF_NODE_LEN), str(P.ANF_INTERNODE_LEN),
             str(P.ANF_SOMA_DIAM), str(P.DT), str(TEMPLATE_MS),
             # peripheral-dendrite myelination geometry and junction arcs
             repr(sorted(hg.junction_arcs_um(side).items())),
             str(P.ANF_TERMINAL_LEN), str(P.ANF_PRESOMA_LEN),
             str(P.ANF_POSTSOMA_LEN), str(P.ANF_PERIPH_N_NODES),
             str(P.ANF_PERIPH_NODE_DIAM),
             # temperature and initial condition
             str(P.BODY_TEMPERATURE_C), str(P.ANF_V_INIT),
             # cable biophysics
             str(_ab.CM_MYELIN_PERIPH), str(_ab.CM_MYELIN_CENTRAL),
             str(_ab.RA), str(_ab.RA_CENTRAL), str(_ab.CM_MEMBRANE),
             str(_ab.G_RATIO_CENTRAL),
             # drive (channel densities: see TEMPLATE_CONTENT_VERSION)
             str(ANF_SYN_TAU1), str(ANF_SYN_TAU2), str(ANF_SYN_E),
             str(ANF_SYN_WEIGHT), str(ANF_STIM_START_MS),
             str(TEMPLATE_CONTENT_VERSION),
             # full head-frame trajectory of both sides (build_anf sizes the
             # central axon from the right side's path)
             np.asarray(hg.ANF_PATH_UM['R']).tobytes().hex(),
             np.asarray(hg.ANF_PATH_UM[side]).tobytes().hex()]
    return hashlib.sha256('|'.join(parts).encode()).hexdigest()[:16]


def template_path(n_bins, side, fiber_diam=None):
    d = os.path.join(paths.ABR_TMP_DIR, 'anf_template')
    os.makedirs(d, exist_ok=True)
    return os.path.join(
        d, f'template_{side}_K{n_bins}_'
           f'{_geometry_hash(n_bins, side, fiber_diam)}.h5')


def segment_positions(arc_um, side):
    """Map each NEURON segment onto the head-frame path by arc length.

    Segments are not uniformly spaced (nseg differs per section), so mapping by
    index would misplace them. `arc_um` is each segment's distance along the
    straight model cable; returns (n_seg, 3) head-frame µm.
    """
    arc = np.asarray(arc_um, dtype=float)
    path = hg.ANF_PATH_UM[side]
    seg_len = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s_path = np.concatenate([[0.0], np.cumsum(seg_len)])
    span = arc.max() - arc.min()
    if span <= 0:
        raise ValueError('degenerate cable: all segments at the same arc length')
    s_want = (arc - arc.min()) / span * s_path[-1]
    return np.stack([np.interp(s_want, s_path, path[:, k]) for k in range(3)],
                    axis=1)


def _run_template_sim(side, stim, fiber_diam=None):
    """One NEURON run of the template cell over TEMPLATE_MS -> (imem, xs, v_terminal).

    stim=True: one suprathreshold EPSC at ANF_STIM_START_MS. stim=False: the
    same cell with no synapse, the spike-independent baseline. The cell is built
    and released within the call.

    imem (n_seg, T) nA and xs (n_seg,) arc length µm in physical cable order.
    """
    sys.path.insert(0, _MODELS)
    import anf_model                                     # noqa: WPS433
    import anf_biophysics                                # noqa: WPS433
    from neuron import h                                 # noqa: WPS433

    # Keep the returned dict alive: it holds the only references to the sections.
    _cell = anf_model.build_anf(fiber_diam=fiber_diam)
    anf_biophysics.decorate_anf()

    # Temperature and fast_imem before the i_membrane_ recording vectors exist.
    h.load_file('stdrun.hoc')
    h.cvode.use_fast_imem(True)
    h.celsius = P.BODY_TEMPERATURE_C

    # Every segment's transmembrane current and arc position, in h.allsec()
    # creation order (peripheral -> central), the order bin_to_dipoles assumes.
    segs, xs = [], []
    for sec in h.allsec():
        x0 = h.x3d(0, sec=sec)
        for seg in sec:
            segs.append(seg)
            xs.append(x0 + seg.x * sec.L)
    xs = np.asarray(xs)
    if np.any(np.diff(xs) < -1e-9):
        raise RuntimeError(
            'segment arc length is not monotone along h.allsec() order -- '
            'bin_to_dipoles assumes physical cable order; refusing to build a '
            'silently-scrambled template.')

    imem_rec = [h.Vector().record(seg._ref_i_membrane_) for seg in segs]
    tvec = h.Vector().record(h._ref_t)
    # Terminal membrane potential, to find the AP initiation sample.
    _term = [sec for sec in h.allsec() if sec.name().startswith('peripheral')][0]
    vterm_rec = h.Vector().record(_term(0.05)._ref_v)

    # One EPSC via NetStim -> Exp2Syn -> NetCon; none at all for the baseline run.
    _keep = []
    if stim:
        periph = [s for s in h.allsec() if s.name().startswith('peripheral')][0]
        ns = h.NetStim()
        ns.number, ns.start, ns.noise = 1, ANF_STIM_START_MS, 0
        syn = h.Exp2Syn(periph(0.05))
        syn.tau1, syn.tau2, syn.e = ANF_SYN_TAU1, ANF_SYN_TAU2, ANF_SYN_E
        nc = h.NetCon(ns, syn)
        nc.weight[0], nc.delay = ANF_SYN_WEIGHT, 0
        _keep = [ns, syn, nc]           # keep alive through h.run()

    # steps_per_ms sets the fixed step; h.dt alone would revert to 0.025 ms.
    h.steps_per_ms = 1.0 / P.DT
    h.dt = P.DT
    h.tstop = TEMPLATE_MS
    h.finitialize(P.ANF_V_INIT)
    h.run()

    if abs(h.dt - P.DT) > 1e-12:
        raise RuntimeError(
            f'h.dt after run() = {h.dt!r}, expected {P.DT!r}: NEURON did not '
            'honour the requested timestep.')
    t_arr = np.asarray(tvec)
    dt_actual = np.diff(t_arr)
    if not np.allclose(dt_actual, P.DT, atol=1e-9):
        raise RuntimeError(
            f'recorded time vector spacing != P.DT: min/max diff '
            f'{dt_actual.min():.6f}/{dt_actual.max():.6f} vs {P.DT}')

    imem = np.array([np.asarray(v) for v in imem_rec])   # (n_seg, T) nA
    return imem, xs, np.asarray(vterm_rec)


def build_template(n_bins=None, side='R', force=False, fiber_diam=None):
    """One fibre, one AP -> (d (K,3,T), r (K,3), srate), cached on disk.

    d[k] is the current dipole moment of bin k in nA·µm in head coordinates. The
    response is debiased: (run with the spike) - (run without it).
    """
    n_bins = P.ANF_N_BINS if n_bins is None else n_bins
    path = template_path(n_bins, side, fiber_diam)
    if os.path.exists(path) and not force:
        with h5py.File(path, 'r') as f:
            _LAST_WHOLE['p'] = f['p_whole'][()]
            _LAST_BIN_ARC_UM['v'] = f['bin_arc_um'][()]
            _LAST_AP_INIT['n'] = int(f['ap_init_sample'][()])
            _LAST_JUNCTION['v'] = {
                'names': [n.decode() if isinstance(n, bytes) else n
                          for n in f['junction_names'][()]],
                'arcs_um': f['junction_arcs_um'][()],
                'i_ax': f['junction_i_ax'][()]}
            return f['d'][()], f['r'][()], float(f['srate'][()])

    imem, xs, vterm = _run_template_sim(side, stim=True, fiber_diam=fiber_diam)
    print(f'ANF template [{side}]: T={imem.shape[1]} samples '
          f'(expected ~{TEMPLATE_MS / P.DT:.1f})')

    # A synaptic drive conserves charge: assert it rather than enforce it.
    resid = np.abs(imem.sum(axis=0)).max()
    print(f'ANF template [{side}]: charge-conservation residual, stim run '
          f'(synaptic drive, no correction applied) = {resid:.3e} nA '
          f'(n_seg={imem.shape[0]})')
    if resid > CHARGE_RESIDUAL_TOL:
        raise RuntimeError(
            f'ANF template [{side}]: charge not conserved: '
            f'max |sum_k i_k(t)| = {resid:.3e} nA > tol {CHARGE_RESIDUAL_TOL:.0e} nA. '
            'A synaptic drive should conserve charge on its own (residual '
            'measured ~7e-13 nA in validation); this size of violation means '
            'something is again injecting current the membrane model does not '
            'see (e.g. a reintroduced IClamp), and must NOT be papered over '
            'by smearing it across segments -- fix the drive instead.')

    # Baseline run: same cell, no synaptic event (the v_init relaxation).
    imem_base, xs_base, _ = _run_template_sim(side, stim=False,
                                             fiber_diam=fiber_diam)
    if imem_base.shape != imem.shape or not np.array_equal(xs_base, xs):
        raise RuntimeError(
            f'ANF template [{side}]: baseline run geometry does not match '
            'the stim run (shape or arc-length positions differ) -- cannot '
            'debias against a mismatched cell build.')
    resid_base = np.abs(imem_base.sum(axis=0)).max()
    print(f'ANF template [{side}]: charge-conservation residual, baseline '
          f'run (no synapse) = {resid_base:.3e} nA')
    if resid_base > CHARGE_RESIDUAL_TOL:
        raise RuntimeError(
            f'ANF template [{side}]: baseline run charge not conserved: '
            f'max |sum_k i_k(t)| = {resid_base:.3e} nA > tol '
            f'{CHARGE_RESIDUAL_TOL:.0e} nA -- refusing to debias against it.')

    # Arc length spanned by each bin (bin_to_dipoles splits by segment count).
    _edges = np.linspace(0, len(xs) - 1, n_bins + 1).astype(int)
    _LAST_BIN_ARC_UM['v'] = np.array(
        [xs[_edges[b + 1]] - xs[_edges[b]] for b in range(n_bins)])

    pos_head = segment_positions(xs, side)

    d_stim, r = ld.bin_to_dipoles(imem, pos_head, n_bins)
    d_base, r_base = ld.bin_to_dipoles(imem_base, pos_head, n_bins)
    if not np.array_equal(r, r_base):
        raise RuntimeError(
            f'ANF template [{side}]: stim and baseline bin positions differ '
            'despite identical geometry -- bin_to_dipoles is not deterministic '
            'for this input, refusing to debias.')

    p_whole_stim = ld.whole_cell_dipole(imem, pos_head)
    p_whole_base = ld.whole_cell_dipole(imem_base, pos_head)
    # Identity checked on both raw runs and on their difference.
    _assert_bin_identity(d_stim, p_whole_stim, f'ANF template [{side}] stim run')
    _assert_bin_identity(d_base, p_whole_base, f'ANF template [{side}] baseline run')

    d = d_stim - d_base
    p_whole = p_whole_stim - p_whole_base
    _assert_bin_identity(d, p_whole, f'ANF template [{side}] debiased')
    srate = 1e3 / P.DT

    # Junction axial currents, debiased the same way.
    j_names, j_arcs, j_iax_stim = junction_axial_currents(imem, xs, side)
    _, _, j_iax_base = junction_axial_currents(imem_base, xs, side)
    j_iax = j_iax_stim - j_iax_base
    _LAST_JUNCTION['v'] = {'names': j_names, 'arcs_um': j_arcs, 'i_ax': j_iax}
    for n, a, row in zip(j_names, j_arcs, j_iax):
        k = int(np.argmax(np.abs(row)))
        print(f'ANF template [{side}]: junction {n:9s} arc {a / 1e3:6.3f} mm  '
              f'peak |I_ax| {np.abs(row[k]):.4g} nA at {k * P.DT:.3f} ms')

    init_sample = ap_init_samples(vterm)
    _LAST_AP_INIT['n'] = init_sample
    print(f'ANF template [{side}]: AP initiates at the terminal on sample '
          f'{init_sample} ({init_sample * P.DT:.3f} ms; NetStim fired at '
          f'{ANF_STIM_START_MS:.3f} ms) -- this is the registration offset '
          'population_dipole is read at (see ap_init_samples).')

    with h5py.File(path, 'w') as f:
        f.create_dataset('ap_init_sample', data=init_sample)
        f.create_dataset('junction_names',
                         data=np.array(j_names, dtype='S16'))
        f.create_dataset('junction_arcs_um', data=j_arcs)
        f.create_dataset('junction_i_ax', data=j_iax)
        f.create_dataset('d', data=d)
        f.create_dataset('r', data=r)
        f.create_dataset('p_whole', data=p_whole)
        f.create_dataset('srate', data=srate)
        f.create_dataset('bin_arc_um', data=_LAST_BIN_ARC_UM['v'])
        f.attrs.update(side=side, n_bins=int(n_bins),
                       units='nA·µm', axes='x=mediolateral, y=anteroposterior, '
                                           'z=inferosuperior',
                       content=TEMPLATE_CONTENT_VERSION)
    _LAST_WHOLE['p'] = p_whole
    print(f'ANF template saved → {path}  K={n_bins} T={d.shape[2]}  '
          f'(debiased: peak|p_stim|={np.abs(p_whole_stim).max():.3e}, '
          f'peak|p_base|={np.abs(p_whole_base).max():.3e}, '
          f'peak|p_debiased|={np.abs(p_whole).max():.3e} nA.um)')
    return d, r, srate


# ---------------------------------------------------------------------------
# Population: convolve the template with the spike histogram
# ---------------------------------------------------------------------------
# All fibres share one morphology and trajectory, so their spikes are summed
# into one histogram before convolving with the template.
def spike_histogram(gdf_path, n_t, dt_ms):
    """All fibres' spikes -> one (n_t,) count histogram at the pipeline's DT."""
    arr = np.loadtxt(gdf_path)
    if arr.size == 0:
        return np.zeros(n_t)
    times = np.atleast_2d(arr)[:, 1]
    idx = np.round(times / dt_ms).astype(int)
    idx = idx[(idx >= 0) & (idx < n_t)]
    hist = np.zeros(n_t)
    np.add.at(hist, idx, 1.0)
    return hist


def population_dipole(d, hist, shift_samples=0):
    """(K,3,T_ap) template * (T,) histogram -> (K,3,T) population dipole.

    shift_samples registers the template that many samples earlier relative to
    the histogram tick; responses that would fall before t = 0 are dropped.
    """
    n_t = len(hist)
    n_pad = n_t + d.shape[2]
    out = np.zeros((d.shape[0], 3, n_t))
    H = np.fft.rfft(hist, n=n_pad)
    for k in range(d.shape[0]):
        for c in range(3):
            D = np.fft.rfft(d[k, c], n=n_pad)
            full = np.fft.irfft(D * H, n=n_pad)
            out[k, c] = full[shift_samples:shift_samples + n_t]
    return out


def _run_direct_fibre(train, side, n_bins):
    """Simulate one fibre driven at exactly `train` (ms) -> (d (K,3,T), r (K,3)),
    binned as build_template does. The cell is built and released within the call.
    """
    sys.path.insert(0, _MODELS)
    import anf_model                                     # noqa: WPS433
    import anf_biophysics                                # noqa: WPS433
    from neuron import h                                 # noqa: WPS433

    _cell = anf_model.build_anf()          # keep alive -- see build_template
    anf_biophysics.decorate_anf()

    h.load_file('stdrun.hoc')
    h.celsius = P.BODY_TEMPERATURE_C

    # One synaptic event per spike time, offset by ANF_STIM_START_MS as in the
    # template so both paths share the same time origin.
    periph = [s for s in h.allsec() if s.name().startswith('peripheral')][0]
    syn = h.Exp2Syn(periph(0.05))
    syn.tau1, syn.tau2, syn.e = ANF_SYN_TAU1, ANF_SYN_TAU2, ANF_SYN_E
    train_drive = np.asarray(train, dtype=float) + ANF_STIM_START_MS
    _vec = h.Vector(train_drive)
    _nc, _keep_ns, _vs = [], [], None
    if hasattr(h, 'VecStim'):
        _vs = h.VecStim()
        _vs.play(_vec)
        _nc = [h.NetCon(_vs, syn)]
    else:                       # no VecStim compiled: one NetStim per spike
        for t0 in train_drive:
            ns = h.NetStim()
            ns.number, ns.start, ns.noise = 1, float(t0), 0.0
            _nc.append(h.NetCon(ns, syn))
            _keep_ns.append(ns)
    for nc in _nc:
        nc.weight[0] = ANF_SYN_WEIGHT
        nc.delay = 0.0

    segs, xs = [], []
    for sec in h.allsec():
        x0 = h.x3d(0, sec=sec)
        for seg in sec:
            segs.append(seg)
            xs.append(x0 + seg.x * sec.L)
    xs = np.asarray(xs)
    h.cvode.use_fast_imem(True)
    recs = [h.Vector().record(seg._ref_i_membrane_) for seg in segs]
    h.steps_per_ms = 1.0 / P.DT      # see build_template: dt alone is a no-op
    h.dt = P.DT
    h.tstop = P.TSTOP
    h.finitialize(P.ANF_V_INIT)
    h.run()

    if abs(h.dt - P.DT) > 1e-12:
        raise RuntimeError(
            f'h.dt after run() = {h.dt!r}, expected {P.DT!r}: NEURON did not '
            'honour the requested timestep.')

    imem = np.array([np.asarray(v) for v in recs])
    resid = float(np.abs(imem.sum(axis=0)).max())
    if resid > CHARGE_RESIDUAL_TOL:
        # Early warning; the identity assertion below is the gate.
        raise RuntimeError(
            f'direct-simulation imem does not conserve charge: {resid:.3e} nA '
            f'> tol {CHARGE_RESIDUAL_TOL:.3e} nA. '
            'Do NOT subtract the mean -- fix the drive.')
    pos = segment_positions(xs, side)          # by arc length, not by index
    d, r = ld.bin_to_dipoles(imem, pos, n_bins)
    # Bin-sum identity on the returned array.
    p_whole = ld.whole_cell_dipole(imem, pos)
    _assert_bin_identity(d, p_whole, 'direct-simulation fibre')
    return d, r


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Validation helpers: does the two-spike response superpose (the population
# convolution's assumption)? Answered within one simulation path.
# ---------------------------------------------------------------------------
FIRE_RATIO_THRESHOLD = 0.5
#: Second-event excess (relative to a single-spike peak) above which the second
#: event counts as a propagating AP; the measured ratio jumps 0.2 -> 1.0 in one
#: sample step, so any value in (0.2, 1.0) is equivalent.

#: Minimum inter-spike interval of the busiest fibre of the 70 dB click, ms.
BUSIEST_FIBRE_MIN_ISI_MS = 1.92

#: Inter-spike intervals swept, ms: from the absolute refractory period to 8 ms.
SUPERPOSITION_DELTAS_MS = (0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0)

#: Larger intervals at which the nonlinearity has decayed (6.2% at 20 ms).
SUPERPOSITION_SANITY_DELTAS_MS = (15.0, 20.0)


def superposition_nonlinearity(deltas_ms=SUPERPOSITION_DELTAS_MS, t0=2.0,
                                side='L', n_bins=None):
    """One fibre, one simulation path: does the two-spike response add?

    R1 = response to [t0]; R2 = response to [t0, t0 + Δ], Δ snapped to a whole
    number of samples; nonlinearity = max|R2 - (R1 + R1 shifted by Δ)| / max|R1|.
    `fired` reports whether the second event produced a propagating AP.
    Returns one dict per Δ.
    """
    n_bins = P.ANF_N_BINS if n_bins is None else n_bins
    d1, _r1 = _run_direct_fibre([t0], side, n_bins)
    p1 = d1.sum(axis=0)                     # (3, T) whole-cell dipole
    peak1 = float(np.abs(p1).max())

    out = []
    for delta_req in deltas_ms:
        n = int(round(delta_req / P.DT))
        delta_ms = n * P.DT                 # SNAPPED -- exact P.DT multiple
        d2, _r2 = _run_direct_fibre([t0, t0 + delta_ms], side, n_bins)
        p2 = d2.sum(axis=0)

        n_common = min(p1.shape[1], p2.shape[1])
        p1c, p2c = p1[:, :n_common], p2[:, :n_common]

        p1_shifted = np.zeros_like(p1c)
        if n < n_common:
            p1_shifted[:, n:] = p1c[:, :n_common - n]
        p2_pred = p1c + p1_shifted

        nonlin = float(np.abs(p2c - p2_pred).max() / peak1)
        excess_ratio = float(np.abs(p2c - p1c).max() / peak1)
        out.append(dict(delta_requested_ms=float(delta_req),
                         delta_ms=float(delta_ms), delta_samples=n,
                         nonlinearity=nonlin, fired=excess_ratio > FIRE_RATIO_THRESHOLD,
                         second_event_excess_ratio=excess_ratio))
    return out


def find_min_firing_delta_samples(t0=2.0, side='L', n_bins=None,
                                   n_lo=1, n_hi=None):
    """Shortest inter-spike interval (in samples) at which a second event still
    fires a propagating AP, by binary search (the response is monotone in Δ).
    """
    n_bins = P.ANF_N_BINS if n_bins is None else n_bins
    n_hi = n_hi if n_hi is not None else int(round(2.0 / P.DT))

    d1, _r1 = _run_direct_fibre([t0], side, n_bins)
    p1 = d1.sum(axis=0)
    peak1 = float(np.abs(p1).max())

    def fired_at(n):
        d2, _r2 = _run_direct_fibre([t0, t0 + n * P.DT], side, n_bins)
        p2 = d2.sum(axis=0)
        n_common = min(p1.shape[1], p2.shape[1])
        excess = float(np.abs(p2[:, :n_common] - p1[:, :n_common]).max())
        return excess / peak1 > FIRE_RATIO_THRESHOLD

    if fired_at(n_lo):
        return n_lo
    if not fired_at(n_hi):
        raise RuntimeError(
            f'no propagating second AP found even at delta={n_hi * P.DT:.3f} '
            'ms -- widen n_hi before trusting this search.')
    lo, hi = n_lo, n_hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        hi, lo = (mid, lo) if fired_at(mid) else (hi, mid)
    return hi


def _align_sup_norm(a, b):
    """Cross-correlation-aligned sup-norm of a vs b, and the lag in samples."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    corr = np.correlate(a - a.mean(), b - b.mean(), mode='full')
    lag = int(corr.argmax() - (n - 1))       # b shifted by +lag aligns to a
    if lag >= 0:
        a_ov, b_ov = a[lag:], b[:n - lag]
    else:
        a_ov, b_ov = a[:n + lag], b[-lag:]
    return float(np.abs(a_ov - b_ov).max()), lag


def _compare_train(train, d, r, srate, side):
    """One spike train through both paths (template convolution, direct
    simulation) -> dict of errors relative to the direct path:
    raw_sup (whole-cell dipole), band_err (band-passed Cz), shift_err and
    shift_lag (Cz after cross-correlation alignment), peak, cz_peak.
    """
    n_bins = d.shape[0]
    n_t = int(round(P.TSTOP / P.DT)) + 1

    # Histogram tick placed with the same round(t/DT) convention NEURON's event
    # queue applies to the template's own event.
    _K0 = int(round(ANF_STIM_START_MS / P.DT))
    hist = np.zeros(n_t)
    idx = np.round((np.asarray(train) + ANF_STIM_START_MS) / P.DT).astype(int) - _K0
    np.add.at(hist, idx[(idx >= 0) & (idx < n_t)], 1.0)
    p_conv = population_dipole(d, hist)                        # (K,3,n_t)

    d_direct, r_direct = _run_direct_fibre(train, side, n_bins)  # (K,3,T)

    n = min(p_conv.shape[2], d_direct.shape[2])
    p_conv, d_direct = p_conv[:, :, :n], d_direct[:, :, :n]

    # Raw sup-norm of the whole-cell dipole (diagnostic).
    p_conv_whole = p_conv.sum(axis=0)
    p_direct_whole = d_direct.sum(axis=0)
    peak = float(np.abs(p_direct_whole).max())
    raw_sup = float(np.abs(p_conv_whole - p_direct_whole).max() / peak)

    # Band-passed Cz trace, each bin projected from its own position.
    sources = ([('conv', p_conv[k], r[k]) for k in range(n_bins)] +
               [('direct', d_direct[k], r_direct[k]) for k in range(n_bins)])
    out = head_model.superpose_sources(
        sources, hg.ABR_ELECTRODES, srate,
        lo=P.BAND_DEFAULT[0], hi=P.BAND_DEFAULT[1])
    cz = hg.ABR_ELECTRODES.index('Cz')
    cz_conv, cz_direct = out['conv'][cz], out['direct'][cz]
    cz_peak = float(np.abs(cz_direct).max())
    band_err = float(np.abs(cz_conv - cz_direct).max() / cz_peak)

    # --- shift-tolerant: same two Cz traces, aligned by cross-correlation.
    shift_diff, shift_lag = _align_sup_norm(cz_conv, cz_direct)
    shift_err = float(shift_diff / cz_peak)

    return dict(n_spikes=len(train), peak=peak, cz_peak=cz_peak,
                raw_sup=raw_sup, band_err=band_err,
                shift_err=shift_err, shift_lag=shift_lag)


def refractoriness_error(side='L', pic_file=None, angle=0):
    """Direct simulation of the busiest real fibre vs its template convolution.

    CONTROL is the fibre's first spike alone (no refractoriness possible); BUSY
    is its full train. The refractoriness cost is BUSY's excess over CONTROL.
    """
    from LFP_reconstruction.main_reconstruct import _extract_spikes

    pic_file = paths.resolve_pic(pic_file)
    stem = paths.pic_stem(pic_file)
    _extract_spikes(angle, side, pic_file=pic_file)
    gdf = os.path.join(paths.spikes_dir_for(stem, angle, side),
                       f'spikes-ANF_{side}-0.gdf')
    arr = np.atleast_2d(np.loadtxt(gdf))
    ids, times = arr[:, 0].astype(int), arr[:, 1]
    busiest = np.bincount(ids).argmax()
    train = np.sort(times[ids == busiest])

    d, r, srate = build_template(P.ANF_N_BINS, side)

    control_train = train[:1]                 # one spike: zero refractoriness
    print(f'refractoriness_error [{side}]: busiest fibre id={busiest}, '
          f'{len(train)} spikes (control: 1 isolated spike from the same '
          'fibre, zero possible refractoriness)')
    control = _compare_train(control_train, d, r, srate, side)
    busy = _compare_train(train, d, r, srate, side)

    def _pair(label, key):
        c, b = control[key] * 100, busy[key] * 100
        print(f'  {label}: control={c:.2f}%  busy={b:.2f}%  excess={b - c:.2f}%')

    print(f'  peak |p_direct| whole-cell (nA.um): control={control["peak"]:.3e}  '
          f'busy={busy["peak"]:.3e}')
    print(f'  peak Cz |p_direct| band-passed (uV): control={control["cz_peak"]:.3e}  '
          f'busy={busy["cz_peak"]:.3e}')
    _pair('raw sup-norm        [DIAGNOSTIC, NOT gated -- sub-sample alignment noise]',
          'raw_sup')
    _pair('band-passed Cz sup-norm   [THE GATE, P.BAND_DEFAULT]', 'band_err')
    _pair('shift-aligned Cz sup-norm [reported, NOT gated]', 'shift_err')
    print(f'  shift-alignment lag (samples): control={control["shift_lag"]}  '
          f'busy={busy["shift_lag"]}')

    return dict(busiest_id=int(busiest), control=control, busy=busy)


def run_side(args, side):
    """Full-N population dipole for one ear -> (p (K,3,T), r (K,3))."""
    from LFP_reconstruction.main_reconstruct import _extract_spikes

    pic_file = paths.resolve_pic(args.pic_file)
    stem = paths.pic_stem(pic_file)
    _extract_spikes(_cond_value(args), side, pic_file=pic_file)
    gdf = os.path.join(paths.spikes_dir_for(stem, _cond_value(args), side),
                       f'spikes-ANF_{side}-0.gdf')

    # Inter-fibre dispersion: one template per diameter, weights normalised, so
    # the total is still P.N_ANF_TOTAL fibres.
    diams = list(P.ANF_DIAM_SET)
    wts = np.asarray(P.ANF_DIAM_WEIGHTS, dtype=float)
    wts = wts / wts.sum()

    n_t = int(round(P.TSTOP / P.DT)) + 1
    hist = spike_histogram(gdf, n_t, P.DT)
    n_total = int(hist.sum())
    print(f'  {side}: {n_total:,} spikes from {P.N_ANF_TOTAL:,} fibres')

    p_pop = None
    r = None
    boundary = {}
    for diam, w in zip(diams, wts):
        d, r_d, _srate = build_template(args.n_bins, side, fiber_diam=diam)
        if r is None:
            r = r_d
        elif not np.array_equal(r, r_d):
            # Bin positions do not depend on diameter.
            raise RuntimeError(
                f'bin positions differ between diameters ({diam} um) -- '
                'refusing to sum templates over inconsistent geometry')
        shift = ap_init_sample_of_last_template()
        contrib = w * population_dipole(d, hist, shift_samples=shift)
        p_pop = contrib if p_pop is None else p_pop + contrib

        jc = junction_currents_of_last_template()
        # Axial-current extent, printed as a diagnostic of the product-form regime.
        ap_um = bd.ap_extent_um(jc['arcs_um'], jc['i_ax'], P.DT)
        # Convolution form: axial current * lead-field transition profile at the
        # volley's measured speed (boundary_dipole.transition_kernel).
        v_um_per_ms = bd.propagation_speed_um_per_ms(
            jc['arcs_um'], jc['i_ax'], P.DT)
        print(f'  {side}: diam {diam:.2f} um -> v = {v_um_per_ms / 1e3:.1f} m/s; '
              f'axial-current extent {ap_um / 1e3:.2f} mm (regime diagnostic)')
        # alpha, kappa and L are per junction (boundary_dipole).
        for name, row in zip(jc['names'], jc['i_ax']):
            if name == 'modiolus' and getattr(
                    args, 'wave1_source', 'modiolus') != 'modiolus':
                continue        # wave I comes from the aperture source
            pos, tan = hg.junction_frame(name, side)
            if name in bd.ORIENT_ALONG_CANAL:
                tan = bd.canal_axis(side)
            k_um = bd.LEAD_FIELD_TRANSITION_UM[name]
            kern = bd.transition_kernel(P.DT, v_um_per_ms, k_um)
            p_one = bd.stationary_dipole(row, bd.ALPHA[name], tan, k_um,
                                         kappa=bd.KAPPA[name], kernel=kern)
            p_j = w * population_dipole(p_one[None, :, :], hist,
                                        shift_samples=shift)
            if name in boundary:
                boundary[name] = (boundary[name][0] + p_j, boundary[name][1])
            else:
                boundary[name] = (p_j, pos[None, :])
    print(f'  {side}: dispersed over {len(diams)} diameters '
          f'{diams[0]:.2f}-{diams[-1]:.2f} um (P.ANF_DIAM_SET)')

    # Zone weighting of the distributed source (boundary_dipole.ZONE_WEIGHTS):
    # scaling the dipole is equivalent to scaling its lead field.
    total_arc = float(np.linalg.norm(
        np.diff(np.asarray(hg.ANF_PATH_UM[side]), axis=0), axis=1).sum())
    K = p_pop.shape[0]
    bin_arcs = (np.arange(K) + 0.5) * total_arc / K
    zw = bd.bin_weights(bin_arcs, hg.junction_arcs_um(side)['porus'])
    if not np.allclose(zw, 1.0):
        zones = {}
        for a_, w_ in zip(bin_arcs, zw):
            zones.setdefault(bd.zone_of(a_, hg.junction_arcs_um(side)['porus']),
                             w_)
        print(f'  {side}: zone weights applied ' +
              ', '.join(f'{k} x{v:.3f}' for k, v in zones.items()))
    p_pop = p_pop * zw[:, None, None]

    dropped = float(hist[:shift].sum())
    print(f'  {side}: template registered on AP-initiation sample {shift} '
          f'({shift * P.DT:.3f} ms); {dropped:,.0f} of {n_total:,} spikes '
          f'({100.0 * dropped / max(n_total, 1):.2f}%) fall in the first '
          f'{shift} samples and are clipped by the shift')

    return p_pop, r, boundary


# ---------------------------------------------------------------------------
# ABR entry point
# ---------------------------------------------------------------------------
# The ANF dipoles are built on the head-frame path, so no model -> head
# rotation is applied.
def _cond_value(args):
    """Raw stimulus key the spike cache is stored under (angle, seconds or dB)."""
    return paths.condition_key(args.angle, args.itd_us, args.ild_db)[0]


def _cond_label(args):
    """Readable stimulus label for the output directory / dipole-record name."""
    return paths.condition_key(args.angle, args.itd_us, args.ild_db)[1]


def _apply_head_model(sources, output_dir, electrode_names):
    """Project every (side, bin) source under the 'ANF' label, sum at the scalp
    and write this run's ABR.h5.
    """
    lo, hi = P.BAND_DEFAULT
    V_out = head_model.superpose_sources(
        [(label, p_head, r) for label, _side, p_head, r in sources],
        electrode_names, P.SRATE, lo=lo, hi=hi)
    with h5py.File(os.path.join(output_dir, 'ABR.h5'), 'w') as f:
        for key, V_uV in V_out.items():
            f.create_dataset(key, data=V_uV)
        f.create_dataset('srate', data=P.SRATE)
        f.create_dataset('electrode_names', data=np.array(electrode_names, dtype='S'))
        f.attrs['units'] = 'µV'
        f.attrs['keys'] = ','.join(V_out)
    print(f'ABR saved -> {os.path.join(output_dir, "ABR.h5")}  keys={list(V_out)}')
    return V_out, P.SRATE


def _plot(output_dir, V_out, electrode_names, srate, cond_label, side,
         derivation='Cz-M1'):
    """Cz trace (top) and the scalp derivation (bottom)."""
    cz = electrode_names.index('Cz')
    tvec = np.arange(next(iter(V_out.values())).shape[1]) / srate * 1e3

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    ax0.plot(tvec, V_out['composite'][cz], color='black', lw=1.2, label='ANF (Cz)')
    ax0.axhline(0, color='k', lw=0.4, ls=':')
    ax0.set_ylabel('Cz potential (µV)')
    ax0.set_title(f'ANF ABR (waves I-II) | {cond_label} | side {side} | '
                 f'N={P.N_ANF_TOTAL:,}')
    ax0.legend(fontsize=9)

    diff, lbl = derive(V_out['composite'], electrode_names, derivation)
    ax1.plot(tvec, diff, color='darkorchid', lw=1.0, label=f'ANF {lbl}')
    ax1.axhline(0, color='k', lw=0.4, ls=':')
    ax1.set_xlabel('Time (ms)'); ax1.set_ylabel('Amplitude (µV)')
    ax1.set_title(f'{lbl}  (vertex-positive upward)')
    ax1.legend(fontsize=9)

    path = os.path.join(output_dir, 'figures', 'anf_abr.png')
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f'ABR figure saved -> {path}')


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='ANF ABR: waves I-II from the fibre\'s own travelling AP')
    parser.add_argument('--pic-file', type=str, default=None, dest='pic_file')
    parser.add_argument('--angle', type=int, default=0)
    parser.add_argument('--itd-us', type=float, default=None, dest='itd_us',
                        help='select an artificial-ITD condition (µs); overrides '
                             '--angle. The pic key is looked up in seconds.')
    parser.add_argument('--ild-db', type=float, default=None, dest='ild_db',
                        help='select an artificial-ILD condition (dB); overrides '
                             '--itd-us and --angle.')
    parser.add_argument('--side', type=str, default='L', choices=['L', 'R', 'both'])
    parser.add_argument('--n-bins', type=int, default=None, dest='n_bins',
                        help='K bins along the fibre (default P.ANF_N_BINS=%d)'
                             % P.ANF_N_BINS)
    parser.add_argument('--force-template', action='store_true',
                        dest='force_template',
                        help='rebuild the cached unitary template(s) even if a '
                             'matching one is already on disk')
    parser.add_argument('--anf-source', type=str, default='boundary',
                        choices=['full', 'boundary'], dest='anf_source',
                        help="nerve representation. full: every bin of the nerve "
                             "(ANF__fibre). boundary (default): wave I from the "
                             "distal nerve, mid-nerve omitted. Both write the "
                             "porus and brainstem stationary potentials.")
    parser.add_argument('--wave1-source', type=str, default='modiolus',
                        choices=['aperture', 'modiolus'], dest='wave1_source',
                        help="wave I source under --anf-source boundary. "
                             "aperture: distal-bin moment projected on "
                             "the canal axis at the modiolus (ANF__distal). "
                             "modiolus (default): boundary source at the canal mouth "
                             "(ANF__modiolus).")
    parser.add_argument('--variant', default=None, choices=['chain'],
                        help='write the dipole records into this variant\'s '
                             'record set (the NEURON chain reads "chain")')
    parser.add_argument('--build-template-only', action='store_true',
                        dest='build_template_only',
                        help='only build/refresh the cached unitary template(s)')
    args = parser.parse_args()

    sides = ['L', 'R'] if args.side == 'both' else [args.side]
    n_bins = P.ANF_N_BINS if args.n_bins is None else args.n_bins

    if args.build_template_only:
        for side in sides:
            build_template(n_bins, side, force=args.force_template)
        return

    if args.force_template:
        # run_side never forces a rebuild, so do it here.
        for side in sides:
            build_template(n_bins, side, force=True)

    stem = paths.pic_stem(paths.resolve_pic(args.pic_file))
    cond_label = _cond_label(args)
    dipoles_dir = paths.dipoles_dir_for(stem, cond_label, variant=args.variant)

    sources = []
    for side in sides:
        p, r, boundary = run_side(args, side)   # (K,3,T) head-frame, (K,3)
        if args.anf_source == 'boundary':
            # Boundary reading: keep only the distal zone (the wave I generator).
            total_arc = float(np.linalg.norm(
                np.diff(np.asarray(hg.ANF_PATH_UM[side]), axis=0), axis=1).sum())
            n_distal = max(1, int(round(
                bd.DISTAL_END_UM / (total_arc / p.shape[0]))))
            p, r = p[:n_distal], r[:n_distal]
            group = P.ANF_DISTAL_GROUP
            print(f'  {side}: BOUNDARY reading -- keeping distal bins '
                  f'0-{n_distal - 1} (0-{n_distal * total_arc / 25 / 1e3:.1f} mm); '
                  'the mid and proximal nerve are suppressed by assertion')
            if args.wave1_source == 'aperture':
                p, r = bd.aperture_source(p, r, side)
                print(f'  {side}: APERTURE source -- the {n_distal} distal bins '
                      f'collapse to one dipole at the modiolus along the canal '
                      f'axis (K={p.shape[0]})')
                write_distributed = True
            else:
                print(f'  {side}: MODIOLUS source -- wave I is the entry-'
                      f'aperture BOUNDARY generator; the distal bins are not '
                      f'written (they would be a second wave I)')
                write_distributed = False
        else:
            group = P.ANF_DISTRIBUTED_GROUP
            write_distributed = True

        # Remove the record of the other --wave1-source, so the composite cannot
        # sum two wave I representations.
        _drop = ('ANF__modiolus' if write_distributed
                 else f'ANF__{P.ANF_DISTAL_GROUP}')
        _stale = os.path.join(dipoles_dir,
                              f'{_drop}__{side}__binaural.h5')
        if os.path.exists(_stale):
            os.remove(_stale)
            print(f'  {side}: removed {_drop} (the other --wave1-source)')

        if write_distributed:
            save_dipole_record(stem, cond_label, 'ANF', group, side, p, r,
                               P.N_ANF_TOTAL, P.N_ANF_TOTAL, P.SRATE,
                               directory=dipoles_dir)
            for k in range(p.shape[0]):
                sources.append(('ANF', side, p[k], r[k]))
        # One record per junction (K=1), kept separate from the wave I record.
        for name, (p_j, r_j) in boundary.items():
            save_dipole_record(stem, cond_label, 'ANF', name, side, p_j, r_j,
                               P.N_ANF_TOTAL, P.N_ANF_TOTAL, P.SRATE,
                               directory=dipoles_dir)
            sources.append(('ANF', side, p_j[0], r_j[0]))

    final_dir = paths.make_output_dirs(
        paths.output_dir_for('abr', stem, cond_label, args.side, prefix='anf'),
        subdirs=('figures',))
    electrode_names = list(P.ELECTRODES)
    V_out, srate = _apply_head_model(sources, final_dir, electrode_names)
    _plot(final_dir, V_out, electrode_names, srate, cond_label, args.side)


if __name__ == '__main__':
    main()
