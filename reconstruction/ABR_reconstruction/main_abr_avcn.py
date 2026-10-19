#!/usr/bin/env python3
"""
AVCN ABR reconstruction: current dipole moment through the 4-sphere head model.

The cochlear nucleus is an early ABR generator (human wave III), upstream of
the MSO. This mirrors main_abr.py but drives morphologically detailed
bushy-cell populations (AVCNPopulation, ANF endbulb input) and places each
dipole at the cochlear-nucleus location.

Two generators are modelled and unified at the ABR stage:
  GBC   globular bushy cell (Type II, VCN_c09 EM), central/caudal VCN
  SBC   spherical bushy cell (Type II-I, SBC_S113 EM), rostral VCN
The spherical-cell area occupies the rostral 1.5-2.0 mm of the nucleus before
the globular region, so the SBC dipole sits ~1.5 mm anterior (+head_y) of the
GBC. Each population's dipole is projected through the 4-sphere model from its
own position and the scalp potentials are summed, since a plain vector sum of
the two dipole moments would only be valid for co-located sources.

CLI (single or MPI):
  python ABR_reconstruction/main_abr_avcn.py --pic-file RESULTS/x.pic \
      --angle 0 --side L --n-cells 200 --generators both
  mpiexec -n 4 python ABR_reconstruction/main_abr_avcn.py ... --side both

Outputs:
  RESULTS/abr_tmp/output_{pop}_{stem}_angle{A}_{side}/population_dipole.h5
  RESULTS/abr_tmp/output_avcn_{stem}_angle{A}_{sidespec}/ABR.h5   (per-pop + composite)
  .../figures/avcn_abr.png                                        (SBC/GBC/composite)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import hybridLFPy

from recon_core import head_model, params as P, paths
from recon_core.io_utils import save_dipole_record
from recon_core.mpi_utils import COMM, RANK, broadcast_from_root, reduce_sum
from recon_core.signal_utils import derive

# AVCN populations + geometry (importing this also loads the mechanisms).
from LFP_reconstruction.main_reconstruct_avcn import (
    GBCSynapticPopulation, GBCSpikingPopulation,
    AVCNPopulation, ELLIPSE_RADIUS_X, ELLIPSE_RADIUS_Y, LAYER_BOUNDARIES, K_YXL,
    _decorate, SYN_DELAY_LOC, SYN_DELAY_SCALE, AXON_TARGET,
)
from LFP_reconstruction import main_reconstruct_sbc as sbc      # SBC deltas
from LFP_reconstruction.main_reconstruct import _extract_spikes

DT, TSTOP, SRATE = P.DT, P.TSTOP, P.SRATE
V_INIT = P.GBC_V_INIT
N_GBC_TOTAL, N_ENDBULBS = P.N_GBC_TOTAL, P.GBC_ENDBULBS

# The rotation is side-specific so the GBC axon, aligned ventromedially in the
# model frame by set_rotations (AXON_TARGET = [-1, 0, -1]), crosses the midline
# on both sides. It also carries the ~32.5 deg outward rostral tilt of the
# human cochlear nucleus (Moore/Osen). _build_rotation(tilt) rebuilds it for
# --avcn-tilt-deg.
from recon_core.head_geometry import (
    AVCN_ROSTRAL_TILT_DEG, build_avcn_rotation as _build_rotation,
)

AVCN_POS_UM = P.AVCN_POS_UM
SBC_POS_UM = P.SBC_POS_UM
ROTATION = P.ROTATION_AVCN

_pic_stem = paths.pic_stem


def superpose_sources(sources, electrode_names, hi=3000., lo=150.):
    """Turn (label, side, p_head, r) sources into band-passed potentials + srate."""
    return (head_model.superpose_sources(
        [(label, p_head, r) for label, _side, p_head, r in sources],
        electrode_names, SRATE, lo=lo, hi=hi), SRATE)


# ---------------------------------------------------------------------------
# Why the dipole IS split -- and why it once was not
#
# The old prohibition was about the WRONG kind of grouping, and the distinction
# is SEGMENTS vs EDGES.
#
# Grouping SEGMENTS is invalid. p = sum(r * i) is translation invariant, and so
# well defined, only when the group's net current is zero. Measured here:
#
#   sum of i_membrane over all segments = 3.0 nA (the injected current; the
#                                         cell as a whole conserves charge)
#   sum over the axonal segments alone  = 3.2 nA (not zero)
#
# Each sub-group carries a large net current, so its dipole is origin dependent,
# and displacing two such groups drops the monopole terms -- with q_axon =
# -q_syn separated by d ~ 3.5 mm the dropped term is itself a large dipole q*d.
# That is what produced the spurious anti-phase cancellation.
#
# Grouping EDGES is valid. With I_j = sum over subtree(j) of i_k, charge
# conservation gives p = sum_j I_j (r_j - r_par(j)): every term is a current
# times a DISPLACEMENT, so every term is individually origin-independent, and
# therefore so is any subset sum. That is the Naess et al. 2021 multi-dipole
# formulation; recon_core/tree_dipole.py implements it from imem and agrees with
# pos.T @ imem to 1.6e-15. See that module for the full argument.
#
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Population registry: everything that differs between GBC and SBC. Defaults in
# AVCNPopulation reproduce the GBC pipeline, so the GBC entry mostly names them.
#
# 'parts' lists the dipole sources a population contributes, each with its own
# head position; 'runs' replaces it for the split GBC, whose positions are
# MEASURED rather than tabulated. The SBC and the lumped GBC each contribute one
# whole-cell dipole, differing in position (SBC 1.5 mm rostral) and in
# morphology: the GBC uses the extended active axon, so its dipole carries the
# travelling volley.
# ---------------------------------------------------------------------------
def _decorate_no_na(cell):
    """LFPy custom_fun for Run A: cnmodel GBC densities with sodium OMITTED.

    decorate_gbc skips `insert` when gbar <= 0, so nacncoop is simply absent --
    no window current, no ena. Every other channel stays: KLT in particular is
    what makes the GBC's EPSP phasic.
    """
    import gbc_biophysics
    gbc_biophysics.decorate_gbc(cell, set_nseg=False,
                                ref_ns=gbc_biophysics.REF_NS_II_NONA)


GBC_EXTENDED_HOC = os.path.join(paths.AVCN_MODELS_DIR, 'morphology',
                                'extended', 'VCN_c09_extended_axon.hoc')

POPULATIONS = {
    # SPLIT, opt-in via --gbc-dipole split: two simulations of the same
    # morphology, three grouped dipoles. "Na absent" and "an injected AP that
    # must propagate" cannot both hold in one run, hence two. Both use the same
    # POPULATIONSEED so the cell positions match cell-for-cell (asserted in
    # _run_split). Costs 1.7x runtime; agrees with lumped to 1% binaurally but
    # differs ~22% monaurally (the superposition bias, which the bilateral
    # cancellation hides).
    'gbc': dict(
        label='GBC', morphology=GBC_EXTENDED_HOC, decorate=_decorate,
        n_post_total=N_GBC_TOTAL, n_endbulbs=N_ENDBULBS, endbulb_weights=None,
        per_pop_syn=AVCNPopulation.PER_POP_SYN, k_yxl=K_YXL,
        ellipse_y=ELLIPSE_RADIUS_Y, seed=44,
        parts=None,                 # positions are MEASURED, not tabulated
        runs=[
            dict(tag='syn', pop_class=GBCSynapticPopulation, X='ANF',
                 decorate=_decorate_no_na, k_yxl=K_YXL,
                 per_pop_syn=AVCNPopulation.PER_POP_SYN,
                 delays=P.GBC_DELAYS, delay_scale=P.SYN_DELAY_SCALE,
                 groups=('GBCsyn',)),
            dict(tag='spike', pop_class=GBCSpikingPopulation, X='GBC',
                 decorate=_decorate, k_yxl=P.GBC_SPIKING_CONVERGENCE,
                 per_pop_syn=P.GBC_SPIKING_SYNAPSES,
                 delays=P.GBC_SPIKING_DELAYS, delay_scale=[None],
                 groups=('GBCspike', 'GBCtrunk')),
        ],
    ),
    # DEFAULT. One whole-cell CurrentDipoleMoment at the nucleus centre, spike
    # timing from NEURON -- the pipeline's long-standing behaviour, unchanged.
    'gbc_lumped': dict(
        label='GBC', morphology=GBC_EXTENDED_HOC, decorate=_decorate,
        n_post_total=N_GBC_TOTAL, n_endbulbs=N_ENDBULBS, endbulb_weights=None,
        per_pop_syn=AVCNPopulation.PER_POP_SYN, k_yxl=K_YXL,
        ellipse_y=ELLIPSE_RADIUS_Y, seed=44,
        parts=[('GBC', None, AVCN_POS_UM)],   # whole-cell dipole, axon included
    ),
    'sbc': dict(
        label='SBC', morphology=sbc.HOC_FILE_SBC, decorate=sbc._decorate_sbc,
        n_post_total=sbc.N_SBC_TOTAL, n_endbulbs=sbc.N_ENDBULBS,
        endbulb_weights=sbc.ENDBULB_WEIGHTS_SBC, per_pop_syn=sbc.PER_POP_SYN_SBC,
        k_yxl=sbc.K_YXL, ellipse_y=sbc.ELLIPSE_RADIUS_Y, seed=45,
        parts=[('SBC', None, SBC_POS_UM)],   # None means plain CurrentDipoleMoment
    ),
}


# ---------------------------------------------------------------------------
# Per-(population, side) simulation, giving a population dipole (3, T) nA.µm
# ---------------------------------------------------------------------------
def _n_cells(args, cfg):
    """How many cells to simulate for THIS population.

    `--n-cells` is one number shared by both populations, and it cannot be full N
    for two populations of different size: the GBC has 3600 cells, the SBC 28000.
    main_abr_full.py sums dipole records with NO scaling, so a shared 3600 lets
    the SBC enter the composite at 13% of its true population.  `--full-n` sizes
    each population from its own `n_post_total` (mirrored live from POP_NUM),
    which is what a physically-weighted composite requires.
    """
    return cfg['n_post_total'] if getattr(args, 'full_n', False) else args.n_cells


def _run_one_grouped(run, cfg, pop_name, side, args, meta):
    """One simulation of a grouped-dipole run. -> (moments, centroids, out_dir).

    Returns model-frame centroids; _project_and_save rotates them with the same
    matrix as the moments (head_model.rotate_offset_to_head).
    """
    X_pops = [f'{run["X"]}_{side}']
    k_arr = np.array(run['k_yxl'])
    n_syn_per_pop = {X: int(k_arr[:, j].sum()) for j, X in enumerate(X_pops)}

    stem = _pic_stem(paths.resolve_pic(args.pic_file))
    cond_val, cond_label = paths.condition_key(args.angle, args.itd_us,
                                               args.ild_db)
    spikes_dir = paths.spikes_dir_for(stem, cond_val, side)
    output_dir = paths.make_output_dirs(
        paths.output_dir_for('abr', stem, cond_label, side,
                             prefix=f'{pop_name}_{run["tag"]}'),
        subdirs=('figures',))

    networkSim = hybridLFPy.CachedNetwork(
        simtime=TSTOP, dt=DT, spike_output_path=spikes_dir,
        label='spikes', ext='gdf',
        GIDs={X: [meta[X]['first_gid'], meta[X]['n_neurons']] for X in X_pops},
        X=X_pops)

    pop = run['pop_class'](
        n_syn_per_pop=n_syn_per_pop,
        axon_target=AXON_TARGET,
        n_post_total=cfg['n_post_total'],
        n_endbulbs=cfg['n_endbulbs'],
        endbulb_weights=cfg['endbulb_weights'],
        per_pop_syn=run['per_pop_syn'],
        y=f'{cfg["label"]}_{side}',
        cellParams={
            'morphology': cfg['morphology'], 'passive': False, 'v_init': V_INIT,
            'dt': DT, 'tstart': 0., 'tstop': TSTOP,
            'nsegs_method': 'lambda_f', 'lambda_f': 100,
            'custom_fun': [run['decorate']], 'custom_fun_args': [{}],
        },
        rand_rot_axis=[],
        simulationParams={'rec_imem': True},
        populationParams={
            'number':   _n_cells(args, cfg),
            'radius':   cfg['ellipse_y'],
            'radius_x': ELLIPSE_RADIUS_X,
            'radius_y': cfg['ellipse_y'],
            'z_min': 0.0, 'z_max': 0.0, 'min_cell_interdist': 1.0,
            'min_r': np.array([[0.], [0.]]),
        },
        layerBoundaries=LAYER_BOUNDARIES,
        probes=[],                      # groups replace the probe output
        savelist=['somapos'],
        savefolder=output_dir,
        dt_output=DT,
        POPULATIONSEED=cfg['seed'],     # SAME for both runs -> same positions
        X=X_pops,
        networkSim=networkSim,
        k_yXL=run['k_yxl'],
        synParams={'section': 'allsec', 'syntype': 'Exp2Syn'},
        synDelayLoc=run['delays'],
        synDelayScale=run['delay_scale'],
        J_yX=[run['per_pop_syn'][run['X']]['weight']],
        tau_yX=[run['per_pop_syn'][run['X']]['tau2']],
    )
    pop.run()
    COMM.Barrier()

    n_t = int(round(TSTOP / DT)) + 1
    moments, centroids = {}, {}
    for g in run['groups']:
        loc_p = loc_c = None
        loc_w = 0.0
        for i in pop.RANK_CELLINDICES:
            d = pop.output[i][g].astype(np.float64)
            loc_p = d.copy() if loc_p is None else loc_p + d
            c = pop.output[i][g + '__c']
            loc_c = c.copy() if loc_c is None else loc_c + c
            loc_w += float(pop.output[i][g + '__w'])
        if loc_p is None:
            loc_p = np.zeros((3, n_t))
            loc_c = np.zeros(3)
        # reduce_sum returns None on non-root ranks (mpi_utils.py:70), so the
        # weighted-centroid division MUST be guarded -- doing it unguarded
        # raises on every non-root rank and deadlocks the rest in the next
        # collective. Only rank 0 uses the centroids (_project_and_save).
        moments[g] = reduce_sum(loc_p)
        tot_c = reduce_sum(loc_c)
        tot_w = reduce_sum(np.array([loc_w]))
        centroids[g] = None
        if RANK == 0:
            centroids[g] = (tot_c / tot_w[0]) if tot_w[0] > 0 else np.zeros(3)
    COMM.Barrier()
    return moments, centroids, output_dir, pop.pop_soma_pos


def _run_split(pop_name, side, args, meta):
    """Both runs of the split GBC generator, merged."""
    cfg = POPULATIONS[pop_name]
    moments, centroids, out_dir, ref_pos = {}, {}, None, None
    for run in cfg['runs']:
        m, c, d, soma_pos = _run_one_grouped(run, cfg, pop_name, side, args, meta)
        if ref_pos is None:
            ref_pos, out_dir = soma_pos, d
        elif RANK == 0:
            # Both runs must place their cells identically, or the two dipoles
            # describe different populations and the soma-centred centroids are
            # inconsistent. Cheap, and it catches a whole class of future edits.
            same = all(np.allclose([ref_pos[i][k] for k in 'xyz'],
                                   [soma_pos[i][k] for k in 'xyz'])
                       for i in range(len(ref_pos)))
            if not same:
                raise RuntimeError(
                    'the split GBC runs placed their cells differently; they '
                    'must share POPULATIONSEED and populationParams')
        moments.update(m)
        centroids.update(c)
    return moments, centroids, out_dir


def _run_one_source(pop_name, side, args, meta):
    from lfpykit import CurrentDipoleMoment
    cfg = POPULATIONS[pop_name]

    X_pops       = [f'ANF_{side}']
    k_yxl_local  = cfg['k_yxl']
    j_yx_local   = [cfg['per_pop_syn']['ANF']['weight']]
    tau_yx_local = [cfg['per_pop_syn']['ANF']['tau2']]

    stem       = _pic_stem(paths.resolve_pic(args.pic_file))
    cond_val, cond_label = paths.condition_key(args.angle, args.itd_us,
                                              args.ild_db)
    spikes_dir = paths.spikes_dir_for(stem, cond_val, side)
    output_dir = paths.make_output_dirs(
        paths.output_dir_for('abr', stem, cond_label, side,
                             prefix=pop_name),
        subdirs=('figures',))

    k_arr         = np.array(k_yxl_local)
    n_syn_per_pop = {X: int(k_arr[:, j].sum()) for j, X in enumerate(X_pops)}

    networkSim = hybridLFPy.CachedNetwork(
        simtime=TSTOP, dt=DT,
        spike_output_path=spikes_dir,
        label='spikes', ext='gdf',
        GIDs={X: [meta[X]['first_gid'], meta[X]['n_neurons']] for X in X_pops},
        X=X_pops,
    )

    # one probe per dipole part (somatodendritic, axonal, or a single plain one)
    probes, part_keys = [], []
    for part_label, probe_cls, _pos in cfg['parts']:
        cls = probe_cls or CurrentDipoleMoment
        probes.append(cls(None))            # cell set by hybridLFPy in cellsim()
        part_keys.append((part_label, cls.__name__))

    pop_label = f'{cfg["label"]}_{side}'
    pop = AVCNPopulation(
        n_syn_per_pop=n_syn_per_pop,
        axon_target=AXON_TARGET,   # fixed ventromedial, laterality via X_pops
        n_post_total=cfg['n_post_total'],
        n_endbulbs=cfg['n_endbulbs'],
        endbulb_weights=cfg['endbulb_weights'],
        per_pop_syn=cfg['per_pop_syn'],
        y=pop_label,
        cellParams={
            'morphology': cfg['morphology'], 'passive': False, 'v_init': V_INIT,
            'dt': DT, 'tstart': 0., 'tstop': TSTOP,
            'nsegs_method': 'lambda_f', 'lambda_f': 100,
            'custom_fun': [cfg['decorate']], 'custom_fun_args': [{}],
        },
        rand_rot_axis=[],   # deterministic ventromedial axon orientation instead
        simulationParams={'rec_imem': True},
        populationParams={
            'number':   _n_cells(args, cfg),
            'radius':   cfg['ellipse_y'],
            'radius_x': ELLIPSE_RADIUS_X,
            'radius_y': cfg['ellipse_y'],
            'z_min': 0.0, 'z_max': 0.0, 'min_cell_interdist': 1.0,
            'min_r': np.array([[0.], [0.]]),
        },
        layerBoundaries=LAYER_BOUNDARIES,
        probes=probes,
        savelist=['somapos'],
        savefolder=output_dir,
        dt_output=DT,
        POPULATIONSEED=cfg['seed'],
        X=X_pops,
        networkSim=networkSim,
        k_yXL=k_yxl_local,
        synParams={'section': 'allsec', 'syntype': 'Exp2Syn'},
        synDelayLoc=SYN_DELAY_LOC,
        synDelayScale=SYN_DELAY_SCALE,
        J_yX=j_yx_local,
        tau_yX=tau_yx_local,
    )

    pop.run()
    COMM.Barrier()

    # Sum per-cell dipole moments (3, T) nA.µm on this rank, per part
    n_t = int(round(TSTOP / DT)) + 1
    dipoles = {}
    for part_label, out_key in part_keys:
        local = None
        for i in pop.RANK_CELLINDICES:
            d = pop.output[i][out_key].astype(np.float64)
            local = d.copy() if local is None else local + d
        if local is None:
            local = np.zeros((3, n_t), dtype=np.float64)
        dipoles[part_label] = reduce_sum(local)
    COMM.Barrier()
    return dipoles, output_dir


# ---------------------------------------------------------------------------
# Head-model projection and saving (rank 0)
# ---------------------------------------------------------------------------
# Settling transient. finitialize starts every cell at V_INIT, which is not its
# equilibrium, so the first timestep carries a large whole-cell relaxation
# current: measured 147,034 nA·µm at t=0 against a 179,345 nA·µm wave-II peak,
# collapsing 29x within 0.1 ms. The zero-phase band-pass rings off that impulse
# into a lobe at ~0.16 ms which is 85% of the real wave II (0.70 vs 0.82 nV at
# Cz) and sits exactly where wave I would be -- the one wave this pipeline does
# not model, so it is mistakable for one.
#
# The window is DERIVED, not chosen: ANF spikes cannot occur before t = 0, so no
# synaptically driven AVCN current can precede the endbulb delay. Everything
# before min(ANFs2GBCs, ANFs2SBCs) is provably non-neural, whatever the stimulus.
#
# NOT P.SETTLE_MS (3.0 ms): that is safe for the LSO, whose NEST spikes start
# after 6 ms, but it would erase the AVCN's 2.5 ms wave II outright.
SETTLE_MS = min(P.GBC_DELAYS[0], P.SBC_DELAYS[0])


def _project_and_save(side, dipoles, output_dir, centroids=None,
                      origin=None):
    """Blank the settling transient, rotate into head coords, save {part: p_head}.

    With `centroids` (model-frame, from the grouped decomposition) each part
    also gets its own measured head position, written as `r__<part>` so the file
    is self-describing. `origin` is the nucleus centre those offsets hang off.
    The SAME rotation matrix is applied to the moment and to the position.
    """
    srate = SRATE
    out, pos_out = {}, {}
    n_settle = int(round(SETTLE_MS / DT))
    with h5py.File(os.path.join(output_dir, 'population_dipole.h5'), 'w') as f:
        for part, p_model in dipoles.items():
            p_model = np.asarray(p_model, dtype=float).copy()
            p_model[:, :n_settle] = 0.0
            p_head = head_model.rotate_to_head(p_model, ROTATION[side])
            f.create_dataset(part, data=p_head)
            out[part] = p_head
            if centroids is not None and part in centroids:
                r = head_model.rotate_offset_to_head(
                    centroids[part], ROTATION[side], origin[side])
                f.create_dataset(f'r__{part}', data=r)
                pos_out[part] = r
        f.create_dataset('srate', data=srate)
        f.attrs['axes']  = 'x=mediolateral, y=anteroposterior, z=inferosuperior'
        f.attrs['units'] = 'nA·µm'
        f.attrs['parts'] = ','.join(out)
    return (out, pos_out) if centroids is not None else out


def _apply_head_model(sources, output_dir, electrode_names):
    """Project each source dipole from its own position and sum at the scalp.

    Thin wrapper over main_abr.superpose_sources, the one 4-sphere
    superposition also used by the cross-nucleus composite in main_abr_full.py,
    that additionally writes this run's ABR.h5. sources is a list of
    (part_label, side, p_head, r_dipole); sources sharing a part_label, such as
    both sides of one generator, are summed.

    Returns (V_by_key, srate), where V_by_key maps each part label and
    'composite' to a band-passed (n_electrodes, n_t) array in µV.
    """
    V_out, srate = superpose_sources(sources, electrode_names, hi=3000., lo=150.)

    with h5py.File(os.path.join(output_dir, 'ABR.h5'), 'w') as f:
        for key, V_uV in V_out.items():
            f.create_dataset(key, data=V_uV)
        f.create_dataset('srate', data=srate)
        f.create_dataset('electrode_names', data=np.array(electrode_names, dtype='S'))
        f.attrs['units'] = 'µV'
        f.attrs['keys']  = ','.join(V_out)
    print(f'ABR saved to {os.path.join(output_dir, "ABR.h5")}  keys={list(V_out)}')
    return V_out, srate


_derivation = derive


def _plot_abr(output_dir, V_out, electrode_names, srate, cond_label, side,
              n_cells, derivation='Cz-M1'):
    """Top: Cz per population and composite. Bottom: composite derivation."""
    cz_idx = electrode_names.index('Cz')
    any_V  = next(iter(V_out.values()))
    tvec   = np.arange(any_V.shape[1]) / srate * 1e3
    colours = {'GBC': 'seagreen', 'SBC': 'darkorange', 'composite': 'black'}

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    for key, V_uV in V_out.items():
        ax0.plot(tvec, V_uV[cz_idx], lw=1.1 if key == 'composite' else 0.9,
                 color=colours.get(key, None),
                 label=f'{key} (Cz)', zorder=3 if key == 'composite' else 2)
    ax0.axhline(0, color='k', lw=0.4, ls=':')
    ax0.set_ylabel('Cz potential (µV)')
    ax0.set_title(f'AVCN ABR (SBC + GBC) | {cond_label} | side {side} | N={n_cells}')
    ax0.legend(fontsize=9)

    diff, lbl = _derivation(V_out['composite'], electrode_names, derivation)
    ax1.plot(tvec, diff, color='darkorchid', lw=1.0, label=f'composite {lbl}')
    ax1.axhline(0, color='k', lw=0.4, ls=':')
    ax1.set_xlabel('Time (ms)'); ax1.set_ylabel('Amplitude (µV)')
    ax1.set_title(f'{lbl}  (vertex-positive upward)')
    ax1.legend(fontsize=9)

    path = os.path.join(output_dir, 'figures', 'avcn_abr.png')
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f'ABR figure saved to {path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cond_value(args):
    """Raw stimulus key the spike cache is stored under (angle, seconds or dB)."""
    return paths.condition_key(args.angle, args.itd_us, args.ild_db)[0]


def _cond_label(args):
    """Readable stimulus label for the output directory name."""
    return paths.condition_key(args.angle, args.itd_us, args.ild_db)[1]


def main():
    import argparse
    parser = argparse.ArgumentParser(description='AVCN (SBC + GBC) ABR reconstruction')
    parser.add_argument('--pic-file',    type=str, default=None, dest='pic_file')
    parser.add_argument('--angle',       type=int, default=0)
    parser.add_argument('--side',        type=str, default='L',
                        choices=['L', 'R', 'both'])
    parser.add_argument('--n-cells',     type=int, default=200, dest='n_cells')
    parser.add_argument('--full-n', action='store_true', dest='full_n',
                        help='size each population from its own n_post_total '
                             '(GBC 3600, SBC 28000) instead of the shared '
                             '--n-cells. Required for a physically-weighted '
                             'composite, since dipoles are summed unscaled.')
    parser.add_argument('--itd-us', type=float, default=None, dest='itd_us',
                        help='select an artificial-ITD condition (µs); overrides '
                             '--angle. The pic key is looked up in seconds.')
    parser.add_argument('--ild-db', type=float, default=None, dest='ild_db',
                        help='select an artificial-ILD condition (dB); overrides '
                             '--itd-us and --angle.')
    parser.add_argument('--generators', type=str, default='both',
                        choices=['gbc', 'sbc', 'both'], dest='generators',
                        help='which bushy-cell generators to model. They are '
                             'distinct cell types, so "both" SUMS them at the scalp.')
    parser.add_argument('--gbc-dipole', type=str, default='lumped',
                        choices=['split', 'lumped'], dest='gbc_dipole',
                        help='how the GBC contributes its dipole. lumped '
                             '(default): one whole-cell GBC record, spike '
                             'timing from NEURON. split: three grouped records '
                             'GBCsyn/GBCspike/GBCtrunk from two runs, with NEST '
                             'spikes driving the axon (1.7x runtime; agrees '
                             'with lumped to 1%% binaurally but differs ~22%% '
                             'monaurally). ORTHOGONAL to --generators, which '
                             'picks the cell type.')
    parser.add_argument('--derivation',  type=str, default='Cz-M1',
                        choices=['Cz-M1', 'Cz-M2', 'Cz-avg'])
    parser.add_argument('--gbc-hoc', type=str, default=None, dest='gbc_hoc',
                        help='override the GBC morphology (default: extended '
                             'active axon; pass the plain VCN_c* EM hoc to '
                             'measure the axonal travelling-wave contribution)')
    parser.add_argument('--avcn-tilt-deg', type=float, default=AVCN_ROSTRAL_TILT_DEG,
                        dest='avcn_tilt_deg',
                        help='outward rostral tilt of the CN about the head '
                             'vertical axis (deg); 0 = untilted (default %(default)s)')
    args = parser.parse_args()

    global ROTATION
    ROTATION = _build_rotation(args.avcn_tilt_deg)

    if args.gbc_hoc:
        POPULATIONS['gbc']['morphology'] = args.gbc_hoc

    sides = ['L', 'R'] if args.side == 'both' else [args.side]
    pops  = ['gbc', 'sbc'] if args.generators == 'both' else [args.generators]
    # --gbc-dipole is ORTHOGONAL to --generators (which picks the cell type,
    # gbc|sbc|both). 'lumped' swaps in the legacy single-dipole registry entry
    # so the old behaviour is the same code, not a copy.
    if args.gbc_dipole == 'lumped':
        pops = ['gbc_lumped' if p == 'gbc' else p for p in pops]
    stem  = _pic_stem(paths.resolve_pic(args.pic_file))

    # Extract presynaptic spikes once per side; both populations share the ANF drive.
    meta_by_side = {
        side: broadcast_from_root(
            lambda side=side: _extract_spikes(_cond_value(args), side,
                                              pic_file=args.pic_file))
        for side in sides
    }

    sources = []   # (part_label, side, p_head, r_dipole) on rank 0
    for pop_name in pops:
        cfg = POPULATIONS[pop_name]
        for side in sides:
            if cfg.get('runs'):
                # split GBC: three grouped dipoles, each at its MEASURED centroid
                dipoles, centroids, output_dir = _run_split(
                    pop_name, side, args, meta_by_side[side])
                if RANK == 0:
                    p_heads, r_heads = _project_and_save(
                        side, dipoles, output_dir, centroids, AVCN_POS_UM)
                    for part_label in dipoles:
                        sources.append((part_label, side, p_heads[part_label],
                                        r_heads[part_label]))
                        save_dipole_record(stem, _cond_label(args), 'AVCN',
                                           part_label, side,
                                           p_heads[part_label],
                                           r_heads[part_label],
                                           cfg['n_post_total'],
                                           _n_cells(args, cfg), SRATE)
                continue

            dipoles, output_dir = _run_one_source(pop_name, side, args,
                                                  meta_by_side[side])
            if RANK == 0:
                p_heads = _project_and_save(side, dipoles, output_dir)
                for part_label, _probe_cls, pos_map in cfg['parts']:
                    sources.append((part_label, side, p_heads[part_label],
                                    pos_map[side]))
                    save_dipole_record(stem, _cond_label(args), 'AVCN',
                                       part_label, side,
                                       p_heads[part_label], pos_map[side],
                                       cfg['n_post_total'],
                                       _n_cells(args, cfg), SRATE)

    if RANK == 0:
        tag = 'avcn' if len(pops) > 1 else pops[0]
        final_dir = paths.make_output_dirs(
            paths.output_dir_for('abr', stem, _cond_label(args), args.side,
                                 prefix=tag),
            subdirs=('figures',))

        electrode_names = list(P.ELECTRODES)
        V_out, srate = _apply_head_model(sources, final_dir, electrode_names)
        n_label = ('/'.join(f'{p.upper()} {POPULATIONS[p]["n_post_total"]}'
                            for p in pops) if args.full_n else str(args.n_cells))
        _plot_abr(final_dir, V_out, electrode_names, srate,
                  _cond_label(args), args.side, n_label,
                  derivation=args.derivation)


if __name__ == '__main__':
    main()
