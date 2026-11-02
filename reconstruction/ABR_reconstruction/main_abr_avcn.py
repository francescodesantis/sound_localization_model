#!/usr/bin/env python3
"""AVCN ABR: globular (GBC) and spherical (SBC) bushy-cell populations through
the 4-sphere head model.

  GBC   Type II, VCN_c09 EM morphology with an extended axon to the MNTB
  SBC   Type II-I, SBC_S113 morphology, ~1.5 mm rostral of the GBC field

The GBC dipole is recorded as two edge groups (GBCsoma, GBCaxon, see
recon_core/physics/tree_dipole.py), each projected from its measured centroid;
the SBC as one whole-cell dipole at its own position.

  mpiexec -n 4 python ABR_reconstruction/main_abr_avcn.py --pic-file RESULTS/<f>.pic \
      --angle 0 --side both --full-n

Outputs: RESULTS/abr_tmp/output_{gbc,sbc,avcn}_<stem>_<stimulus>_<side>/
(population dipoles, ABR.h5, figures) and the dipole records.
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

from recon_core.physics import head_model
from recon_core.config import params as P
from recon_core.utils import paths
from recon_core.utils.io_utils import save_dipole_record
from recon_core.utils.mpi_utils import COMM, RANK, broadcast_from_root, reduce_sum
from recon_core.utils.signal_utils import derive
from ABR_reconstruction.plots.common import nucleus_colour

# AVCN populations + geometry (importing this also loads the mechanisms).
from LFP_reconstruction.main_reconstruct_avcn import (
    AVCNPopulation, ELLIPSE_RADIUS_X, ELLIPSE_RADIUS_Y, LAYER_BOUNDARIES, K_YXL,
    _decorate, SYN_DELAY_LOC, SYN_DELAY_SCALE, AXON_TARGET, GBCGroupedPopulation)
from LFP_reconstruction import main_reconstruct_sbc as sbc      # SBC deltas
from LFP_reconstruction.main_reconstruct import _extract_spikes

DT, TSTOP, SRATE = P.DT, P.TSTOP, P.SRATE
V_INIT = P.GBC_V_INIT
N_GBC_TOTAL, N_ENDBULBS = P.N_GBC_TOTAL, P.GBC_ENDBULBS

# Side-specific rotation: the GBC axon crosses the midline on both sides, and the
# nucleus carries the outward rostral tilt of the human cochlear nucleus.
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
# Population registry: what differs between GBC and SBC. 'parts' lists the
# dipole sources of a population with their head positions; 'run' replaces it
# for the GBC, whose two edge groups sit at measured centroids.
# ---------------------------------------------------------------------------
GBC_EXTENDED_HOC = os.path.join(paths.AVCN_MODELS_DIR, 'morphology',
                                'extended', 'VCN_c09_extended_axon.hoc')

POPULATIONS = {
    # One endbulb-driven run recorded as two edge groups (GBCsoma, GBCaxon) that
    # partition the cell.
    'gbc': dict(
        label='GBC', morphology=GBC_EXTENDED_HOC, decorate=_decorate,
        n_post_total=N_GBC_TOTAL, n_endbulbs=N_ENDBULBS, endbulb_weights=None,
        per_pop_syn=AVCNPopulation.PER_POP_SYN, k_yxl=K_YXL,
        ellipse_y=ELLIPSE_RADIUS_Y, seed=44,
        parts=None,                 # positions are MEASURED, not tabulated
        run=dict(tag='grouped', pop_class=GBCGroupedPopulation, X='ANF',
                 decorate=_decorate, k_yxl=K_YXL,
                 per_pop_syn=AVCNPopulation.PER_POP_SYN,
                 delays=P.GBC_DELAYS, delay_scale=P.SYN_DELAY_SCALE,
                 groups=P.GBC_GROUPED_GROUPS),
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
    if getattr(args, 'n_cells', None) is not None:
        return args.n_cells          # explicit --n-cells N: a small-N probe
    return cfg['n_post_total']       # DEFAULT: each population at its own full N


def _run_grouped(pop_name, side, args, meta):
    """The grouped GBC simulation. -> (moments, centroids, out_dir, soma_pos).

    Returns model-frame centroids; _project_and_save rotates them with the same
    matrix as the moments (head_model.rotate_offset_to_head).
    """
    cfg = POPULATIONS[pop_name]
    run = cfg['run']
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
        axon_target=AXON_TARGET[side],
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
        POPULATIONSEED=cfg['seed'],
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
        # reduce_sum returns None off rank 0: guard the centroid division.
        moments[g] = reduce_sum(loc_p)
        tot_c = reduce_sum(loc_c)
        tot_w = reduce_sum(np.array([loc_w]))
        centroids[g] = None
        if RANK == 0:
            centroids[g] = (tot_c / tot_w[0]) if tot_w[0] > 0 else np.zeros(3)
    COMM.Barrier()
    return moments, centroids, output_dir, pop.pop_soma_pos


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
        axon_target=AXON_TARGET[side],   # atlas chord to the contra MNTB
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
# Settling transient: cells start at V_INIT, not at equilibrium, and the first
# steps carry a large relaxation current that the band-pass would ring into a
# spurious early lobe. No synaptically driven current can precede the endbulb
# delay, so everything before it is blanked.
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
    V_out, srate = superpose_sources(sources, electrode_names,
                                     lo=P.BAND_DEFAULT[0], hi=P.BAND_DEFAULT[1])

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

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    for key, V_uV in V_out.items():
        ax0.plot(tvec, V_uV[cz_idx], lw=1.1 if key == 'composite' else 0.9,
                 color='black' if key == 'composite'
                 else nucleus_colour(f'AVCN:{key}'),
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
    # Default (None): full N for each population; see _n_cells.
    parser.add_argument('--n-cells',     type=int, default=None, dest='n_cells')
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
    args = parser.parse_args()

    sides = ['L', 'R'] if args.side == 'both' else [args.side]
    pops  = ['gbc', 'sbc']
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
            if cfg.get('run'):
                # GBC: two grouped dipoles, each at its MEASURED centroid
                dipoles, centroids, output_dir, _ = _run_grouped(
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
        tag = 'avcn'
        final_dir = paths.make_output_dirs(
            paths.output_dir_for('abr', stem, _cond_label(args), args.side,
                                 prefix=tag),
            subdirs=('figures',))

        electrode_names = list(P.ELECTRODES)
        V_out, srate = _apply_head_model(sources, final_dir, electrode_names)
        n_label = ('/'.join(f'{p.upper()} {POPULATIONS[p]["n_post_total"]}'
                            for p in pops) if args.full_n else str(args.n_cells))
        _plot_abr(final_dir, V_out, electrode_names, srate,
                  _cond_label(args), args.side, n_label)


if __name__ == '__main__':
    main()
