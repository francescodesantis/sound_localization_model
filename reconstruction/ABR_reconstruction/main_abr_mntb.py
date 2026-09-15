#!/usr/bin/env python3
"""
MNTB ABR reconstruction: current dipole moment through the 4-sphere head model.

Two co-located generators, each a whole-cell CurrentDipoleMoment summed at the
scalp, following the SBC+GBC composite pattern of main_abr_avcn.py:
  PRINCIPAL   MNTB principal cell, the calyx EPSC somatodendritic sink.
  CALYX       calyx of Held presynaptic terminal, the prespike, leading by
              ~0.5 ms since the postsynaptic sink carries the GBCs2MNTBCs delay.

Both are driven by the contralateral GBC spike train and placed at the MNTB
position (~400 µm from the midline, medial and anterior to the MSO).

The GBC axon crossing the midline, the dominant wave-III generator (Karadas
2021), is modelled in the AVCN GBC ABR (main_abr_avcn.py,
VCN_c09_extended_axon.hoc) and is not re-added here, so nothing is counted
twice. The MNTB's own contribution (closed-field soma plus the radial calyx
current) is expected to be small.

CLI:
  mpiexec -n 4 python ABR_reconstruction/main_abr_mntb.py --pic-file RESULTS/<f>.pic \
      --angle 0 --side both --n-cells 200 [--generators principal|calyx|both]

Outputs go to RESULTS/abr_tmp/output_mntb_{stem}_angle{A}_{sidespec}/:
  ABR.h5 (principal, calyx and composite), figures/mntb_abr.png
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

# MNTB LFP machinery (importing it also loads the MNTB/MSO mechanisms).
from LFP_reconstruction.main_reconstruct_mntb import (
    MNTBPopulation, ELLIPSE_RADIUS_X, ELLIPSE_RADIUS_Y, LAYER_BOUNDARIES,
    HOC_FILE as MNTB_HOC,
)
from LFP_reconstruction.main_reconstruct_calyx import CalyxPopulation, CALYX_HOC
from LFP_reconstruction.main_reconstruct import _extract_spikes

DT, TSTOP, SRATE = P.DT, P.TSTOP, P.SRATE
V_INIT = P.MNTB_V_INIT
N_MNTB_TOTAL = P.N_MNTB_TOTAL

# The rotation is side-specific: the pre-calyx axon (model x) becomes the
# mediolateral crossing trapezoid-body fibre, mirrored across sides so the
# calyx dipoles sum coherently, and a single tilt about head x lifts the
# dendritic axis to MNTB_LONGAXIS_FROM_HORIZ_DEG (Kulesza) while leaving that
# dominant mediolateral dipole invariant.
MNTB_POS_UM = P.MNTB_POS_UM
ROTATION = P.ROTATION_MNTB

_pic_stem = paths.pic_stem

# per-generator config
POPULATIONS = {
    'principal': dict(popclass=MNTBPopulation, hoc=MNTB_HOC, seed=46,
                      syn_delay=0.5,  label='principal'),
    'calyx':     dict(popclass=CalyxPopulation, hoc=CALYX_HOC, seed=47,
                      syn_delay=0.05, label='calyx'),
}


def _run_one_source(pop_name, side, args, meta):
    """Run one MNTB generator, summing a whole-cell dipole (3, T) nA.µm on rank 0."""
    from lfpykit import CurrentDipoleMoment
    cfg = POPULATIONS[pop_name]
    contra_side = 'R' if side == 'L' else 'L'

    X_pops       = [f'GBC_{contra_side}']
    k_yxl_local  = P.MNTB_CONVERGENCE
    per_pop_syn  = cfg['popclass'].PER_POP_SYN
    weight       = per_pop_syn['GBC']['weight']
    j_yx_local   = [weight]
    tau_yx_local = [per_pop_syn['GBC']['tau2']]

    stem       = _pic_stem(paths.resolve_pic(args.pic_file))
    cond_val, cond_label = paths.condition_key(args.angle, args.itd_us,
                                              args.ild_db)
    spikes_dir = paths.spikes_dir_for(stem, cond_val, side)
    output_dir = paths.make_output_dirs(
        paths.output_dir_for('abr', stem, cond_label, side,
                             prefix=f'mntb_{pop_name}'),
        subdirs=('figures',))

    k_arr         = np.array(k_yxl_local)
    n_syn_per_pop = {X: int(k_arr[:, j].sum()) for j, X in enumerate(X_pops)}

    networkSim = hybridLFPy.CachedNetwork(
        simtime=TSTOP, dt=DT, spike_output_path=spikes_dir,
        label='spikes', ext='gdf',
        GIDs={X: [meta[X]['first_gid'], meta[X]['n_neurons']] for X in X_pops},
        X=X_pops)

    probe = CurrentDipoleMoment(None)   # cell set by hybridLFPy in cellsim()

    pop_label = f'{cfg["label"]}_{side}'
    pop = cfg['popclass'](
        n_syn_per_pop=n_syn_per_pop, y=pop_label,
        cellParams={'morphology': cfg['hoc'], 'passive': False, 'v_init': V_INIT,
                    'dt': DT, 'tstart': 0., 'tstop': TSTOP, 'nsegs_method': None},
        # No random rotation for the ABR: the GBC axons cross the midline in a
        # common mediolateral direction, so the pre-calyx axon dipoles have to
        # sum coherently (Karadas 2021 aligned axial current). The random
        # z-rotation used for the near-field LFP would zero the x-dipole.
        rand_rot_axis=[],
        simulationParams={'rec_imem': True},
        populationParams={'number': args.n_cells,
                          'radius': ELLIPSE_RADIUS_Y,
                          'radius_x': ELLIPSE_RADIUS_X, 'radius_y': ELLIPSE_RADIUS_Y,
                          'z_min': 0.0, 'z_max': 0.0, 'min_cell_interdist': 1.0,
                          'min_r': np.array([[0.], [0.]])},
        layerBoundaries=LAYER_BOUNDARIES, probes=[probe], savelist=['somapos'],
        savefolder=output_dir, dt_output=DT, POPULATIONSEED=cfg['seed'], X=X_pops,
        networkSim=networkSim, k_yXL=k_yxl_local,
        synParams={'section': 'allsec', 'syntype': 'Exp2Syn'},
        synDelayLoc=[cfg['syn_delay']], synDelayScale=[None],
        J_yX=j_yx_local, tau_yX=tau_yx_local)

    pop.run()
    COMM.Barrier()

    n_t = int(round(TSTOP / DT)) + 1
    local = None
    for i in pop.RANK_CELLINDICES:
        d = pop.output[i]['CurrentDipoleMoment'].astype(np.float64)
        local = d.copy() if local is None else local + d
    if local is None:
        local = np.zeros((3, n_t), dtype=np.float64)
    glob = reduce_sum(local)
    COMM.Barrier()
    return glob, output_dir


def _apply_head_model(sources, output_dir, electrode_names):
    """Project each generator from the MNTB position and sum at the scalp."""
    lo, hi = P.BAND_CLINICAL
    V_out = head_model.superpose_sources(
        [(part_label, p_head, r) for part_label, _side, p_head, r in sources],
        electrode_names, SRATE, lo=lo, hi=hi)
    srate = SRATE

    with h5py.File(os.path.join(output_dir, 'ABR.h5'), 'w') as f:
        for key, V_uV in V_out.items():
            f.create_dataset(key, data=V_uV)
        f.create_dataset('srate', data=srate)
        f.create_dataset('electrode_names', data=np.array(electrode_names, dtype='S'))
        f.attrs['units'] = 'µV'
        f.attrs['keys']  = ','.join(V_out)
    print(f'ABR saved -> {os.path.join(output_dir, "ABR.h5")}  keys={list(V_out)}')
    return V_out, srate


def _plot_abr(output_dir, V_out, electrode_names, srate, cond_label, side,
              n_cells):
    cz = electrode_names.index('Cz')
    m1 = electrode_names.index('M1')
    tvec = np.arange(next(iter(V_out.values())).shape[1]) / srate * 1e3
    colours = {'principal': 'steelblue', 'calyx': 'darkorange', 'composite': 'black'}
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    for key, V_uV in V_out.items():
        ax0.plot(tvec, V_uV[cz], lw=1.2 if key == 'composite' else 0.9,
                 color=colours.get(key), label=f'{key} (Cz)',
                 zorder=3 if key == 'composite' else 2)
    ax0.axhline(0, color='k', lw=0.4, ls=':')
    ax0.set_ylabel('Cz potential (µV)')
    ax0.set_title(f'MNTB ABR (principal + calyx prespike) | {cond_label} | '
                  f'side {side} | N={n_cells}')
    ax0.legend(fontsize=9)
    diff = V_out['composite'][cz] - V_out['composite'][m1]
    ax1.plot(tvec, diff, color='darkorchid', lw=1.0, label='composite Cz-M1')
    ax1.axhline(0, color='k', lw=0.4, ls=':')
    ax1.set_xlabel('Time (ms)'); ax1.set_ylabel('Amplitude (µV)')
    ax1.set_title('Cz-M1 (vertex-positive upward)')
    ax1.legend(fontsize=9)
    path = os.path.join(output_dir, 'figures', 'mntb_abr.png')
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f'ABR figure saved -> {path}')


def _cond_value(args):
    """Raw stimulus key the spike cache is stored under (angle, seconds or dB)."""
    return paths.condition_key(args.angle, args.itd_us, args.ild_db)[0]


def _cond_label(args):
    """Readable stimulus label for the output directory name."""
    return paths.condition_key(args.angle, args.itd_us, args.ild_db)[1]


def main():
    import argparse
    parser = argparse.ArgumentParser(description='MNTB ABR reconstruction')
    parser.add_argument('--pic-file', type=str, default=None, dest='pic_file')
    parser.add_argument('--angle',    type=int, default=0)
    parser.add_argument('--side',     type=str, default='L', choices=['L', 'R', 'both'])
    parser.add_argument('--n-cells',  type=int, default=200, dest='n_cells')
    parser.add_argument('--itd-us', type=float, default=None, dest='itd_us',
                        help='select an artificial-ITD condition (µs); overrides '
                             '--angle. The pic key is looked up in seconds.')
    parser.add_argument('--ild-db', type=float, default=None, dest='ild_db',
                        help='select an artificial-ILD condition (dB); overrides '
                             '--itd-us and --angle.')
    parser.add_argument('--generators', type=str, default='both',
                        choices=['principal', 'calyx', 'both'], dest='generators',
                        help='which MNTB generators to model. The postsynaptic cell '
                             'and the presynaptic terminal are distinct sources, so '
                             '"both" SUMS them at the scalp.')
    args = parser.parse_args()

    sides = ['L', 'R'] if args.side == 'both' else [args.side]
    parts = (['principal', 'calyx'] if args.generators == 'both'
             else [args.generators])
    stem  = _pic_stem(paths.resolve_pic(args.pic_file))

    meta_by_side = {
        side: broadcast_from_root(
            lambda side=side: _extract_spikes(_cond_value(args), side,
                                              pic_file=args.pic_file))
        for side in sides
    }

    sources = []
    for pop_name in parts:
        for side in sides:
            dipole, _ = _run_one_source(pop_name, side, args, meta_by_side[side])
            if RANK == 0:
                p_head = head_model.rotate_to_head(dipole, ROTATION[side])
                sources.append((pop_name, side, p_head, MNTB_POS_UM[side]))
                save_dipole_record(stem, _cond_label(args), 'MNTB',
                                   pop_name, side,
                                   p_head, MNTB_POS_UM[side], N_MNTB_TOTAL,
                                   args.n_cells, SRATE)

    if RANK == 0:
        final_dir = paths.make_output_dirs(
            paths.output_dir_for('abr', stem, _cond_label(args), args.side,
                                 prefix='mntb'),
            subdirs=('figures',))
        electrode_names = list(P.ELECTRODES)
        V_out, srate = _apply_head_model(sources, final_dir, electrode_names)
        _plot_abr(final_dir, V_out, electrode_names, srate,
                  _cond_label(args), args.side, args.n_cells)


if __name__ == '__main__':
    main()
