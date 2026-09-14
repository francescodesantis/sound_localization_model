#!/usr/bin/env python3
"""
LSO ABR reconstruction: current dipole moment through the 4-sphere head model.

This is the spiking-output generator the scalp BIC reflects (Tolnai et al.):
each LSO projection neuron fires one AP per NEST output spike and the AP
travels up the ~4 mm ascending lateral lemniscus active axon as a moving
current dipole (main_reconstruct_lso.LSOSpikingPopulation with
models/mso/lso_model_active_axon.hoc). main_abr.py (MSO, synaptic dipole) is
the other half of the ablation compared in plots/tolnai.py.

Mirrors main_abr_avcn.py: own anatomical position (LSO_POS_UM) and rotation,
with the whole-cell dipole (soma, dendrites and axonal travelling wave)
projected once per cell, not split into synaptic and axonal parts (see the
comment block in main_abr_avcn.py).

CLI (single or MPI):
  python ABR_reconstruction/main_abr_lso.py --pic-file RESULTS/x.pic --angle 0 --side both --n-cells 200
  mpiexec -n 4 python ABR_reconstruction/main_abr_lso.py --itd-us 500 --side both --n-cells 200

Condition selection (mutually exclusive): --angle (default), --itd-us, --ild-db.

Outputs:
  RESULTS/abr_tmp/output_lso_{stem}_{cond}_{side}/population_dipole.h5   (per side)
  RESULTS/abr_tmp/output_lso_{stem}_{cond}_{sidespec}/ABR.h5            (Cz/M1/M2, µV)
  .../figures/lso_abr.png
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import hybridLFPy

from recon_core import head_model, io_utils, params as P, paths
from recon_core.io_utils import save_dipole_record
from recon_core.mpi_utils import COMM, RANK, broadcast_from_root, reduce_sum
from recon_core.signal_utils import derive

# LSO populations + model geometry (importing this also loads the mechanisms).
from LFP_reconstruction.main_reconstruct_lso import (
    LSOPopulation, LSOSpikingPopulation, HOC_STUB, HOC_AXON,
    ELLIPSE_RADIUS_X, ELLIPSE_RADIUS_Y, LAYER_BOUNDARIES,
)
from LFP_reconstruction.main_reconstruct import _extract_spikes

DT, TSTOP, SRATE = P.DT, P.TSTOP, P.SRATE
V_INIT = P.LSO_V_INIT
N_LSO_TOTAL = P.N_LSO_TOTAL

# The rotation maps the extended LSO axon (build_lso_axon.AXON_DIR) onto the
# head inferosuperior axis on both sides, mirroring the mediolateral and
# anteroposterior axes per side.
LSO_POS_UM = P.LSO_POS_UM
ROTATION = P.ROTATION_LSO

_pic_stem = paths.pic_stem


# ---------------------------------------------------------------------------
# Condition selection (angle, itd-us, ild-db) to (.pic key, readable label)
# ---------------------------------------------------------------------------
def _condition(args):
    return paths.condition_key(args.angle, getattr(args, 'itd_us', None),
                               getattr(args, 'ild_db', None))


# ---------------------------------------------------------------------------
# Per-side simulation, giving a population dipole (3, T) nA.µm on rank 0
# ---------------------------------------------------------------------------
def _side_condition(condition, side):
    """Map left_ear/right_ear to per-side ipsilateral/contralateral, as main_abr.py.
    For side L the ipsi ear is 'left'; for side R it is 'right'."""
    if condition == 'binaural':
        return 'binaural'
    if (condition == 'left_ear' and side == 'L') or \
       (condition == 'right_ear' and side == 'R'):
        return 'ipsilateral'
    return 'contralateral'


def _run_one_side(side, args, meta, cond_label):
    from lfpykit import CurrentDipoleMoment
    spiking = (args.generators == 'spiking')

    if spiking:
        pop_class     = LSOSpikingPopulation
        hoc_file      = HOC_AXON[side]
        rot_axis      = []                   # coherent rostro-dorsal volley
        X_pops        = [f'LSO_{side}']
        k_yxl_local   = P.LSO_SPIKING_CONVERGENCE   # one suprathreshold AIS synapse
        j_yx_local, tau_yx_local = P.LSO_SPIKING_J_YX, P.LSO_SPIKING_TAU_YX
        syn_delay_loc, syn_delay_scale = P.LSO_SPIKING_DELAYS, [None]
    else:
        # Synaptic LSO scalp dipole, the analogue of the MSO ABR: integrate the
        # SBC/MNTBC currents, monaural via ear-specific silencing (single-layer).
        pop_class     = LSOPopulation
        hoc_file      = HOC_STUB[side]
        rot_axis      = []
        X_pops        = [f'SBC_{side}', f'MNTBC_{side}']
        k_sbc, k_mntbc = P.LSO_CONVERGENCE[0]   # SBC on dendrites, MNTBC on soma
        sc = _side_condition(args.condition, side)
        if sc == 'ipsilateral':             # ipsi ear: keep SBC, drop MNTBC
            k_mntbc = 0
        elif sc == 'contralateral':         # contra ear: drop SBC, keep MNTBC
            k_sbc = 0
        k_yxl_local   = [[k_sbc, k_mntbc]]
        j_yx_local    = P.LSO_J_YX
        tau_yx_local  = P.LSO_TAU_YX
        syn_delay_loc, syn_delay_scale = P.LSO_DELAYS, [None, None]

    stem       = _pic_stem(paths.resolve_pic(args.pic_file))
    cond, _    = _condition(args)
    spikes_dir = paths.spikes_dir_for(stem, cond, side)
    drive_tag  = '' if spiking else '_syn'
    cond_tag   = '' if args.condition == 'binaural' else f'_{args.condition}'
    output_dir = paths.make_output_dirs(
        paths.output_dir_for('abr', stem, cond_label, side, prefix='lso',
                             suffix=f'{drive_tag}{cond_tag}'),
        subdirs=('figures',))

    k_arr = np.array(k_yxl_local)
    n_syn_per_pop = {X: int(k_arr[:, j].sum()) for j, X in enumerate(X_pops)}

    networkSim = hybridLFPy.CachedNetwork(
        simtime=TSTOP, dt=DT,
        spike_output_path=spikes_dir,
        label='spikes', ext='gdf',
        GIDs={X: [meta[X]['first_gid'], meta[X]['n_neurons']] for X in X_pops},
        X=X_pops,
    )

    probe     = CurrentDipoleMoment(None)   # cell set by hybridLFPy in cellsim()
    pop_label = f'LSO_{side}'

    pop = pop_class(
        n_syn_per_pop=n_syn_per_pop,
        y=pop_label,
        cellParams={
            'morphology': hoc_file, 'passive': False, 'v_init': V_INIT,
            'dt': DT, 'tstart': 0., 'tstop': TSTOP, 'nsegs_method': None,
        },
        rand_rot_axis=rot_axis,
        simulationParams={'rec_imem': True},
        populationParams={
            'number':   args.n_cells,
            'radius':   ELLIPSE_RADIUS_Y,
            'radius_x': ELLIPSE_RADIUS_X,
            'radius_y': ELLIPSE_RADIUS_Y,
            'z_min': 0.0, 'z_max': 0.0, 'min_cell_interdist': 1.0,
            'min_r': np.array([[0.], [0.]]),
        },
        layerBoundaries=LAYER_BOUNDARIES,
        probes=[probe],
        savelist=['somapos'],
        savefolder=output_dir,
        dt_output=DT,
        POPULATIONSEED=43,
        X=X_pops,
        networkSim=networkSim,
        k_yXL=k_yxl_local,
        synParams={'section': 'allsec', 'syntype': 'Exp2Syn'},
        synDelayLoc=syn_delay_loc,
        synDelayScale=syn_delay_scale,
        J_yX=j_yx_local,
        tau_yX=tau_yx_local,
    )

    pop.run()
    COMM.Barrier()

    # Sum per-cell whole-cell dipole moments (3, T) nA·µm on this rank.
    local = None
    for i in pop.RANK_CELLINDICES:
        d = pop.output[i]['CurrentDipoleMoment'].astype(np.float64)
        local = d.copy() if local is None else local + d
    if local is None:
        local = np.zeros((3, int(round(TSTOP / DT)) + 1), dtype=np.float64)
    glob = reduce_sum(local)
    COMM.Barrier()
    return glob, output_dir


# The long (~4 mm) active axon turns finitialize's settling capacitive currents,
# times the large lever arm, into a big spurious dipole at t=0 that the
# zero-phase bandpass rings off. NEST LSO spikes only start after 6 ms, so
# 0 to SETTLE_MS is silent and is blanked before projection and filtering.
SETTLE_MS = P.SETTLE_MS


def _project_and_save(side, p_model, output_dir):
    """Blank the settling transient, rotate into the head frame, store p_head."""
    p_model = np.asarray(p_model, dtype=float).copy()
    p_model[:, :int(round(SETTLE_MS / DT))] = 0.0
    p_head = head_model.rotate_to_head(p_model, ROTATION[side])
    io_utils.write_population_dipole(output_dir, p_head, SRATE)
    return p_head


def _apply_head_model(p_head_by_side, output_dir, electrode_names):
    """Project each side's LSO dipole from its own position and sum at the scalp."""
    lo, hi = P.BAND_TOLNAI
    V_uV, srate = head_model.project_by_side(
        p_head_by_side, LSO_POS_UM, electrode_names, SRATE, lo=lo, hi=hi)
    io_utils.write_abr(output_dir, V_uV, electrode_names, srate)
    return V_uV, srate


def _plot_abr(output_dir, V_uV, electrode_names, srate, cond_label, side, n_cells,
              derivation='Cz-M1'):
    n_t  = V_uV.shape[1]
    tvec = np.arange(n_t) / srate * 1e3
    idx  = {e: electrode_names.index(e) for e in electrode_names}
    cz, m1 = V_uV[idx['Cz']], V_uV[idx['M1']]
    diff, lbl = derive(V_uV, electrode_names, derivation)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    ax0.plot(tvec, cz, color='steelblue', lw=0.9, label='Cz')
    ax0.plot(tvec, m1, color='firebrick', lw=0.7, label='M1')
    ax0.axhline(0, color='k', lw=0.4, ls=':')
    ax0.set_ylabel('Potential (µV)')
    ax0.set_title(f'LSO ABR (spiking) | {cond_label} | side {side} | N={n_cells}')
    ax0.legend(fontsize=9)

    ax1.plot(tvec, diff, color='darkorchid', lw=1.0, label=lbl)
    ax1.axhline(0, color='k', lw=0.4, ls=':')
    ax1.set_xlabel('Time (ms)'); ax1.set_ylabel('Amplitude (µV)')
    ax1.set_title(f'{lbl}  (vertex-positive upward)')
    ax1.legend(fontsize=9)

    path = os.path.join(output_dir, 'figures', 'lso_abr.png')
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f'ABR figure saved to {path}')


def main():
    import argparse
    parser = argparse.ArgumentParser(description='LSO (spiking) ABR reconstruction')
    parser.add_argument('--pic-file',   type=str, default=None, dest='pic_file')
    parser.add_argument('--angle',      type=int, default=0)
    parser.add_argument('--itd-us',     type=float, default=None, dest='itd_us',
                        help='Select an artificial-ITD condition (µs); key = µs*1e-6 s.')
    parser.add_argument('--ild-db',     type=float, default=None, dest='ild_db',
                        help='Select an artificial-ILD condition (dB); key = dB.')
    parser.add_argument('--side',       type=str, default='L',
                        choices=['L', 'R', 'both'])
    parser.add_argument('--n-cells',    type=int, default=200, dest='n_cells')
    parser.add_argument('--derivation', type=str, default='Cz-M1',
                        choices=['Cz-M1', 'Cz-M2', 'Cz-avg'])
    parser.add_argument('--generators', type=str, default='spiking',
                        choices=['spiking', 'synaptic'], dest='generators',
                        help='which LSO generator to model. NOTE: unlike AVCN/MNTB '
                             'these are ALTERNATIVE models of the same cells and are '
                             'never summed, so there is no "both". '
                             'spiking=LSO output train up the LL axon (travelling-wave '
                             'dipole; monaural needs monaural pics); synaptic=SBC/MNTBC '
                             'synaptic dipole -> scalp (analog of the MSO ABR; monaural '
                             'via --condition, like main_abr.py)')
    parser.add_argument('--condition', type=str, default='binaural',
                        choices=['binaural', 'left_ear', 'right_ear'],
                        help='synaptic drive only: acoustic condition for the BIC '
                             '(binaural / left-ear-only / right-ear-only)')
    args = parser.parse_args()

    sides = ['L', 'R'] if args.side == 'both' else [args.side]
    cond, cond_label = _condition(args)
    stem = _pic_stem(paths.resolve_pic(args.pic_file))

    p_head_by_side, output_dirs = {}, {}
    for side in sides:
        meta = broadcast_from_root(
            lambda: _extract_spikes(cond, side, pic_file=args.pic_file))

        p_model, output_dir = _run_one_side(side, args, meta, cond_label)
        output_dirs[side] = output_dir
        if RANK == 0:
            p_head_by_side[side] = _project_and_save(side, p_model, output_dir)
            save_dipole_record(stem, cond_label, 'LSO', args.generators, side,
                               p_head_by_side[side], LSO_POS_UM[side],
                               N_LSO_TOTAL, args.n_cells, SRATE,
                               condition=args.condition)

    if RANK == 0:
        if len(sides) > 1:
            drive_tag = '' if args.generators == 'spiking' else '_syn'
            cond_tag  = '' if args.condition == 'binaural' else f'_{args.condition}'
            final_dir = paths.make_output_dirs(
                paths.output_dir_for('abr', stem, cond_label, 'both', prefix='lso',
                                     suffix=f'{drive_tag}{cond_tag}'),
                subdirs=('figures',))
        else:
            final_dir = output_dirs[sides[0]]

        electrode_names = list(P.ELECTRODES)
        V_uV, srate = _apply_head_model(p_head_by_side, final_dir, electrode_names)
        _plot_abr(final_dir, V_uV, electrode_names, srate,
                  cond_label, args.side, args.n_cells, derivation=args.derivation)


if __name__ == '__main__':
    main()
