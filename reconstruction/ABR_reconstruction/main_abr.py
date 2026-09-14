#!/usr/bin/env python3
"""
MSO ABR reconstruction: current dipole moment through the 4-sphere head model.

CLI:
  python ABR_reconstruction/main_abr.py [options]
  mpiexec -n 4 python ABR_reconstruction/main_abr.py --angle 45 --side R --n-cells 200

Options:
  --pic-file FILE          Path to .pic simulation result
  --angle DEGREES          Sound azimuth angle (default: 0)
  --side L|R|both          Brain side (default: L)
  --n-cells N              MSO cells to simulate (default: 100)
  --derivation Cz-M1|Cz-M2|Cz-avg  Differential to plot (default: Cz-M1)
  --condition binaural|left_ear|right_ear  Acoustic condition (default: binaural)

Outputs go to RESULTS/abr_tmp/output_<stem>_angle<A>_<side>/:
  ABR.h5                   scalp potentials (n_electrodes, n_t), mV
  population_dipole.h5     population dipole moment (3, n_t), nA.µm
  figures/mso_abr.png      ABR waveforms and differentials
  figures/mso_abr_phase_cycle.png  phase-averaged ABR

Units are lfpykit native, with no conversion needed:
  Positions / electrode coords : µm
  Dipole moment (CurrentDipoleMoment output) : nA.µm
  FourSphereVolumeConductor input p : nA.µm
  FourSphereVolumeConductor input dipole_location : µm
  FourSphereVolumeConductor output : mV
  Saved / plotted ABR : µV (mV x 1000)
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
from recon_core.mpi_utils import COMM, RANK, broadcast_from_root, reduce_sum
from recon_core.signal_utils import bandpass

from LFP_reconstruction.main_reconstruct import (
    MSOPopulation, HOC_FILE, ELLIPSE_RADIUS_X, ELLIPSE_RADIUS_Y,
    LAYER_BOUNDARIES, _extract_spikes,
)

DT, TSTOP = P.DT, P.TSTOP
V_INIT = P.MSO_V_INIT
N_CELLS = P.N_MSO_TOTAL
SRATE = P.SRATE

# Re-exported for the figure scripts, which read the same head model.
FOUR_SPHERE_RADII = P.FOUR_SPHERE_RADII
FOUR_SPHERE_SIGMAS = P.FOUR_SPHERE_SIGMAS
ELECTRODE_POS = P.ELECTRODE_POS
MSO_POS_UM = P.MSO_POS_UM
ROTATION = P.ROTATION_MSO

_pic_stem = paths.pic_stem
dipoles_dir_for = paths.dipoles_dir_for
save_dipole_record = io_utils.save_dipole_record


def _bandpass(signal, lo=150., hi=None, fs=None, order=4):
    """Wrapper kept for older callers; new code calls signal_utils.bandpass."""
    return bandpass(signal, fs=fs or SRATE, lo=lo, hi=hi, order=order)


def superpose_sources(sources, electrode_names, hi=3000., lo=150.):
    """Kept for the figure scripts: sources are (label, side, p_head, r)."""
    return (head_model.superpose_sources(
        [(label, p_head, r) for label, _side, p_head, r in sources],
        electrode_names, SRATE, lo=lo, hi=hi), SRATE)


def _inh_tag(args):
    """Directory suffix for an inhibition-blocked run ('' when inhibition intact)."""
    return '_noinh' if getattr(args, 'block_inhibition', False) else ''


def _block_inhibition():
    """Zero the MSO inhibitory synaptic weights (MNTBC contra, LNTBC ipsi).

    The conductances that reach the cell live in MSOPopulation.PER_POP_SYN,
    which insert_all_synapses() reads. The J_yX argument does not drive them.
    """
    for pop in ('MNTBC', 'LNTBC'):
        MSOPopulation.PER_POP_SYN[pop]['weight'] = 0.0


def _side_condition(condition, side):
    """Per-MSO acoustic condition, given the stimulated ear and the MSO side.

    condition : 'binaural' | 'left_ear' | 'right_ear'
    side      : 'L' | 'R'
    returns   : 'binaural' | 'ipsilateral' | 'contralateral'
    """
    if condition == 'binaural':
        return 'binaural'
    if (condition == 'left_ear' and side == 'L') or \
       (condition == 'right_ear' and side == 'R'):
        return 'ipsilateral'
    return 'contralateral'


# ---------------------------------------------------------------------------
# Per-side simulation
# ---------------------------------------------------------------------------
def _run_one_side(side, args, meta):
    """
    Simulate the MSO population for one brain side.
    Returns the population dipole (3, T) nA.µm, meaningful on rank 0 only,
    and the output directory path.
    """
    from lfpykit import CurrentDipoleMoment

    contra_side = 'R' if side == 'L' else 'L'
    X_pops = [f'SBC_{contra_side}', f'SBC_{side}',
               f'MNTBC_{side}', f'LNTBC_{side}']
    k_yxl_local = [row[:] for row in P.MSO_CONVERGENCE]
    side_cond = _side_condition(args.condition, side)
    # Silence the inputs the absent ear would have driven by zeroing that
    # population's column, so the surviving counts stay tied to MSO_CONVERGENCE.
    #   ipsilateral   ear: contra SBC (col 0) and MNTBC (col 2, contra-driven) off
    #   contralateral ear: ipsi SBC   (col 1) and LNTBC (col 3, ipsi-driven)   off
    _SILENCED = {'ipsilateral': (0, 2), 'contralateral': (1, 3)}
    for col in _SILENCED.get(side_cond, ()):
        for row in k_yxl_local:
            row[col] = 0
    # J_yX does not set the synapse conductance: MSOPopulation.insert_all_synapses
    # overrides it with PER_POP_SYN[pop]['weight']. See _block_inhibition().
    j_yx_local    = P.MSO_J_YX
    tau_yx_local  = P.MSO_TAU_YX
    syn_delay_loc = P.MSO_DELAYS

    pic_file   = paths.resolve_pic(args.pic_file)
    stem       = _pic_stem(pic_file)
    cond, cond_label = _condition(args)
    # spikes_dir is keyed by the raw condition value so it matches the cache
    # _extract_spikes builds; output_dir uses the readable label.
    spikes_dir = paths.spikes_dir_for(stem, cond, side)
    cond_tag   = f'_{args.condition}' if args.condition != 'binaural' else ''
    output_dir = paths.make_output_dirs(
        paths.output_dir_for('abr', stem, cond_label, side,
                             suffix=f'{cond_tag}{_inh_tag(args)}'),
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

    # CurrentDipoleMoment(None): cell=None is valid; hybridLFPy sets probe.cell
    # before calling get_transformation_matrix() inside cellsim().
    probe     = CurrentDipoleMoment(None)
    pop_label = f'MSO_{side}'

    pop = MSOPopulation(
        n_syn_per_pop=n_syn_per_pop,
        y=pop_label,
        cellParams={
            'morphology': HOC_FILE, 'passive': False, 'v_init': V_INIT,
            'dt': DT, 'tstart': 0., 'tstop': TSTOP, 'nsegs_method': None,
        },
        rand_rot_axis=['z'],
        simulationParams={'rec_imem': True},
        populationParams={
            'number':             args.n_cells,
            'radius':             ELLIPSE_RADIUS_Y,
            'radius_x':           ELLIPSE_RADIUS_X,
            'radius_y':           ELLIPSE_RADIUS_Y,
            'z_min': -100.0, 'z_max': 100.0,
            'min_cell_interdist': 1.0,
            'min_r':              np.array([[0.], [0.]]),
        },
        layerBoundaries=LAYER_BOUNDARIES,
        probes=[probe],
        savelist=['somapos'],
        savefolder=output_dir,
        dt_output=DT,
        POPULATIONSEED=42,
        X=X_pops,
        networkSim=networkSim,
        k_yXL=k_yxl_local,
        synParams={'section': 'allsec', 'syntype': 'Exp2Syn'},
        synDelayLoc=syn_delay_loc,
        synDelayScale=[None] * len(X_pops),
        J_yX=j_yx_local,
        tau_yX=tau_yx_local,
    )

    pop.run()   # calls COMM.Barrier() internally
    COMM.Barrier()

    # Sum per-cell dipole moments on this rank.
    # pop.output[i]['CurrentDipoleMoment'] has shape (3, T) float32, nA.µm.
    # Must be grabbed before any collect_data() call.
    local_dipole = None
    for i in pop.RANK_CELLINDICES:
        d = pop.output[i]['CurrentDipoleMoment'].astype(np.float64)  # (3, T)
        if local_dipole is None:
            local_dipole = d.copy()
        else:
            local_dipole += d

    if local_dipole is None:   # this rank got no cells
        local_dipole = np.zeros((3, int(round(TSTOP / DT)) + 1), dtype=np.float64)

    global_dipole = reduce_sum(local_dipole)
    COMM.Barrier()
    return global_dipole, output_dir   # only valid on rank 0


# ---------------------------------------------------------------------------
# Head model projection + saving (rank 0 only)
# ---------------------------------------------------------------------------
def _project_and_save(side, p_model, output_dir):
    """Rotate the dipole into the head frame and store it, as (p_head, srate)."""
    p_head = head_model.rotate_to_head(p_model, ROTATION[side])
    io_utils.write_population_dipole(output_dir, p_head, SRATE)
    return p_head, SRATE


def _apply_head_model(p_head_by_side, output_dir, electrode_names):
    """Project each side's dipole from its own position and sum at the scalp."""
    lo, hi = P.BAND_TOLNAI
    V_uV, srate = head_model.project_by_side(
        p_head_by_side, MSO_POS_UM, electrode_names, SRATE, lo=lo, hi=hi)
    io_utils.write_abr(output_dir, V_uV, electrode_names, srate)
    return V_uV, srate


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plot_abr(output_dir, V_uV, electrode_names, srate, cond_label, side, n_cells,
              derivation='Cz-M1'):
    import matplotlib.gridspec as gridspec

    n_t  = V_uV.shape[1]
    tvec = np.arange(n_t) / srate * 1e3   # ms

    fig = plt.figure(figsize=(13, 8), constrained_layout=True)
    gs  = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1])

    ax_cz  = fig.add_subplot(gs[0, 0])
    ax_m1  = fig.add_subplot(gs[0, 1])
    ax_bot = fig.add_subplot(gs[1, :])

    def _plot_single(ax, label, color, title):
        if label in electrode_names:
            idx = electrode_names.index(label)
            ax.plot(tvec, V_uV[idx], color=color, lw=0.9, label=label)
        ax.axhline(0, color='k', lw=0.4, ls=':')
        ax.set_ylabel('Potential (µV)')
        ax.set_xlabel('Time (ms)')
        ax.set_title(title)
        ax.legend(fontsize=9)

    _plot_single(ax_cz, 'Cz', 'steelblue',
                 f'Cz (vertex)  |  {cond_label}  |  side {side}  |  N={n_cells}')
    _plot_single(ax_m1, 'M1', 'firebrick', 'M1 (left mastoid)')

    # Compute chosen differential
    cz = V_uV[electrode_names.index('Cz')]
    m1 = V_uV[electrode_names.index('M1')]
    m2 = V_uV[electrode_names.index('M2')]
    if derivation == 'Cz-M1':
        diff, deriv_label = cz - m1, 'Cz−M1'
    elif derivation == 'Cz-M2':
        diff, deriv_label = cz - m2, 'Cz−M2'
    else:
        diff, deriv_label = cz - (m1 + m2) / 2.0, 'Cz−(M1+M2)/2'

    ax_bot.plot(tvec, diff, color='darkorchid', lw=0.9, label=deriv_label)
    ax_bot.axhline(0, color='k', lw=0.4, ls=':')
    ax_bot.set_ylabel('Amplitude (µV)')
    ax_bot.set_xlabel('Time (ms)')
    ax_bot.set_title(f'{deriv_label}  (vertex-positive upward)')
    ax_bot.legend(fontsize=9)

    fig.suptitle('MSO ABR  |  vertex-positive upward',
                 fontsize=11, fontweight='bold')

    fig_path = os.path.join(output_dir, 'figures', 'mso_abr.png')
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f'ABR figure saved to {fig_path}')

    # Second figure: derivation only, more square aspect ratio
    fig2, ax2 = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax2.plot(tvec, diff, color='darkorchid', lw=0.9, label=deriv_label)
    ax2.axhline(0, color='k', lw=0.4, ls=':')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Amplitude (µV)')
    ax2.set_title(f'{deriv_label}  |  {cond_label}  |  side {side}  |  N={n_cells}')
    ax2.legend(fontsize=9)
    fig2.suptitle('MSO ABR  |  vertex-positive upward',
                  fontsize=10, fontweight='bold')
    fig2_path = os.path.join(output_dir, 'figures', 'mso_abr_derivation.png')
    fig2.savefig(fig2_path, dpi=150)
    plt.close(fig2)
    print(f'Derivation figure saved to {fig2_path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _condition(args):
    """Return (condition_value, label) for key lookup and directory naming.

    Without --itd-us the value is the integer angle and the label is
    'angle{N}'. With --itd-us the value is the ITD in seconds (us * 1e-6) used
    for the .pic key lookup, and the label is 'itd{us}us'.
    """
    if getattr(args, 'ild_db', None) is not None:
        return float(args.ild_db), f'ild{args.ild_db:g}dB'
    if getattr(args, 'itd_us', None) is None:
        return args.angle, f'angle{args.angle}'
    return args.itd_us * 1e-6, f'itd{args.itd_us:g}us'


def main():
    import argparse
    parser = argparse.ArgumentParser(description='MSO ABR reconstruction')
    parser.add_argument('--pic-file',   type=str, default=None, dest='pic_file')
    parser.add_argument('--angle',      type=int, default=0)
    parser.add_argument('--itd-us',     type=float, default=None, dest='itd_us',
                        help='Select an artificial-ITD condition (microseconds). '
                             'Overrides --angle; key looked up in seconds (us*1e-6).')
    parser.add_argument('--ild-db',     type=float, default=None, dest='ild_db',
                        help='Select an artificial-ILD condition (dB). Overrides '
                             '--itd-us/--angle; key looked up in dB.')
    parser.add_argument('--side',       type=str, default='L',
                        choices=['L', 'R', 'both'])
    parser.add_argument('--n-cells',    type=int, default=N_CELLS, dest='n_cells')
    parser.add_argument('--derivation', type=str, default='Cz-M1',
                        choices=['Cz-M1', 'Cz-M2', 'Cz-avg'],
                        help='Differential derivation to plot (Cz-avg = Cz − mean(M1,M2))')
    parser.add_argument('--condition', type=str, default='binaural',
                        choices=['binaural', 'left_ear', 'right_ear'])
    parser.add_argument('--block-inhibition', action='store_true',
                        dest='block_inhibition',
                        help='Block MSO inhibition: zero the MNTBC and LNTBC synaptic '
                             'weights. Output goes to a separate *_noinh directory.')
    args = parser.parse_args()

    if args.block_inhibition:
        _block_inhibition()
        if RANK == 0:
            print('[block-inhibition] MSO inhibitory weights (MNTBC, LNTBC) set to 0.0')

    sides = ['L', 'R'] if args.side == 'both' else [args.side]

    p_head_by_side = {}   # side to (3, T) nA.µm in head coords (rank 0 only)
    output_dirs    = {}

    for side in sides:
        meta = broadcast_from_root(
            lambda: _extract_spikes(_condition(args)[0], side,
                                    pic_file=args.pic_file))

        global_dipole, output_dir = _run_one_side(side, args, meta)
        output_dirs[side] = output_dir

        if RANK == 0:
            p_head, _ = _project_and_save(side, global_dipole, output_dir)
            p_head_by_side[side] = p_head
            _stem = _pic_stem(paths.resolve_pic(args.pic_file))
            # The variant tag has to reach the record name: records are keyed
            # <nucleus>__<generator>__<side>__<condition>, so an
            # inhibition-blocked run would otherwise overwrite the intact
            # binaural record main_abr_full.py superposes. Tagging it also keeps
            # the variant out of the composite, which selects on condition.
            save_dipole_record(_stem, _condition(args)[1], 'MSO',
                               'postsynaptic', side,
                               p_head, MSO_POS_UM[side], N_CELLS, args.n_cells,
                               SRATE,
                               condition=f'{args.condition}{_inh_tag(args)}')

    if RANK == 0:
        # For bilateral, create a joint output directory
        if len(sides) > 1:
            stem = _pic_stem(paths.resolve_pic(args.pic_file))
            _, cond_label = _condition(args)
            cond_tag = f'_{args.condition}' if args.condition != 'binaural' else ''
            final_dir = paths.make_output_dirs(
                paths.output_dir_for('abr', stem, cond_label, 'both',
                                     suffix=f'{cond_tag}{_inh_tag(args)}'),
                subdirs=('figures',))
        else:
            final_dir = output_dirs[sides[0]]

        electrode_names = list(P.ELECTRODES)
        V_uV, srate = _apply_head_model(p_head_by_side, final_dir, electrode_names)

        _plot_abr(final_dir, V_uV, electrode_names, srate,
                  _condition(args)[1], args.side, args.n_cells, derivation=args.derivation)


if __name__ == '__main__':
    main()
