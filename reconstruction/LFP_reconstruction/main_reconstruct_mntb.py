#!/usr/bin/env python3
"""
HybridLFPy LFP reconstruction for the MNTB (medial nucleus of the trapezoid body).

The MNTB principal cell is driven by a single giant calyx of Held from the
contralateral globular bushy cell. This script reconstructs the postsynaptic
field (the calyx EPSC somatic sink and its passive spread). The presynaptic
calyx current, the prespike, is a separate co-located generator added by
--with-calyx (see calyx_model.hoc).

  MNTB drive (per BrainstemModel.py):
    GBC_{contra} to MNTB_{side}   excitatory calyx, 1 per cell, decussating

The GBC axon crossing the midline, the dominant wave-III generator, is
modelled in the AVCN GBC pipeline (VCN_c09_extended_axon.hoc) and not here, so
nothing is counted twice. This stage is the MNTB's own somatodendritic and
calyx-terminal field.

CLI:
  python LFP_reconstruction/main_reconstruct_mntb.py --pic-file RESULTS/<f>.pic \
      --angle 0 --side L --n-cells 100 [--with-calyx]
  mpiexec -n 4 python LFP_reconstruction/main_reconstruct_mntb.py ... --n-cells 3600

Outputs go to RESULTS/lfp_tmp/output_mntb_{stem}_angle{A}_{S}/figures/:
  mntb_lfp_reconstruction.png, mntb_lfp_single_cells.png,
  mntb_lfp_phase_cycle.png, plus output_mntb_calyx_* and a combined figure
  when --with-calyx
"""

import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import lfpykit.models as lfpykit_models
import hybridLFPy

from recon_core import io_utils, params as P, paths
from recon_core.population import ReconstructionPopulation
from LFP_reconstruction import figures
from recon_core.mpi_utils import (COMM, RANK, broadcast_from_root,
                                  load_mechanisms, set_temperature)

MNTB_DIR = paths.MNTB_MODELS_DIR
HOC_FILE = os.path.join(MNTB_DIR, 'mntb_model_active.hoc')

# namntb lives in models/mntb; klt/kht/ih come from models/mso.
load_mechanisms(MNTB_DIR, paths.MSO_MODELS_DIR)
set_temperature(P.BODY_TEMPERATURE_C)

DT, TSTOP = P.DT, P.TSTOP
V_INIT = P.MNTB_V_INIT
N_MNTB_TOTAL = P.N_MNTB_TOTAL

ELLIPSE_RADIUS_X, ELLIPSE_RADIUS_Y = P.MNTB_RADIUS_X, P.MNTB_RADIUS_Y

N_CH = P.N_CH
PROBE_Z = P.probe_z(P.MNTB_PROBE_HALF_SPAN)   # brackets the ±67 µm dendrite tips
PROBE_X = np.zeros(N_CH)
PROBE_Y = np.zeros(N_CH)
SIGMA = P.SIGMA_EXTRACELLULAR

LAYER_BOUNDARIES = P.MNTB_LAYERS
CALYX_SYN = P.MNTB_SYNAPSES


# ---------------------------------------------------------------------------
# MNTBPopulation subclass
# ---------------------------------------------------------------------------
class MNTBPopulation(ReconstructionPopulation):
    """MNTB principal cell, driven by one giant calyx from the contralateral GBC."""

    PER_POP_SYN = P.MNTB_SYNAPSES
    N_POST_TOTAL = N_MNTB_TOTAL

    def select_synapse_idx(self, cell, pop_type, idx, layer):
        """The calyx of Held is axosomatic: it engulfs the soma, so place it there."""
        soma_segs = cell.get_idx('soma')
        if pop_type == 'GBC' and len(soma_segs) > 0:
            return np.random.choice(soma_segs, size=len(idx),
                                    replace=True).astype('int32')
        return idx

    def draw_rand_pos(self, radius_x=ELLIPSE_RADIUS_X, radius_y=ELLIPSE_RADIUS_Y,
                      z_min=0.0, z_max=0.0, min_cell_interdist=1.0, **kwargs):
        """Fill the MNTB cylinder (z collapsed), ordered tonotopically along x."""
        return self.rejection_sample_ellipse(
            extents={'x': (-radius_x, radius_x), 'y': (-radius_y, radius_y),
                     'z': (z_min, z_max)},
            sample_order=('x', 'y', 'z'), ellipse_axes=('x', 'y'),
            min_cell_interdist=min_cell_interdist, sort_axis='x')


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description='MNTB LFP reconstruction')
    parser.add_argument('--pic-file', type=str, default=None, dest='pic_file')
    parser.add_argument('--angle',    type=int, default=0)
    parser.add_argument('--side',     type=str, default='L', choices=['L', 'R'])
    parser.add_argument('--n-cells',  type=int, default=100, dest='n_cells')
    parser.add_argument('--n-single', type=int, default=5, dest='n_single')
    parser.add_argument('--with-calyx', action='store_true', dest='with_calyx',
                        help='also simulate the co-located calyx-of-Held prespike '
                             'generator and sum its LFP (Phase 3)')
    args = parser.parse_args()

    side        = args.side
    contra_side = 'R' if side == 'L' else 'L'
    pic_file    = paths.resolve_pic(args.pic_file)

    from LFP_reconstruction.main_reconstruct import _extract_spikes

    meta = broadcast_from_root(
        lambda: _extract_spikes(args.angle, side, pic_file=pic_file))

    # The MNTB is driven by the contralateral GBC, since the calyx decussates.
    X_pops      = [f'GBC_{contra_side}']
    k_yxl_local = P.MNTB_CONVERGENCE     # dend_A, dend_B, soma <- 1 calyx on soma
    j_yx_local  = P.MNTB_J_YX
    tau_yx_local = P.MNTB_TAU_YX
    syn_delay_loc   = P.MNTB_DELAYS
    syn_delay_scale = [None]

    stem       = paths.pic_stem(pic_file)
    spikes_dir = paths.spikes_dir_for(stem, args.angle, side)
    output_dir = paths.make_output_dirs(paths.output_dir_for(
        'lfp', stem, f'angle{args.angle}', side, prefix='mntb'))

    k_arr         = np.array(k_yxl_local)
    n_syn_per_pop = {X: int(k_arr[:, j].sum()) for j, X in enumerate(X_pops)}

    _run_population(
        MNTBPopulation, HOC_FILE, X_pops, meta, spikes_dir, output_dir,
        k_yxl_local, j_yx_local, tau_yx_local, syn_delay_loc, syn_delay_scale,
        n_syn_per_pop, args, seed=46, title='MNTB', fig_prefix='mntb',
        per_pop_syn=None, v_init=V_INIT)

    # ----- co-located calyx prespike generator -----
    if args.with_calyx:
        from LFP_reconstruction import main_reconstruct_calyx as calyx_mod
        calyx_mod.run_calyx(args, meta, spikes_dir, stem, contra_side)
        if RANK == 0:
            _plot_combined(output_dir, args, stem)


def _run_population(PopClass, hoc_file, X_pops, meta, spikes_dir, output_dir,
                    k_yxl_local, j_yx_local, tau_yx_local, syn_delay_loc,
                    syn_delay_scale, n_syn_per_pop, args, seed, title, fig_prefix,
                    per_pop_syn=None, v_init=V_INIT):
    """Shared hybridLFPy run and plotting for a hand-written-hoc population."""
    side = args.side
    networkSim = hybridLFPy.CachedNetwork(
        simtime=TSTOP, dt=DT, spike_output_path=spikes_dir,
        label='spikes', ext='gdf',
        GIDs={X: [meta[X]['first_gid'], meta[X]['n_neurons']] for X in X_pops},
        X=X_pops,
    )
    probe = lfpykit_models.PointSourcePotential(
        cell=None, x=PROBE_X, y=PROBE_Y, z=PROBE_Z, sigma=SIGMA)

    pop_label = f'{title}_{side}'
    pop = PopClass(
        n_syn_per_pop=n_syn_per_pop, y=pop_label,
        cellParams={'morphology': hoc_file, 'passive': False, 'v_init': v_init,
                    'dt': DT, 'tstart': 0., 'tstop': TSTOP, 'nsegs_method': None},
        rand_rot_axis=['z'],
        simulationParams={'rec_imem': True},
        populationParams={'number': args.n_cells,
                          'radius': ELLIPSE_RADIUS_Y,
                          'radius_x': ELLIPSE_RADIUS_X, 'radius_y': ELLIPSE_RADIUS_Y,
                          'z_min': 0.0, 'z_max': 0.0, 'min_cell_interdist': 1.0,
                          'min_r': np.array([[0.], [0.]])},
        layerBoundaries=LAYER_BOUNDARIES, probes=[probe], savelist=['somapos'],
        savefolder=output_dir, dt_output=DT, POPULATIONSEED=seed, X=X_pops,
        networkSim=networkSim, k_yXL=k_yxl_local,
        synParams={'section': 'allsec', 'syntype': 'Exp2Syn'},
        synDelayLoc=syn_delay_loc, synDelayScale=syn_delay_scale,
        J_yX=j_yx_local, tau_yX=tau_yx_local,
    )
    if per_pop_syn is not None:
        pop.PER_POP_SYN = per_pop_syn

    pop.run()
    COMM.Barrier()

    n_grab = min(args.n_single, len(pop.RANK_CELLINDICES))
    cell_indices = random.sample(list(pop.RANK_CELLINDICES), n_grab)
    single_contribs = np.stack(
        [pop.output[i]['PointSourcePotential'] * 1e3 for i in cell_indices], axis=0)
    soma_pos = np.array([[pop.pop_soma_pos[i]['x'], pop.pop_soma_pos[i]['y'],
                          pop.pop_soma_pos[i]['z']] for i in cell_indices])

    pop.collect_data()
    COMM.Barrier()

    postproc = hybridLFPy.PostProcess(
        y=[pop_label], dt_output=DT, mapping_Yy=[(pop_label, pop_label)],
        savelist=['somapos'], probes=[probe], savefolder=output_dir)
    if RANK == 0:
        postproc.run()
    COMM.Barrier()

    if RANK == 0:
        figures.plot_all(output_dir, (PROBE_X, PROBE_Y, PROBE_Z), side, args.angle,
                         args.n_cells, figures.FigureStyle(name=title,
                                                           file_prefix=fig_prefix),
                         stimulus_freq=meta.get('stim_freq_hz'),
                         single_contribs=single_contribs, soma_pos=soma_pos,
                         cell_gids=cell_indices, dt_ms=DT)


def _plot_combined(output_dir, args, stem):
    """Sum principal and calyx PointSourcePotential, then plot the composite."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    calyx_dir = paths.output_dir_for('lfp', stem, f'angle{args.angle}', args.side,
                                     prefix='mntb_calyx')
    if not all(os.path.exists(os.path.join(d, 'PointSourcePotential_sum.h5'))
               for d in (output_dir, calyx_dir)):
        print('Warning: missing principal or calyx LFP; skipping combined plot.')
        return
    princ, srate = io_utils.read_lfp_sum(output_dir)
    calyx, _ = io_utils.read_lfp_sum(calyx_dir)
    comp = princ + calyx
    tvec = np.arange(comp.shape[1]) / srate * 1e3

    best = int(np.argmax(np.abs(comp).max(axis=1)))
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(tvec, princ[best], color='steelblue', lw=1.0, label='principal (postsynaptic)')
    ax.plot(tvec, calyx[best], color='darkorange', lw=1.0, label='calyx (prespike)')
    ax.plot(tvec, comp[best],  color='k', lw=1.3, label='composite')
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('LFP (uV)')
    ax.set_title(f'MNTB composite LFP (probe z={PROBE_Z[best]:.0f} um) | '
                 f'side {args.side} | angle {args.angle}')
    ax.legend(fontsize=8)
    out = os.path.join(output_dir, 'figures', 'mntb_lfp_composite.png')
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f'Combined LFP figure saved -> {out}')


if __name__ == '__main__':
    main()
