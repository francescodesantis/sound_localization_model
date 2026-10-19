#!/usr/bin/env python3
"""
HybridLFPy LFP reconstruction for the AVCN spherical bushy cell (SBC) population.

Parallel to main_reconstruct_avcn.py (the globular bushy cell), reusing the
same AVCNPopulation subclass and plotting. Only the SBC deltas are here:

  morphology    SBC EM reconstruction SBC_S113.hoc (NeuroMorpho Atoh7+,
                converted by models/avcn/swc_to_hoc.py), not GBC's VCN_c09
  channels      cnmodel XM13_nacncoop Type II-I (less KLT and Na, so higher
                input resistance), not GBC's Type II
  convergence   3 ANF endbulbs per SBC (params.py ANFs2SBCs), not 20
  endbulbs      few, large, axosomatic (85% soma, 15% proximal dendrite),
                not GBC's 70/20/5/5 modified endbulbs
  population    28,000 SBC per side (params.py n_SBCs), not 3,600
  geometry      rostral spherical-cell area ~1.5-2.0 mm rostrocaudal, so
                ELLIPSE_RADIUS_Y = 875 µm rather than 600

Presynaptic drive, taus, coherent ventromedial axon orientation, probe and
tonotopy are identical to the GBC pipeline.

CLI (single or MPI):
  python LFP_reconstruction/main_reconstruct_sbc.py --pic-file RESULTS/x.pic \
      --angle 0 --side L --n-cells 100
  mpiexec -n 4 python LFP_reconstruction/main_reconstruct_sbc.py ... --n-cells 3600

Outputs go to RESULTS/lfp_tmp/output_sbc_{stem}_angle{A}_{S}/figures/:
  sbc_lfp_reconstruction.png, sbc_lfp_phase_cycle.png, sbc_lfp_single_cells.png
"""

import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import lfpykit.models as lfpykit_models
import hybridLFPy

from recon_core import params as P, paths
from LFP_reconstruction import figures
from recon_core.mpi_utils import COMM, RANK, broadcast_from_root

sys.path.insert(0, paths.AVCN_MODELS_DIR)
import gbc_biophysics                                    # noqa: E402

# The SBC reuses the whole GBC infrastructure; importing it also loads the AVCN
# NEURON mechanisms. Only the deltas below are SBC-specific.
from LFP_reconstruction.main_reconstruct_avcn import (    # noqa: E402
    AVCNPopulation, DT, TSTOP, PROBE_X, PROBE_Y, PROBE_Z, SIGMA,
    ELLIPSE_RADIUS_X, AXON_TARGET, LAYER_BOUNDARIES,
    SYN_DELAY_LOC, SYN_DELAY_SCALE, AVCN_DIR,
)

V_INIT = P.SBC_V_INIT
FIGURE_STYLE = figures.FigureStyle(name='AVCN (SBC)', file_prefix='sbc',
                                   trace_gain=60.0, blank_onset_ms=0.2)

# ---------------------------------------------------------------------------
# SBC-specific parameters
# ---------------------------------------------------------------------------
HOC_FILE_SBC = os.path.join(AVCN_DIR, 'morphology', 'neuromorpho', 'SBC_S113.hoc')

N_CELLS       = 100
N_SBC_TOTAL   = P.N_SBC_TOTAL
N_ENDBULBS    = P.SBC_ENDBULBS

ELLIPSE_RADIUS_Y = P.SBC_RADIUS_Y

ENDBULB_WEIGHTS_SBC = P.SBC_ENDBULB_WEIGHTS

PER_POP_SYN_SBC = P.SBC_SYNAPSES

K_YXL = P.SBC_CONVERGENCE


def _decorate_sbc(cell):
    """LFPy custom_fun: cnmodel Type II-I (SBC) channel densities."""
    gbc_biophysics.decorate_gbc(cell, set_nseg=False,
                                ref_ns=gbc_biophysics.REF_NS_II_I)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description='AVCN (SBC) LFP reconstruction')
    parser.add_argument('--pic-file', type=str, default=None, dest='pic_file')
    parser.add_argument('--angle',    type=int, default=0)
    parser.add_argument('--side',     type=str, default='L', choices=['L', 'R'])
    parser.add_argument('--n-cells',  type=int, default=N_CELLS, dest='n_cells')
    parser.add_argument('--n-single', type=int, default=5, dest='n_single')
    parser.add_argument('--hoc-file', type=str, default=HOC_FILE_SBC, dest='hoc_file',
                        help='SBC morphology hoc (default: SBC_S113 reconstruction)')
    args = parser.parse_args()

    side     = args.side
    pic_file = paths.resolve_pic(args.pic_file)

    from LFP_reconstruction.main_reconstruct import _extract_spikes

    meta = broadcast_from_root(
        lambda: _extract_spikes(args.angle, side, pic_file=pic_file))

    X_pops       = [f'ANF_{side}']
    j_yx_local   = [PER_POP_SYN_SBC['ANF']['weight']]
    tau_yx_local = [PER_POP_SYN_SBC['ANF']['tau2']]

    stem       = paths.pic_stem(pic_file)
    spikes_dir = paths.spikes_dir_for(stem, args.angle, side)
    output_dir = paths.make_output_dirs(paths.output_dir_for(
        'lfp', stem, f'angle{args.angle}', side, prefix='sbc'))

    k_arr         = np.array(K_YXL)
    n_syn_per_pop = {X: int(k_arr[:, j].sum()) for j, X in enumerate(X_pops)}

    networkSim = hybridLFPy.CachedNetwork(
        simtime=TSTOP, dt=DT,
        spike_output_path=spikes_dir,
        label='spikes', ext='gdf',
        GIDs={X: [meta[X]['first_gid'], meta[X]['n_neurons']] for X in X_pops},
        X=X_pops,
    )
    probe = lfpykit_models.PointSourcePotential(
        cell=None, x=PROBE_X, y=PROBE_Y, z=PROBE_Z, sigma=SIGMA,
    )
    pop_label = f'SBC_{side}'
    pop = AVCNPopulation(
        n_syn_per_pop=n_syn_per_pop,
        axon_target=AXON_TARGET,
        n_post_total=N_SBC_TOTAL,
        n_endbulbs=N_ENDBULBS,
        endbulb_weights=ENDBULB_WEIGHTS_SBC,
        per_pop_syn=PER_POP_SYN_SBC,
        y=pop_label,
        cellParams={
            'morphology': args.hoc_file, 'passive': False, 'v_init': V_INIT,
            'dt': DT, 'tstart': 0., 'tstop': TSTOP,
            'nsegs_method': 'lambda_f', 'lambda_f': 100,
            'custom_fun': [_decorate_sbc], 'custom_fun_args': [{}],
        },
        rand_rot_axis=[],
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
        POPULATIONSEED=45,      # distinct from GBC (44)
        X=X_pops,
        networkSim=networkSim,
        k_yXL=K_YXL,
        synParams={'section': 'allsec', 'syntype': 'Exp2Syn'},
        synDelayLoc=SYN_DELAY_LOC,
        synDelayScale=SYN_DELAY_SCALE,
        J_yX=j_yx_local,
        tau_yX=tau_yx_local,
    )

    pop.run()
    COMM.Barrier()

    n_grab = min(args.n_single, len(pop.RANK_CELLINDICES))
    cell_indices = random.sample(list(pop.RANK_CELLINDICES), n_grab) if n_grab else []
    single_contribs = (np.stack([pop.output[i]['PointSourcePotential'] * 1e3
                                 for i in cell_indices], axis=0)
                       if cell_indices else None)
    soma_pos = (np.array([[pop.pop_soma_pos[i]['x'], pop.pop_soma_pos[i]['y'],
                           pop.pop_soma_pos[i]['z']] for i in cell_indices])
                if cell_indices else None)

    pop.collect_data()
    COMM.Barrier()

    postproc = hybridLFPy.PostProcess(
        y=[pop_label], dt_output=DT,
        mapping_Yy=[(pop_label, pop_label)],
        savelist=['somapos'], probes=[probe],
        savefolder=output_dir,
    )
    if RANK == 0:
        postproc.run()
    COMM.Barrier()

    if RANK == 0:
        figures.plot_all(output_dir, (PROBE_X, PROBE_Y, PROBE_Z), side, args.angle,
                         args.n_cells, FIGURE_STYLE,
                         stimulus_freq=meta.get('stim_freq_hz'),
                         single_contribs=single_contribs, soma_pos=soma_pos,
                         cell_gids=cell_indices, dt_ms=DT)


if __name__ == '__main__':
    main()
