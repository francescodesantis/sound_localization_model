#!/usr/bin/env python3
"""
HybridLFPy LFP reconstruction for the LSO population.

CLI usage (single or multi-process):
  python LFP_reconstruction/main_reconstruct_lso.py [options]
  mpiexec -n 4 python LFP_reconstruction/main_reconstruct_lso.py --angle 0 --side L --n-cells 200

Options:
  --pic-file FILE   Path to .pic simulation result (default: RESULTS/baseline_simulation.pic)
  --angle DEGREES   Sound azimuth angle (default: 0)
  --side  L|R       Brain side (default: L)
  --n-cells N       LSO cells to simulate (default: 100)
  --n-single N      Single-cell contribution plots to save (default: 5)

LSO connectivity (per BrainstemModel.py):
  SBC_{side}   to LSO_{side}  excitatory, ipsilateral, 40 synapses/cell
  MNTBC_{side} to LSO_{side}  inhibitory, ipsilateral,  8 synapses/cell

Reuses the presynaptic GDF files main_reconstruct.py produced for the same
pic/angle/side (spikes_{stem}_angle{ANGLE}_{SIDE}/).

Outputs go to RESULTS/lfp_tmp/output_lso_{stem}_angle{ANGLE}_{SIDE}/figures/:
  lso_lfp_reconstruction.png
  lso_lfp_single_cells.png
"""

import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import lfpykit.models as lfpykit_models
import hybridLFPy

from recon_core import params as P, paths
from recon_core.population import ReconstructionPopulation
from LFP_reconstruction import figures
from recon_core.mpi_utils import (COMM, RANK, broadcast_from_root,
                                  load_mechanisms, set_temperature)

load_mechanisms(paths.MSO_MODELS_DIR)
set_temperature(P.BODY_TEMPERATURE_C)

# One morphology per side: the left cell is the mirror of the right (model x
# negated), which keeps the lemniscal axon on head +z while tonotopy and the
# dendritic tilt stay mirror-symmetric. See build_lso_axon.axon_dir().
#   stub = synaptic drive; axon = spiking drive plus ~4 mm ascending-LL cable
HOC_STUB = {s: os.path.join(paths.MSO_MODELS_DIR, f)
            for s, f in (('R', 'lso_model_active.hoc'),
                         ('L', 'lso_model_active_left.hoc'))}
HOC_AXON = {s: os.path.join(paths.MSO_MODELS_DIR, f)
            for s, f in (('R', 'lso_model_active_axon.hoc'),
                         ('L', 'lso_model_active_axon_left.hoc'))}
HOC_FILE = HOC_STUB['R']            # kept for callers that import the default
HOC_FILE_AXON = HOC_AXON['R']

DT, TSTOP = P.DT, P.TSTOP
V_INIT = P.LSO_V_INIT
N_LSO_TOTAL = P.N_LSO_TOTAL

HALF_HEIGHT_Y = P.LSO_HALF_HEIGHT_Y
RADIUS_X = P.LSO_RADIUS_X
RADIUS_Z = P.LSO_RADIUS_Z
# populationParams keys only; draw_rand_pos uses the constants above directly,
# so the radius_x/radius_y passed through hybridLFPy are not authoritative.
ELLIPSE_RADIUS_X = RADIUS_X
ELLIPSE_RADIUS_Y = RADIUS_Z

N_CH = P.N_CH
# Along y, the dendritic axis, so the probe follows the current dipole instead
# of running across the tonotopic gradient. See params.probe_positions().
PROBE_X, PROBE_Y, PROBE_Z = P.probe_positions(P.LSO_PROBE_HALF_SPAN,
                                              P.LSO_PROBE_AXIS)
SIGMA = P.SIGMA_EXTRACELLULAR

LAYER_BOUNDARIES = P.LSO_LAYERS

# Per-channel scaling: the axonal travelling wave spans orders of magnitude
# across the probe, so one global scale would flatten most channels.
FIGURE_STYLE = figures.FigureStyle(
    name='LSO', file_prefix='lso', trace_scale='per_channel', trace_gain=0.100,
    probe_axis=P.LSO_PROBE_AXIS,
    index_label=lambda gid, n: f'LSO idx '
    f'{int(round(gid * (N_LSO_TOTAL - 1) / (n - 1))) if n > 1 else 0}')


# ---------------------------------------------------------------------------
# LSOPopulation subclass
# ---------------------------------------------------------------------------
class LSOPopulation(ReconstructionPopulation):
    """LSO: ipsi SBC excitation on the dendrites, MNTBC inhibition on the soma."""

    PER_POP_SYN = P.LSO_SYNAPSES
    N_POST_TOTAL = N_LSO_TOTAL

    def select_synapse_idx(self, cell, pop_type, idx, layer):
        """Place by section name rather than by depth.

        The reoriented nucleus spreads its somas along z, so hybridLFPy's depth
        bands would miss most cells. Excitation goes onto the dendrites with a
        distal bias, inhibition onto the soma as the MNTB's glycinergic
        terminals do.
        """
        dend_segs = np.concatenate([cell.get_idx('dend_A'), cell.get_idx('dend_B')])
        soma_segs = cell.get_idx('soma')

        if pop_type == 'SBC' and len(dend_segs) > 0:
            soma_mid = np.array([cell.x[soma_segs].mean(), cell.y[soma_segs].mean(),
                                 cell.z[soma_segs].mean()])
            seg_mids = np.column_stack([cell.x[dend_segs].mean(axis=1),
                                        cell.y[dend_segs].mean(axis=1),
                                        cell.z[dend_segs].mean(axis=1)])
            dist = np.linalg.norm(seg_mids - soma_mid, axis=1)
            total = dist.sum()
            weights = dist / total if total > 0 else np.ones(len(dist)) / len(dist)
            return np.random.choice(dend_segs, size=len(idx), p=weights,
                                    replace=True).astype('int32')
        if pop_type == 'MNTBC' and len(soma_segs) > 0:
            return np.random.choice(soma_segs, size=len(idx),
                                    replace=True).astype('int32')
        return idx

    def draw_rand_pos(self, min_cell_interdist=1.0, **kwargs):
        """Fill the reoriented LSO, ordered mediolaterally (z), low to high CF.

        Geometry comes from the module constants, not from the radius_x/radius_y
        hybridLFPy passes: the long axis is rostrocaudal (y) and the elliptic
        cross-section lies in the x-z plane, so the two do not correspond.
        """
        return self.rejection_sample_ellipse(
            extents={'x': (-RADIUS_X, RADIUS_X), 'z': (-RADIUS_Z, RADIUS_Z),
                     'y': (-HALF_HEIGHT_Y, HALF_HEIGHT_Y)},
            sample_order=('x', 'z', 'y'), ellipse_axes=('x', 'z'),
            min_cell_interdist=min_cell_interdist, sort_axis='z')


# ---------------------------------------------------------------------------
# LSOSpikingPopulation, the spiking-output driver (Tolnai BIC generator)
#
# Instead of integrating SBC/MNTBC synaptic currents, each cell is driven by its
# own NEST LSO output train (X = LSO_{side}) through a single suprathreshold
# synapse on the AIS, so it fires one AP per spike. The AP propagates up the
# ~4 mm ascending-LL active axon (lso_model_active_axon.hoc) as a travelling
# wave, the integrated spiking output the scalp BIC reflects. Same pattern as
# main_reconstruct_calyx.CalyxPopulation. Reuses LSOPopulation.get_all_SpCells
# and draw_rand_pos.
# ---------------------------------------------------------------------------
class LSOSpikingPopulation(LSOPopulation):
    """LSO driven suprathreshold by its own output train, giving an axonal wave."""

    PER_POP_SYN = P.LSO_SPIKING_SYNAPSES

    def select_synapse_idx(self, cell, pop_type, idx, layer):
        """One suprathreshold synapse on the axon initial segment.

        Firing the AIS initiates the action potential and seeds the saltatory
        volley up the ascending lemniscal axon, the travelling-wave dipole this
        population exists to produce. Falls back to the soma if the morphology
        has no AIS.
        """
        drive_segs = cell.get_idx('ais')
        if len(drive_segs) == 0:
            drive_segs = cell.get_idx('soma')
        if len(drive_segs) == 0:
            return idx
        return np.random.choice(drive_segs, size=len(idx),
                                replace=True).astype('int32')


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description='LSO LFP reconstruction')
    parser.add_argument('--pic-file', type=str,  default=None, dest='pic_file')
    parser.add_argument('--angle',    type=int,  default=0)
    parser.add_argument('--side',     type=str,  default='L', choices=['L', 'R'])
    parser.add_argument('--n-cells',  type=int,  default=N_LSO_TOTAL, dest='n_cells')
    parser.add_argument('--n-single', type=int,  default=5,  dest='n_single')
    parser.add_argument('--condition', type=str, default='binaural',
                        choices=['binaural', 'ipsilateral', 'contralateral'],
                        help='binaural=both; ipsilateral=silence MNTBC; contralateral=silence SBC')
    parser.add_argument('--generators', type=str, default='synaptic',
                        dest='generators',
                        choices=['synaptic', 'spiking'],
                        help='synaptic=integrate SBC/MNTBC currents (near-field LFP, '
                             'default); spiking=drive the LSO output train up the '
                             'extended active axon (travelling-wave dipole, BIC generator)')
    parser.add_argument('--itd-us', type=float, default=None, dest='itd_us',
                        help='Select an artificial-ITD condition (µs). Overrides '
                             '--angle; pic key looked up in seconds (µs*1e-6).')
    parser.add_argument('--ild-db', type=float, default=None, dest='ild_db',
                        help='Select an artificial-ILD condition (dB). Overrides '
                             '--itd-us/--angle; pic key looked up in dB.')
    args = parser.parse_args()

    side = args.side

    pic_file = paths.resolve_pic(args.pic_file)
    cond_val, cond_label = paths.condition_key(args.angle, args.itd_us, args.ild_db)

    # Reuse the same spikes directory as the MSO script (same pops extracted).
    from LFP_reconstruction.main_reconstruct import _extract_spikes

    meta = broadcast_from_root(
        lambda: _extract_spikes(cond_val, side, pic_file=pic_file))

    if args.generators == 'spiking':
        # Drive each cell suprathreshold from its own LSO output train up the
        # extended active axon (travelling-wave dipole). One synapse on the AIS
        # (soma layer, k row 2); insert_all_synapses reassigns it to the AIS.
        pop_class     = LSOSpikingPopulation
        hoc_file      = HOC_AXON[side]
        rot_axis      = []   # no rotation, so the rostro-dorsal volley stays
                             # coherent (the axon is off the principal axis)
        X_pops        = [f'LSO_{side}']
        k_yxl_local   = P.LSO_SPIKING_CONVERGENCE   # 1 synapse -> AIS via override
        j_yx_local    = P.LSO_SPIKING_J_YX
        tau_yx_local  = P.LSO_SPIKING_TAU_YX
        syn_delay_loc = P.LSO_SPIKING_DELAYS
        syn_delay_scale = [None]
    else:
        pop_class     = LSOPopulation
        hoc_file      = HOC_STUB[side]
        rot_axis      = ['y']   # spin about the dendritic axis: randomises the
                                # cells without touching the dipole they carry
        X_pops = [f'SBC_{side}', f'MNTBC_{side}']

        # Single layer: 40 SBC on the dendrites and 8 MNTBC on the soma, placed
        # by the section-name override in LSOPopulation.insert_all_synapses.
        k_sbc, k_mntbc = P.LSO_CONVERGENCE[0]
        if args.condition == 'ipsilateral':
            k_mntbc = 0                    # silence MNTBC (contralateral inhibition)
        elif args.condition == 'contralateral':
            k_sbc = 0                      # silence SBC (ipsilateral excitation)
        k_yxl_local     = [[k_sbc, k_mntbc]]
        j_yx_local      = P.LSO_J_YX
        tau_yx_local    = P.LSO_TAU_YX
        syn_delay_loc   = P.LSO_DELAYS
        syn_delay_scale = [None, None]

    stem       = paths.pic_stem(pic_file)
    spikes_dir = paths.spikes_dir_for(stem, cond_val, side)
    cond_tag   = '' if args.condition == 'binaural' else f'_cond_{args.condition}'
    drive_tag  = '_spiking' if args.generators == 'spiking' else ''
    output_dir = paths.make_output_dirs(paths.output_dir_for(
        'lfp', stem, cond_label, side, prefix='lso', suffix=f'{cond_tag}{drive_tag}'))

    k_arr         = np.array(k_yxl_local)
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

    n_available = len(pop.RANK_CELLINDICES)
    n_grab = min(args.n_single, n_available)
    cell_indices = random.sample(list(pop.RANK_CELLINDICES), n_grab)

    single_contribs = np.stack(
        [pop.output[i]['PointSourcePotential'] * 1e3 for i in cell_indices],
        axis=0,
    )   # (n_grab, n_ch, n_t)  uV
    soma_pos = np.array([[pop.pop_soma_pos[i]['x'],
                          pop.pop_soma_pos[i]['y'],
                          pop.pop_soma_pos[i]['z']] for i in cell_indices])

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
