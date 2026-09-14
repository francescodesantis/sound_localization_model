#!/usr/bin/env python3
"""
HybridLFPy LFP reconstruction for the MSO population.

CLI usage (single or multi-process):
  python LFP_reconstruction/main_reconstruct.py [options]
  mpiexec -n 4 python LFP_reconstruction/main_reconstruct.py --angle 45 --side R --n-cells 200

Options:
  --pic-file FILE   Path to .pic simulation result (default: RESULTS/baseline_simulation.pic)
  --angle DEGREES   Sound azimuth angle (default: 0)
  --side  L|R       Brain side (default: L)
  --n-cells N       MSO cells to simulate (default: 100)
  --n-single N      Single-cell contribution plots to save (default: 5)

Outputs go to RESULTS/lfp_tmp/output_angle{ANGLE}_{SIDE}/figures/:
  mso_lfp_reconstruction.png   compound LFP (stacked traces and colourmap)
  mso_lfp_single_cells.png     per-cell LFP colourmap and best-channel trace,
                               annotated with the min soma to probe distance
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import lfpykit.models as lfpykit_models
import hybridLFPy

from recon_core import params as P
from recon_core import paths
from recon_core.population import ReconstructionPopulation
from LFP_reconstruction import figures
from recon_core.mpi_utils import (COMM, RANK, broadcast_from_root,
                                  load_mechanisms, set_temperature)

load_mechanisms(paths.MSO_MODELS_DIR)
set_temperature(P.BODY_TEMPERATURE_C)

HOC_FILE = os.path.join(paths.MSO_MODELS_DIR, 'mso_model.hoc')

# Module-level aliases kept because ABR_reconstruction/main_abr.py and the figure
# scripts import the MSO geometry from here.
DT, TSTOP = P.DT, P.TSTOP
V_INIT = P.MSO_V_INIT
N_CELLS = N_MSO_TOTAL = P.N_MSO_TOTAL
ELLIPSE_RADIUS_X, ELLIPSE_RADIUS_Y = P.MSO_RADIUS_X, P.MSO_RADIUS_Y
LAYER_BOUNDARIES = P.MSO_LAYERS
MSO_FREQ_MIN, MSO_FREQ_MAX = P.MSO_FREQ_MIN, P.MSO_FREQ_MAX

N_CH = P.N_CH
PROBE_Z = P.probe_z(P.MSO_PROBE_HALF_SPAN)
PROBE_X = np.zeros(N_CH)
PROBE_Y = np.zeros(N_CH)
SIGMA = P.SIGMA_EXTRACELLULAR


def mso_freq_to_idx(freq_hz):
    """MSO tonotopic index of a characteristic frequency (Hz)."""
    return P.tonotopic_index(freq_hz, N_MSO_TOTAL)


FIGURE_STYLE = figures.FigureStyle(name='MSO', file_prefix='mso')


# ---------------------------------------------------------------------------
# MSOPopulation subclass
# ---------------------------------------------------------------------------
class MSOPopulation(ReconstructionPopulation):
    """MSO: bipolar dendrites, each driven by one ear.

    The tonotopic range can be narrowed to a frequency band
    (--mso-freq-min/max), which restricts both the presynaptic window and the
    x-slice the somas occupy, so the probe at x = 0 keeps the same relative
    position within the modelled band.
    """

    PER_POP_SYN = P.MSO_SYNAPSES
    N_POST_TOTAL = N_MSO_TOTAL
    # The MSO's inhibitory inputs use hybridLFPy's layer indices unchanged, so
    # its empty layers must still reach insert_synapses.
    SKIP_EMPTY_LAYERS = False

    def __init__(self, n_syn_per_pop=None, mso_idx_lo=0, mso_idx_hi=None, **kwargs):
        self.mso_idx_lo = mso_idx_lo
        self.mso_idx_hi = mso_idx_hi if mso_idx_hi is not None else N_MSO_TOTAL - 1
        super().__init__(n_syn_per_pop=n_syn_per_pop, **kwargs)

    def post_index(self, cellindex):
        """Map the sampled cell onto [mso_idx_lo, mso_idx_hi], not the whole nucleus."""
        n_sim = self.POPULATION_SIZE
        if n_sim <= 1:
            return self.mso_idx_lo
        span = self.mso_idx_hi - self.mso_idx_lo
        return self.mso_idx_lo + int(round(cellindex * span / (n_sim - 1)))

    def select_synapse_idx(self, cell, pop_type, idx, layer):
        """Excitation is spread over the dendrite with a distal bias.

        Each SBC synapse lands on a segment of this layer's depth band with a
        probability proportional to its distance from the soma, reproducing the
        distal-dominant endbulb distribution. Inhibition keeps the default
        somatic placement.
        """
        if pop_type != 'SBC' or len(idx) == 0:
            return idx
        z_lo, z_hi = min(self.layerBoundaries[layer]), max(self.layerBoundaries[layer])
        z_mid = cell.z.mean(axis=1)
        layer_segs = np.where((z_mid >= z_lo) & (z_mid <= z_hi))[0]
        soma_segs = cell.get_idx('soma')
        soma_mid = np.array([cell.x[soma_segs].mean(), cell.y[soma_segs].mean(),
                             cell.z[soma_segs].mean()])
        seg_mids = np.column_stack([cell.x[layer_segs].mean(axis=1),
                                    cell.y[layer_segs].mean(axis=1),
                                    cell.z[layer_segs].mean(axis=1)])
        dist = np.linalg.norm(seg_mids - soma_mid, axis=1)
        total = dist.sum()
        weights = dist / total if total > 0 else np.ones(len(dist)) / len(dist)
        return np.random.choice(layer_segs, size=len(idx), p=weights,
                                replace=True).astype('int32')

    def draw_rand_pos(self, radius_x=ELLIPSE_RADIUS_X, radius_y=ELLIPSE_RADIUS_Y,
                      z_min=0.0, z_max=0.0, min_cell_interdist=1.0, **kwargs):
        """Fill the tonotopic x-slice of the ellipse, sorted along x.

        Sampled column-wise rather than by rejection: x is drawn inside the
        slice and y within that column's chord, so a narrow band stays filled.
        """
        n_cells = self.POPULATION_SIZE
        x_lo = -radius_x + self.mso_idx_lo / (N_MSO_TOTAL - 1) * 2.0 * radius_x
        x_hi = -radius_x + self.mso_idx_hi / (N_MSO_TOTAL - 1) * 2.0 * radius_x

        def draw(n):
            xi = np.random.uniform(x_lo, x_hi, n)
            chord = radius_y * np.sqrt(np.maximum(0.0, 1.0 - (xi / radius_x) ** 2))
            yi = (np.random.rand(n) * 2.0 - 1.0) * chord
            zi = np.random.rand(n) * (z_max - z_min) + z_min
            return xi, yi, zi

        x, y, z = draw(n_cells)
        crowded = np.where(self.calc_min_cell_interdist(x, y, z) < min_cell_interdist)[0]
        while len(crowded):
            x[crowded], y[crowded], z[crowded] = draw(len(crowded))
            crowded = np.where(
                self.calc_min_cell_interdist(x, y, z) < min_cell_interdist)[0]

        soma_pos = [{'x': x[i], 'y': y[i], 'z': z[i]} for i in range(n_cells)]
        soma_pos.sort(key=lambda p: p['x'])
        return soma_pos


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    import argparse
    import random
    parser = argparse.ArgumentParser(description='MSO LFP reconstruction')
    parser.add_argument('--pic-file', type=str,  default=None,
                        dest='pic_file',
                        help='Path to .pic file (default: RESULTS/baseline_simulation.pic)')
    parser.add_argument('--angle',    type=int,  default=0,
                        help='Sound angle in degrees (default: 0)')
    parser.add_argument('--side',     type=str,  default='L', choices=['L', 'R'],
                        help='Side (default: L)')
    parser.add_argument('--n-cells',  type=int,  default=N_CELLS, dest='n_cells',
                        help=f'Number of MSO cells (default: {N_CELLS})')
    parser.add_argument('--n-single', type=int,  default=5, dest='n_single',
                        help='Number of single-cell contribution plots (default: 5)')
    parser.add_argument('--condition', type=str, default='binaural',
                        choices=['binaural', 'ipsilateral', 'contralateral'],
                        help='binaural=both ears; ipsilateral=ipsi-ear inputs only '
                             '(silence contra SBC medial dend + MNTBC); contralateral='
                             'contra-ear inputs only (silence ipsi SBC lateral dend + '
                             'LNTBC). Mirrors main_reconstruct_lso and main_abr.py.')
    parser.add_argument('--itd-us', type=float, default=None, dest='itd_us',
                        help='Select an artificial-ITD condition (µs). Overrides '
                             '--angle; pic key looked up in seconds (µs*1e-6).')
    parser.add_argument('--ild-db', type=float, default=None, dest='ild_db',
                        help='Select an artificial-ILD condition (dB). Overrides '
                             '--itd-us/--angle; pic key looked up in dB.')
    parser.add_argument('--hoc-file', type=str, default=None, dest='hoc_file',
                        help='HOC morphology file (default: models/mso/mso_model.hoc)')
    parser.add_argument('--mso-freq-min', type=float, default=MSO_FREQ_MIN, dest='mso_freq_min',
                        help=f'Lower CF bound for MSO input band in Hz (default: {MSO_FREQ_MIN})')
    parser.add_argument('--mso-freq-max', type=float, default=MSO_FREQ_MAX, dest='mso_freq_max',
                        help=f'Upper CF bound for MSO input band in Hz (default: {MSO_FREQ_MAX})')
    args = parser.parse_args()

    if args.hoc_file is None:
        args.hoc_file = HOC_FILE

    mso_idx_lo = mso_freq_to_idx(args.mso_freq_min)
    mso_idx_hi = mso_freq_to_idx(args.mso_freq_max)
    if RANK == 0:
        print(f'[MSO band] {args.mso_freq_min:.0f}–{args.mso_freq_max:.0f} Hz '
              f'-> idx [{mso_idx_lo}, {mso_idx_hi}] / {N_MSO_TOTAL}')

    side        = args.side
    contra_side = 'R' if side == 'L' else 'L'

    # Stimulus condition key: --ild-db, then --itd-us, then --angle (default 0).
    # The pic key is the raw value (seconds for ITD, dB for ILD, int for angle);
    # the readable label goes into the output dir, as in main_abr.py::_condition.
    cond_val, cond_label = paths.condition_key(args.angle, args.itd_us, args.ild_db)

    # Spike extraction on rank 0 only; broadcast metadata to all ranks
    meta = broadcast_from_root(
        lambda: _extract_spikes(cond_val, side, pic_file=args.pic_file))

    # MSO anatomy: medial dendrite from contra SBC, lateral dendrite from ipsi
    # SBC (Cant & Hyson 1992; Joris et al. 1998). MNTBC is contra-ear-driven
    # inhibition, LNTBC ipsi-ear-driven. Ear-specific conditions silence the
    # inputs of the absent ear (as in main_abr.py::_side_condition).
    X_pops = [f'SBC_{contra_side}', f'SBC_{side}',
              f'MNTBC_{side}', f'LNTBC_{side}']
    k_yxl_local = [row[:] for row in P.MSO_CONVERGENCE]
    # Silence the inputs the absent ear would have driven by zeroing that
    # population's column, so the surviving counts stay tied to MSO_CONVERGENCE.
    #   ipsilateral   ear: contra SBC (col 0) and MNTBC (col 2, contra-driven) off
    #   contralateral ear: ipsi SBC   (col 1) and LNTBC (col 3, ipsi-driven)   off
    _SILENCED = {'ipsilateral': (0, 2), 'contralateral': (1, 3)}
    for col in _SILENCED.get(args.condition, ()):
        for row in k_yxl_local:
            row[col] = 0
    j_yx_local      = P.MSO_J_YX
    tau_yx_local    = P.MSO_TAU_YX
    syn_delay_loc   = P.MSO_DELAYS
    syn_delay_scale = [None] * len(X_pops)

    pic_file   = paths.resolve_pic(args.pic_file)
    stem       = _pic_stem(pic_file)
    spikes_dir = paths.spikes_dir_for(stem, cond_val, side)
    cond_tag   = '' if args.condition == 'binaural' else f'_cond_{args.condition}'
    hoc_tag    = '_active' if args.hoc_file != HOC_FILE else ''
    freq_tag   = (f'_f{int(args.mso_freq_min)}-{int(args.mso_freq_max)}Hz'
                  if (args.mso_freq_min != MSO_FREQ_MIN or args.mso_freq_max != MSO_FREQ_MAX)
                  else '')
    output_dir = paths.make_output_dirs(paths.output_dir_for(
        'lfp', stem, cond_label, side, suffix=f'{cond_tag}{hoc_tag}{freq_tag}'))

    # Synapses per presynaptic pop per MSO cell (column sums of k_yxl_local)
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
    pop_label = f'MSO_{side}'
    pop = MSOPopulation(
        n_syn_per_pop=n_syn_per_pop,
        mso_idx_lo=mso_idx_lo,
        mso_idx_hi=mso_idx_hi,
        y=pop_label,
        cellParams={
            'morphology': args.hoc_file, 'passive': False, 'v_init': V_INIT,
            'dt': DT, 'tstart': 0., 'tstop': TSTOP, 'nsegs_method': None,
        },
        rand_rot_axis=['z'],
        simulationParams={'rec_imem': True},
        populationParams={
            'number':   args.n_cells,
            'radius':   ELLIPSE_RADIUS_Y,   # bounding value required by parent __init__
            'radius_x': ELLIPSE_RADIUS_X,
            'radius_y': ELLIPSE_RADIUS_Y,
            'z_min': -100.0, 'z_max': 100.0, 'min_cell_interdist': 1.0,
            'min_r': np.array([[0.], [0.]]),
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
        synDelayScale=syn_delay_scale,
        J_yX=j_yx_local,
        tau_yX=tau_yx_local,
    )

    pop.run()
    COMM.Barrier()

    # Grab single-cell outputs before collect_data() clears pop.output
    
    # Never sample more cells than exist on this MPI rank
    n_available = len(pop.RANK_CELLINDICES)
    n_grab = min(args.n_single, n_available)
    
    # Select cells randomly instead of taking the first contiguous block
    cell_indices = random.sample(list(pop.RANK_CELLINDICES), n_grab)

    single_contribs = np.stack(
        [pop.output[i]['PointSourcePotential'] * 1e3 for i in cell_indices],
        axis=0,
    )   # (n_grab, n_ch, n_t)  µV
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


# ---------------------------------------------------------------------------
# Spike extraction (runs extract_spikes.py if the cache is missing)
# ---------------------------------------------------------------------------
_pic_stem = paths.pic_stem


def _extract_spikes(angle, side, pic_file=None):
    """Load cached metadata or run extraction from .pic file."""
    import json
    pic_file   = paths.resolve_pic(pic_file)
    stem       = _pic_stem(pic_file)
    spikes_dir = paths.spikes_dir_for(stem, angle, side)
    meta_path  = os.path.join(spikes_dir, 'metadata.json')
    contra_side = 'R' if side == 'L' else 'L'
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        # Accept the cache only if it matches the current extract format: contra
        # SBC + GBC, ipsi ANF and ipsi LSO output. Contra GBC is the MNTB calyx
        # drive and LSO_{side} the spiking-LSO ABR drive, so requiring them
        # forces re-extraction of stale caches.
        if (f'SBC_{contra_side}' in meta and f'GBC_{contra_side}' in meta
                and f'ANF_{side}' in meta and f'LSO_{side}' in meta):
            return meta
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import extract_spikes as _es
    return _es.extract_and_save(pic_file, angle, side, spikes_dir)


if __name__ == '__main__':
    main()