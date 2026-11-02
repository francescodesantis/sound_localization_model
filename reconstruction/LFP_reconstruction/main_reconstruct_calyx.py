#!/usr/bin/env python3
"""Calyx-of-Held prespike population, co-located with the MNTB principal cells
and summed with their LFP (main_reconstruct_mntb.py --with-calyx).

Each calyx, driven by the GBC_{contra} spike train through a suprathreshold
synapse, fires one presynaptic AP per input; its transmembrane current is the
extracellular prespike. Probe and nucleus geometry match the principal
cells, so the two LFPs add exactly. Ephaptic feedback onto the apposed MNTB
soma is not modelled.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from recon_core.config import params as P
from recon_core.utils import paths
from LFP_reconstruction.main_reconstruct_mntb import MNTBPopulation, _run_population

CALYX_HOC = os.path.join(paths.MNTB_MODELS_DIR, 'calyx_model.hoc')


class CalyxPopulation(MNTBPopulation):
    """Presynaptic calyx terminal: suprathreshold GBC drive on the pre-calyx axon."""

    PER_POP_SYN = P.CALYX_SYNAPSES

    def select_synapse_idx(self, cell, pop_type, idx, layer):
        """Drive the PROXIMAL end of the neck -- the heminode boundary.

        The prespike is the presynaptic action potential invading the terminal,
        so the drive has to arrive upstream of the arbor and propagate in. The
        heminode itself belongs to the GBC trunk (models/avcn/axon_builder.py)
        and is deliberately absent from this cell, so neck(0) IS the boundary:
        everything distal to it is this generator's, everything proximal is the
        AVCN's, and no membrane is counted twice.

        Not a random draw over the neck any more: every calyx is invaded at the
        same place, as every trunk delivers its action potential to the same
        place.
        """
        neck_segs = cell.get_idx('neck')
        if pop_type == 'GBC' and len(neck_segs) > 0:
            return np.full(len(idx), int(neck_segs[0]), dtype='int32')
        return idx


def run_calyx(args, meta, spikes_dir, stem, contra_side):
    X_pops      = [f'GBC_{contra_side}']
    k_yxl_local = P.MNTB_CONVERGENCE   # single synapse, re-placed by the override
    j_yx_local  = P.CALYX_J_YX
    tau_yx_local = P.CALYX_TAU_YX
    # The calyx fires when the AP reaches the heminode (CALYX_DELAYS = axonal
    # conduction); the postsynaptic EPSC follows GBC_MNTB_SYNAPTIC_MS later.
    syn_delay_loc   = P.CALYX_DELAYS
    syn_delay_scale = [None]

    output_dir = paths.make_output_dirs(paths.output_dir_for(
        'lfp', stem, f'angle{args.angle}', args.side, prefix='mntb_calyx'))

    k_arr = np.array(k_yxl_local)
    n_syn_per_pop = {X: int(k_arr[:, j].sum()) for j, X in enumerate(X_pops)}

    _run_population(
        CalyxPopulation, CALYX_HOC, X_pops, meta, spikes_dir, output_dir,
        k_yxl_local, j_yx_local, tau_yx_local, syn_delay_loc, syn_delay_scale,
        n_syn_per_pop, args, seed=47, title='CALYX', fig_prefix='calyx',
        v_init=P.MNTB_V_INIT)
