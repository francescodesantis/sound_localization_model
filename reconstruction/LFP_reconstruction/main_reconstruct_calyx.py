#!/usr/bin/env python3
"""
Co-located calyx-of-Held prespike population, summed with the MNTB principal
cell LFP by linear superposition. Invoked by main_reconstruct_mntb.py
--with-calyx; not usually run standalone.

Each calyx is the presynaptic terminal of a contralateral GBC axon. It is
driven by the GBC_{contra} spike train through a suprathreshold somatic-axon
synapse so it fires one presynaptic AP per input, and its transmembrane
current is the extracellular prespike. Probe and nucleus geometry match the
principal cells, so the two LFPs add exactly (output_mntb_calyx_* is summed in
main_reconstruct_mntb._plot_combined).

The cleft leak conductance g_cl ~ 1 uS is characterised single-cell in
models/mntb/validate_calyx.py (V_cleft of several mV). The population prespike
here is the calyx transmembrane current, the dominant term. Cross-cell
ephaptic feedback onto the apposed MNTB soma is not modelled, since LFPy cells
are independent.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from recon_core import params as P, paths
from LFP_reconstruction.main_reconstruct_mntb import MNTBPopulation, _run_population

CALYX_HOC = os.path.join(paths.MNTB_MODELS_DIR, 'calyx_model.hoc')


class CalyxPopulation(MNTBPopulation):
    """Presynaptic calyx terminal: suprathreshold GBC drive on the pre-calyx axon."""

    PER_POP_SYN = P.CALYX_SYNAPSES

    def select_synapse_idx(self, cell, pop_type, idx, layer):
        """Drive the pre-calyx axon, not the terminal.

        The prespike is the presynaptic action potential invading the terminal,
        so the drive has to arrive upstream of it and propagate in.
        """
        pre_segs = cell.get_idx('precalyx_axon')
        if pop_type == 'GBC' and len(pre_segs) > 0:
            return np.random.choice(pre_segs, size=len(idx),
                                    replace=True).astype('int32')
        return idx


def run_calyx(args, meta, spikes_dir, stem, contra_side):
    X_pops      = [f'GBC_{contra_side}']
    k_yxl_local = P.MNTB_CONVERGENCE   # single synapse, re-placed by the override
    j_yx_local  = P.CALYX_J_YX
    tau_yx_local = P.CALYX_TAU_YX
    # The calyx AP happens at the GBC spike, while the postsynaptic MNTB EPSC
    # carries the full GBCs2MNTBCs = 0.5 ms synaptic delay, so the prespike
    # leads the postsynaptic sink by ~0.5 ms as it does in vivo.
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
