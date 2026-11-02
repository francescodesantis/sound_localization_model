"""Populations of the NEURON chain: the existing cell populations plus spike
publishing (SpikeRecorderMixin).

Each class subclasses the population the NEST-driven pipeline uses; the
morphologies, biophysics and synapse placement are unchanged. The SBC is
AVCNPopulation with SBC morphology and synapses. The hooks are called from
GroupedDipoleMixin.cellsim (GBC) or ReconstructionPopulation.cellsim (the
others). SpikeRecorderMixin comes first in the MRO. Spike thresholds are per
recording site (params.SOMA/NODE_SPIKE_THRESHOLD_MV).
"""

import sys

from recon_core.config import params as _P
from recon_core.utils import paths

sys.path.insert(0, paths.AVCN_MODELS_DIR)
import gbc_biophysics                                          # noqa: E402

from LFP_reconstruction.main_reconstruct_avcn import (AVCNPopulation,       # noqa: E402
                                                     GBCGroupedPopulation)
from LFP_reconstruction.main_reconstruct_mntb import MNTBPopulation         # noqa: E402
from LFP_reconstruction.main_reconstruct import MSOPopulation               # noqa: E402
from LFP_reconstruction.main_reconstruct_lso import LSOPopulation           # noqa: E402

from NEURON_chain import emitter                                           # noqa: E402
from NEURON_chain.emitter import SpikeRecorderMixin                        # noqa: E402


class ChainGBCPopulation(SpikeRecorderMixin, GBCGroupedPopulation):
    """Stage 1. Endbulb-driven, recorded as two edge groups; spikes taken at the
    distal heminode, so the axonal crossing is inside the spike time.
    """

    RECORD_CLASSES = ('node',)
    RECORD_SELECT = 'distal'
    #: NODE, not soma: the record site is a node of Ranvier.
    SPIKE_THRESHOLD_MV = _P.NODE_SPIKE_THRESHOLD_MV
    GROUPS = {'GBCsoma': gbc_biophysics.GBC_SPIKE_CLASSES,
              'GBCaxon': gbc_biophysics.GBC_TRUNK_CLASSES}


class ChainSBCPopulation(SpikeRecorderMixin, AVCNPopulation):
    """Stage 1. Spikes recorded at the soma (the SBC axon is truncated), located
    through the compartment-class map.
    """

    RECORD_CLASSES = ('soma',)
    RECORD_SELECT = 'soma'
    SPIKE_THRESHOLD_MV = _P.SOMA_SPIKE_THRESHOLD_MV


class ChainMNTBPopulation(SpikeRecorderMixin, MNTBPopulation):
    """Stage 2. Driven by the chained contralateral GBC; spikes recorded at the
    soma, located by section name (this hoc's sections are named and carry no
    compartment classes).
    """

    #: Hoc section to watch. Named, not classified -- see the class docstring.
    RECORD_SECTION = 'soma'
    SPIKE_THRESHOLD_MV = _P.SOMA_SPIKE_THRESHOLD_MV

    def _chain_prepare(self, cell):
        idx = cell.get_idx(section=self.RECORD_SECTION)
        if len(idx) == 0:
            raise ValueError(
                f'no segments in section {self.RECORD_SECTION!r} of the MNTB '
                'morphology; do NOT fall back to segment 0 or to the '
                'compartment-class map -- a silently wrong record site produces '
                'a plausible but wrong spike train')
        seg_idx = int(idx[len(idx) // 2])
        self._chain_recorder = emitter.attach_recorder(
            cell, seg_idx, self.SPIKE_THRESHOLD_MV)
        return seg_idx


class ChainMSOPopulation(MSOPopulation):
    """Stage 3, terminal: nothing downstream reads MSO spikes, so it emits none."""


class ChainLSOPopulation(LSOPopulation):
    """Stage 3, TERMINAL. Emits no spikes, for the same reason as the MSO."""
