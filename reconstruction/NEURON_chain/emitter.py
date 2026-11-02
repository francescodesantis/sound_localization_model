"""Spike recording for chained populations.

A NetCon on one segment per cell records threshold crossings; after the
population run every cell's times are gathered on rank 0 and written as GDF.

Record sites:
  GBC   the distal-most node (heminode): the axonal crossing is inside the
        spike time, and the calyx is driven from the same point.
  SBC   soma (its axon is truncated).
  MNTB  soma (its axon is a stub).

On the baked extended morphologies sections are named `sections[N]`, so
`cell.get_idx('<name>')` finds nothing; the compartment-class map is used.
"""

import numpy as np
from neuron import h

from recon_core.utils.mpi_utils import COMM, RANK
from NEURON_chain import spike_io


def record_segment_index(cell, classes, select='distal'):
    """Index of the segment whose membrane potential is watched for spikes.

    classes : tuple of compartment-class names, e.g. ('node',) or ('soma',)
    select  : 'distal' -> farthest from the soma along the axon (the heminode)
              'soma'   -> the soma's own mid segment
    """
    import gbc_biophysics as gb

    segs = gb.seg_idx_for_classes(cell, classes)
    if len(segs) == 0:
        raise ValueError(
            f'no segments in classes {classes}; do NOT fall back to segment 0 '
            '-- a silently wrong record site produces a plausible but wrong '
            'spike train')
    if select == 'soma' or len(segs) == 1:
        return int(segs[len(segs) // 2])

    soma = gb.seg_idx_for_classes(cell, ('soma',))
    origin = np.array([cell.x.mean(-1)[soma].mean(),
                       cell.y.mean(-1)[soma].mean(),
                       cell.z.mean(-1)[soma].mean()])
    mid = np.c_[cell.x.mean(-1), cell.y.mean(-1), cell.z.mean(-1)]
    dist = np.linalg.norm(mid[segs] - origin, axis=1)
    return int(segs[int(np.argmax(dist))])


class _Recorder:
    """Holds the spike-time Vector alongside the NetCon that feeds it.

    NEURON does not own the NetCon; Python would garbage-collect it the
    moment attach_recorder returns, silently recording nothing. Bundling both
    on one object (rather than stashing the NetCon as an attribute of the
    h.Vector) keeps the same guarantee without relying on h.Vector accepting
    arbitrary Python attributes.
    """

    __slots__ = ('vec', 'netcon', 'section')

    def __init__(self, vec, netcon, section):
        self.vec = vec
        self.netcon = netcon
        self.section = section


def attach_recorder(cell, seg_idx, threshold_mv):
    """NetCon on one segment; returns a _Recorder wrapping the h.Vector.

    The NetCon and its source segment are kept alive on the returned object:
    NEURON does not own them and Python would garbage-collect the NetCon,
    silently recording nothing.
    """
    sec, seg_x = _section_and_x(cell, seg_idx)
    vec = h.Vector()
    nc = h.NetCon(sec(seg_x)._ref_v, None, sec=sec)
    nc.threshold = float(threshold_mv)
    nc.record(vec)
    return _Recorder(vec, nc, sec)


def _section_and_x(cell, seg_idx):
    """(section, normalised position) for a flat LFPy segment index."""
    i = 0
    for sec in cell.allseclist:
        for seg in sec:
            if i == seg_idx:
                return sec, seg.x
            i += 1
    raise IndexError(f'segment index {seg_idx} beyond {i} segments')


def recorded_times(rec):
    """Spike times from a recorder, ascending, as a float array."""
    return np.sort(np.asarray(list(rec.vec), dtype=float))


class SpikeRecorderMixin:
    """Records each cell's spikes during cellsim and publishes them as GDF.

    Mix in BEFORE the population class, so _chain_prepare/_chain_collect and
    emit_spikes resolve here rather than on the parent:
        class ChainGBCPopulation(SpikeRecorderMixin, GBCGroupedPopulation)
    """

    #: Compartment classes to record from, and which of them to pick.
    RECORD_CLASSES = ('soma',)
    RECORD_SELECT = 'soma'
    #: NetCon threshold. Recorded into metadata because the count depends on it.
    SPIKE_THRESHOLD_MV = -20.0
    #: GDF label, e.g. 'GBC_L'. Set per run.
    EMIT_LABEL = None

    # The hooks are called from the population's cellsim (the cell exists only
    # inside it), between construction and simulate and after simulate.

    def _chain_prepare(self, cell):
        """Attach the recorder. Called by the forked cellsim before simulate."""
        idx = record_segment_index(cell, self.RECORD_CLASSES,
                                   self.RECORD_SELECT)
        self._chain_recorder = attach_recorder(cell, idx,
                                              self.SPIKE_THRESHOLD_MV)
        return idx

    def _chain_collect(self, cellindex):
        """Store this cell's spikes. Called by the forked cellsim after simulate."""
        if not hasattr(self, '_chain_spikes'):
            self._chain_spikes = {}
        self._chain_spikes[cellindex] = recorded_times(self._chain_recorder)
        self._chain_recorder = None

    def emit_spikes(self, spikes_dir, n_total):
        """Gather every rank's spikes on rank 0 and write one GDF.

        Returns the metadata dict on rank 0 and None elsewhere -- deliberately
        mirroring reduce_sum's contract, so a caller that forgets the RANK guard
        fails on None rather than deadlocking on a later collective.
        """
        local = getattr(self, '_chain_spikes', {})
        gathered = COMM.gather(local, root=0)
        if RANK != 0:
            return None

        merged = {}
        for part in gathered:
            merged.update(part)
        n_sim = len(merged)
        if n_sim != n_total:
            raise ValueError(
                f'{self.EMIT_LABEL}: simulated {n_sim} of {n_total} cells. A '
                'partial-N run must not emit spikes: the missing cells would be '
                'written as silent, and downstream that is indistinguishable '
                'from genuine silence. Re-run at full N.')

        by_cell = [merged[i] for i in range(n_total)]
        meta = spike_io.write_gdf(spikes_dir, self.EMIT_LABEL, by_cell,
                                  n_total, self.SPIKE_THRESHOLD_MV)
        return meta
