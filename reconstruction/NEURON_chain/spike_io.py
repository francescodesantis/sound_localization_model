"""Spike interchange of the chain: NEURON spike times -> GDF files.

Every population takes its presynaptic drive from hybridLFPy.CachedNetwork
reading GDF files, so a NEURON population that writes its spikes in that format
drives the next nucleus unchanged. The format matches
LFP_reconstruction/extract_spikes.py: tab-separated, '%d' neuron id (1-based,
contiguous, = cellindex + 1) and '%.4f' time in ms. NEURON cell i stands for
NEST cell i (the tonotopic connector is reproduced exactly).
"""

import contextlib
import fcntl
import json
import os

import numpy as np

from recon_core.utils import paths


def chain_spikes_dir(stem, cond_value, side):
    """The chain's GDF directory, a sibling of the NEST one.

    Separate directories rather than a flag: the two producers have different
    invalidation rules and `extract_spikes.py` rewrites its own directory
    wholesale, so sharing one would make staleness undiagnosable.
    """
    nest_dir = paths.spikes_dir_for(stem, cond_value, side)
    parent, name = os.path.split(nest_dir)
    from NEURON_chain.params import CHAIN_VARIANT
    return os.path.join(parent, f'{CHAIN_VARIANT}_{name}')


def write_gdf(spikes_dir, label, spike_times_by_cell, n_total, threshold_mv):
    """Write one population's spikes as GDF; return its metadata dict.

    spike_times_by_cell : list of length n_total; element i is a 1-D array of
        spike times in ms for cell i. An empty array means that cell was silent.
        It must be silent, not unsimulated (see the full-N gate in emitter.py).
    """
    os.makedirs(spikes_dir, exist_ok=True)
    if len(spike_times_by_cell) != n_total:
        raise ValueError(f'{label}: got {len(spike_times_by_cell)} cells, '
                         f'expected n_total={n_total}; a partial-N run must not '
                         'emit spikes (silence would be indistinguishable from '
                         'a cell that was never simulated)')

    ids, times = [], []
    for cellindex, ts in enumerate(spike_times_by_cell):
        ts = np.atleast_1d(np.asarray(ts, dtype=float))
        for t in ts:
            ids.append(cellindex + 1)
            times.append(float(t))

    order = np.argsort(times, kind='stable')   # global time order
    ids = np.asarray(ids, dtype=int)[order]
    times = np.asarray(times, dtype=float)[order]

    gdf_path = os.path.join(spikes_dir, f'spikes-{label}-0.gdf')
    # savetxt on an empty (0,2) array writes an empty file, which is what a
    # wholly silent population must produce -- not a missing file.
    np.savetxt(gdf_path, np.column_stack([ids, times]) if ids.size
               else np.empty((0, 2)),
               fmt=['%d', '%.4f'], delimiter='\t')

    return {
        'first_gid': 1,
        'n_neurons': int(n_total),
        'n_spikes': int(ids.size),
        't_min_ms': float(times.min()) if times.size else None,
        't_max_ms': float(times.max()) if times.size else None,
        'threshold_mv': float(threshold_mv),
        'source': 'NEURON_chain',
    }


@contextlib.contextmanager
def metadata_lock(spikes_dir):
    """File lock (fcntl.flock) around the read-modify-write of metadata.json:
    jobs publishing into the same side's directory run concurrently.
    """
    os.makedirs(spikes_dir, exist_ok=True)
    lock_path = os.path.join(spikes_dir, '.metadata.lock')
    with open(lock_path, 'w') as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)


def write_metadata(spikes_dir, metadata):
    """Write metadata.json in the shape CachedNetwork's GIDs argument needs."""
    os.makedirs(spikes_dir, exist_ok=True)
    path = os.path.join(spikes_dir, 'metadata.json')
    with open(path, 'w') as f:
        json.dump(metadata, f, indent=2)
    return path


def read_metadata(spikes_dir):
    with open(os.path.join(spikes_dir, 'metadata.json')) as f:
        return json.load(f)


def read_gdf(spikes_dir, label):
    """(ids, times) from one GDF. For tests and for the staleness hash."""
    path = os.path.join(spikes_dir, f'spikes-{label}-0.gdf')
    arr = np.loadtxt(path, delimiter='\t', ndmin=2)
    if arr.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)
    return arr[:, 0].astype(int), arr[:, 1].astype(float)
