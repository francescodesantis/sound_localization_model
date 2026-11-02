"""HDF5 files exchanged between the pipeline stages.

population_dipole.h5    (3, T) nA·µm head-frame dipole of one run.
dipole record           a dipole plus the position it is projected from, one
                        file per (nucleus, generator, side, condition); the
                        input of main_abr_full.
ABR.h5                  (n_electrodes, T) µV scalp potentials, band-passed.
"""

import os

import h5py
import numpy as np

from recon_core.utils import paths

_AXES = 'x=mediolateral, y=anteroposterior, z=inferosuperior'


# ---------------------------------------------------------------------------
# Scalp potentials
# ---------------------------------------------------------------------------
def write_abr(output_dir, V_uV, electrode_names, srate, **attrs):
    """Write ABR.h5; returns its path."""
    path = os.path.join(output_dir, 'ABR.h5')
    with h5py.File(path, 'w') as f:
        f.create_dataset('data', data=np.asarray(V_uV))
        f.create_dataset('srate', data=float(srate))
        f.create_dataset('electrode_names',
                         data=np.array(list(electrode_names), dtype='S'))
        f.attrs['units'] = 'µV'
        f.attrs.update({k: str(v) for k, v in attrs.items()})
    print(f'ABR saved to {path}')
    return path


def read_abr(path):
    """Read ABR.h5 as (V_uV, electrode_names, srate)."""
    with h5py.File(path, 'r') as f:
        V = f['data'][()]
        names = [n.decode() for n in f['electrode_names'][:]]
        srate = float(f['srate'][()])
    return V, names, srate


def write_named_traces(path, traces, electrode_names, srate, **attrs):
    """Write several named (n_electrodes, T) arrays (per generator, per nucleus,
    composite) into one ABR file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, 'w') as f:
        for label, V in traces.items():
            f.create_dataset(label.replace(':', '__'), data=np.asarray(V))
        f.create_dataset('srate', data=float(srate))
        f.create_dataset('electrode_names',
                         data=np.array(list(electrode_names), dtype='S'))
        f.attrs['units'] = 'µV'
        f.attrs.update({k: str(v) for k, v in attrs.items()})
    return path


def read_named_traces(path):
    """Inverse of write_named_traces: (traces, electrode_names, srate)."""
    reserved = {'srate', 'electrode_names'}
    with h5py.File(path, 'r') as f:
        names = [n.decode() for n in f['electrode_names'][:]]
        srate = float(f['srate'][()])
        traces = {k: f[k][()] for k in f if k not in reserved}
    return traces, names, srate


# ---------------------------------------------------------------------------
# Dipoles
# ---------------------------------------------------------------------------
def write_population_dipole(output_dir, p_head, srate):
    """Write population_dipole.h5 (head frame, nA.um); returns its path."""
    path = os.path.join(output_dir, 'population_dipole.h5')
    with h5py.File(path, 'w') as f:
        f.create_dataset('data', data=np.asarray(p_head))
        f.create_dataset('srate', data=float(srate))
        f.attrs['axes'] = _AXES
        f.attrs['units'] = 'nA·µm'
    print(f'Population dipole saved to {path}')
    return path


def read_population_dipole(path):
    """Read population_dipole.h5 as (p_head, srate)."""
    with h5py.File(path, 'r') as f:
        return f['data'][()], float(f['srate'][()])


#: Synaptic-weight convention written into every dipole record.
SYN_WEIGHTS_TAG = 'nest'


def save_dipole_record(stem, cond_label, nucleus, generator, side, p_head,
                       r_dipole, n_total, n_cells, srate, condition='binaural',
                       directory=None):
    """Write one head-frame dipole record into dipoles_dir_for(stem, cond_label)
    (or `directory`).

    `p_head` is (3, T) for a compact generator or (K, 3, T) for an extended one
    (the auditory nerve, binned along its path), with `r_dipole` (3,) or (K, 3).

    File name: <nucleus>__<generator>__<side>__<condition>.h5. AVCN and MNTB
    always write 'binaural': each side is driven by one ear, so its dipole does
    not depend on the condition. Consumers ask for a condition and fall back to
    binaural.
    """
    p = np.asarray(p_head, dtype=float)
    r = np.asarray(r_dipole, dtype=float)
    if p.ndim == 2:
        p, r = p[None, ...], r[None, ...]
    if p.ndim != 3 or r.shape != (p.shape[0], 3):
        raise ValueError(f'p_head {p.shape} and r_dipole {r.shape} disagree; '
                         'expected (K,3,T) and (K,3)')

    directory = directory or paths.dipoles_dir_for(stem, cond_label)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory,
                        f'{nucleus}__{generator}__{side}__{condition}.h5')
    with h5py.File(path, 'w') as f:
        f.create_dataset('p_head', data=p)                                    # (K,3,T) nA·µm
        f.create_dataset('r_dipole', data=r)                                  # (K,3) µm
        f.create_dataset('srate', data=float(srate))
        f.attrs.update(nucleus=nucleus, generator=generator, side=side,
                       condition=condition, n_total=int(n_total),
                       n_cells=int(n_cells), stem=str(stem),
                       cond_label=str(cond_label), n_bins=int(p.shape[0]),
                       axes=_AXES, units='nA·µm', syn_weights=SYN_WEIGHTS_TAG)
    print(f'dipole record saved to {path}  K={p.shape[0]}')
    return path


def read_dipole_record(path):
    """One dipole record -> (attrs dict, p_head (K,3,T), r_dipole (K,3)).

    (3, T) records are promoted to K=1. attrs['srate'] is read from the dataset.
    """
    with h5py.File(path, 'r') as f:
        attrs = dict(f.attrs)
        attrs.setdefault('condition', 'binaural')
        attrs['srate'] = float(f['srate'][()])
        p = f['p_head'][()]
        r = f['r_dipole'][()]
    if p.ndim == 2:
        p, r = p[None, ...], np.asarray(r)[None, ...]
    attrs.setdefault('n_bins', p.shape[0])
    return attrs, p, r


def collapse_dipole_record(p_head, r_dipole):
    """K-source record -> one equivalent (p_head (3,T), r_dipole (3,)).

    The moment is the exact sum over bins; the position is the centroid weighted
    by each bin's total |moment|. A no-op for K = 1.
    """
    p_head = np.asarray(p_head, dtype=float)
    r_dipole = np.asarray(r_dipole, dtype=float)
    p_sum = p_head.sum(axis=0)
    w = np.abs(p_head).sum(axis=(1, 2))              # (K,) total |current| per bin
    if not np.any(w > 0):
        return p_sum, r_dipole.mean(axis=0)
    r = (w[:, None] * r_dipole).sum(axis=0) / w.sum()
    return p_sum, r


# ---------------------------------------------------------------------------
# Near-field LFP (written by hybridLFPy.PostProcess)
# ---------------------------------------------------------------------------
def read_lfp_sum(output_dir, probe='PointSourcePotential'):
    """Read <probe>_sum.h5 as (lfp_uV (n_channels, T), srate); hybridLFPy stores mV."""
    path = os.path.join(output_dir, f'{probe}_sum.h5')
    with h5py.File(path, 'r') as f:
        return f['data'][()] * 1e3, float(f['srate'][()])
