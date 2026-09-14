"""
HDF5 payloads exchanged between the pipeline stages.

Three file kinds, each written by one stage and read by several:

population_dipole.h5    (3, T) nA.um head-frame dipole for one run, the raw
                        output a producer keeps for its own figures.
dipole record           the same dipole plus the anatomical position it must be
                        projected from, one file per (nucleus, generator, side,
                        condition). This is the hand-off that lets main_abr_full
                        superpose nuclei without re-simulating.
ABR.h5                  (n_electrodes, T) uV scalp potentials, band-passed.

Readers and writers live together so the layout cannot drift between them.
"""

import os

import h5py
import numpy as np

from recon_core import paths

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
    """Write a multi-trace ABR file (per-generator, per-nucleus, composite).

    Used by main_abr_full (ABR_full.h5), main_abr_bi (BI.h5) and the
    multi-population per-nucleus scripts, which store several named
    (n_electrodes, T) arrays side by side instead of a single data array.
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


def save_dipole_record(stem, cond_label, nucleus, generator, side, p_head,
                       r_dipole, n_total, n_cells, srate, condition='binaural'):
    """Write one standardised head-frame dipole record.

    cond_label is the stimulus label from paths.condition_key ('angle0',
    'itd500us', 'ild-10dB') and selects the directory, so records from two
    stimulus conditions never land on each other.

    The filename encodes the acoustic condition
    (<nucleus>__<generator>__<side>__<condition>.h5) so a monaural run cannot
    overwrite the binaural record for the same stimulus. AVCN and MNTB have no
    --condition flag and always write binaural: a given cochlear-nucleus or
    MNTB side is driven by one ear whatever the other ear does, so its per-side
    dipole does not depend on the condition. Consumers ask for a condition and
    fall back to binaural.
    """
    directory = paths.dipoles_dir_for(stem, cond_label)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory,
                        f'{nucleus}__{generator}__{side}__{condition}.h5')
    with h5py.File(path, 'w') as f:
        f.create_dataset('p_head', data=np.asarray(p_head))                   # (3, T) nA·µm
        f.create_dataset('r_dipole', data=np.asarray(r_dipole, dtype=float))  # (3,) µm
        f.create_dataset('srate', data=float(srate))
        f.attrs.update(nucleus=nucleus, generator=generator, side=side,
                       condition=condition, n_total=int(n_total),
                       n_cells=int(n_cells), stem=str(stem),
                       cond_label=str(cond_label),
                       axes=_AXES, units='nA·µm')
    print(f'dipole record saved to {path}')
    return path


def read_dipole_record(path):
    """Read one dipole record as (attrs dict, p_head, r_dipole).

    attrs['srate'] is filled in from the dataset, so a caller superposing
    several records can take the time base from the records themselves.
    """
    with h5py.File(path, 'r') as f:
        attrs = dict(f.attrs)
        attrs.setdefault('condition', 'binaural')
        attrs['srate'] = float(f['srate'][()])
        return attrs, f['p_head'][()], f['r_dipole'][()]


# ---------------------------------------------------------------------------
# Near-field LFP (written by hybridLFPy.PostProcess)
# ---------------------------------------------------------------------------
def read_lfp_sum(output_dir, probe='PointSourcePotential'):
    """Read <probe>_sum.h5 as (lfp_uV (n_channels, T), srate).

    hybridLFPy stores mV and the pipelines work in µV throughout, so the
    conversion happens here rather than in each caller.
    """
    path = os.path.join(output_dir, f'{probe}_sum.h5')
    with h5py.File(path, 'r') as f:
        return f['data'][()] * 1e3, float(f['srate'][()])
