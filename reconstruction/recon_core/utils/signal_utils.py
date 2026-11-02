"""Signal processing shared by the LFP and ABR pipelines and the figure scripts.

No pipeline imports (no params, no NEURON); sample rates are passed explicitly.
"""

import numpy as np
import scipy.signal


def srate_from_dt(dt_ms):
    """Sampling rate in Hz for a simulation timestep given in ms."""
    return 1.0 / (dt_ms * 1e-3)


def bandpass(signal, fs, lo=150., hi=None, order=4):
    """Zero-phase Butterworth filter, applied row-wise to a (n, T) array.

    hi=None gives a high-pass at lo. Zero phase keeps peak latencies unshifted.
    The filter is linear, so filtered sources can be summed afterwards.
    """
    if hi is None:
        sos = scipy.signal.butter(order, lo, btype='high', fs=fs, output='sos')
    else:
        sos = scipy.signal.butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    signal = np.asarray(signal, dtype=float)
    if signal.ndim == 1:
        return scipy.signal.sosfiltfilt(sos, signal)
    return np.stack([scipy.signal.sosfiltfilt(sos, row) for row in signal])


def derive(V, electrode_names, kind='Cz-M1'):
    """Scalp derivation from a (n_electrodes, T) array -> (trace, label).

    Vertex positive upward (Cz minus a reference).
    """
    idx = {name: i for i, name in enumerate(electrode_names)}

    def chan(*aliases):
        for a in aliases:
            if a in idx:
                return V[idx[a]]
        raise KeyError(f'none of {aliases} in {electrode_names}')

    if kind == 'Cz-M1':
        return chan('Cz') - chan('M1', 'A1'), 'Cz−M1'
    if kind == 'Cz-M2':
        return chan('Cz') - chan('M2', 'A2'), 'Cz−M2'
    if kind == 'Cz-avg':
        return (chan('Cz') - 0.5 * (chan('M1', 'A1') + chan('M2', 'A2')),
                'Cz−(M1+M2)/2')
    if kind == 'FPz-M1':
        return chan('FPz') - chan('M1', 'A1'), 'FPz−M1'
    if kind == 'FPz-M2':
        return chan('FPz') - chan('M2', 'A2'), 'FPz−M2'
    raise ValueError(f'unknown derivation {kind!r}')


def onset_peak(trace, t_ms, window=(1.0, 15.0)):
    """Largest deflection inside window, as (latency_ms, signed_amplitude).

    The onset peak, not the global extremum: later ABR waves are often larger,
    and the window pins the measurement to the wave of interest.
    """
    t = np.asarray(t_ms, dtype=float)
    mask = (t >= window[0]) & (t <= window[1])
    if not mask.any():
        return float('nan'), float('nan')
    seg = np.asarray(trace, dtype=float)[mask]
    i = int(np.argmax(np.abs(seg)))
    return float(t[mask][i]), float(seg[i])


def onset_latency(trace, t_ms, frac=0.3, skip_ms=1.0):
    """First time |trace| exceeds frac of its own peak, after skip_ms.

    More robust than peak latency for ordering generators, since it does not
    depend on waveform shape. skip_ms steps over the band-pass edge transient.
    """
    t = np.asarray(t_ms, dtype=float)
    mask = t > skip_ms
    a = np.abs(np.asarray(trace, dtype=float)) * mask
    if not a.any():
        return float('nan')
    return float(t[int(np.argmax(a > frac * a.max()))])


def zscore(values):
    """Standardise a curve; all-equal input returns zeros rather than NaN."""
    a = np.asarray(values, dtype=float)
    sd = a.std()
    return np.zeros_like(a) if sd == 0 else (a - a.mean()) / sd


def time_axis(n_samples, srate):
    """Time vector in ms for `n_samples` at `srate` Hz."""
    return np.arange(n_samples) / srate * 1e3
