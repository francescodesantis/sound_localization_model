#!/usr/bin/env python3
"""
Near-field LFP amplitude as a function of sound azimuth, the tuning curve.

For each angle it phase-folds the compound LFP on the centre probe channel and
measures the peak-to-peak of the averaged cycle. Folding first keeps only what
is phase-locked to the stimulus, so the curve reflects the population response
rather than the noise floor.

Reads finished runs and re-simulates nothing. Angles with no run are skipped
and reported rather than silently dropped.

Usage:
  python reconstruction/main.py lfp tuning --pic-file RESULTS/<f>.pic --side L \\
      --angles -90 -45 0 45 90
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

from recon_core import io_utils, paths

NUCLEUS_PREFIX = {'mso': None, 'gbc': 'avcn', 'sbc': 'sbc', 'lso': 'lso',
                  'mntb': 'mntb'}


def cycle_amplitude(output_dir, stimulus_freq, skip_ms=10.0):
    """Peak-to-peak of the cycle-averaged LFP on the centre channel, in µV.

    Returns None when the run is missing or too short to hold a full cycle.
    """
    if not os.path.exists(os.path.join(output_dir, 'PointSourcePotential_sum.h5')):
        return None
    lfp, srate = io_utils.read_lfp_sum(output_dir)

    centre = lfp[lfp.shape[0] // 2]
    dt_ms = 1e3 / srate
    steady = scipy.signal.detrend(centre[int(skip_ms / dt_ms):])

    samples_per_cycle = max(1, int(round(1e3 / stimulus_freq / dt_ms)))
    n_cycles = len(steady) // samples_per_cycle
    if n_cycles < 1:
        return None
    folded = (steady[:n_cycles * samples_per_cycle]
              .reshape(n_cycles, samples_per_cycle).mean(axis=0))
    return float(folded.max() - folded.min())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pic-file', dest='pic_file', default=None)
    ap.add_argument('--nucleus', default='mso', choices=list(NUCLEUS_PREFIX))
    ap.add_argument('--side', default='L', choices=['L', 'R'])
    ap.add_argument('--angles', type=int, nargs='+',
                    default=[-90, -45, 0, 45, 90])
    ap.add_argument('--stim-freq', dest='stim_freq', type=float, default=None,
                    help='stimulus frequency in Hz (default: from the spike metadata)')
    ap.add_argument('--skip-ms', dest='skip_ms', type=float, default=10.0,
                    help='onset ramp skipped before folding (default 10)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    stem = paths.pic_stem(paths.resolve_pic(args.pic_file))
    prefix = NUCLEUS_PREFIX[args.nucleus]

    stim_freq = args.stim_freq
    if stim_freq is None:
        import json
        meta = os.path.join(paths.spikes_dir_for(stem, args.angles[0], args.side),
                            'metadata.json')
        if os.path.exists(meta):
            with open(meta) as f:
                stim_freq = json.load(f).get('stim_freq_hz')
    if stim_freq is None:
        sys.exit('error: stimulus frequency unknown, pass --stim-freq. '
                 'A tuning curve needs a periodic stimulus.')

    angles, amplitudes, missing = [], [], []
    for angle in args.angles:
        directory = paths.output_dir_for('lfp', stem, f'angle{angle}', args.side,
                                         prefix=prefix)
        amplitude = cycle_amplitude(directory, stim_freq, args.skip_ms)
        if amplitude is None:
            missing.append(angle)
            continue
        angles.append(angle)
        amplitudes.append(amplitude)
        print(f'  angle {angle:>+4d}°  peak-to-peak = {amplitude:.4g} µV')

    if missing:
        print('no usable run for angles: ' + ', '.join(str(a) for a in missing))
    if not angles:
        sys.exit('no finished runs found, run the LFP pipeline for these angles first.')

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(angles, amplitudes, marker='o', color='firebrick', lw=2, markersize=8)
    best = int(np.argmax(amplitudes))
    ax.axvline(angles[best], color='gray', ls='--', alpha=0.5)
    ax.text(angles[best] + 2, min(amplitudes), f'max @ {angles[best]}°', color='gray')
    ax.set_title(f'{args.nucleus.upper()} LFP tuning curve  |  side {args.side}  '
                 f'|  {stim_freq:.0f} Hz', fontsize=13, fontweight='bold')
    ax.set_xlabel('Sound azimuth (°)')
    ax.set_ylabel('Cycle-averaged peak-to-peak LFP (µV)')
    ax.set_xticks(angles)
    ax.grid(True, ls='--', alpha=0.7)

    out = args.out or os.path.join(paths.LFP_TMP_DIR,
                                   f'{args.nucleus}_tuning_curve_{args.side}.png')
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'figure saved to {out}')


if __name__ == '__main__':
    main()
