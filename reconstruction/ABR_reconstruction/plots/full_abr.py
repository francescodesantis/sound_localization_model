#!/usr/bin/env python3
"""
Views of the complete brainstem ABR written by main_abr_full.py.

--layout electrodes   per-nucleus decomposition and composite, one panel per
                      scalp electrode (Cz, M1, M2)
--layout derivations  composite only, Cz-M1 above Cz-M2

Both read the band-passed ABR_full.h5; nothing is recomputed.

Usage:
  python ABR_reconstruction/plots/full_abr.py --dir RESULTS/full_abr/<stem>_...
  python ABR_reconstruction/plots/full_abr.py --layout derivations
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from recon_core import io_utils, paths
from recon_core.signal_utils import derive, time_axis
from ABR_reconstruction.plots import common

def _newest_run():
    """Most recently written RESULTS/full_abr/<run>, or None if there is none."""
    runs = [d for d in glob.glob(os.path.join(paths.FULL_ABR_DIR, '*'))
            if os.path.exists(os.path.join(d, 'ABR_full.h5'))]
    return max(runs, key=os.path.getmtime) if runs else None


def plot_per_electrode(traces, names, srate, side, out_png):
    """One panel per electrode, each showing every nucleus plus the composite."""
    composite = traces['composite']
    nuclei = {k.split('__', 1)[1]: v for k, v in traces.items()
              if k.startswith('nucleus__')}
    t = time_axis(composite.shape[1], srate)

    fig, axes = plt.subplots(3, 1, figsize=(9, 12), constrained_layout=True,
                             sharex=True, sharey=True)
    for ax, elec in zip(axes, common.ELECTRODES):
        i = names.index(elec)
        for name in sorted(nuclei):
            ax.plot(t, nuclei[name][i], lw=0.9, label=name)
        ax.plot(t, composite[i], color='k', lw=1.6, label='composite', zorder=5)
        ax.axhline(0, color='k', lw=0.4, ls=':')
        ax.set_ylabel(f'{elec} potential (µV)')
        ax.set_title(f'composite ABR and per-nucleus decomposition ({elec})'
                     f'  |  side {side}')
        ax.legend(fontsize=8, ncol=2)
    axes[-1].set_xlabel('Time (ms)')
    common.save(fig, out_png)


def plot_derivations(traces, names, srate, side, out_png):
    """Composite only, in the two clinical derivations."""
    composite = traces['composite']
    t = time_axis(composite.shape[1], srate)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), constrained_layout=True,
                             sharex=True, sharey=True)
    for ax, kind, colour in ((axes[0], 'Cz-M1', 'darkorchid'),
                             (axes[1], 'Cz-M2', 'teal')):
        trace, label = derive(composite, names, kind)
        ax.plot(t, trace, color=colour, lw=1.4, label=f'composite {label}')
        ax.axhline(0, color='k', lw=0.4, ls=':')
        ax.set_ylabel('Amplitude (µV)')
        ax.set_title(f'Composite ABR, derivation: {label}  |  side {side}')
        ax.legend(fontsize=9)
    axes[-1].set_xlabel('Time (ms)')
    common.save(fig, out_png)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', default=None,
                    help='RESULTS/full_abr/<run> directory holding ABR_full.h5 '
                         '(default: the most recently written one)')
    ap.add_argument('--layout', choices=['electrodes', 'derivations'],
                    default='electrodes')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    run_dir = args.dir or _newest_run()
    if run_dir is None:
        sys.exit(f'error: no composite found under {paths.FULL_ABR_DIR}, '
                 'run `python reconstruction/main.py abr full ...` first')
    path = os.path.join(run_dir, 'ABR_full.h5')
    if not os.path.exists(path):
        sys.exit(f'error: {path} not found, run main_abr_full.py first')

    traces, names, srate = io_utils.read_named_traces(path)
    import h5py
    with h5py.File(path, 'r') as f:
        side = f.attrs.get('side', '?')

    out = args.out or os.path.join(run_dir, 'figures',
                                   f'full_abr_{args.layout}.png')
    if args.layout == 'electrodes':
        plot_per_electrode(traces, names, srate, side, out)
    else:
        plot_derivations(traces, names, srate, side, out)


if __name__ == '__main__':
    main()
