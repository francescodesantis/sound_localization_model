#!/usr/bin/env python3
"""
Views of the complete brainstem ABR written by main_abr_full.py.

--layout electrodes    per-nucleus decomposition and composite, one panel per
                       scalp electrode (Cz, M1, M2).  --no-composite drops the
                       summed trace and writes full_abr_electrodes_nuclei.png,
                       leaving the composite figure in place; the y-axis then
                       rescales to the contributions instead of to the sum.
--layout derivations   composite only, Cz-M1 above Cz-M2
--layout contributions the decomposition in ONE clinical derivation: composite
                       over the per-nucleus traces, then the same traces alone.
                       Defaults to the ipsilateral montage for the run's
                       acoustic condition (left_ear -> Cz-M1, right_ear ->
                       Cz-M2); override with --derivation.

Both read the band-passed ABR_full.h5; nothing is recomputed.

Usage:
  python ABR_reconstruction/plots/full_abr.py --dir RESULTS/full_abr/<stem>_...
  python ABR_reconstruction/plots/full_abr.py --layout derivations
  python ABR_reconstruction/plots/full_abr.py --layout electrodes --no-composite
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


def plot_per_electrode(traces, names, srate, side, out_png,
                       with_composite=True):
    """One panel per electrode, each showing every nucleus.

    with_composite=False drops the summed trace.  That is not only a cosmetic
    choice: the composite is the largest trace in every panel and the three
    panels share one y-axis, so it sets the scale and flattens the nuclei that
    are small at that electrode.  Without it the axes rescale to the
    contributions themselves.
    """
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
        if with_composite:
            ax.plot(t, composite[i], color='k', lw=1.6, label='composite',
                    zorder=5)
        ax.axhline(0, color='k', lw=0.4, ls=':')
        ax.set_ylabel(f'{elec} potential (µV)')
        head = ('composite ABR and per-nucleus decomposition'
                if with_composite else 'per-nucleus contributions only')
        ax.set_title(f'{head} ({elec})  |  side {side}')
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


def ipsilateral_derivation(condition):
    """The montage a clinician would use for this acoustic condition.

    The reference mastoid is the one on the stimulated side, so it is Cz-M1 for
    a left-ear stimulus and Cz-M2 for a right-ear one. A binaural run has no
    ipsilateral side; Cz-M1 is used and the caller is told.
    """
    return {'left_ear': 'Cz-M1', 'right_ear': 'Cz-M2'}.get(condition, 'Cz-M1')


def run_condition(run_dir, attrs):
    """Acoustic condition of a run, from its attrs or failing that its name.

    Runs written before `condition` was stored carry it only in the directory
    name, so fall back to that rather than silently mislabelling the montage.
    """
    condition = attrs.get('condition')
    if condition:
        return str(condition), 'file'
    name = os.path.basename(os.path.normpath(run_dir))
    for cand in ('left_ear', 'right_ear'):
        if cand in name:
            return cand, 'directory name'
    return 'binaural', 'assumed'


def plot_contributions(traces, names, srate, side, kind, out_png, subtitle=''):
    """One derivation, twice: with the composite over it, then without.

    The second panel exists because the composite dominates the shared y-axis
    of the first: dropping it lets the axis fit the generators themselves, so
    the smaller ones stay readable.
    """
    composite = traces['composite']
    nuclei = {k.split('__', 1)[1]: v for k, v in traces.items()
              if k.startswith('nucleus__')}
    t = time_axis(composite.shape[1], srate)
    label = derive(composite, names, kind)[1]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9.5, 9),
                                   constrained_layout=True, sharex=True)

    for ax, with_composite, title in (
            (ax0, True, f'Composite ABR {label} and its per-nucleus contributions'),
            (ax1, False, f'Per-nucleus contributions only, {label}')):
        for nuc in sorted(nuclei):
            ax.plot(t, derive(nuclei[nuc], names, kind)[0], lw=1.0, label=nuc)
        if with_composite:
            ax.plot(t, derive(composite, names, kind)[0], color='k', lw=1.7,
                    label='composite', zorder=5)
        ax.axhline(0, color='k', lw=0.4, ls=':')
        ax.set_ylabel('Amplitude (µV)')
        ax.set_title(f'{title}  |  side {side}', fontsize=11)
        ax.legend(fontsize=8, ncol=2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    ax1.set_xlabel('Time (ms)')

    if subtitle:
        fig.suptitle(subtitle, fontsize=10)
    common.save(fig, out_png)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', default=None,
                    help='RESULTS/full_abr/<run> directory holding ABR_full.h5 '
                         '(default: the most recently written one)')
    ap.add_argument('--layout',
                    choices=['electrodes', 'derivations', 'contributions'],
                    default='electrodes')
    ap.add_argument('--derivation', default=None,
                    help='contributions layout only: which montage to plot '
                         '(default: the ipsilateral one for the run)')
    ap.add_argument('--no-composite', action='store_true',
                    dest='no_composite',
                    help='electrodes layout only: draw the per-nucleus '
                         'contributions without the summed composite, and '
                         'write full_abr_electrodes_nuclei.png so the '
                         'composite figure is not overwritten')
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
        attrs = dict(f.attrs)
    side = attrs.get('side', '?')

    name = args.layout + ('_nuclei' if args.no_composite
                          and args.layout == 'electrodes' else '')
    out = args.out or os.path.join(run_dir, 'figures', f'full_abr_{name}.png')
    if args.layout == 'electrodes':
        plot_per_electrode(traces, names, srate, side, out,
                           with_composite=not args.no_composite)
    elif args.layout == 'derivations':
        plot_derivations(traces, names, srate, side, out)
    else:
        condition, source = run_condition(run_dir, attrs)
        kind = args.derivation or ipsilateral_derivation(condition)
        note = (f'condition {condition} (from the {source})  ->  {kind}'
                if args.derivation is None
                else f'condition {condition} (from the {source}), '
                     f'montage forced to {kind}')
        if condition == 'binaural' and args.derivation is None:
            note += '   [binaural has no ipsilateral side]'
        print(f'  {note}')
        plot_contributions(traces, names, srate, side, kind, out,
                           subtitle=f'{attrs.get("stem", "")}  |  {note}')


if __name__ == '__main__':
    main()
