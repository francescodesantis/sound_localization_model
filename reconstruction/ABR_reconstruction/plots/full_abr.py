#!/usr/bin/env python3
"""
Figures of the complete brainstem ABR written by main_abr_full.py.

--layout electrodes    composite and per-nucleus traces, one panel per
                       electrode (Cz, M1, M2) -> full_abr_electrodes.png
--layout contributions composite and per-nucleus traces in the ipsilateral
                       derivation (left_ear: Cz-M1, right_ear: Cz-M2), then the
                       per-nucleus traces alone -> full_abr_contributions.png.
                       A binaural run also writes full_abr_contributions_Cz-M2.png.

Reads the band-passed ABR_full.h5 in --dir; writes into <dir>/figures/.

Usage:
  python ABR_reconstruction/plots/full_abr.py --dir <run> --layout contributions --tmax 20
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from recon_core.utils import io_utils
from recon_core.utils.signal_utils import derive, time_axis
from ABR_reconstruction.plots import common

def _apply_tmax(axes, tmax):
    """Limit the x-range of every axis, in ms.

    The ABR's waves all fall inside the first ~10-15 ms; the default 50 ms
    window devotes most of its width to the flat tail and squeezes the waves
    into the left edge, which is exactly where they need to be read. Axes share
    x, but set_xlim is applied to each so the helper works regardless.
    """
    if not tmax:
        return
    try:
        seq = list(axes)                 # array or tuple of axes
    except TypeError:
        seq = [axes]                     # a single Axes
    for ax in seq:
        ax.set_xlim(0, float(tmax))


def plot_per_electrode(traces, names, srate, side, out_png, tmax=None):
    """One panel per electrode, each showing every nucleus and the composite."""
    composite = traces['composite']
    nuclei = {k.split('__', 1)[1]: v for k, v in traces.items()
              if k.startswith('nucleus__')}
    t = time_axis(composite.shape[1], srate)

    fig, axes = plt.subplots(3, 1, figsize=(9, 12), constrained_layout=True,
                             sharex=True, sharey=True)
    for ax, elec in zip(axes, common.ELECTRODES):
        i = names.index(elec)
        for name in sorted(nuclei):
            ax.plot(t, nuclei[name][i], lw=0.9, label=name,
                    color=common.nucleus_colour(name))
        ax.plot(t, composite[i], color='k', lw=1.6, label='composite', zorder=5)
        ax.axhline(0, color='k', lw=0.4, ls=':')
        ax.set_ylabel(f'{elec} potential (µV)')
        ax.set_title(f'composite ABR and per-nucleus decomposition ({elec})  |  side {side}')
        ax.legend(fontsize=8, ncol=2)
    axes[-1].set_xlabel('Time (ms)')
    _apply_tmax(axes, tmax)
    common.save(fig, out_png)


def ipsilateral_derivation(condition):
    """Reference on the stimulated side: Cz-M1 for left_ear, Cz-M2 for right_ear,
    Cz-M1 for binaural."""
    return {'left_ear': 'Cz-M1', 'right_ear': 'Cz-M2'}.get(condition, 'Cz-M1')


def plot_contributions(traces, names, srate, side, kind, out_png,
                       subtitle='', tmax=None):
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
            ax.plot(t, derive(nuclei[nuc], names, kind)[0], lw=1.0, label=nuc,
                    color=common.nucleus_colour(nuc))
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
    _apply_tmax((ax0, ax1), tmax)
    common.save(fig, out_png)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', required=True,
                    help='run directory holding ABR_full.h5')
    ap.add_argument('--layout', choices=['electrodes', 'contributions'],
                    default='electrodes')
    ap.add_argument('--tmax', type=float, default=None,
                    help='upper limit of the time axis in ms (default: whole window)')
    args = ap.parse_args()

    path = os.path.join(args.dir, 'ABR_full.h5')
    if not os.path.exists(path):
        sys.exit(f'error: {path} not found, run main_abr_full.py first')
    traces, names, srate = io_utils.read_named_traces(path)
    with h5py.File(path, 'r') as f:
        attrs = dict(f.attrs)
    side = attrs.get('side', '?')
    condition = str(attrs.get('condition', 'binaural'))
    fig_dir = os.path.join(args.dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    if args.layout == 'electrodes':
        plot_per_electrode(traces, names, srate, side,
                           os.path.join(fig_dir, 'full_abr_electrodes.png'),
                           tmax=args.tmax)
        return
    outputs = [(ipsilateral_derivation(condition), 'full_abr_contributions.png')]
    if condition == 'binaural':
        outputs.append(('Cz-M2', 'full_abr_contributions_Cz-M2.png'))
    for kind, name in outputs:
        plot_contributions(traces, names, srate, side, kind,
                           os.path.join(fig_dir, name),
                           subtitle=f'{attrs.get("stem", "")}  |  {condition}  ->  {kind}',
                           tmax=args.tmax)


if __name__ == '__main__':
    main()
