#!/usr/bin/env python3
"""
Reading an ABR sweep across artificial ITDs.

--figure summary  onset-peak latency and amplitude versus ITD, plus the CSV of
                  the numbers behind them
--figure overlay  the waveforms themselves, one colour per ITD, at Cz, M1 and
                  in the Cz-M1 derivation
--figure compare  latency versus ITD with and without MSO inhibition, read from
                  two summary CSVs, so it tests whether the ITD dependence
                  survives when the inhibition is removed

All three locate their inputs by rebuilding the directory main_abr.py wrote:
one multi-ITD .pic, so the stem is fixed and only the ITD label varies.

Usage:
  python ABR_reconstruction/plots/itd_sweep.py --figure summary \\
      --pic-file RESULTS/artificial_itd.pic --out RESULTS/click_itd_sweep_L --side L
  python ABR_reconstruction/plots/itd_sweep.py --figure overlay --itds 0 -400 400
  python ABR_reconstruction/plots/itd_sweep.py --figure compare
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from recon_core import params as P, paths
from ABR_reconstruction.plots import common

DEFAULT_ITDS = [-1000, -800, -600, -400, -200, 0, 200, 400, 600, 800, 1000]
BLUE = '#4C82B5'
# Colour-blind-friendly, assigned in sorted ITD order.
PALETTE = ['#000000', '#4C82B5', '#D1615D', '#3E9B72', '#B07AA1',
           '#E49444', '#5778A4', '#85B6B2', '#A87C9F', '#E7CA60', '#6A9F58']


def itd_dir(stem, itd_us, side, tag=''):
    """The ABR output directory main_abr.py --itd-us wrote."""
    return common.abr_dir(stem, common.condition_label(itd_us, 'itd'), side,
                          suffix=tag)


def _despine(ax, panel_letter=None):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axvline(0, color='k', lw=0.5, ls=':')       # midline, ITD = 0
    if panel_letter:
        ax.text(-0.10, 1.04, panel_letter, transform=ax.transAxes, fontsize=17,
                fontweight='bold', va='bottom', ha='left')


def collect_peaks(stem, itds, side, tag, derivation, window):
    """Onset peak per ITD, as (itds, latencies_ms, amplitudes_µV)."""
    used, latencies, amplitudes = [], [], []
    print(f'{"ITD(us)":>8} {"latency(ms)":>12} {"amplitude(uV)":>15}')
    for itd in sorted(itds):
        loaded = common.load_derivation(itd_dir(stem, itd, side, tag), derivation)
        if loaded is None:
            print(f'{itd:>8g}   MISSING')
            continue
        trace, t = loaded
        latency, amplitude = common.onset_peak(trace, t, window=window)
        used.append(itd)
        latencies.append(latency)
        amplitudes.append(amplitude)
        print(f'{itd:>8g} {latency:>12.3f} {amplitude:>15.4e}')
    return np.array(used), np.array(latencies), np.array(amplitudes)


def plot_summary(out_dir, stem, itds, side, tag, derivation, window, stim_label,
                 n_cells):
    """Latency-vs-ITD and amplitude-vs-ITD, plus the CSV behind them."""
    itds, latencies, amplitudes = collect_peaks(stem, itds, side, tag,
                                                derivation, window)
    if not len(itds):
        sys.exit('no ABR.h5 found for any ITD, did the sweep finish?')

    os.makedirs(out_dir, exist_ok=True)
    title = (f'{stim_label}   |   {derivation}, onset peak '
             f'({window[0]:g}–{window[1]:g} ms)   |   N = {n_cells}')

    for values, ylabel, panel, name in (
            (latencies, 'Peak latency (ms)', 'A', 'abr_itd_peak_latency.png'),
            (amplitudes, 'Peak amplitude (µV)', 'B', 'abr_itd_peak_amplitude.png')):
        fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
        ax.plot(itds, values, '-o', color=BLUE, lw=1.8, markersize=6,
                markerfacecolor=BLUE, markeredgecolor='white', markeredgewidth=0.8)
        ax.set_xlabel('ITD (µs)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(itds)
        _despine(ax, panel)
        common.save(fig, os.path.join(out_dir, name), dpi=170)

    csv = os.path.join(out_dir, 'abr_itd_peaks.csv')
    np.savetxt(csv, np.column_stack([itds, latencies, amplitudes]),
               header='itd_us,peak_latency_ms,peak_amplitude_uV',
               delimiter=',', comments='')
    print(f'  saved {csv}')
    return csv


def plot_overlay(out_png, stem, itds, side, tag, stim_label, n_cells):
    """One curve per ITD at Cz and M1, and in the Cz-M1 derivation."""
    fig = plt.figure(figsize=(13, 8), constrained_layout=True)
    grid = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1])
    ax_cz = fig.add_subplot(grid[0, 0])
    ax_m1 = fig.add_subplot(grid[0, 1])
    ax_diff = fig.add_subplot(grid[1, :])

    missing = []
    for k, itd in enumerate(itds):
        directory = itd_dir(stem, itd, side, tag)
        loaded = common.load_trace(directory)
        if loaded is None:
            missing.append(itd)
            continue
        V, names, srate = loaded
        t = np.arange(V.shape[1]) / srate * 1e3
        colour = PALETTE[k % len(PALETTE)]
        label = f'ITD {itd:+g} µs' if itd else 'ITD 0 µs'
        ax_cz.plot(t, V[names.index('Cz')], color=colour, lw=1.0, label=label)
        ax_m1.plot(t, V[names.index('M1')], color=colour, lw=1.0, label=label)
        diff, _ = common.derive(V, names, 'Cz-M1')
        ax_diff.plot(t, diff, color=colour, lw=1.1, label=label)

    if missing:
        print('WARNING: no ABR.h5 for ITDs ' +
              ', '.join(f'{v:g}' for v in missing))

    for ax in (ax_cz, ax_m1, ax_diff):
        ax.axhline(0, color='k', lw=0.4, ls=':')
        ax.set_xlabel('Time (ms)')
        ax.legend(fontsize=9)
    ax_cz.set_ylabel('Potential (µV)')
    ax_m1.set_ylabel('Potential (µV)')
    ax_diff.set_ylabel('Amplitude (µV)')
    ax_cz.set_title(f'Cz (vertex)  |  {stim_label}  |  side {side}  |  N={n_cells}')
    ax_m1.set_title('M1 (left mastoid)')
    ax_diff.set_title('Cz−M1  (vertex-positive upward)')
    fig.suptitle('MSO ABR  |  vertex-positive upward', fontsize=11,
                 fontweight='bold')
    common.save(fig, out_png)


def plot_inhibition_compare(out_dir, sides, n_cells):
    """Latency vs ITD with and without MSO inhibition, from the summary CSVs.

    Reads what --figure summary wrote for the intact and the _noinh sweeps, so
    the two curves are the same measurement.
    """
    os.makedirs(out_dir, exist_ok=True)
    for side in sides:
        curves = {}
        for label, suffix in (('inhibition intact', ''),
                              ('inhibition blocked', '_noinh')):
            csv = os.path.join(paths.RESULTS_DIR,
                               f'click_itd_sweep_{side}{suffix}', 'abr_itd_peaks.csv')
            if not os.path.exists(csv):
                print(f'[{side}] missing {csv} -> skip')
                continue
            data = np.genfromtxt(csv, delimiter=',', names=True)
            curves[label] = (data['itd_us'], data['peak_latency_ms'])
        if not curves:
            continue

        fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
        for (label, (itds, latencies)), colour in zip(curves.items(),
                                                      (BLUE, '#D1615D')):
            ax.plot(itds, latencies, '-o', color=colour, lw=1.8, markersize=6,
                    markerfacecolor=colour, markeredgecolor='white',
                    markeredgewidth=0.8, label=label)
        ax.set_xlabel('ITD (µs)')
        ax.set_ylabel('Peak latency (ms)')
        ax.set_title(f'MSO ABR onset-peak latency vs ITD  |  side {side}  '
                     f'|  N = {n_cells}')
        ax.legend(fontsize=9)
        _despine(ax)
        common.save(fig, os.path.join(out_dir, f'abr_itd_latency_inh_vs_noinh_{side}.png'),
                    dpi=170)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(ap, side=True)
    ap.add_argument('--figure', choices=['summary', 'overlay', 'compare'],
                    default='summary')
    ap.add_argument('--itds', type=float, nargs='+', default=DEFAULT_ITDS)
    ap.add_argument('--tag', default='',
                    help="trailing tag on the ABR output dirs, e.g. '_noinh'")
    ap.add_argument('--sides', nargs='+', default=['L', 'R'],
                    help='sides to compare (compare figure only)')
    ap.add_argument('--n-cells', dest='n_cells', type=int, default=P.N_MSO_TOTAL)
    args = ap.parse_args()

    stem = common.resolve_stem(args)
    window = tuple(args.win)

    if args.figure == 'summary':
        plot_summary(args.out or os.path.join(paths.RESULTS_DIR, 'click_itd_sweep'),
                     stem, args.itds, args.side, args.tag, args.derivation,
                     window, args.stim_label, args.n_cells)
    elif args.figure == 'overlay':
        out = args.out or os.path.join(paths.RESULTS_DIR, 'click_itd_sweep',
                                       'mso_abr_overlay_itd.png')
        plot_overlay(out, stem, args.itds, args.side, args.tag, args.stim_label,
                     args.n_cells)
    else:
        plot_inhibition_compare(
            args.out or os.path.join(paths.RESULTS_DIR, 'click_itd_sweep_compare'),
            args.sides, args.n_cells)


if __name__ == '__main__':
    main()
