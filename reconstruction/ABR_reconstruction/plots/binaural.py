#!/usr/bin/env python3
"""
Binaural interaction, per nucleus: does RL equal R + L?

--figure additivity  RL overlaid on R+L for one nucleus, with the residual
                     BI = RL - (R+L) underneath. AVCN and MNTB are monaural by
                     construction, so their residual must be about 0; a
                     non-zero one would mean a monaural generator had leaked
                     binaural information.
--figure residual    the residual alone, all four nuclei side by side on one
                     shared y-range, so their relative size is readable.

Both call main_abr_full.assemble() for the three acoustic conditions; nothing
is re-simulated.

Usage:
  python ABR_reconstruction/plots/binaural.py --pic-file RESULTS/<f>.pic
  python ABR_reconstruction/plots/binaural.py --figure residual
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from recon_core import paths
from recon_core.signal_utils import derive as _derive, time_axis
from ABR_reconstruction.main_abr_full import assemble, ELECTRODES
from ABR_reconstruction.main_abr_bi import CONDITIONS, DERIVATIONS, DERIV_LABEL
from ABR_reconstruction.plots import common

ALL_NUCLEI = ['MSO', 'LSO', 'MNTB', 'AVCN']
GROUPS = {'monaural': ['AVCN', 'MNTB'],   # RL = R+L, no interaction possible
          'binaural': ['MSO', 'LSO'],     # RL != R+L, the interaction is the signal
          'all': ALL_NUCLEI}


def _nucleus_composites(stem, cond_label, nucleus, band, lso_generator='synaptic'):
    """Return {tag: composite (n_e,T) µV} for RL/R/L, restricted to one nucleus."""
    comp = {}
    srate = None
    for tag, cond in CONDITIONS.items():
        V_gen, _V_nuc, sr = assemble(stem, cond_label, {nucleus}, ['L', 'R'], cond,
                                     lso_generator=lso_generator, band=band)
        comp[tag] = V_gen['composite']
        srate = sr
    return comp, srate


def plot_additivity(out_dir, nucleus, comp, srate, band):
    t = time_axis(comp['RL'].shape[1], srate)
    fig, axes = plt.subplots(2, len(DERIVATIONS), figsize=(11, 6.5),
                             sharex=True, constrained_layout=True)

    max_resid = 0.0
    top_data, bot_data = [], []
    for j, deriv in enumerate(DERIVATIONS):
        R, _  = _derive(comp['R'],  ELECTRODES, deriv)
        L, _  = _derive(comp['L'],  ELECTRODES, deriv)
        RL, _ = _derive(comp['RL'], ELECTRODES, deriv)
        sumRL = R + L
        resid = RL - sumRL
        max_resid = max(max_resid, np.abs(resid).max())
        top_data.append((R, L, RL, sumRL))
        bot_data.append(resid)

        # --- top: overlay RL against R+L, plus the R and L components ---
        # colour standard: right ear green, left ear light purple, RL dark grey
        # so the dashed red R+L overlay stays visible.
        ax = axes[0, j]
        ax.plot(t, R,  color='seagreen',    lw=0.9, alpha=0.7, label='R (right)')
        ax.plot(t, L,  color='mediumpurple', lw=0.9, alpha=0.7, label='L (left)')
        ax.plot(t, RL, color='#3a3a3a',     lw=2.4, label='RL (binaural)')
        ax.plot(t, sumRL, color='crimson',  lw=1.3, ls='--', label='R + L')
        ax.axhline(0, color='k', lw=0.4, ls=':')
        ax.set_xlim(0, 15)
        ax.set_title(DERIV_LABEL[deriv], fontsize=11, fontweight='bold')
        ax.set_ylabel('µV')
        if j == 0:
            ax.legend(fontsize=8, loc='upper right')

        # --- bottom: residual RL-(R+L), the binaural interaction ---
        # About 0 for the monaural nuclei (AVCN, MNTB), the real BI for MSO/LSO.
        axr = axes[1, j]
        axr.plot(t, resid, color='crimson', lw=1.2)
        axr.axhline(0, color='k', lw=0.4, ls=':')
        axr.set_xlim(0, 15)
        axr.set_xlabel('Time (ms)')
        axr.set_ylabel('BI = RL − (R+L)  µV')
        axr.set_title(f'binaural interaction  (max |·| = {np.abs(resid).max():.2e} µV)',
                      fontsize=9)

    # share the y-range across the top row (ABR traces) and, separately, across
    # the bottom row (residuals); the two rows are not forced to match each
    # other. A 10% margin keeps curves off the plot boundary.
    PAD = 0.10
    top_lo = min(min(x.min() for x in quad) for quad in top_data)
    top_hi = max(max(x.max() for x in quad) for quad in top_data)
    top_margin = (top_hi - top_lo) * PAD
    top_lo, top_hi = top_lo - top_margin, top_hi + top_margin
    for j in range(len(DERIVATIONS)):
        axes[0, j].set_ylim(top_lo, top_hi)

    bot_lo = min(r.min() for r in bot_data)
    bot_hi = max(r.max() for r in bot_data)
    bot_margin = (bot_hi - bot_lo) * PAD
    bot_lo, bot_hi = bot_lo - bot_margin, bot_hi + bot_margin
    for j in range(len(DERIVATIONS)):
        axes[1, j].set_ylim(bot_lo, bot_hi)

    is_mono = nucleus in ('AVCN', 'MNTB')
    headline = ('RL = R + L  ⇒  NO binaural interaction (BI ≈ 0)' if is_mono else
                'RL ≠ R + L  ⇒  binaural interaction  BI = RL − (R+L)')
    fig.suptitle(f'{nucleus}: {headline}\n{band[0]:.0f}–{band[1]:.0f} Hz  |  '
                 f'vertex-positive up  |  max |BI| = {max_resid:.2e} µV',
                 fontsize=11, fontweight='bold')

    path = os.path.join(out_dir, 'figures', f'{nucleus.lower()}_additivity.png')
    common.save(fig, path)
    print(f'{nucleus}: max|RL-(R+L)| = {max_resid:.3e} µV')
    return max_resid


def plot_residual(out_dir, stem, cond_label, band, nuclei, derivation='Cz-M1'):
    """The BI residual for every nucleus, side by side on one shared y-range."""
    residuals, t = {}, None
    for nuc in nuclei:
        comp, srate = _nucleus_composites(stem, cond_label, nuc, band)
        R, _  = _derive(comp['R'],  ELECTRODES, derivation)
        L, _  = _derive(comp['L'],  ELECTRODES, derivation)
        RL, _ = _derive(comp['RL'], ELECTRODES, derivation)
        residuals[nuc] = RL - (R + L)
        if t is None:
            t = time_axis(RL.shape[0], srate)

    fig, axes = plt.subplots(1, len(nuclei), figsize=(4 * len(nuclei), 4.2),
                             sharex=True, sharey=True, constrained_layout=True)

    PAD = 0.10
    all_lo = min(r.min() for r in residuals.values())
    all_hi = max(r.max() for r in residuals.values())
    margin = (all_hi - all_lo) * PAD
    ylim = (all_lo - margin, all_hi + margin)

    for ax, nuc in zip(axes, nuclei):
        resid = residuals[nuc]
        ax.plot(t, resid, color='red', lw=1.2)
        ax.axhline(0, color='k', lw=0.4, ls=':')
        ax.set_xlim(0, 15)
        ax.set_ylim(*ylim)
        ax.set_xlabel('Time (ms)')
        ax.set_title(f'{nuc}  (max |·| = {np.abs(resid).max():.2e} µV)', fontsize=10)

    axes[0].set_ylabel('BI = RL − (R+L)  µV')
    fig.suptitle(f'Binaural interaction residual  |  {derivation.replace("Cz-", "Cz/")}  |  '
                f'{band[0]:.0f}–{band[1]:.0f} Hz', fontsize=12, fontweight='bold')

    path = os.path.join(out_dir, 'figures', 'bi_residual_all_nuclei.png')
    return common.save(fig, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pic-file', dest='pic_file', default=None)
    ap.add_argument('--angle', type=int, default=0)
    ap.add_argument('--figure', choices=['additivity', 'residual'],
                    default='additivity')
    ap.add_argument('--nucleus', default='binaural',
                    choices=ALL_NUCLEI + list(GROUPS),
                    help='one nucleus, or a group (additivity figure only)')
    ap.add_argument('--band', default='150,3000', help='band-pass lo,hi in Hz')
    args = ap.parse_args()

    stem = paths.pic_stem(paths.resolve_pic(args.pic_file))
    band = tuple(float(x) for x in args.band.split(','))
    _, cond_label = paths.condition_key(args.angle)
    out_dir = os.path.join(paths.ABR_TMP_DIR, f'output_bi_{stem}_{cond_label}')

    if args.figure == 'residual':
        plot_residual(out_dir, stem, cond_label, band, ALL_NUCLEI)
        return

    for nucleus in GROUPS.get(args.nucleus, [args.nucleus]):
        comp, srate = _nucleus_composites(stem, cond_label, nucleus, band)
        plot_additivity(out_dir, nucleus, comp, srate, band)


if __name__ == '__main__':
    main()
