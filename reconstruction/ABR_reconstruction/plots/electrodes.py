#!/usr/bin/env python3
"""
Per-nucleus scalp ABR at Cz, M1 and M2, one figure per generator.

Two sources of the same picture, selected by --filter:

  --filter band  (default) read the stored, band-passed ABR.h5
  --filter none  re-project the raw head-frame dipole records through the same
                 4-sphere model without the band-pass. The stored traces are
                 already filtered and that cannot be undone, so an unfiltered
                 view has to be rebuilt from the dipoles.

--validate re-applies each nucleus's own band to the unfiltered projection and
checks it reproduces the stored ABR.h5. That is the end-to-end gate on the ABR
chain: if projection then filter does not equal what the producer wrote, the
two have diverged.

Usage:
  python ABR_reconstruction/plots/electrodes.py --pic-file RESULTS/<f>.pic
  python ABR_reconstruction/plots/electrodes.py --filter none
  python ABR_reconstruction/plots/electrodes.py --validate
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

from recon_core import head_model, io_utils, params as P, paths
from recon_core.signal_utils import bandpass, time_axis
from ABR_reconstruction.plots import common

def generators(lso_generator='synaptic'):
    """Each generator: how its dipole records are named, which stored ABR.h5
    holds it, under which dataset key and directory suffix, and the band that
    producer used.

    The LSO row follows lso_generator: spiking and synaptic are alternative
    models of the same cells, never summed, written by different runs into
    different directories (main_abr_lso.py tags the synaptic one _syn).
    """
    lso_suffix = '' if lso_generator == 'spiking' else '_syn'
    return [
        # key            label             dipole (nucleus, generator)     prefix  key         band              suffix
        ('mso', 'MSO', [('MSO', 'postsynaptic')], None, 'data', P.BAND_TOLNAI, ''),
        ('lso', 'LSO', [('LSO', lso_generator)], 'lso', 'data', P.BAND_TOLNAI,
         lso_suffix),
        ('mntb', 'MNTB', [('MNTB', 'principal'), ('MNTB', 'calyx')], 'mntb',
         'composite', P.BAND_CLINICAL, ''),
        ('avcn_combined', 'AVCN (GBC+SBC)', [('AVCN', 'GBC'), ('AVCN', 'SBC')],
         'avcn', 'composite', P.BAND_CLINICAL, ''),
        ('avcn_gbc', 'AVCN GBC', [('AVCN', 'GBC')], 'avcn', 'GBC',
         P.BAND_CLINICAL, ''),
        ('avcn_sbc', 'AVCN SBC', [('AVCN', 'SBC')], 'avcn', 'SBC',
         P.BAND_CLINICAL, ''),
    ]

FULL_NAME = {'Cz': 'Cz (vertex)', 'M1': 'M1 (left mastoid)',
             'M2': 'M2 (right mastoid)'}


def project_unfiltered(stem, cond_label, generators, electrodes):
    """Sum the raw dipole records at the scalp, in µV, with no band-pass.

    Exactly head_model.superpose_sources minus the filter, so the two stay
    comparable, which is what makes --validate meaningful.
    """
    directory = paths.dipoles_dir_for(stem, cond_label)
    sources, srate = [], None
    for nucleus, generator in generators:
        for side in ('L', 'R'):
            path = os.path.join(directory,
                                f'{nucleus}__{generator}__{side}__binaural.h5')
            if not os.path.exists(path):
                return None, None
            attrs, p_head, r_dipole = io_utils.read_dipole_record(path)
            srate = attrs['srate']
            sources.append(('all', p_head, r_dipole))
    V_mV = head_model.project_sources(sources, electrodes)['all']
    return V_mV * 1e3, srate


def plot(label, t_ms, traces, electrodes, out_png, filtered):
    """Three side-by-side panels sharing one y-range, so they are comparable."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, elec, trace in zip(axes, electrodes, traces):
        ax.plot(t_ms, trace, color='tab:blue', lw=1.0, label=elec)
        ax.axhline(0, color='0.6', lw=0.5)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Potential (µV)')
        ax.legend(loc='best', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(FULL_NAME.get(elec, elec), fontsize=11)

    lo = min(ax.get_ylim()[0] for ax in axes)
    hi = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(lo, hi)

    band = 'band-passed' if filtered else 'NO bandpass'
    fig.suptitle(f'{label} ABR  |  {" / ".join(electrodes)}  |  {band}  |  '
                 'vertex-positive upward', fontsize=13, fontweight='bold')
    common.save(fig, out_png, dpi=170)


def validate(stem, cond, electrodes, gens):
    """Re-filter the projection and compare with what the producer stored."""
    ok = True
    for _key, label, sources, prefix, dataset, band, suffix in gens:
        V_raw, srate = project_unfiltered(stem, cond, sources, electrodes)
        if V_raw is None:
            print(f'[{label:16s}] missing dipole records -> skip')
            ok = False
            continue
        stored = common.load_trace(
            common.abr_dir(stem, cond, 'both', prefix=prefix, suffix=suffix),
            key=dataset)
        if stored is None:
            print(f'[{label:16s}] missing stored ABR.h5 -> skip')
            ok = False
            continue
        refiltered = bandpass(V_raw, fs=srate, lo=band[0], hi=band[1])
        reference = stored[0]
        n = min(refiltered.shape[1], reference.shape[1])
        rel = (np.abs(refiltered[:, :n] - reference[:, :n]).max()
               / (np.abs(reference).max() + 1e-30))
        verdict = 'OK' if rel < 1e-6 else 'MISMATCH'
        ok &= rel < 1e-6
        print(f'[{label:16s}] band {band[0]:.0f}-{band[1]:.0f} Hz  '
              f'rel-err={rel:.2e}  {verdict}')
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(ap, window=False, stim_label=False)
    ap.add_argument('--filter', choices=['band', 'none'], default='band',
                    help='band = stored band-passed traces; none = re-project raw dipoles')
    ap.add_argument('--validate', action='store_true',
                    help='check projection+filter reproduces the stored ABR.h5')
    ap.add_argument('--lso-generator', choices=['spiking', 'synaptic'],
                    default='synaptic', dest='lso_generator',
                    help='which LSO model to show; they are alternatives, never '
                         'summed, and live in different output directories')
    args = ap.parse_args()

    stem = common.resolve_stem(args)
    cond = common.resolve_condition(args)
    electrodes = common.ELECTRODES
    gens = generators(args.lso_generator)

    if args.validate:
        sys.exit(0 if validate(stem, cond, electrodes, gens) else 1)

    out_dir = args.out or os.path.join(
        paths.RESULTS_DIR, 'cz_m1_m2' + ('' if args.filter == 'band' else '_nofilter'))
    os.makedirs(out_dir, exist_ok=True)

    for key, label, sources, prefix, dataset, _band, dir_suffix in gens:
        if args.filter == 'band':
            loaded = common.load_trace(
                common.abr_dir(stem, cond, 'both', prefix=prefix,
                               suffix=dir_suffix), key=dataset)
            if loaded is None:
                print(f'[{label}] no stored {dataset} trace -> skip')
                continue
            V, names, srate = loaded
            traces = [V[names.index(e)] for e in electrodes]
            name_suffix = ''
        else:
            V, srate = project_unfiltered(stem, cond, sources, electrodes)
            if V is None:
                print(f'[{label}] missing dipole records -> skip')
                continue
            traces = list(V)
            name_suffix = '_nofilter'
        plot(label, time_axis(len(traces[0]), srate), traces, electrodes,
             os.path.join(out_dir, f'cz_m1_m2_{key}{name_suffix}.png'),
             filtered=args.filter == 'band')


if __name__ == '__main__':
    main()
