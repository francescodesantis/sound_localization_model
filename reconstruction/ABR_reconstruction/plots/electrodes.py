#!/usr/bin/env python3
"""
Per-nucleus scalp ABR at Cz, M1 and M2: one figure per generator group.

Projects each nucleus's head-frame dipole records through the 4-sphere model
and band-passes them with P.BAND_DEFAULT, the same selection and filter the
composite uses (main_abr_full._discover).

Usage:
  python ABR_reconstruction/plots/electrodes.py --pic-file RESULTS/<f>.pic \
      --itd-us 0 --condition binaural --side both --out <dir>
      [--variant chain] [--anf-source boundary --wave1-source modiolus]

Writes <out>/cz_m1_m2_{anf,gbc,sbc,mntb,mso,lso}.png.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from recon_core.physics import head_model
from recon_core.utils import io_utils, paths
from recon_core.config import params as P
from recon_core.utils.signal_utils import bandpass, time_axis
from ABR_reconstruction.main_abr_full import MONAURAL_SIDES, anf_wave1_group
from ABR_reconstruction.plots import common

FULL_NAME = {'Cz': 'Cz (vertex)', 'M1': 'M1 (left mastoid)',
             'M2': 'M2 (right mastoid)'}


def generators(anf_source='boundary', wave1_source='modiolus'):
    """(key, label, [(nucleus, generator), ...]) for every figure."""
    return [
        ('anf', 'ANF (auditory nerve)',
         [('ANF', anf_wave1_group(anf_source, wave1_source)),
          *[('ANF', g) for g in P.ANF_STATIONARY_GROUPS]]),
        ('gbc', 'AVCN GBC', [('AVCN', 'GBCsoma'), ('AVCN', 'GBCaxon')]),
        ('sbc', 'AVCN SBC', [('AVCN', 'SBC')]),
        ('mntb', 'MNTB', [('MNTB', 'principal'), ('MNTB', 'calyx')]),
        ('mso', 'MSO', [('MSO', 'postsynaptic')]),
        ('lso', 'LSO', [('LSO', 'synaptic')]),
    ]


def record_sides(nucleus, condition, side='both'):
    """Hemispheres of a nucleus included under this condition and --side."""
    wanted = ('L', 'R') if side == 'both' else (side,)
    mono = MONAURAL_SIDES.get(condition)
    if mono is None or nucleus in ('MSO', 'LSO'):
        driven = ('L', 'R')
    elif nucleus in ('ANF', 'AVCN'):
        driven = (mono[0],)
    else:                                   # MNTB: the GBC -> calyx path decussates
        driven = (mono[1],)
    return tuple(s for s in driven if s in wanted)


def project(directory, gens, electrodes, condition, side):
    """Sum the records of `gens` at the scalp, band-passed, in µV.

    Returns (V, srate), or (None, reason) when a record is missing or no
    hemisphere is selected. Extended records contribute one source per bin.
    """
    sources, srate = [], None
    for nucleus, generator in gens:
        sides = record_sides(nucleus, condition, side)
        if not sides:
            return None, 'no hemisphere selected'
        for s in sides:
            base = os.path.join(directory, f'{nucleus}__{generator}__{s}__')
            path = base + f'{condition}.h5'
            if not os.path.exists(path):
                path = base + 'binaural.h5'
            if not os.path.exists(path):
                return None, f'missing {os.path.basename(path)}'
            attrs, p_head, r_dipole = io_utils.read_dipole_record(path)
            srate = attrs['srate']
            sources += [('all', p_head[k], r_dipole[k]) for k in range(p_head.shape[0])]
    V_mV = head_model.project_sources(sources, electrodes)['all']
    lo, hi = P.BAND_DEFAULT
    return bandpass(V_mV * 1e3, fs=srate, lo=lo, hi=hi), srate


def plot(label, t_ms, traces, electrodes, out_png, condition):
    """Three side-by-side panels sharing one y-range."""
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
    fig.suptitle(f'{label} ABR  |  {" / ".join(electrodes)}  |  {condition}  |  '
                 f'band-passed  |  vertex-positive upward',
                 fontsize=13, fontweight='bold')
    common.save(fig, out_png, dpi=170)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pic-file', dest='pic_file', required=True)
    sel = ap.add_mutually_exclusive_group()
    sel.add_argument('--angle', type=int, default=0)
    sel.add_argument('--itd-us', dest='itd_us', type=float, default=None)
    sel.add_argument('--ild-db', dest='ild_db', type=float, default=None)
    ap.add_argument('--condition', default='binaural',
                    choices=['binaural', 'left_ear', 'right_ear'])
    ap.add_argument('--side', default='both', choices=['L', 'R', 'both'])
    ap.add_argument('--variant', default=None, choices=['chain'])
    ap.add_argument('--anf-source', dest='anf_source', default='boundary',
                    choices=['full', 'boundary'])
    ap.add_argument('--wave1-source', dest='wave1_source', default='modiolus',
                    choices=['aperture', 'modiolus'])
    ap.add_argument('--out', required=True, help='output directory')
    args = ap.parse_args()

    stem = paths.pic_stem(args.pic_file)
    _, cond_label = paths.condition_key(args.angle, args.itd_us, args.ild_db)
    directory = paths.dipoles_dir_for(stem, cond_label, variant=args.variant)
    electrodes = list(P.ELECTRODES)
    os.makedirs(args.out, exist_ok=True)

    for key, label, gens in generators(args.anf_source, args.wave1_source):
        V, info = project(directory, gens, electrodes, args.condition, args.side)
        if V is None:
            print(f'[{label}] skipped: {info}')
            continue
        plot(label, time_axis(V.shape[1], info), list(V), electrodes,
             os.path.join(args.out, f'cz_m1_m2_{key}.png'), args.condition)


if __name__ == '__main__':
    main()
