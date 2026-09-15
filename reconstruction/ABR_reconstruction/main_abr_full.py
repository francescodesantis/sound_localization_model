"""
Complete brainstem ABR: superpose all nuclei into one composite.

Each per-nucleus ABR script (main_abr{,_avcn,_lso,_mntb}.py) writes a
standardised head-frame dipole record per (generator, side) into

    RESULTS/abr_tmp/dipoles/<stem>_<cond>/<nucleus>__<generator>__<side>.h5

(see main_abr.save_dipole_record), where <cond> is the stimulus label the
producers ran with ('angle0', 'itd500us', 'ild-10dB'). This orchestrator
collects those records for one stimulus, selected with the same --angle,
--itd-us or --ild-db flag, projects every generator through the 4-sphere model
from its own anatomical position, and sums the scalp potentials. No NEURON
simulation is re-run: run each nucleus once at full N, then this.

All four pipelines share DT=0.026 ms and TSTOP=50 ms, so every dipole lives on
the same time axis and the inter-wave latencies fall out of the NEST spike
timing. Full-count runs are assumed (no N_total/n_cells scaling): the raw head
dipoles are summed as they are.

Generators, each modelled exactly once so nothing is counted twice:
  AVCN:GBC   waves II to III (GBC soma, dendrites and 4 mm crossing axon)
  AVCN:SBC   wave II
  MNTB:principal / MNTB:calyx   small own field (wave III proper is the GBC axon)
  MSO:postsynaptic            waves IV to V
  LSO:spiking|synaptic        waves IV to V (LL travelling wave, BIC)
Wave I (ANF) is not modelled, so the earliest generator is the GBC.

Output:
  RESULTS/full_abr/<stem>_<cond>_<sidespec>/ABR_full.h5
        (per-generator, per-nucleus and composite, µV) + figures/full_abr.png

Usage:
  python ABR_reconstruction/main_abr_full.py --pic-file RESULTS/<f>.pic \
        --angle 0 --side both [--nuclei MSO,AVCN,LSO,MNTB] [--derivation Cz-M1]
"""

import os
import sys
import glob
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from recon_core import head_model, io_utils, paths
from recon_core.head_geometry import ABR_ELECTRODES
from recon_core.signal_utils import derive, onset_latency, time_axis

# No NEURON here: this orchestrator only reads dipole records the producers wrote.
ELECTRODES = list(ABR_ELECTRODES)
REPO_ROOT = paths.REPO_ROOT

_pic_stem = paths.pic_stem
dipoles_dir_for = paths.dipoles_dir_for


# ---------------------------------------------------------------------------
def _discover(stem, cond_label, want_nuclei, want_sides, want_condition,
              lso_generator='spiking'):
    """Load matching dipole records as (records, srate).

    Each record is (label, nucleus, side, p_head, r_dipole). srate comes from
    the records themselves, so this orchestrator needs no simulation constants.

    Per (nucleus, generator, side) the requested want_condition is used if a
    record for it exists, else the binaural record. So a monaural composite
    reuses the condition-invariant AVCN/MNTB records, only ever written as
    binaural, while picking the condition-specific MSO/LSO records.

    lso_generator selects the LSO generator: 'spiking' (output travelling wave)
    and 'synaptic' (postsynaptic dipole) are alternative models of the LSO's
    ABR contribution and are never summed, so only the requested one is kept.

    Under a monaural condition (left_ear/right_ear) the nuclei that are
    monaural by construction are restricted to the ear-appropriate hemisphere:
    the silent ear drives no click-locked response, so AVCN keeps only its
    ipsilateral side and MNTB the opposite one (the GBC to calyx projection
    decussates). MSO and LSO keep both hemispheres.
    """
    # ear to (AVCN side, MNTB side) for monaural conditions
    _EAR_SIDES = {'right_ear': ('R', 'L'), 'left_ear': ('L', 'R')}
    mono = _EAR_SIDES.get(want_condition)

    ddir = dipoles_dir_for(stem, cond_label)
    record_paths = sorted(glob.glob(os.path.join(ddir, '*.h5')))
    if not record_paths:
        raise FileNotFoundError(
            f'no dipole records in {ddir}\n  Run each nucleus ABR first, with '
            f'the same stimulus selector, e.g.\n'
            f'    python ABR_reconstruction/main_abr_avcn.py --pic-file ... '
            f'--side both')

    cand = {}    # (nucleus, generator, side) to {condition: record}
    srates = set()
    for path in record_paths:
        attrs, p_head, r_dipole = io_utils.read_dipole_record(path)
        nucleus, generator = attrs['nucleus'], attrs['generator']
        side, cond = attrs['side'], attrs['condition']
        if want_nuclei and nucleus not in want_nuclei:
            continue
        if side not in want_sides:
            continue
        if nucleus == 'LSO' and generator != lso_generator:
            continue   # keep only the requested LSO drive (mutually exclusive)
        if mono is not None:
            # monaural: keep only the ear-driven hemisphere for AVCN/MNTB
            if nucleus == 'AVCN' and side != mono[0]:
                continue
            if nucleus == 'MNTB' and side != mono[1]:
                continue
        srates.add(attrs['srate'])
        cand.setdefault((nucleus, generator, side), {})[cond] = (
            f'{nucleus}:{generator}', nucleus, side, p_head, r_dipole)

    recs = []
    for by_cond in cand.values():
        if want_condition in by_cond:
            recs.append(by_cond[want_condition])
        elif 'binaural' in by_cond:
            recs.append(by_cond['binaural'])
    if not recs:
        raise FileNotFoundError(
            f'records exist in {ddir} but none match nuclei={want_nuclei} '
            f'sides={want_sides} condition={want_condition}')
    if len(srates) > 1:
        raise ValueError(f'dipole records disagree on the sample rate: {srates}. '
                         'Re-run the producers so every nucleus shares one DT.')
    return recs, srates.pop()


def assemble(stem, cond_label, want_nuclei, sides, condition,
             lso_generator='synaptic', band=(150., 3000.)):
    """Collect dipole records for one stimulus and superpose them.

    cond_label is the stimulus label the producers filed their records under
    ('angle0', 'itd500us', 'ild-10dB'); condition is the acoustic condition
    ('binaural', 'left_ear', 'right_ear').

    Returns (V_gen, V_nuc, srate): V_gen maps 'nucleus:generator' and
    'composite' to band-passed (n_e, T) µV; V_nuc maps nucleus to summed µV.
    Reused by main() and by the Curio & Weigel BI reproduction (main_abr_bi.py).
    """
    lo, hi = band
    recs, srate = _discover(stem, cond_label, want_nuclei, set(sides), condition,
                            lso_generator=lso_generator)
    V_gen = head_model.superpose_sources(
        [(label, p_head, r) for (label, _nuc, _side, p_head, r) in recs],
        ELECTRODES, srate, lo=lo, hi=hi)

    # per-nucleus is the sum of that nucleus's traces (safe, the filter is linear)
    nuc_of = {f'{n}:{g}': n for (lbl, n, s, _p, _r) in recs
              for (g,) in [(lbl.split(':', 1)[1],)]}
    V_nuc = defaultdict(lambda: 0.0)
    for label, V in V_gen.items():
        if label == 'composite':
            continue
        V_nuc[nuc_of[label]] = V_nuc[nuc_of[label]] + V
    return V_gen, dict(V_nuc), srate


_derive = derive


def _plot(out_dir, V_gen, V_nuc, srate, side, derivation):
    t = time_axis(V_gen['composite'].shape[1], srate)
    cz = ELECTRODES.index('Cz')
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 8), constrained_layout=True)

    for nuc in sorted(V_nuc):
        ax0.plot(t, V_nuc[nuc][cz], lw=0.9, label=nuc)
    ax0.plot(t, V_gen['composite'][cz], color='k', lw=1.6, label='composite', zorder=5)
    ax0.axhline(0, color='k', lw=0.4, ls=':')
    ax0.set_xlabel('Time (ms)'); ax0.set_ylabel('Cz potential (µV)')
    ax0.set_title(f'composite ABR AND per-nucleus decomposition (Cz)  |  side {side}')
    ax0.legend(fontsize=8, ncol=2)

    diff, lbl = _derive(V_gen['composite'], ELECTRODES, derivation)
    ax1.plot(t, diff, color='darkorchid', lw=1.1, label=f'composite {lbl}')
    ax1.axhline(0, color='k', lw=0.4, ls=':')
    ax1.set_xlabel('Time (ms)'); ax1.set_ylabel('Amplitude (µV)')
    ax1.set_title(f'Composite ABR {lbl}  (vertex-positive upward)')
    ax1.legend(fontsize=9)

    path = os.path.join(out_dir, 'figures', 'full_abr.png')
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f'figure saved to {path}')


def _summary(V_nuc, V_gen, srate):
    """Per-nucleus Cz onset and peak table, the wave-ordering check.

    Onset (first crossing of 30% of the peak, after the band-pass edge) is the
    robust ordering metric; peak latency depends on waveform shape.
    """
    t = time_axis(V_gen['composite'].shape[1], srate)
    cz = ELECTRODES.index('Cz')
    m = t > 1.0   # skip the band-pass edge transient

    def onset(v):
        return onset_latency(v, t)

    print('\n  nucleus        onset (ms)   peak (ms)   peak|Cz| (µV)')
    print('  ' + '-' * 52)
    for nuc in sorted(V_nuc):
        v = V_nuc[nuc][cz]; i = int(np.argmax(np.abs(v) * m))
        print(f'  {nuc:<12s}   {onset(v):>8.2f}   {t[i]:>8.2f}   {np.abs(v[i]):>12.3e}')
    v = V_gen['composite'][cz]; i = int(np.argmax(np.abs(v) * m))
    print(f'  {"composite":<12s}   {onset(v):>8.2f}   {t[i]:>8.2f}   {np.abs(v[i]):>12.3e}\n')


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='Complete brainstem ABR (all nuclei)')
    ap.add_argument('--pic-file', type=str, default=None, dest='pic_file')
    ap.add_argument('--angle',    type=int, default=0)
    ap.add_argument('--itd-us', type=float, default=None, dest='itd_us',
                    help='artificial ITD in µs; overrides --angle, and selects '
                         'the dipole records the producers filed under it.')
    ap.add_argument('--ild-db', type=float, default=None, dest='ild_db',
                    help='artificial ILD in dB; overrides --itd-us and --angle.')
    ap.add_argument('--side',     type=str, default='both', choices=['L', 'R', 'both'])
    ap.add_argument('--nuclei',   type=str, default=None,
                    help='comma-separated subset (e.g. MSO,AVCN); default = all present')
    ap.add_argument('--derivation', type=str, default='Cz-M1',
                    choices=['Cz-M1', 'Cz-M2', 'Cz-avg'])
    ap.add_argument('--condition', type=str, default='binaural',
                    choices=['binaural', 'left_ear', 'right_ear'],
                    help='acoustic condition: picks condition-specific MSO/LSO '
                         'records, falling back to binaural for AVCN/MNTB')
    ap.add_argument('--lso-generator', type=str, default='synaptic',
                    choices=['spiking', 'synaptic'], dest='lso_generator',
                    help='which LSO generator to include (alternatives, not summed)')
    ap.add_argument('--band', type=str, default='150,3000',
                    help='band-pass lo,hi in Hz (default 150,3000)')
    args = ap.parse_args()

    pic_file = args.pic_file or os.path.join(REPO_ROOT, 'RESULTS',
                                             'baseline_simulation.pic')
    stem  = _pic_stem(pic_file)
    _, cond_label = paths.condition_key(args.angle, args.itd_us, args.ild_db)
    sides = ['L', 'R'] if args.side == 'both' else [args.side]
    want_nuclei = (set(n.strip() for n in args.nuclei.split(',')) if args.nuclei
                   else None)
    lo, hi = (float(x) for x in args.band.split(','))

    V_gen, V_nuc, srate = assemble(stem, cond_label, want_nuclei, sides,
                                   args.condition, lso_generator=args.lso_generator,
                                   band=(lo, hi))
    print(f'collected generators ({args.condition}, LSO={args.lso_generator}): '
          f'{sorted(k for k in V_gen if k != "composite")}')

    cond_tag = '' if args.condition == 'binaural' else f'_{args.condition}'
    lso_tag  = '' if args.lso_generator == 'spiking' else f'_lso{args.lso_generator}'
    # cond_label names the dipole set this composite was built from
    # (RESULTS/abr_tmp/dipoles/<stem>_<cond_label>).
    out_dir = paths.make_output_dirs(
        os.path.join(paths.FULL_ABR_DIR,
                     f'{stem}_{cond_label}_{args.side}{cond_tag}{lso_tag}'),
        subdirs=('figures',))

    with h5py.File(os.path.join(out_dir, 'ABR_full.h5'), 'w') as f:
        for label, V in V_gen.items():          # per-generator and composite
            f.create_dataset(label.replace(':', '__'), data=V)
        for nuc, V in V_nuc.items():            # per-nucleus
            f.create_dataset(f'nucleus__{nuc}', data=V)
        f.create_dataset('srate', data=srate)
        f.create_dataset('electrode_names', data=np.array(ELECTRODES, dtype='S'))
        f.attrs.update(units='µV', stem=stem, cond_label=cond_label,
                       side=args.side, band=f'{lo}-{hi} Hz')
    print(f'ABR_full.h5 saved to {out_dir}')

    _summary(V_nuc, V_gen, srate)
    _plot(out_dir, V_gen, V_nuc, srate, args.side, args.derivation)


if __name__ == '__main__':
    main()
