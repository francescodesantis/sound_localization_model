"""Complete brainstem ABR: superpose every nucleus into one composite.

Each per-nucleus script writes head-frame dipole records to
RESULTS/abr_tmp/dipoles/<stem>_<stimulus>[_chain]/<nucleus>__<generator>__<side>__<condition>.h5.
This script collects them for one stimulus, projects each generator through
the 4-sphere model from its own position and sums the scalp potentials. No
NEURON run is repeated. All records share DT and TSTOP; they are summed
unscaled, so every nucleus must be simulated at full N.

Generators, each modelled once:
  ANF        wave I source + porus (I_n) + brainstem entry (II_p)
  AVCN       GBCsoma, GBCaxon (the crossing axon), SBC
  MNTB       principal, calyx
  MSO        postsynaptic
  LSO        synaptic

  python ABR_reconstruction/main_abr_full.py --pic-file RESULTS/<f>.pic \
      --angle 0 --side both [--condition ...] [--variant chain] [--out-dir DIR]

Output: <out-dir>/ABR_full.h5 (per generator, per nucleus and composite, µV).
"""

import os
import sys
import glob
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import h5py

from recon_core.physics import head_model
from recon_core.utils import io_utils, paths
from recon_core.config import params as P
from recon_core.config.head_geometry import ABR_ELECTRODES
from recon_core.utils.signal_utils import onset_latency, time_axis

# No NEURON here: this orchestrator only reads dipole records the producers wrote.
ELECTRODES = list(ABR_ELECTRODES)
REPO_ROOT = paths.REPO_ROOT

_pic_stem = paths.pic_stem
dipoles_dir_for = paths.dipoles_dir_for

#: Monaural condition -> (side of the ear-driven AVCN and ANF, side of the
#: ear-driven MNTB); the GBC -> calyx projection decussates.
MONAURAL_SIDES = {'right_ear': ('R', 'L'), 'left_ear': ('L', 'R')}


#: Generators each nucleus writes (ANF: see P.ANF_WAVE1_GROUPS and
#: P.ANF_STATIONARY_GROUPS). Records of any other generator are skipped.
KNOWN_GENERATORS = {
    'AVCN': P.GBC_GROUPED_GROUPS + ('SBC',),
    'MNTB': ('principal', 'calyx'),
    'MSO': ('postsynaptic',),
    'LSO': ('synaptic',),
}


def anf_wave1_group(anf_source, wave1_source):
    """Record name of the wave I generator main_abr_anf.py writes in this mode."""
    if anf_source == 'full':
        return P.ANF_DISTRIBUTED_GROUP
    return P.ANF_MODIOLUS_GROUP if wave1_source == 'modiolus' else P.ANF_DISTAL_GROUP


# ---------------------------------------------------------------------------
def _discover(stem, cond_label, want_nuclei, want_sides, want_condition,
              anf_source='boundary', variant=None):
    """Load matching dipole records as (records, srate).

    Each record is (label, nucleus, side, p_head, r_dipole). Per (nucleus,
    generator, side) the record for want_condition is used if present, else the
    binaural one (AVCN and MNTB write only binaural). Under a monaural condition
    AVCN and ANF keep the ear-driven side and MNTB the opposite one (the GBC ->
    calyx projection decussates); MSO and LSO keep both. `variant` selects the
    record set (None: NEST-driven, 'chain': NEURON chain).
    """
    mono = MONAURAL_SIDES.get(want_condition)

    ddir = dipoles_dir_for(stem, cond_label, variant=variant)
    record_paths = sorted(glob.glob(os.path.join(ddir, '*.h5')))
    if not record_paths:
        raise FileNotFoundError(
            f'no dipole records in {ddir}\n  Run each nucleus ABR first, with '
            f'the same stimulus selector, e.g.\n'
            f'    python ABR_reconstruction/main_abr_avcn.py --pic-file ... '
            f'--side both')

    gbc_seen, avcn_seen = False, False

    cand = {}    # (nucleus, generator, side) to {condition: record}
    srates = set()
    anf_wave1_seen = set()      # wave I representations actually COLLECTED
    anf_wave1_on_disk = {}      # every one present, to name in an error
    anf_any_on_disk = False
    for path in record_paths:
        attrs, p_head, r_dipole = io_utils.read_dipole_record(path)
        nucleus, generator = attrs['nucleus'], attrs['generator']
        side, cond = attrs['side'], attrs['condition']
        name = os.path.basename(path)
        if nucleus != 'ANF' and generator not in KNOWN_GENERATORS.get(nucleus, ()):
            print(f'  [discover] skipping {name}: unknown generator {nucleus}:{generator}')
            continue
        if attrs.get('syn_weights') != io_utils.SYN_WEIGHTS_TAG:
            print(f'  [discover] skipping {name}: written with other synaptic weights; re-run it')
            continue
        if attrs.get('n_cells') != attrs.get('n_total'):
            print(f'  [discover] warning: {name} is at {attrs.get("n_cells")} of '
                  f'{attrs.get("n_total")} cells; amplitude ratios are not physical')
        if want_nuclei and nucleus not in want_nuclei:
            continue
        if side not in want_sides:
            continue
        if mono is not None:
            # Monaural: AVCN and ANF keep the ear-driven side, MNTB the opposite
            # one (the GBC -> calyx projection decussates).
            if nucleus in ('AVCN', 'ANF') and side != mono[0]:
                continue
            if nucleus == 'MNTB' and side != mono[1]:
                continue
        if nucleus == 'ANF':
            anf_any_on_disk = True
            if generator not in P.ANF_STATIONARY_GROUPS:
                anf_wave1_on_disk.setdefault(
                    generator, P.ANF_WAVE1_GROUPS.get(generator))
        if nucleus == 'AVCN':
            avcn_seen = True
        if nucleus == 'AVCN' and generator in P.GBC_GROUPED_GROUPS:
            gbc_seen = True
        # Wave I representations are alternatives: keep only the one selected by
        # anf_source. Every ANF generator must be declared in P.ANF_WAVE1_GROUPS
        # or P.ANF_STATIONARY_GROUPS.
        if nucleus == 'ANF' and generator not in P.ANF_STATIONARY_GROUPS:
            mode = P.ANF_WAVE1_GROUPS.get(generator)
            if mode is None:
                raise ValueError(
                    f'{os.path.basename(path)}: unknown ANF generator '
                    f'{generator!r}. Every ANF generator is either a '
                    f'stationary junction (P.ANF_STATIONARY_GROUPS = '
                    f'{list(P.ANF_STATIONARY_GROUPS)}) or a wave I '
                    f'representation, which must be declared in '
                    f'P.ANF_WAVE1_GROUPS = {P.ANF_WAVE1_GROUPS} together with '
                    f'the --anf-source mode that selects it. Refusing to '
                    f'guess: an undeclared generator would be summed on top '
                    f'of the selected wave I source.')
            if mode != anf_source:
                continue
            anf_wave1_seen.add(generator)
        srates.add(attrs['srate'])
        # Each bin of an extended record is a source at its own position, under one label.
        cand.setdefault((nucleus, generator, side), {})[cond] = [
            (f'{nucleus}:{generator}', nucleus, side, p_head[k], r_dipole[k])
            for k in range(p_head.shape[0])]

    # Exactly one wave I representation must have been collected.
    if anf_any_on_disk:
        if len(anf_wave1_seen) > 1:
            raise ValueError(
                f'DOUBLE-COUNTED WAVE I: --anf-source {anf_source!r} collected '
                f'{sorted(anf_wave1_seen)} from {ddir}. These are ALTERNATIVE '
                f'representations of the nerve and must never be summed.\n'
                f'  main_abr_anf.py writes exactly one per run; more than one '
                f'on disk means a stale leftover from an earlier mode.\n'
                f'  Delete the ones you are not using (they are regenerable) '
                f'and re-assemble.')
        if not anf_wave1_seen:
            raise ValueError(
                f'NO WAVE I SOURCE: --anf-source {anf_source!r} selected none '
                f'of the ANF wave I representations in {ddir}.\n'
                f'  on disk: {anf_wave1_on_disk or "none"}  (generator -> the '
                f'--anf-source mode that selects it)\n'
                f'  The stationary junctions (I_n, II_p) would still be '
                f'summed, so the composite would silently LOSE wave I while '
                f'keeping the deflections around it. Pass the matching '
                f'--anf-source, or re-run main_abr_anf.py in the mode you '
                f'want.')

    if avcn_seen and not gbc_seen:
        raise ValueError(f'NO GBC: {ddir} has AVCN records but no GBCsoma/GBCaxon; '
                         're-run the AVCN (main_abr_avcn.py).')

    recs = []
    for by_cond in cand.values():
        if want_condition in by_cond:
            recs.extend(by_cond[want_condition])
        elif 'binaural' in by_cond:
            recs.extend(by_cond['binaural'])
    if not recs:
        raise FileNotFoundError(
            f'records exist in {ddir} but none match nuclei={want_nuclei} '
            f'sides={want_sides} condition={want_condition}')
    if len(srates) > 1:
        raise ValueError(f'dipole records disagree on the sample rate: {srates}. '
                         'Re-run the producers so every nucleus shares one DT.')
    return recs, srates.pop()


def assemble(stem, cond_label, want_nuclei, sides, condition,
             band=P.BAND_DEFAULT,
             anf_source='boundary', variant=None):
    """Collect the dipole records of one stimulus and superpose them.

    Returns (V_gen, V_nuc, srate): V_gen maps 'nucleus:generator' and
    'composite' to band-passed (n_electrodes, T) µV; V_nuc maps nucleus to its
    summed µV.
    """
    lo, hi = band
    recs, srate = _discover(stem, cond_label, want_nuclei, set(sides), condition,
                            anf_source=anf_source,
                            variant=variant)
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
    ap.add_argument('--condition', type=str, default='binaural',
                    choices=['binaural', 'left_ear', 'right_ear'],
                    help='acoustic condition: picks condition-specific MSO/LSO '
                         'records, falling back to binaural for AVCN/MNTB')
    ap.add_argument('--anf-source', type=str, default='boundary',
                    choices=['full', 'boundary'], dest='anf_source',
                    help='ANF representation to assemble; must match the one '
                         'main_abr_anf.py wrote (default boundary)')
    ap.add_argument('--variant', default=None,
                    help='dipole-record variant to assemble; "chain" reads the '
                         'NEURON_chain records instead of the NEST-driven ones. '
                         'The two never share a directory, so no exclusivity '
                         'rule is needed between them.')
    ap.add_argument('--out-dir', type=str, default=None, dest='out_dir',
                    help='write here instead of the derived '
                         'RESULTS/full_abr/<stem>_<cond>_<side>[...] name. A '
                         'bare name is taken relative to RESULTS/full_abr/.')
    args = ap.parse_args()

    pic_file = args.pic_file or os.path.join(REPO_ROOT, 'RESULTS',
                                             'baseline_simulation.pic')
    stem  = _pic_stem(pic_file)
    _, cond_label = paths.condition_key(args.angle, args.itd_us, args.ild_db)
    sides = ['L', 'R'] if args.side == 'both' else [args.side]
    lo, hi = P.BAND_DEFAULT

    V_gen, V_nuc, srate = assemble(stem, cond_label, None, sides,
                                   args.condition,
                                   anf_source=args.anf_source, variant=args.variant)
    print(f'collected generators ({args.condition}): '
          f'{sorted(k for k in V_gen if k != "composite")}')

    cond_tag = '' if args.condition == 'binaural' else f'_{args.condition}'
    # Directory tags: ANF representation and record set.
    anf_tag  = '' if args.anf_source == 'boundary' else f'_anf{args.anf_source}'
    variant_tag = '' if not args.variant else f'_{args.variant}'
    default_name = (f'{stem}_{cond_label}_{args.side}{cond_tag}'
                    f'{anf_tag}{variant_tag}')
    if args.out_dir:
        # absolute/relative path as given; a bare name lands in RESULTS/full_abr/
        target = (args.out_dir if os.path.sep in args.out_dir
                  else os.path.join(paths.FULL_ABR_DIR, args.out_dir))
    else:
        target = os.path.join(paths.FULL_ABR_DIR, default_name)
    out_dir = paths.make_output_dirs(target, subdirs=('figures',))

    with h5py.File(os.path.join(out_dir, 'ABR_full.h5'), 'w') as f:
        for label, V in V_gen.items():          # per-generator and composite
            f.create_dataset(label.replace(':', '__'), data=V)
        for nuc, V in V_nuc.items():            # per-nucleus
            f.create_dataset(f'nucleus__{nuc}', data=V)
        f.create_dataset('srate', data=srate)
        f.create_dataset('electrode_names', data=np.array(ELECTRODES, dtype='S'))
        # `condition` is recorded so a downstream figure can pick the
        # ipsilateral mastoid without parsing the directory name.
        f.attrs.update(units='µV', stem=stem, cond_label=cond_label,
                       side=args.side, condition=args.condition,
                       band=f'{lo}-{hi} Hz')
    print(f'ABR_full.h5 saved to {out_dir}')

    _summary(V_nuc, V_gen, srate)


if __name__ == '__main__':
    main()
