"""
Complete ABR run, from the auditory nerve to the superior olive.

Two pipelines share one interface and one output layout:

  nest   every nucleus is driven by the NEST spike trains in the .pic
  chain  GBC/SBC -> MNTB -> MSO/LSO are chained in NEURON; only the ANF input
         comes from the .pic

Each run simulates every nucleus at full N, superposes the dipole records at
the scalp (main_abr_full.py) and draws the figures:

  <out-dir>/ABR_full.h5
  <out-dir>/figures/full_abr_contributions.png   (+ _Cz-M2 when binaural)
  <out-dir>/figures/full_abr_electrodes.png
  <out-dir>/cz_m1_m2/cz_m1_m2_{anf,gbc,sbc,mntb,mso,lso}.png
"""

import argparse
import os
import subprocess
import sys

PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PACKAGE_ROOT)

from recon_core.config import brainstem                       # noqa: E402
from recon_core.utils import io_utils, paths                   # noqa: E402
from ABR_reconstruction.main_abr_full import (                 # noqa: E402
    MONAURAL_SIDES, anf_wave1_group)

PYTHON = sys.executable
_ENV_MPIEXEC = os.path.join(os.path.dirname(PYTHON), 'mpiexec')
MPIEXEC = _ENV_MPIEXEC if os.path.exists(_ENV_MPIEXEC) else 'mpiexec'
RESULTS_ROOT = os.path.join(PACKAGE_ROOT, 'abr_results')
ABR = os.path.join(PACKAGE_ROOT, 'ABR_reconstruction')

NUCLEI = ('anf', 'avcn', 'mntb', 'mso', 'lso')
NUCLEUS_UPPER = {'anf': 'ANF', 'avcn': 'AVCN', 'mntb': 'MNTB', 'mso': 'MSO', 'lso': 'LSO'}
SCRIPT = {'anf': 'main_abr_anf.py', 'avcn': 'main_abr_avcn.py',
          'mntb': 'main_abr_mntb.py', 'mso': 'main_abr.py', 'lso': 'main_abr_lso.py'}
# Generators each nucleus writes. AVCN and MNTB file theirs under 'binaural'
# (their response does not depend on the other ear); MSO and LSO under the condition.
GENERATORS = {'avcn': ('GBCsoma', 'GBCaxon', 'SBC'), 'mntb': ('principal', 'calyx'),
              'mso': ('postsynaptic',), 'lso': ('synaptic',)}


def build_parser(mode):
    """Command-line interface, identical for 'nest' and 'chain'."""
    ap = argparse.ArgumentParser(
        prog=f'main.py {mode}',
        description=f'Complete ABR, {mode} pipeline: every nucleus at full N, '
                    'composite and figures.')
    ap.add_argument('--pic-file', required=True, dest='pic_file',
                    help='NEST simulation result (.pic)')
    ap.add_argument('--out-dir', dest='out_dir', default=None,
                    help='output directory (default: reconstruction/abr_results/'
                         '<stem>_<stimulus>_<condition>[_<side>][_anf<mode>]_<pipeline>)')
    ap.add_argument('--condition', default='binaural',
                    choices=['binaural', 'left_ear', 'right_ear'],
                    help='which ear(s) receive the sound')
    ap.add_argument('--side', default='both', choices=['L', 'R', 'both'],
                    help='brain hemisphere(s) included in the composite and figures')
    sel = ap.add_mutually_exclusive_group()
    sel.add_argument('--angle', type=int, default=0,
                     help='stimulus key: HRTF azimuth in degrees (default 0)')
    sel.add_argument('--itd-us', dest='itd_us', type=float, default=None,
                     help='stimulus key: artificial ITD in µs (diotic click: 0)')
    sel.add_argument('--ild-db', dest='ild_db', type=float, default=None,
                     help='stimulus key: artificial ILD in dB')
    ap.add_argument('--ranks', type=int, default=4, help='MPI ranks per NEURON job')
    ap.add_argument('--anf-source', dest='anf_source', default='boundary',
                    choices=['full', 'boundary'],
                    help='auditory-nerve representation (default boundary)')
    ap.add_argument('--wave1-source', dest='wave1_source', default='modiolus',
                    choices=['aperture', 'modiolus'],
                    help='wave I source under --anf-source boundary (default modiolus)')
    ap.add_argument('--no-force', action='store_true', dest='no_force',
                    help='skip nuclei whose full-N records already exist')
    ap.add_argument('--dry-run', action='store_true', dest='dry_run',
                    help='print the commands without running them')
    return ap


def resolve_pic_path(pic):
    """Absolute .pic path, relative to the working directory or the repository root."""
    for cand in (pic, os.path.join(paths.REPO_ROOT, pic)):
        if os.path.exists(cand):
            return os.path.abspath(cand)
    sys.exit(f'error: .pic not found: {pic}')


def condition_key(args):
    """(.pic key, directory label) of the stimulus selector."""
    return paths.condition_key(args.angle, args.itd_us, args.ild_db)


def check_selector(pic_file, args):
    """Exit before any simulation if the .pic has no entry for the selector."""
    import dill
    sys.path.insert(0, brainstem.SIMULATE_DIR)
    from LFP_reconstruction.extract_spikes import _select_condition
    with open(pic_file, 'rb') as f:
        result = dill.load(f, ignore=True)
    rate_map = result.get('angle_to_rate') or result['cue_to_rate']
    key, label = condition_key(args)
    try:
        _select_condition(rate_map, key)
    except KeyError:
        sys.exit(f'error: {os.path.basename(pic_file)} has no stimulus {label} '
                 f'(key {key}); available keys: {sorted(rate_map)}. '
                 'Use --angle, --itd-us or --ild-db to match.')


def sides_for(nucleus, condition, side):
    """Hemispheres to simulate: ear-driven ones under a monaural condition, ∩ --side."""
    wanted = ('L', 'R') if side == 'both' else (side,)
    mono = MONAURAL_SIDES.get(condition)
    if mono is None or nucleus in ('mso', 'lso'):
        driven = ('L', 'R')
    elif nucleus in ('anf', 'avcn'):
        driven = (mono[0],)
    else:
        driven = (mono[1],)
    return tuple(s for s in driven if s in wanted)


def anf_generators(args):
    return (anf_wave1_group(args.anf_source, args.wave1_source), 'porus', 'brainstem')


def record_paths(nucleus, dipoles_dir, args, sides=None):
    """Dipole records `nucleus` must leave behind for this run."""
    gens = anf_generators(args) if nucleus == 'anf' else GENERATORS[nucleus]
    cond = args.condition if nucleus in ('mso', 'lso') else 'binaural'
    sides = sides_for(nucleus, args.condition, args.side) if sides is None else sides
    return [os.path.join(dipoles_dir, f'{NUCLEUS_UPPER[nucleus]}__{g}__{s}__{cond}.h5')
            for g in gens for s in sides]


def needs_run(record_list):
    """True unless every record exists, is at full N and uses NEST weights."""
    for path in record_list:
        if not os.path.exists(path):
            return True
        attrs, _, _ = io_utils.read_dipole_record(path)
        if attrs.get('syn_weights') != io_utils.SYN_WEIGHTS_TAG:
            return True
        if attrs.get('n_cells') != attrs.get('n_total'):
            return True
    return False


def default_out_dir(args, mode):
    stem = paths.pic_stem(args.pic_file)
    _, label = condition_key(args)
    side = '' if args.side == 'both' else f'_{args.side}'
    anf = ('' if (args.anf_source, args.wave1_source) == ('boundary', 'modiolus')
           else '_anffull' if args.anf_source == 'full' else f'_anf{args.wave1_source}')
    return os.path.join(RESULTS_ROOT,
                        f'{stem}_{label}_{args.condition}{side}{anf}_{mode}')


def _selector(args):
    if args.ild_db is not None:
        return ['--ild-db', str(args.ild_db)]
    if args.itd_us is not None:
        return ['--itd-us', str(args.itd_us)]
    return ['--angle', str(args.angle)]


def _side_arg(sides):
    return sides[0] if len(sides) == 1 else 'both'


def anf_command(args, variant=None):
    sides = sides_for('anf', args.condition, args.side)
    if not sides:
        return None
    cmd = [PYTHON, os.path.join(ABR, SCRIPT['anf']), '--pic-file', args.pic_file,
           *_selector(args), '--side', _side_arg(sides),
           '--anf-source', args.anf_source, '--wave1-source', args.wave1_source]
    return cmd + (['--variant', variant] if variant else [])


def nest_nucleus_command(nucleus, args):
    sides = sides_for(nucleus, args.condition, args.side)
    if not sides:
        return None
    cmd = [MPIEXEC, '-n', str(args.ranks), PYTHON, os.path.join(ABR, SCRIPT[nucleus]),
           '--pic-file', args.pic_file, *_selector(args), '--side', _side_arg(sides)]
    if nucleus == 'avcn':
        cmd.append('--full-n')
    if nucleus in ('mso', 'lso'):
        cmd += ['--condition', args.condition]
    return cmd


def chain_command(args):
    return [PYTHON, os.path.join(PACKAGE_ROOT, 'NEURON_chain', 'main_chain.py'),
            '--pic-file', args.pic_file, *_selector(args),
            '--ranks', str(args.ranks), '--condition', args.condition]


def composite_command(args, out_dir, variant=None):
    cmd = [PYTHON, os.path.join(ABR, 'main_abr_full.py'), '--pic-file', args.pic_file,
           *_selector(args), '--side', args.side, '--condition', args.condition,
           '--anf-source', args.anf_source, '--out-dir', out_dir]
    return cmd + (['--variant', variant] if variant else [])


def figure_commands(args, out_dir, variant=None):
    plots = os.path.join(ABR, 'plots')
    electrodes = [PYTHON, os.path.join(plots, 'electrodes.py'), '--pic-file', args.pic_file,
                  *_selector(args), '--condition', args.condition, '--side', args.side,
                  '--anf-source', args.anf_source, '--wave1-source', args.wave1_source,
                  '--out', os.path.join(out_dir, 'cz_m1_m2')]
    return [
        ('composite + contributions', [PYTHON, os.path.join(plots, 'full_abr.py'),
                                       '--dir', out_dir, '--layout', 'contributions',
                                       '--tmax', '20']),
        ('composite per electrode', [PYTHON, os.path.join(plots, 'full_abr.py'),
                                     '--dir', out_dir, '--layout', 'electrodes',
                                     '--tmax', '15']),
        ('per-nucleus Cz/M1/M2', electrodes + (['--variant', variant] if variant else [])),
    ]


def _execute(cmd, what, dry_run):
    print(f'\n--- {what}\n    {" ".join(cmd)}', flush=True)
    if dry_run:
        return
    env = dict(os.environ, PATH=os.path.dirname(PYTHON) + os.pathsep + os.environ['PATH'])
    if subprocess.run(cmd, env=env).returncode != 0:
        sys.exit(f'FAILED: {what}')


def _plan(mode, args, out_dir):
    """Ordered (what, cmd) list for one run."""
    stem = paths.pic_stem(args.pic_file)
    _, label = condition_key(args)
    variant = 'chain' if mode == 'chain' else None
    ddir = paths.dipoles_dir_for(stem, label, variant=variant)
    force = not args.no_force
    steps = []

    if force or needs_run(record_paths('anf', ddir, args)):
        cmd = anf_command(args, variant)
        if cmd:
            steps.append(('ANF', cmd))
    if mode == 'nest':
        for nucleus in NUCLEI[1:]:
            if force or needs_run(record_paths(nucleus, ddir, args)):
                cmd = nest_nucleus_command(nucleus, args)
                if cmd:
                    steps.append((NUCLEUS_UPPER[nucleus], cmd))
    else:
        chained = [p for n in NUCLEI[1:]
                   for p in record_paths(n, ddir, args, sides=('L', 'R'))]
        if force or needs_run(chained):
            steps.append(('NEURON chain (GBC/SBC -> MNTB -> MSO/LSO)', chain_command(args)))
    steps.append(('composite ABR', composite_command(args, out_dir, variant)))
    steps += figure_commands(args, out_dir, variant)
    return steps


def run(mode, argv=None):
    """Run the `mode` pipeline ('nest' or 'chain'); return the output directory."""
    args = build_parser(mode).parse_args(argv)
    args.pic_file = resolve_pic_path(args.pic_file)
    out_dir = os.path.abspath(args.out_dir or default_out_dir(args, mode))
    _, label = condition_key(args)

    print(f'pipeline   {mode}')
    print(f'stimulus   {paths.pic_stem(args.pic_file)}  ({label})')
    print(f'condition  {args.condition}   hemispheres {args.side}')
    print(f'output     {out_dir}')
    check_selector(args.pic_file, args)
    if not args.dry_run:
        os.makedirs(out_dir, exist_ok=True)
    for what, cmd in _plan(mode, args, out_dir):
        _execute(cmd, what, args.dry_run)
    if not args.dry_run:
        print(f'\n=== done: {out_dir}')
    return out_dir
