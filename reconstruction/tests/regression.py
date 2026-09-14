#!/usr/bin/env python3
"""
Golden-output regression harness for the LFP and ABR reconstruction pipelines.

Runs every entry point at a small cell count against a dedicated stimulus stem,
then fingerprints every HDF5 array produced. Refactors are proven behaviour
preserving by diffing two fingerprints:

    python tests/regression.py --label before
    ...refactor...
    python tests/regression.py --label after
    python tests/regression.py --compare before after

The stimulus is a symlink (RESULTS/_regr_tone12.pic) rather than the real .pic,
so every output directory carries the _regr_tone12 stem and cannot collide with
a production run. The symlink is created on first use.

Runs are serial by design: under MPI the cells are distributed across ranks, so
a fingerprint taken with -n 4 is not comparable to one taken with -n 1.

Only .h5 payloads are fingerprinted. Figures are skipped because single-cell
panels pick their cells with an unseeded random.sample, so the PNGs are
legitimately non-deterministic while the population HDF5 output is not.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import h5py
import numpy as np

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACKAGE_ROOT)

from recon_core import paths                                    # noqa: E402

RESULTS = paths.RESULTS_DIR
REGR_DIR = os.path.join(RESULTS, 'regression')

SOURCE_PIC = 'tone_1.2kHz_59dB.pic'
REGR_PIC = os.path.join(RESULTS, '_regr_tone12.pic')
STEM = '_regr_tone12'          # what _pic_stem() makes of the symlink's basename

N_LFP = '20'                   # MSO/LSO cell count for the small matrix
N_SMALL = '8'                  # bushy/MNTB cells (full EM morphologies are slower)


def _lfp(script, *extra):
    return [os.path.join('LFP_reconstruction', script),
            '--pic-file', REGR_PIC, '--angle', '0', '--side', 'L', *extra]


def _abr(script, *extra):
    return [os.path.join('ABR_reconstruction', script),
            '--pic-file', REGR_PIC, '--angle', '0', '--side', 'both', *extra]


# Ordered: the LFP runs populate the spike cache, the per-nucleus ABR runs write
# the dipole records that main_abr_full / main_abr_bi then consume.
RUNS = [
    ('lfp_mso',        _lfp('main_reconstruct.py',      '--n-cells', N_LFP, '--n-single', '2')),
    ('lfp_gbc',        _lfp('main_reconstruct_avcn.py', '--n-cells', N_SMALL, '--n-single', '2')),
    ('lfp_sbc',        _lfp('main_reconstruct_sbc.py',  '--n-cells', N_SMALL, '--n-single', '2')),
    ('lfp_lso_syn',    _lfp('main_reconstruct_lso.py',  '--n-cells', N_SMALL, '--n-single', '2')),
    ('lfp_lso_spike',  _lfp('main_reconstruct_lso.py',  '--n-cells', N_SMALL, '--n-single', '2',
                            '--generators', 'spiking')),
    ('lfp_mntb',       _lfp('main_reconstruct_mntb.py', '--n-cells', N_SMALL, '--n-single', '2')),
    ('lfp_mntb_calyx', _lfp('main_reconstruct_mntb.py', '--n-cells', N_SMALL, '--n-single', '2',
                            '--with-calyx')),

    ('abr_mso',        _abr('main_abr.py',      '--n-cells', N_LFP)),
    ('abr_avcn',       _abr('main_abr_avcn.py', '--n-cells', N_SMALL, '--generators', 'both')),
    ('abr_lso_spike',  _abr('main_abr_lso.py',  '--n-cells', N_SMALL, '--generators', 'spiking')),
    ('abr_lso_syn',    _abr('main_abr_lso.py',  '--n-cells', N_SMALL, '--generators', 'synaptic')),
    ('abr_mntb',       _abr('main_abr_mntb.py', '--n-cells', N_SMALL, '--generators', 'both')),

    # Monaural records, prerequisites for the binaural-interaction reproduction.
    ('abr_mso_left',   _abr('main_abr.py', '--n-cells', N_LFP, '--condition', 'left_ear')),
    ('abr_mso_right',  _abr('main_abr.py', '--n-cells', N_LFP, '--condition', 'right_ear')),
    ('abr_lso_left',   _abr('main_abr_lso.py', '--n-cells', N_SMALL, '--generators', 'synaptic',
                            '--condition', 'left_ear')),
    ('abr_lso_right',  _abr('main_abr_lso.py', '--n-cells', N_SMALL, '--generators', 'synaptic',
                            '--condition', 'right_ear')),

    ('abr_full',       _abr('main_abr_full.py')),
    ('abr_bi',         [os.path.join('ABR_reconstruction', 'main_abr_bi.py'),
                        '--pic-file', REGR_PIC, '--angle', '0']),
]


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------
def _fingerprint_array(arr):
    """Bit-exact hash plus human-readable stats for one dataset."""
    a = np.ascontiguousarray(arr)
    entry = {'shape': list(a.shape), 'dtype': str(a.dtype),
             'sha256': hashlib.sha256(a.tobytes()).hexdigest()[:32]}
    if a.dtype.kind == 'f' and a.size:
        finite = a[np.isfinite(a)]
        if finite.size:
            entry['stats'] = {
                'min': float(finite.min()), 'max': float(finite.max()),
                'mean': float(finite.mean()), 'l2': float(np.linalg.norm(finite)),
                'argmax_abs': int(np.argmax(np.abs(a))),
            }
    return entry


def _fingerprint_h5(path):
    out = {}

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            out[name] = _fingerprint_array(obj[()])

    with h5py.File(path, 'r') as f:
        f.visititems(visit)
        # visititems does not reach datasets sitting at the root of some files
        for key, obj in f.items():
            if isinstance(obj, h5py.Dataset) and key not in out:
                out[key] = _fingerprint_array(obj[()])
    return out


def collect_fingerprints():
    """Every .h5 under RESULTS/{lfp_tmp,abr_tmp} belonging to the regression stem."""
    prints = {}
    for sub in ('lfp_tmp', 'abr_tmp'):
        root = os.path.join(RESULTS, sub)
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in sorted(filenames):
                if not fn.endswith('.h5'):
                    continue
                path = os.path.join(dirpath, fn)
                rel = os.path.relpath(path, RESULTS)
                if STEM not in rel:
                    continue
                try:
                    prints[rel] = _fingerprint_h5(path)
                except OSError as exc:
                    prints[rel] = {'__error__': str(exc)}
    return prints


# ---------------------------------------------------------------------------
# Driving the pipelines
# ---------------------------------------------------------------------------
def ensure_symlink():
    src = os.path.join(RESULTS, SOURCE_PIC)
    if not os.path.exists(src):
        sys.exit(f'source stimulus missing: {src}')
    if not os.path.islink(REGR_PIC):
        os.symlink(SOURCE_PIC, REGR_PIC)


def run_all(only=None, verbose=False):
    """Execute the matrix; returns {run_name: {'rc', 'seconds'}}."""
    report = {}
    for name, argv in RUNS:
        if only and name not in only:
            continue
        cmd = [sys.executable, *argv]
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=PACKAGE_ROOT, capture_output=not verbose, text=True)
        dt = time.time() - t0
        report[name] = {'rc': proc.returncode, 'seconds': round(dt, 1)}
        status = 'ok ' if proc.returncode == 0 else 'FAIL'
        print(f'  [{status}] {name:<16s} {dt:6.1f}s')
        if proc.returncode != 0 and not verbose:
            tail = (proc.stderr or proc.stdout or '').strip().splitlines()[-15:]
            print('\n'.join(f'         {line}' for line in tail))
    return report


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def compare(before, after, rtol=0.0):
    """Print a diff of two fingerprint files. Returns True when they match."""
    a, b = before['fingerprints'], after['fingerprints']
    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    changed = []

    for rel in sorted(set(a) & set(b)):
        for key in sorted(set(a[rel]) | set(b[rel])):
            fa, fb = a[rel].get(key), b[rel].get(key)
            if fa is None or fb is None:
                changed.append((rel, key, 'dataset added/removed'))
            elif fa['sha256'] != fb['sha256']:
                sa, sb = fa.get('stats'), fb.get('stats')
                if sa and sb and rtol and np.isclose(sa['l2'], sb['l2'], rtol=rtol):
                    continue
                detail = (f"l2 {sa['l2']:.6g} -> {sb['l2']:.6g}" if sa and sb
                          else 'bytes differ')
                changed.append((rel, key, detail))

    if only_a:
        print(f'\nonly in {before["label"]} ({len(only_a)}):')
        for rel in only_a:
            print(f'  - {rel}')
    if only_b:
        print(f'\nonly in {after["label"]} ({len(only_b)}):')
        for rel in only_b:
            print(f'  + {rel}')
    if changed:
        print(f'\nchanged arrays ({len(changed)}):')
        for rel, key, detail in changed:
            print(f'  ~ {rel}::{key}  {detail}')

    identical = not (only_a or only_b or changed)
    n_arrays = sum(len(v) for v in a.values())
    print(f'\n{"IDENTICAL" if identical else "DIFFERENT"}: '
          f'{len(a)} files / {n_arrays} arrays in {before["label"]}, '
          f'{len(b)} files in {after["label"]}')
    return identical


def _load(label):
    path = os.path.join(REGR_DIR, f'{label}.json')
    if not os.path.exists(path):
        sys.exit(f'no such fingerprint: {path}')
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--label', help='run the matrix and save the fingerprint under this name')
    ap.add_argument('--compare', nargs=2, metavar=('BEFORE', 'AFTER'),
                    help='diff two saved fingerprints')
    ap.add_argument('--only', nargs='+', help='restrict to these run names')
    ap.add_argument('--fingerprint-only', action='store_true',
                    help='skip the runs, just fingerprint what is already on disk')
    ap.add_argument('--rtol', type=float, default=0.0,
                    help='treat arrays whose L2 norms agree within this relative '
                         'tolerance as unchanged (default 0 = bit-exact)')
    ap.add_argument('--list', action='store_true', help='list run names and exit')
    ap.add_argument('-v', '--verbose', action='store_true', help='stream subprocess output')
    args = ap.parse_args()

    if args.list:
        for name, argv in RUNS:
            print(f'  {name:<16s} {" ".join(argv)}')
        return

    if args.compare:
        sys.exit(0 if compare(_load(args.compare[0]), _load(args.compare[1]),
                              rtol=args.rtol) else 1)

    if not args.label:
        ap.error('need --label, --compare or --list')

    ensure_symlink()
    os.makedirs(REGR_DIR, exist_ok=True)

    runs = {}
    if not args.fingerprint_only:
        print(f'running matrix ({len(args.only or RUNS)} entry points, serial)...')
        runs = run_all(only=args.only, verbose=args.verbose)

    print('fingerprinting...')
    payload = {'label': args.label, 'stem': STEM, 'runs': runs,
               'fingerprints': collect_fingerprints()}
    out = os.path.join(REGR_DIR, f'{args.label}.json')
    with open(out, 'w') as f:
        json.dump(payload, f, indent=1, sort_keys=True)

    n_arrays = sum(len(v) for v in payload['fingerprints'].values())
    failed = [k for k, v in runs.items() if v['rc'] != 0]
    print(f'\n{len(payload["fingerprints"])} files / {n_arrays} arrays -> {out}')
    if failed:
        print(f'FAILED runs: {", ".join(failed)}')
        sys.exit(1)


if __name__ == '__main__':
    main()
