#!/usr/bin/env python3
"""NEURON chain orchestrator: stage order, upstream freshness, per-stage dispatch.

    stage 1   GBC_L || GBC_R || SBC_L || SBC_R    ANF-driven, independent
    stage 2   MNTBC_L || MNTBC_R                  each needs the contralateral GBC
    stage 3   MSO || LSO                          need both SBC sides and the MNTB

Each job is its own `mpiexec -n RANKS` subprocess (runner.py); jobs of one
stage run in parallel as far as the cores allow, stages in order.

Freshness: every job is gated on the content hash of each upstream GDF it
reads, recorded in chain_manifest.json beside the dipole records, both when a
producer emits and when a consumer first reads a label. A label with no
recorded hash is trusted once and recorded.

  python reconstruction/NEURON_chain/main_chain.py --pic-file RESULTS/<f>.pic \
      --angle 0 [--condition binaural] [--stage 1|2|3] [--force] [--ranks 4]
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recon_core.utils import paths
from NEURON_chain import params as C
from NEURON_chain import spike_io

#: The DAG. Jobs within one stage are mutually independent and may run together.
STAGES = {
    1: [('GBC', 'L'), ('GBC', 'R'), ('SBC', 'L'), ('SBC', 'R')],
    2: [('MNTB', 'L'), ('MNTB', 'R')],
    3: [('MSO', 'both'), ('LSO', 'both')],
}

#: Which GDF labels each stage's job consumes. The MNTB reads the CONTRALATERAL
#: GBC, which is why both GBC sides must run even for a monaural stimulus.
def upstream_labels(nucleus, side):
    contra = {'L': 'R', 'R': 'L'}
    if nucleus == 'MNTB':
        return [f'GBC_{contra[side]}']
    if nucleus in ('MSO', 'LSO'):
        return ['SBC_L', 'SBC_R', 'MNTBC_L', 'MNTBC_R']
    return []                      # GBC and SBC read ANF, which the chain does
                                   # not produce


def upstream_fingerprint(spikes_dir, label):
    """Content hash of one upstream GDF plus its metadata entry (content, not
    mtime: a rewrite within the mtime granularity would pass an mtime check).
    """
    gdf = os.path.join(spikes_dir, f'spikes-{label}-0.gdf')
    if not os.path.exists(gdf):
        return None
    h = hashlib.sha256()
    with open(gdf, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    meta_path = os.path.join(spikes_dir, 'metadata.json')
    if os.path.exists(meta_path):
        import json
        with open(meta_path) as f:
            entry = json.load(f).get(label)
        h.update(repr(sorted((entry or {}).items())).encode())
    return h.hexdigest()


def is_fresh(spikes_dir, label, stored):
    """True when the upstream still hashes to what the downstream recorded."""
    current = upstream_fingerprint(spikes_dir, label)
    return current is not None and current == stored


def assert_fresh(spikes_dir, label, stored):
    """Halt rather than run on a stale or missing upstream."""
    current = upstream_fingerprint(spikes_dir, label)
    if current is None:
        raise SystemExit(
            f'missing upstream: spikes-{label}-0.gdf not in {spikes_dir}\n'
            f'  run the stage that produces {label} first')
    if current != stored:
        raise SystemExit(
            f'STALE upstream: {label} in {spikes_dir} has changed since the '
            f'downstream record was written.\n'
            f'  stored {stored[:16]}...  current {current[:16]}...\n'
            f'  re-run this stage, or pass --force to use it anyway')


#: nucleus -> params attribute holding its full-N total.
N_TOTAL_ATTR = {
    'GBC': 'N_GBC_TOTAL',
    'SBC': 'N_SBC_TOTAL',
    'MNTB': 'N_MNTB_TOTAL',
    'MSO': 'N_MSO_TOTAL',
    'LSO': 'N_LSO_TOTAL',
}


def _n_total(nucleus):
    """Full-N total of a nucleus, from params (runner.assert_full_n requires it)."""
    attr = N_TOTAL_ATTR.get(nucleus)
    if attr is None:
        raise ValueError(
            f'{nucleus}: no params attribute registered in N_TOTAL_ATTR -- '
            'add it there rather than typing a literal here')
    if not hasattr(C, attr):
        raise ValueError(
            f'{nucleus}: NEURON_chain.params has no {attr} (and neither does '
            'recon_core.config.params, which it falls through to) -- this '
            'nucleus\'s full-N total is not available anywhere in params')
    return getattr(C, attr)


def run_stage_subprocess(nucleus, side, args, condition=None):
    """Launch one job as its own `mpiexec -n ranks runner.py ...` and return the
    Popen (None under --dry-run). MSO/LSO jobs run with --side both.
    """
    runner = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'runner.py')
    cmd = ['mpiexec', '-n', str(args.ranks), sys.executable, runner,
           '--nucleus', nucleus, '--pic-file', args.pic_file,
           '--angle', str(args.angle),
           '--n-cells', str(_n_total(nucleus)),
           '--expect-ranks', str(args.ranks)]
    # The stimulus selector is forwarded too: runner.py resolves the same
    # condition_key, and --angle alone silently dropped --itd-us/--ild-db.
    if args.itd_us is not None:
        cmd += ['--itd-us', str(args.itd_us)]
    if args.ild_db is not None:
        cmd += ['--ild-db', str(args.ild_db)]
    if side:
        cmd += ['--side', side]
    if condition:                  # stage 3 only; stages 1-2 are condition-free
        cmd += ['--condition', condition]
    print('  $ ' + ' '.join(cmd), flush=True)
    if getattr(args, 'dry_run', False):
        return None
    # Popen: all jobs of a stage are launched first, then joined.
    return subprocess.Popen(cmd)


def max_concurrent_jobs(ranks):
    """How many `mpiexec -n ranks` jobs fit on this machine's cores (at least 1)."""
    return max(1, (os.cpu_count() or 1) // max(1, int(ranks)))


# ---------------------------------------------------------------------------
# Freshness bookkeeping
# ---------------------------------------------------------------------------
def _manifest_path(dipoles_dir):
    return os.path.join(dipoles_dir, 'chain_manifest.json')


def _load_manifest(dipoles_dir):
    path = _manifest_path(dipoles_dir)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _store_fingerprint(dipoles_dir, label, fingerprint):
    manifest = _load_manifest(dipoles_dir)
    manifest[label] = fingerprint
    os.makedirs(dipoles_dir, exist_ok=True)
    with open(_manifest_path(dipoles_dir), 'w') as f:
        json.dump(manifest, f, indent=2)


def _label_side(label):
    return label.rsplit('_', 1)[-1]


def _emit_label_for(nucleus, side):
    """The GDF label a nucleus/side job emits, per runner.NUCLEI -- or None
    for a nucleus that does not emit (terminal, or not yet registered)."""
    from NEURON_chain import runner
    spec = runner.NUCLEI.get(nucleus)
    if not spec or not spec.get('emits'):
        return None
    return spec['emit_label'].format(side=side)


def check_upstream(nucleus, side, stem, cond_val, dipoles_dir, force,
                   dry_run=False):
    """Gate one job on every upstream label it reads and record the fingerprints.

    A stale fingerprint raises (unless `force`); a label with no recorded
    fingerprint is trusted once and recorded. `dry_run` validates without
    recording.
    """
    manifest = _load_manifest(dipoles_dir)
    for label in upstream_labels(nucleus, side):
        input_side = _label_side(label)
        spikes_dir = spike_io.chain_spikes_dir(stem, cond_val, input_side)
        stored = manifest.get(label)
        current = upstream_fingerprint(spikes_dir, label)
        if stored is None:
            if current is None:
                raise SystemExit(
                    f'missing upstream: spikes-{label}-0.gdf not in '
                    f'{spikes_dir}\n  run the stage that produces {label} '
                    'first')
            print(f'  [chain] NO RECORDED FINGERPRINT yet for {label} -- '
                  'no staleness protection applies to it on this run; '
                  'trusting its current content'
                  + (' (dry-run: not recording it)' if dry_run else
                     ' and recording it now so later runs are held to it'))
            if not dry_run:
                _store_fingerprint(dipoles_dir, label, current)
            continue
        if force:
            if not is_fresh(spikes_dir, label, stored):
                print(f'  [chain] --force: ignoring stale fingerprint for '
                      f'{label} in {spikes_dir}')
            if current is not None and not dry_run:
                _store_fingerprint(dipoles_dir, label, current)
            continue
        assert_fresh(spikes_dir, label, stored)


def record_emitted(nucleus, side, stem, cond_val, dipoles_dir):
    """After a job that emits spikes, record the fingerprint of its own output."""
    label = _emit_label_for(nucleus, side)
    if label is None:
        return
    spikes_dir = spike_io.chain_spikes_dir(stem, cond_val, side)
    fingerprint = upstream_fingerprint(spikes_dir, label)
    if fingerprint is None:
        return
    _store_fingerprint(dipoles_dir, label, fingerprint)


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='NEURON chain orchestrator: walks the three-stage DAG in '
                    'order, gating every job on the freshness of every '
                    'upstream GDF it reads.')
    parser.add_argument('--pic-file', required=True, dest='pic_file')
    parser.add_argument('--angle', type=int, default=0)
    parser.add_argument('--itd-us', type=float, default=None, dest='itd_us')
    parser.add_argument('--ild-db', type=float, default=None, dest='ild_db')
    parser.add_argument('--stage', type=int, default=None, choices=sorted(STAGES),
                        help='run only this stage; default runs 1, 2, 3 in order')
    parser.add_argument('--force', action='store_true',
                        help='skip the freshness gate; prints exactly which '
                             'fingerprints it ignored, rather than silently '
                             'proceeding')
    parser.add_argument('--ranks', type=int, default=4,
                        help='MPI ranks per stage job (mpiexec -n RANKS)')
    parser.add_argument('--condition', nargs='+', default=['binaural'],
                        choices=['binaural', 'left_ear', 'right_ear'],
                        help='acoustic condition(s) for the stage-3 jobs (MSO, '
                             'LSO); one stage-3 pass per value. Stages 1-2 are '
                             'condition-independent and run once, tagged '
                             'binaural.')
    parser.add_argument('--dry-run', action='store_true', dest='dry_run',
                        help='print each mpiexec command (and run the '
                             'freshness gate) without launching anything')
    args = parser.parse_args()

    pic_file = paths.resolve_pic(args.pic_file)
    stem = paths.pic_stem(pic_file)
    cond_val, cond_label = paths.condition_key(args.angle, args.itd_us,
                                               args.ild_db)
    dipoles_dir = paths.dipoles_dir_for(stem, cond_label, variant=C.CHAIN_VARIANT)

    stage_nums = [args.stage] if args.stage is not None else sorted(STAGES)
    for stage_num in stage_nums:
        # Stages 1-2 are shared by every condition; stage 3 runs once per one.
        if stage_num == 3:
            jobs = [(n, sd, c) for c in args.condition for n, sd in STAGES[3]]
        else:
            jobs = [(n, sd, None) for n, sd in STAGES[stage_num]]
        print(f'\n[chain] stage {stage_num}: {jobs}')
        width = max_concurrent_jobs(args.ranks)
        print(f'[chain] launching up to {width} of {len(jobs)} job(s) at a '
              f'time ({os.cpu_count()} cores / {args.ranks} ranks per job)')
        for batch_start in range(0, len(jobs), width):
            batch = jobs[batch_start:batch_start + width]
            running = []
            # On any exit path, terminate and join the jobs already launched.
            try:
                for nucleus, side, condition in batch:
                    check_upstream(nucleus, side, stem, cond_val, dipoles_dir,
                                  args.force, dry_run=args.dry_run)
                    proc = run_stage_subprocess(nucleus, side, args, condition)
                    if proc is not None:
                        running.append(((nucleus, side, condition), proc))
                # Join the batch and collect every exit code.
                failed = [job for job, proc in running if proc.wait() != 0]
            finally:
                for _, proc in running:
                    if proc.poll() is None:
                        proc.terminate()
                for _, proc in running:
                    proc.wait()
            if failed:
                raise SystemExit(
                    '[chain] stage %s FAILED: %s' % (
                        stage_num,
                        ', '.join(f'{n}{"/" + s if s else ""}{"/" + c if c else ""}'
                                  for n, s, c in failed)))
        for nucleus, side, _ in jobs:
            # A dry run launches nothing, so nothing is recorded.
            if not args.dry_run:
                record_emitted(nucleus, side, stem, cond_val, dipoles_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
