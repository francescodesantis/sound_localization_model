#!/usr/bin/env python3
"""
One entry point for the whole reconstruction.

The pipeline has three stages and this dispatches the last two:

    NEST spiking simulation      ../simulate/main.py       (run separately)
      near-field LFP             main.py lfp <nucleus>
      far-field scalp ABR        main.py abr <nucleus|full|bi>

Run it from the repository root (the folder holding reconstruction/ and
simulate/) so that RESULTS/... arguments resolve.

abr full superposes every nucleus's stored dipole into one composite ABR, so
the per-nucleus abr runs have to happen first. See commands.txt for the recipe.

This is only a dispatcher: each target is the same module you can run directly
(python ABR_reconstruction/main_abr.py ...) and every unrecognised flag is
passed through. That also keeps it MPI transparent, since all ranks reach the
same module.

    python reconstruction/main.py lfp mso --pic-file RESULTS/<f>.pic --side L
    python reconstruction/main.py abr full --pic-file RESULTS/<f>.pic --side both
    python reconstruction/main.py plot positions --view coronal
    python reconstruction/main.py validate params
    python reconstruction/main.py list
"""

import argparse
import os
import runpy
import sys

# This file sits at the top of reconstruction/, so its own directory is the
# package root every target path is relative to.
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PACKAGE_ROOT)

# target to module path, relative to reconstruction/
LFP_TARGETS = {
    'mso': 'LFP_reconstruction/main_reconstruct.py',
    'gbc': 'LFP_reconstruction/main_reconstruct_avcn.py',
    'sbc': 'LFP_reconstruction/main_reconstruct_sbc.py',
    'lso': 'LFP_reconstruction/main_reconstruct_lso.py',
    'mntb': 'LFP_reconstruction/main_reconstruct_mntb.py',
    'replot': 'LFP_reconstruction/replot.py',
    'tuning': 'LFP_reconstruction/tuning_curve.py',
}

ABR_TARGETS = {
    'mso': 'ABR_reconstruction/main_abr.py',
    'avcn': 'ABR_reconstruction/main_abr_avcn.py',
    'lso': 'ABR_reconstruction/main_abr_lso.py',
    'mntb': 'ABR_reconstruction/main_abr_mntb.py',
    'full': 'ABR_reconstruction/main_abr_full.py',
    'bi': 'ABR_reconstruction/main_abr_bi.py',
}

PLOT_TARGETS = {
    'positions': 'ABR_reconstruction/plots/positions.py',
    'morphology': 'ABR_reconstruction/plots/morphology.py',
    'electrodes': 'ABR_reconstruction/plots/electrodes.py',
    'full-abr': 'ABR_reconstruction/plots/full_abr.py',
    'itd-sweep': 'ABR_reconstruction/plots/itd_sweep.py',
    'tolnai': 'ABR_reconstruction/plots/tolnai.py',
    'binaural': 'ABR_reconstruction/plots/binaural.py',
}

VALIDATE_TARGETS = {
    'params': 'tests/test_params.py',
    'head-geometry': 'ABR_reconstruction/test_head_geometry.py',
    'regression': 'tests/regression.py',
    'mntb': 'models/mntb/validate_mntb.py',
    'calyx': 'models/mntb/validate_calyx.py',
    'gbc': 'models/avcn/validate_gbc.py',
    'gbc-axon': 'models/avcn/validate_gbc_axon.py',
    'lso-axon': 'models/mso/validate_lso_axon.py',
}

STAGES = {'lfp': LFP_TARGETS, 'abr': ABR_TARGETS, 'plot': PLOT_TARGETS,
          'validate': VALIDATE_TARGETS}

DESCRIPTIONS = {
    'lfp': 'near-field local field potential, one nucleus at a time',
    'abr': 'far-field scalp ABR; "full" superposes every nucleus',
    'plot': 'figures from results that already exist',
    'validate': 'single-cell checks and the regression harness',
}


def run(script, argv):
    """Execute a target exactly as python <script> would.

    That means putting the script's own directory first on sys.path, as the
    interpreter does for a directly-run file. Several targets rely on it to
    reach sibling modules (models/avcn/validate_gbc.py imports gbc_biophysics
    that way).
    """
    path = os.path.join(PACKAGE_ROOT, script)
    if not os.path.exists(path):
        sys.exit(f'error: {script} not found')
    sys.path.insert(0, os.path.dirname(path))
    sys.argv = [path, *argv]
    runpy.run_path(path, run_name='__main__')


def print_targets():
    print(__doc__.strip().splitlines()[0])
    for stage, targets in STAGES.items():
        print(f'\n{stage}  {DESCRIPTIONS[stage]}')
        width = max(len(name) for name in targets)
        for name, script in targets.items():
            print(f'    {name:<{width}}  {script}')
    print('\nSee commands.txt for complete, runnable workflows.')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False)
    ap.add_argument('stage', nargs='?', choices=list(STAGES) + ['list'],
                    help='which stage to run')
    ap.add_argument('target', nargs='?', help='what to run within that stage')
    ap.add_argument('-h', '--help', action='store_true', dest='want_help')
    args, passthrough = ap.parse_known_args()

    if args.stage is None or args.stage == 'list' or (args.want_help and not args.stage):
        print_targets()
        return

    targets = STAGES[args.stage]
    if args.target not in targets:
        known = ', '.join(targets)
        sys.exit(f'error: unknown {args.stage} target {args.target!r}\n'
                 f'       choose one of: {known}')

    if args.want_help:
        passthrough.append('--help')
    run(targets[args.target], passthrough)


if __name__ == '__main__':
    main()
