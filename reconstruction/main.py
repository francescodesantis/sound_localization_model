#!/usr/bin/env python3
"""
Entry point of the LFP and ABR reconstruction.

    python reconstruction/main.py nest  --pic-file RESULTS/<f>.pic [options]
    python reconstruction/main.py chain --pic-file RESULTS/<f>.pic [options]
    python reconstruction/main.py lfp <mso|gbc|sbc|lso|mntb|replot> [options]

nest / chain   complete ABR (ANF -> AVCN -> MNTB -> MSO/LSO), composite and
               figures. Same options for both; `main.py nest --help` lists them.
lfp            near-field LFP of one nucleus.

Run from the repository root so that RESULTS/... paths resolve. See
commands.txt for complete examples.
"""

import os
import runpy
import sys

PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PACKAGE_ROOT)

LFP_TARGETS = {
    'mso': 'LFP_reconstruction/main_reconstruct.py',
    'gbc': 'LFP_reconstruction/main_reconstruct_avcn.py',
    'sbc': 'LFP_reconstruction/main_reconstruct_sbc.py',
    'lso': 'LFP_reconstruction/main_reconstruct_lso.py',
    'mntb': 'LFP_reconstruction/main_reconstruct_mntb.py',
    'replot': 'LFP_reconstruction/replot.py',
}


def run_script(script, argv):
    """Execute a script exactly as `python <script>` would."""
    path = os.path.join(PACKAGE_ROOT, script)
    sys.path.insert(0, os.path.dirname(path))
    sys.argv = [path, *argv]
    runpy.run_path(path, run_name='__main__')


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] in ('-h', '--help'):
        print(__doc__.strip())
        return
    command, rest = argv[0], argv[1:]
    if command in ('nest', 'chain'):
        import pipeline
        pipeline.run(command, rest)
    elif command == 'lfp':
        if not rest or rest[0] not in LFP_TARGETS:
            sys.exit(f'usage: main.py lfp <{"|".join(LFP_TARGETS)}> [options]')
        run_script(LFP_TARGETS[rest[0]], rest[1:])
    else:
        sys.exit(f'error: unknown command {command!r}; choose nest, chain or lfp')


if __name__ == '__main__':
    main()
