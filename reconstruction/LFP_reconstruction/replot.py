#!/usr/bin/env python3
"""
Regenerate a finished run's LFP figures without re-simulating.

Reads PointSourcePotential_sum.h5 from an existing output directory and redraws
the compound-LFP and phase-cycle figures. Use it after changing how the figures
look, or to recover a plot from a run whose figures were lost.

Works for every nucleus: the nucleus, side and angle are read from the
directory name and the figure style comes from the same registry the pipelines
use, so this cannot drift from what a real run would draw.

Usage:
  python LFP_reconstruction/replot.py \
      --output-dir RESULTS/lfp_tmp/output_avcn_tone_1_2kHz_59dB_angle0_L
  # optional: --stim-freq 1200 --n-cells 3600 --skip-ms 10
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recon_core import params as P
from LFP_reconstruction import figures

# Directory prefix to (figure style, probe half-span). Longest prefix wins, so
# 'output_mntb_calyx_' is matched before 'output_mntb_'.
NUCLEI = {
    'output_avcn_': (figures.FigureStyle('AVCN (GBC)', 'avcn', trace_gain=60.0,
                                         blank_onset_ms=0.2),
                     P.AVCN_PROBE_HALF_SPAN),
    'output_sbc_': (figures.FigureStyle('AVCN (SBC)', 'sbc', trace_gain=60.0,
                                        blank_onset_ms=0.2),
                    P.AVCN_PROBE_HALF_SPAN),
    'output_lso_': (figures.FigureStyle('LSO', 'lso', trace_scale='per_channel',
                                        trace_gain=0.100,
                                        probe_axis=P.LSO_PROBE_AXIS),
                    P.LSO_PROBE_HALF_SPAN),
    'output_mntb_calyx_': (figures.FigureStyle('CALYX', 'calyx'),
                           P.MNTB_PROBE_HALF_SPAN),
    'output_mntb_': (figures.FigureStyle('MNTB', 'mntb'), P.MNTB_PROBE_HALF_SPAN),
    'output_': (figures.FigureStyle('MSO', 'mso'), P.MSO_PROBE_HALF_SPAN),
}


def _resolve(output_dir):
    """Read (style, probe (x,y,z), angle, side) off a directory name."""
    base = os.path.basename(os.path.normpath(output_dir))
    for prefix in sorted(NUCLEI, key=len, reverse=True):
        if base.startswith(prefix):
            style, half_span = NUCLEI[prefix]
            break
    else:
        sys.exit(f'error: {base!r} is not a recognised LFP output directory')

    match = re.search(r'_angle(-?\d+)_([LR])', base)
    angle, side = (int(match.group(1)), match.group(2)) if match else (0, 'L')
    return style, P.probe_positions(half_span, style.probe_axis), angle, side


def _detect_stim_freq(output_dir):
    """Stimulus frequency from the spike metadata that fed this run."""
    path = os.path.normpath(output_dir)
    base, parent = os.path.basename(path), os.path.dirname(path)
    spikes = re.sub(r'^output(_[a-z_]+?)?_', 'spikes_', base, count=1)
    meta = os.path.join(parent, spikes, 'metadata.json')
    if os.path.exists(meta):
        with open(meta) as f:
            return json.load(f).get('stim_freq_hz')
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output-dir', required=True, dest='output_dir',
                    help='RESULTS/lfp_tmp/output_* directory of a finished run')
    ap.add_argument('--stim-freq', type=float, default=None, dest='stim_freq',
                    help='stimulus frequency in Hz (default: read from the spike metadata)')
    ap.add_argument('--n-cells', type=int, default=None, dest='n_cells',
                    help='cell count, for the figure titles only')
    ap.add_argument('--skip-ms', type=float, default=10.0, dest='skip_ms',
                    help='onset ramp skipped before phase-averaging (default 10)')
    args = ap.parse_args()

    if not os.path.exists(os.path.join(args.output_dir, 'PointSourcePotential_sum.h5')):
        sys.exit(f'error: no PointSourcePotential_sum.h5 in {args.output_dir}, '
                 'is this a finished LFP output directory?')

    style, probe_xyz, angle, side = _resolve(args.output_dir)
    freq = args.stim_freq if args.stim_freq is not None else \
        _detect_stim_freq(args.output_dir)
    n_cells = args.n_cells if args.n_cells is not None else '?'
    os.makedirs(os.path.join(args.output_dir, 'figures'), exist_ok=True)

    depth = probe_xyz[style.depth_index]
    figures.plot_compound_lfp(args.output_dir, depth, side, angle, n_cells, style)
    figures.plot_phase_cycle(args.output_dir, freq, depth, side, angle, n_cells,
                             style, skip_ms=args.skip_ms)


if __name__ == '__main__':
    main()
