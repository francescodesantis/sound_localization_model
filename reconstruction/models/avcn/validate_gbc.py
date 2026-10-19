#!/usr/bin/env python3
"""
Validation: load a GBC morphology, decorate it with cnmodel XM13_nacncoop
channels, and run a somatic current-clamp step.

A correct Type II bushy cell should show:
  strong sub-threshold rectification (low-threshold K, KLT),
  at most a single onset action potential to a supra-threshold step,
  a depolarising sag on hyperpolarising steps (Ih).

Usage:
  python models/avcn/validate_gbc.py \
      [--morphology models/avcn/morphology/bushy_stick.hoc]

Saves models/avcn/figures/gbc_iclamp_validation.png
"""
import os
import sys
import argparse

import numpy as np
import neuron
from neuron import h

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from recon_core import params as _P                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--morphology', default=os.path.join(HERE, 'morphology', 'bushy_stick.hoc'))
    ap.add_argument('--ref-ns', choices=['II', 'II-I'], default='II', dest='ref_ns',
                    help='channel density set: II = GBC (default), II-I = SBC')
    ap.add_argument('--amps', type=float, nargs='+',
                    default=[-0.2, -0.1, 0.1, 0.2, 0.5, 1.0],  # nA
                    help='current step amplitudes (nA)')
    args = ap.parse_args()

    try:
        neuron.load_mechanisms(HERE)      # models/avcn/x86_64 (may be auto-loaded from CWD)
    except RuntimeError as e:
        if 'already exists' not in str(e):
            raise
    h.load_file('stdrun.hoc')
    h.load_file(args.morphology)

    import gbc_biophysics as gb
    ref_ns = gb.REF_NS_II_I if args.ref_ns == 'II-I' else gb.REF_NS_II
    print(f'  [validate] decoration = Type {args.ref_ns} '
          f'({"SBC" if args.ref_ns == "II-I" else "GBC"})')
    gb.decorate_gbc(verbose=True, ref_ns=ref_ns)

    # first soma section for recording + injection
    soma_sec = list(h.soma)[0]

    h.celsius = _P.BODY_TEMPERATURE_C   # the pipeline temperature
    h.finitialize(-65.0)

    t = h.Vector().record(h._ref_t)
    v = h.Vector().record(soma_sec(0.5)._ref_v)

    traces = []
    delay, dur, tstop = 10.0, 100.0, 130.0
    for amp in args.amps:
        ic = h.IClamp(soma_sec(0.5))
        ic.delay, ic.dur, ic.amp = delay, dur, amp
        h.finitialize(-65.0)
        h.continuerun(tstop)
        tv, vv = np.array(t), np.array(v)
        # spike count during the step
        step = (tv >= delay) & (tv <= delay + dur)
        nsp = int(np.sum((vv[:-1] < 0) & (vv[1:] >= 0) & step[1:]))
        traces.append((amp, tv.copy(), vv.copy(), nsp))
        print(f'  I={amp:+.2f} nA  Vrest={vv[0]:.1f} mV  '
              f'Vmin={vv[step].min():.1f}  Vmax={vv[step].max():.1f}  spikes={nsp}')
        ic = None

    _plot(traces, args.ref_ns)


def _plot(traces, ref_ns='II'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    label = 'SBC (Type II-I)' if ref_ns == 'II-I' else 'GBC (Type II)'
    tag   = 'sbc' if ref_ns == 'II-I' else 'gbc'
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for amp, tv, vv, nsp in traces:
        ax.plot(tv, vv, lw=0.9, label=f'{amp:+.2f} nA ({nsp} sp)')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Soma V (mV)')
    ax.set_title(f'{label} current-clamp validation (XM13_nacncoop decoration)')
    ax.legend(fontsize=8, ncol=2)
    outdir = os.path.join(HERE, 'figures')
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f'{tag}_iclamp_validation.png')
    fig.savefig(path, dpi=150)
    print(f'Figure saved -> {path}')


if __name__ == '__main__':
    main()
