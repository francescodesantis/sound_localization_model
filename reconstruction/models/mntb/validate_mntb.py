#!/usr/bin/env python3
"""
Validation of the human MNTB principal-cell morphology
(models/mntb/mntb_model_active.hoc; klt/kht/ih from models/mso, fast Na namntb
from models/mntb).

The MNTB principal-cell signature checked here:
  1. Fires a fast action potential (Kv3.1/KHT gives a brief half-width).
  2. Sub-threshold rectification from the low-threshold K current (KLT): a
     depolarising step stays well below the ohmic prediction.
  3. Depolarising sag on hyperpolarising steps (Ih).
  4. Phase-locks to fast calyx-of-Held EPSCs: about one spike per event, and it
     follows a high-frequency train. The cell is phasic to DC like the other
     R&M nuclei but follows brief suprathreshold EPSCs.

Usage:
  python models/mntb/validate_mntb.py

Saves models/mntb/figures/mntb_iclamp_validation.png and a PASS/FAIL table
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

HERE       = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.dirname(HERE)          # reconstruction/models
MSO_DIR    = os.path.join(MODELS_DIR, 'mso')   # klt / kht / ih live there


def _load_mech(d):
    try:
        neuron.load_mechanisms(d)
    except RuntimeError as e:
        if 'already exists' not in str(e):
            raise


def _upcrossings(tv, vv, lo, hi, thr=0.0):
    m = (tv >= lo) & (tv <= hi)
    return np.where((vv[:-1] < thr) & (vv[1:] >= thr) & m[1:])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--morphology', default=os.path.join(HERE, 'mntb_model_active.hoc'))
    ap.add_argument('--amps', type=float, nargs='+',
                    default=[-0.5, -0.2, 0.1, 0.2, 0.5, 1.0])   # nA
    ap.add_argument('--syn-weight', type=float, default=0.08, dest='syn_weight',
                    help='calyx Exp2Syn weight (uS) for the following test')
    args = ap.parse_args()

    _load_mech(HERE)   # namntb
    _load_mech(MSO_DIR)                                  # klt/kht/ih (also CWD auto-load)
    h.load_file('stdrun.hoc')
    h.load_file(args.morphology)

    soma = h.soma
    h.celsius = _P.BODY_TEMPERATURE_C   # the pipeline temperature
    t = h.Vector().record(h._ref_t)
    v = h.Vector().record(soma(0.5)._ref_v)

    # ---- DC step family: AP, KLT rectification, Ih sag ----
    delay, dur, tstop = 20.0, 100.0, 140.0
    traces, sags = [], {}
    v_defl = {}
    ap_halfwidth = None
    for amp in args.amps:
        ic = h.IClamp(soma(0.5)); ic.delay, ic.dur, ic.amp = delay, dur, amp
        h.finitialize(-70.0); h.continuerun(tstop)
        tv, vv = np.array(t), np.array(v)
        step = (tv >= delay) & (tv <= delay + dur)
        up = _upcrossings(tv, vv, delay, delay + dur)
        nsp = len(up)
        traces.append((amp, tv.copy(), vv.copy(), nsp))
        vmin, vmax = vv[step].min(), vv[step].max()
        # peak sub-threshold deflection from rest (for KLT rectification test)
        v_defl[amp] = (vmax - vv[0]) if amp > 0 else (vmin - vv[0])
        if amp < 0:
            vend = vv[(tv >= delay + dur - 5) & (tv <= delay + dur)].mean()
            sags[amp] = vend - vmin
        # AP half-width from the first supra-threshold step that spikes
        if nsp >= 1 and ap_halfwidth is None:
            i0 = up[0]
            vpk = vv[i0:i0 + int(3 / (tv[1] - tv[0]))].max()
            half = (vpk + vv[i0]) / 2.0
            above = np.where(vv[i0:i0 + int(3 / (tv[1] - tv[0]))] >= half)[0]
            if len(above) > 1:
                ap_halfwidth = (above[-1] - above[0]) * (tv[1] - tv[0])
        print(f'  I={amp:+.2f} nA  Vrest={vv[0]:.1f}  Vmin={vmin:.1f}  '
              f'Vmax={vmax:.1f}  spikes={nsp}')
        ic = None

    # ---- calyx-EPSC following ----
    def follow(freq, weight, dur_train=40.0):
        isi = 1e3 / freq
        n_ev = int(dur_train / isi)
        ns = h.NetStim(); ns.number = n_ev
        ns.start = 5.0; ns.interval = isi; ns.noise = 0
        syn = h.Exp2Syn(soma(0.5)); syn.tau1 = 0.1; syn.tau2 = 0.17; syn.e = 0.0
        nc = h.NetCon(ns, syn); nc.weight[0] = weight; nc.delay = 0
        h.finitialize(-70.0); h.continuerun(5.0 + dur_train + 5)
        tv, vv = np.array(t), np.array(v)
        return n_ev, len(_upcrossings(tv, vv, 0, 1e9))

    print('\n  calyx-EPSC following (weight=%.3f uS):' % args.syn_weight)
    follow_rates = {}
    for f in (100, 150, 200, 300):
        n_in, n_sp = follow(f, args.syn_weight)
        ratio = n_sp / max(n_in, 1)
        follow_rates[f] = ratio
        print(f'    {f:>3d} Hz : {n_sp:2d}/{n_in:2d} spikes  ratio={ratio:.2f}')

    # ---- criteria ----
    fired      = any(nsp >= 1 for _, _, _, nsp in traces)
    fast_ap    = (ap_halfwidth is not None) and (ap_halfwidth < 1.0)   # ms
    sag_ok     = any(s > 1.0 for s in sags.values())
    # KLT rectification: the depolarising deflection is sublinear (outward
    # rectification), so doubling the current (0.1 to 0.2 nA) gives less than
    # twice the deflection because KLT activates and shunts it.
    rect_ok    = (0.1 in v_defl and 0.2 in v_defl
                  and 0 < v_defl[0.2] < 2.0 * v_defl[0.1])
    max_follow = max([f for f, r in follow_rates.items() if r >= 0.8], default=0)
    follow_ok  = max_follow >= 150

    print('\n  --- MNTB signature ---')
    print(f'  fires AP              : {"yes" if fired else "no":>4}          '
          f'[{"PASS" if fired else "FAIL"}]')
    print(f'  fast AP half-width    : '
          f'{ap_halfwidth if ap_halfwidth else float("nan"):.2f} ms   '
          f'[{"PASS" if fast_ap else "FAIL"}] (<1 ms)')
    print(f'  KLT rectification     : dV(+0.2)={v_defl.get(0.2,0):.1f} '
          f'< 2*dV(+0.1)={2*v_defl.get(0.1,0):.1f} mV  '
          f'[{"PASS" if rect_ok else "FAIL"}] (sublinear)')
    print(f'  Ih sag                : {max(sags.values()) if sags else 0:.1f} mV  '
          f'[{"PASS" if sag_ok else "FAIL"}] (>1 mV)')
    print(f'  calyx following       : up to {max_follow} Hz @ratio>=0.8  '
          f'[{"PASS" if follow_ok else "FAIL"}] (>=150 Hz)')
    ok = fired and fast_ap and sag_ok and rect_ok and follow_ok
    print(f'\n  {"ALL PASS" if ok else "SOME FAILED"}')

    _plot(traces, max_follow)
    return 0 if ok else 1


def _plot(traces, max_follow):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for amp, tv, vv, nsp in traces:
        ax.plot(tv, vv, lw=0.9, label=f'{amp:+.2f} nA ({nsp} sp)')
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('Soma V (mV)')
    ax.set_title(f'MNTB principal cell current-clamp validation '
                 f'(calyx following to {max_follow} Hz)')
    ax.legend(fontsize=8, ncol=2)
    outdir = os.path.join(HERE, 'figures'); os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, 'mntb_iclamp_validation.png')
    fig.savefig(path, dpi=150)
    print(f'Figure saved -> {path}')


if __name__ == '__main__':
    import sys
    sys.exit(main())
