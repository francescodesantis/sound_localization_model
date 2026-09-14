#!/usr/bin/env python3
"""
Single-cell validation of the calyx of Held prespike and cleft potential.

Loads models/mntb/calyx_model.hoc, inserts NEURON's extracellular mechanism on
the terminal to represent the sub-calyceal cleft leak conductance
  g_cl ~ 1.0 uS  (R_cleft ~ 1 MOhm; Sierksma & Borst 2021 [7]),
drives the pre-calyx axon to fire one presynaptic AP, and checks:
  1. the terminal fires an AP (the prespike source),
  2. the cleft potential V_cleft (extracellular vext under the terminal)
     reaches several mV during the AP, as reported for the juvenile calyx [7],
  3. a current-dipole moment p(t) = sum(r * i_membrane) is produced.

Usage:
  python models/mntb/validate_calyx.py

Saves models/mntb/figures/calyx_prespike_validation.png and a PASS/FAIL line.
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

G_CL_US = 1.0          # cleft leak conductance (uS)


def _load(d):
    try:
        neuron.load_mechanisms(d)
    except RuntimeError as e:
        if 'already exists' not in str(e):
            raise


def _area_cm2(sec):
    return sum(seg.area() for seg in sec) * 1e-8   # um2 -> cm2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--morphology', default=os.path.join(HERE, 'calyx_model.hoc'))
    ap.add_argument('--weight', type=float, default=0.15,
                    help='suprathreshold Exp2Syn weight (uS) onto the pre-calyx axon')
    ap.add_argument('--g-cl', type=float, default=G_CL_US, dest='g_cl',
                    help='cleft leak conductance (uS)')
    args = ap.parse_args()

    _load(HERE)    # namntb
    _load(MSO_DIR)                                   # klt/kht
    h.load_file('stdrun.hoc')
    h.load_file(args.morphology)
    h.cvode.use_fast_imem(1)
    h.celsius = _P.BODY_TEMPERATURE_C   # the pipeline temperature

    calyx = h.calyx
    pre   = h.precalyx_axon

    # ---- cleft: extracellular mechanism on the terminal, xg*area = g_cl ----
    a_cm2 = _area_cm2(calyx)
    xg = (args.g_cl * 1e-6) / a_cm2            # S/cm2 so xg*area = g_cl (S)
    calyx.insert('extracellular')
    for seg in calyx:
        seg.xg[0] = xg
        seg.xc[0] = 0.0
        seg.xraxial[0] = 1e9                   # isolate the cleft node (MOhm/cm)
    print(f'  calyx area = {a_cm2*1e8:.1f} um2 ; g_cl = {args.g_cl:.2f} uS -> '
          f'xg = {xg:.3f} S/cm2')

    # ---- drive: one presynaptic AP via a strong EPSC on the pre-calyx axon ----
    ns = h.NetStim(); ns.number = 1; ns.start = 2.0; ns.noise = 0
    syn = h.Exp2Syn(pre(0.2)); syn.tau1 = 0.1; syn.tau2 = 0.17; syn.e = 0.0
    nc = h.NetCon(ns, syn); nc.weight[0] = args.weight; nc.delay = 0

    # ---- recordings ----
    t     = h.Vector().record(h._ref_t)
    v_pre = h.Vector().record(pre(0.5)._ref_v)
    v_cal = h.Vector().record(calyx(0.5)._ref_v)
    vext  = h.Vector().record(calyx(0.5)._ref_vext[0])

    segs, im_vecs, pos = [], [], []
    for sec in h.allsec():
        n3d = int(h.n3d(sec=sec))
        for seg in sec:
            im_vecs.append(h.Vector().record(seg._ref_i_membrane_))
            segs.append(seg)
            # xyz via arc interpolation
            if n3d >= 2:
                arcs = np.array([h.arc3d(i, sec=sec) for i in range(n3d)])
                xs = np.array([h.x3d(i, sec=sec) for i in range(n3d)])
                ys = np.array([h.y3d(i, sec=sec) for i in range(n3d)])
                zs = np.array([h.z3d(i, sec=sec) for i in range(n3d)])
                aa = seg.x * arcs[-1]
                pos.append([np.interp(aa, arcs, xs), np.interp(aa, arcs, ys),
                            np.interp(aa, arcs, zs)])
            else:
                pos.append([0., 0., 0.])

    h.finitialize(-70.0)
    h.continuerun(8.0)

    tv    = np.array(t)
    vpre  = np.array(v_pre); vcal = np.array(v_cal); vc = np.array(vext)
    im    = np.array([np.array(x) for x in im_vecs])   # (Nseg, Nt) nA
    r     = np.array(pos)                               # (Nseg, 3) um
    p     = r.T @ im                                    # (3, Nt) nA.um

    fired    = vcal.max() >= 0.0
    vcleft_pk = np.abs(vc).max()
    cleft_ok = vcleft_pk >= 1.0                         # several mV expected
    p_pk     = np.abs(p).max()
    dip_ok   = p_pk > 0.0

    print(f'  terminal AP peak     : {vcal.max():.1f} mV   '
          f'[{"PASS" if fired else "FAIL"}]')
    print(f'  cleft potential |Vc| : {vcleft_pk:.2f} mV   '
          f'[{"PASS" if cleft_ok else "FAIL"}] (>=1 mV, "several mV" [7])')
    print(f'  prespike dipole |p|  : {p_pk:.1f} nA.um  '
          f'[{"PASS" if dip_ok else "FAIL"}]')
    ok = fired and cleft_ok and dip_ok
    print(f'\n  {"ALL PASS" if ok else "SOME FAILED"}')

    _plot(tv, vpre, vcal, vc, p)
    return 0 if ok else 1


def _plot(tv, vpre, vcal, vc, p):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True, constrained_layout=True)
    ax[0].plot(tv, vpre, label='pre-calyx axon Vm', color='firebrick')
    ax[0].plot(tv, vcal, label='calyx terminal Vm', color='darkorange')
    ax[0].set_ylabel('Vm (mV)'); ax[0].legend(fontsize=8)
    ax[0].set_title('Calyx of Held presynaptic AP')
    ax[1].plot(tv, vc, color='purple')
    ax[1].set_ylabel('V_cleft (mV)')
    ax[1].set_title('Cleft potential (g_cl = 1 uS)')
    for i, lab in enumerate('xyz'):
        ax[2].plot(tv, p[i], label=f'p_{lab}')
    ax[2].plot(tv, np.linalg.norm(p, axis=0), 'k', label='|p|')
    ax[2].set_ylabel('dipole (nA.um)'); ax[2].set_xlabel('Time (ms)')
    ax[2].legend(fontsize=8, ncol=4); ax[2].set_title('Prespike current dipole')
    outdir = os.path.join(HERE, 'figures'); os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, 'calyx_prespike_validation.png')
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f'Figure saved -> {path}')


if __name__ == '__main__':
    import sys
    sys.exit(main())
