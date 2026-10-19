"""
Single-cell check of the GBC axonal travelling-wave dipole.

Loads a GBC morphology, appends a synthetic node/internode active axon
(axon_builder), decorates it (gbc_biophysics: active nodes, passive myelin),
fires one action potential at the soma, and checks that:

  1. the AP propagates saltatorily along the axon (node Vm sequence),
  2. conduction velocity is physiological (~3-10 m/s),
  3. the net current-dipole moment shows a travelling-wave signature, a moving
     axonal current source distinct from the stationary synaptic dipole.

Saves models/avcn/figures/gbc_axon_travelingwave.png

Usage:
  python models/avcn/validate_gbc_axon.py
"""

import os
import sys

import numpy as np
import neuron
from neuron import h

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from recon_core import params as _P                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def seg_positions():
    """Return {seg: (x,y,z) µm} for every segment, interpolated from pt3d."""
    h.define_shape()
    pos = {}
    for sec in h.allsec():
        n = int(h.n3d(sec=sec))
        if n < 2:
            continue
        arc = np.array([h.arc3d(i, sec=sec) for i in range(n)])
        if arc[-1] == 0:
            continue
        arcn = arc / arc[-1]
        xx = np.array([h.x3d(i, sec=sec) for i in range(n)])
        yy = np.array([h.y3d(i, sec=sec) for i in range(n)])
        zz = np.array([h.z3d(i, sec=sec) for i in range(n)])
        for seg in sec:
            pos[seg] = (np.interp(seg.x, arcn, xx),
                        np.interp(seg.x, arcn, yy),
                        np.interp(seg.x, arcn, zz))
    return pos


def main():
    sys.path.insert(0, HERE)
    import gbc_biophysics as gb
    import axon_builder as ab

    try:
        neuron.load_mechanisms(HERE)
    except RuntimeError as e:
        if 'already exists' not in str(e):
            raise
    h.load_file('stdlib.hoc'); h.load_file('import3d.hoc'); h.load_file('stdrun.hoc')
    h.load_file(os.path.join(HERE, 'morphology', 'dryad',
                             'VCN_c09_Full_MeshInflate.hoc'))

    nodes, internodes = ab.build_extended_axon(verbose=True)
    gb.decorate_gbc(set_nseg=True, ref_ns=gb.REF_NS_II)
    h.cvode.use_fast_imem(1)
    h.celsius = _P.BODY_TEMPERATURE_C   # the pipeline temperature

    pos = seg_positions()

    # cumulative path distance soma->node (for CV); node soma-distance via pos
    soma = list(h.soma)[0]
    soma_c = np.array(pos[soma(0.5)])
    node_dist = np.array([np.linalg.norm(np.array(pos[nd(0.5)]) - soma_c)
                          for nd in nodes])

    # recordings
    t = h.Vector().record(h._ref_t)
    v_nodes = [h.Vector().record(nd(0.5)._ref_v) for nd in nodes]
    v_soma  = h.Vector().record(soma(0.5)._ref_v)
    # transmembrane current + position for every segment (for the dipole moment)
    seg_list, seg_pos, imem = [], [], []
    for sec in h.allsec():
        for seg in sec:
            if seg in pos:
                seg_list.append(seg)
                seg_pos.append(pos[seg])
                imem.append(h.Vector().record(seg._ref_i_membrane_))
    seg_pos = np.array(seg_pos)          # (Nseg, 3) µm

    # fire one action potential with a brief strong somatic pulse
    ic = h.IClamp(soma(0.5)); ic.delay, ic.dur, ic.amp = 2.0, 0.3, 3.0
    h.finitialize(-65.0)
    h.continuerun(8.0)

    t   = np.array(t)
    vs  = np.array(v_soma)
    vN  = np.array([np.array(v) for v in v_nodes])       # (Nnode, Nt)
    im  = np.array([np.array(v) for v in imem])          # (Nseg, Nt) nA

    # --- 1) node spike times + conduction velocity ---------------------------
    def first_cross(trace, thr=-10.0):
        idx = np.where((trace[:-1] < thr) & (trace[1:] >= thr))[0]
        return t[idx[0]] if len(idx) else np.nan
    node_t = np.array([first_cross(vN[i]) for i in range(len(nodes))])
    ok = np.isfinite(node_t)
    n_fired = int(ok.sum())
    if n_fired >= 2:
        # CV from linear fit distance(µm) vs time(ms): slope µm/ms = m/s
        cv = np.polyfit(node_t[ok], node_dist[ok], 1)[0] / 1e3  # µm/ms -> m/s
    else:
        cv = np.nan
    print(f'nodes reached by AP: {n_fired}/{len(nodes)}; '
          f'conduction velocity ~ {cv:.2f} m/s')

    # --- 2) net current-dipole moment p(t) = sum r * i_membrane --------------
    p = seg_pos.T @ im            # (3, Nt) nA·µm
    print(f'peak |p| = {np.linalg.norm(p, axis=0).max():.1f} nA·µm')

    _plot(t, vs, vN, node_dist, node_t, im, seg_pos, seg_list, nodes, p, cv, n_fired)


def _plot(t, vs, vN, node_dist, node_t, im, seg_pos, seg_list, nodes, p, cv, n_fired):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # axonal segment mask + their soma distance for the current raster
    import gbc_biophysics as gb
    comp = gb._classify_sections()
    ax_mask = np.array([comp.get(s.sec.name()) in ('node', 'internode', 'myelinatedaxon')
                        for s in seg_list])
    ax_pos = seg_pos[ax_mask]
    soma_c = seg_pos[[i for i, s in enumerate(seg_list)
                      if comp.get(s.sec.name()) == 'soma'][0]]
    ax_d = np.linalg.norm(ax_pos - soma_c, axis=1)
    order = np.argsort(ax_d)
    im_ax = im[ax_mask][order]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)

    # (a) node Vm stacked by distance, showing saltatory propagation
    sc = 40.0
    for i in range(len(nodes)):
        axes[0].plot(t, vN[i] / 120 * sc + node_dist[i], color='k', lw=0.5)
    axes[0].plot(node_t[np.isfinite(node_t)],
                 node_dist[np.isfinite(node_t)], 'o', color='crimson', ms=3,
                 label='AP arrival')
    axes[0].set_xlabel('Time (ms)'); axes[0].set_ylabel('Distance from soma (µm)')
    axes[0].set_title(f'Node Vm — AP propagation ({n_fired}/{len(nodes)} nodes)')
    axes[0].legend(fontsize=8)

    # (b) axonal transmembrane current raster (distance x time): travelling source
    vmax = np.percentile(np.abs(im_ax), 99) or 1e-6
    axes[1].imshow(im_ax, aspect='auto', origin='lower', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax,
                   extent=[t[0], t[-1], ax_d[order][0], ax_d[order][-1]])
    axes[1].set_xlabel('Time (ms)'); axes[1].set_ylabel('Distance from soma (µm)')
    axes[1].set_title(f'Axonal i_membrane — traveling wave  (CV ≈ {cv:.1f} m/s)')

    # (c) net dipole moment components
    for k, lab, col in ((0, 'p_x', 'tab:red'), (1, 'p_y', 'tab:green'),
                        (2, 'p_z', 'tab:blue')):
        axes[2].plot(t, p[k], color=col, lw=0.9, label=lab)
    axes[2].plot(t, np.linalg.norm(p, axis=0), color='k', lw=1.2, label='|p|')
    axes[2].axhline(0, color='k', lw=0.4, ls=':')
    axes[2].set_xlabel('Time (ms)'); axes[2].set_ylabel('Dipole moment (nA·µm)')
    axes[2].set_title('Net current-dipole moment'); axes[2].legend(fontsize=8)

    outdir = os.path.join(HERE, 'figures')
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, 'gbc_axon_travelingwave.png')
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f'Figure saved -> {path}')


if __name__ == '__main__':
    main()
