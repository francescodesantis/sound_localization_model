"""
Curio & Weigel (1990) scalp binaural-interaction (BI) reproduction.

Recreates the human vertex-to-mastoid BAEP under monaural R, monaural L and
binaural RL click stimulation, and the binaural-interaction trace

    BI = (RL) - (R + L)                        (Wernick & Starr 1967)

in the paper's two derivations Cz/A1 (Cz-M1) and Cz/A2 (Cz-M2), band-pass
0.15-3 kHz. The three condition composites are assembled from the standardised
dipole records (main_abr_full.assemble); no NEURON is re-run here.

BI is MSO plus LSO only. Subtracting the summed monaural from the binaural
response cancels any generator whose binaural response equals the sum of its
two monaural ones. AVCN and MNTB are monaural by construction, each driven by
a single ear, so they cancel and the scalp BI comes entirely from the binaural
nuclei MSO and LSO, the SOC origin the paper attributes BI to (around wave III
and later). This is verified numerically below.

Monaural conditions come from post-hoc silencing of the binaural pic
(main_abr.py/_run_one_side --condition), so the LSO uses its synaptic
generator: a spiking output volley cannot be un-mixed afterwards. AVCN and
MNTB reuse their binaural records restricted to the ear-driven hemisphere
(the ear to side rule in main_abr_full._discover).

Prerequisite records (run once, full N, both sides) in
RESULTS/abr_tmp/dipoles/<stem>_angle<A>/ (the <cond> the producers used):
  binaural           MSO, LSO(synaptic), AVCN(GBC+SBC), MNTB(principal+calyx)
  right_ear/left_ear MSO, LSO(synaptic)
  (main_abr.py --condition {right_ear,left_ear};
   main_abr_lso.py --generators synaptic --condition {right_ear,left_ear})

Output:
  RESULTS/abr_tmp/output_bi_<stem>_angle<A>/BI.h5   (R, L, RL, BI + per-nucleus BI)
  .../figures/curio_weigel_bi.png                   (Cz/A1 and Cz/A2 x R, L, RL, BI)

Usage:
  python ABR_reconstruction/main_abr_bi.py --pic-file RESULTS/click_70dBbaseline.pic --angle 0
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from recon_core import paths
from ABR_reconstruction.main_abr_full import assemble, ELECTRODES, _derive

_pic_stem = paths.pic_stem

# paper conditions to our assemble() condition keys
CONDITIONS = {'RL': 'binaural', 'R': 'right_ear', 'L': 'left_ear'}
DERIVATIONS = ['Cz-M1', 'Cz-M2']        # Cz/A1, Cz/A2
DERIV_LABEL = {'Cz-M1': 'Cz/M1', 'Cz-M2': 'Cz/M2'}


def _bi(a, b, c):
    """Binaural interaction: RL - (R + L), elementwise."""
    return a - (b + c)


def main():
    ap = argparse.ArgumentParser(description='Curio & Weigel scalp BI reproduction')
    ap.add_argument('--pic-file', type=str, default=None, dest='pic_file')
    ap.add_argument('--angle',    type=int, default=0)
    ap.add_argument('--band', type=str, default='150,3000',
                    help='band-pass lo,hi Hz (paper 0.15-3 kHz; default 150,3000)')
    ap.add_argument('--lso-generator', type=str, default='synaptic',
                    choices=['synaptic', 'spiking'], dest='lso_generator',
                    help='synaptic required for post-hoc monaural BI (default)')
    args = ap.parse_args()

    stem = _pic_stem(paths.resolve_pic(args.pic_file))
    _, cond_label = paths.condition_key(args.angle)
    lo, hi = (float(x) for x in args.band.split(','))
    band = (lo, hi)

    # --- assemble the three condition composites (all generators, both sides) ---
    comp, nuc, srate = {}, {}, None
    for tag, cond in CONDITIONS.items():
        V_gen, V_nuc, sr = assemble(stem, cond_label, None, ['L', 'R'], cond,
                                    lso_generator=args.lso_generator, band=band)
        comp[tag] = V_gen['composite']          # (n_e, T) µV
        nuc[tag]  = V_nuc                        # nucleus to (n_e, T) µV
        srate = sr
        print(f'{tag:2s} ({cond}): generators {sorted(k for k in V_gen if k!="composite")}')

    # --- BI = RL - (R + L) at the composite and per-nucleus levels ---
    bi_comp = _bi(comp['RL'], comp['R'], comp['L'])
    all_nuc = sorted(set().union(*[set(n) for n in nuc.values()]))
    bi_nuc = {}
    for nm in all_nuc:
        z = np.zeros_like(bi_comp)
        bi_nuc[nm] = _bi(nuc['RL'].get(nm, z), nuc['R'].get(nm, z), nuc['L'].get(nm, z))

    # --- validation: AVCN/MNTB cancel, so BI is about MSO_BI + LSO_BI ---
    binaural_only = sum(bi_nuc.get(nm, np.zeros_like(bi_comp)) for nm in ('MSO', 'LSO'))
    resid = np.max(np.abs(bi_comp - binaural_only))
    print(f'\n[validation] max|BI - (MSO_BI + LSO_BI)| = {resid:.3e} µV '
          f'(AVCN/MNTB should cancel to ~0)')
    for nm in all_nuc:
        cz = ELECTRODES.index('Cz')
        print(f'    per-nucleus BI  {nm:<5s} peak|Cz| = {np.abs(bi_nuc[nm][cz]).max():.3e} µV')

    # --- wave III and wave V ancillary latencies (the paper's IIIR, VL lines) ---
    # Wave to generator here: III is the AVCN crossing-axon volley, V the LSO
    # (SOC output). III is taken from the right-ear response and V from the
    # left-ear one, as the paper marks IIIR and VL throughout Fig. 2.
    t = np.arange(bi_comp.shape[1]) / srate * 1e3
    cz = ELECTRODES.index('Cz')

    def _peak_ms(trace, win):
        # largest peak by amplitude within a latency window (standard ABR peak
        # picking; the raw generator traces are multiphasic, so a global argmax
        # can land on a late rebound lobe).
        m = (t >= win[0]) & (t <= win[1])
        a = np.where(m, np.abs(trace[cz]), 0.0)
        return float(t[int(np.argmax(a))])

    t_III = _peak_ms(nuc['R']['AVCN'], (2., 8.))    # wave III (right ear, AVCN)
    t_V   = _peak_ms(nuc['L']['LSO'],  (5., 12.))   # wave V   (left ear, LSO)
    print(f'\n[waves] III_R (AVCN, right ear) = {t_III:.2f} ms   '
          f'V_L (LSO, left ear) = {t_V:.2f} ms')

    # --- save ---
    # No side suffix: the BI is a whole-head measure, both sides always included.
    out_dir = paths.make_output_dirs(
        os.path.join(paths.ABR_TMP_DIR, f'output_bi_{stem}_{cond_label}'),
        subdirs=('figures',))
    with h5py.File(os.path.join(out_dir, 'BI.h5'), 'w') as f:
        for tag in ('R', 'L', 'RL'):
            f.create_dataset(tag, data=comp[tag])
        f.create_dataset('BI', data=bi_comp)
        for nm, v in bi_nuc.items():
            f.create_dataset(f'BI_nucleus__{nm}', data=v)
        f.create_dataset('srate', data=srate)
        f.create_dataset('electrode_names', data=np.array(ELECTRODES, dtype='S'))
        f.attrs.update(units='µV', stem=stem, angle=str(args.angle),
                       band=f'{lo}-{hi} Hz', lso_generator=args.lso_generator,
                       bi_residual_uV=float(resid),
                       wave_III_R_ms=t_III, wave_V_L_ms=t_V)
    print(f'BI.h5 saved to {out_dir}')

    _plot_fig2(out_dir, comp, bi_comp, srate, band)


def _plot_fig2(out_dir, comp, bi_comp, srate, band):
    """Paper Fig. 2 layout: rows are {Cz/A1, Cz/A2}, columns {R, L, RL, BI}."""
    t = np.arange(bi_comp.shape[1]) / srate * 1e3
    cols = ['R', 'L', 'RL', 'BI']
    traces = dict(comp); traces['BI'] = bi_comp
    # colour standard: right ear = green, left ear = light purple, RL = dark grey
    colours = {'R': 'seagreen', 'L': 'mediumpurple', 'RL': '#3a3a3a', 'BI': 'crimson'}

    fig, axes = plt.subplots(len(DERIVATIONS), len(cols),
                             figsize=(13, 3.6), sharex=True, constrained_layout=True)
    # shared y-scale within R/L/RL, which are comparable; BI on its own scale
    rl_max = max(np.abs(_derive(traces[c], ELECTRODES, DERIVATIONS[0])[0]).max()
                 for c in ('R', 'L', 'RL'))
    for i, deriv in enumerate(DERIVATIONS):
        for j, col in enumerate(cols):
            ax = axes[i, j]
            y, lbl = _derive(traces[col], ELECTRODES, deriv)
            ax.plot(t, y, color=colours[col], lw=0.9)
            ax.axhline(0, color='k', lw=0.4, ls=':')
            ax.set_xlim(0, 15)
            if col != 'BI':
                ax.set_ylim(-1.15 * rl_max, 1.15 * rl_max)
            if i == 0:
                ax.set_title(col, fontsize=11, fontweight='bold')
            if j == 0:
                ax.set_ylabel(f'{DERIV_LABEL[deriv]}\n(µV)')
            if i == len(DERIVATIONS) - 1:
                ax.set_xlabel('Time (ms)')
    fig.suptitle(f'Curio & Weigel (1990) scalp BAEP + binaural interaction  '
                 f'|  BI = RL − (R+L)  |  {band[0]:.0f}–{band[1]:.0f} Hz  '
                 f'|  vertex-positive up', fontsize=10, fontweight='bold')
    path = os.path.join(out_dir, 'figures', 'curio_weigel_bi.png')
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f'figure saved to {path}')


if __name__ == '__main__':
    main()
