#!/usr/bin/env python3
"""
AVCN ABR reconstruction: current dipole moment through the 4-sphere head model.

The cochlear nucleus is an early ABR generator (human wave III), upstream of
the MSO. This mirrors main_abr.py but drives morphologically detailed
bushy-cell populations (AVCNPopulation, ANF endbulb input) and places each
dipole at the cochlear-nucleus location.

Two generators are modelled and unified at the ABR stage:
  GBC   globular bushy cell (Type II, VCN_c09 EM), central/caudal VCN
  SBC   spherical bushy cell (Type II-I, SBC_S113 EM), rostral VCN
The spherical-cell area occupies the rostral 1.5-2.0 mm of the nucleus before
the globular region, so the SBC dipole sits ~1.5 mm anterior (+head_y) of the
GBC. Each population's dipole is projected through the 4-sphere model from its
own position and the scalp potentials are summed, since a plain vector sum of
the two dipole moments would only be valid for co-located sources.

CLI (single or MPI):
  python ABR_reconstruction/main_abr_avcn.py --pic-file RESULTS/x.pic \
      --angle 0 --side L --n-cells 200 --generators both
  mpiexec -n 4 python ABR_reconstruction/main_abr_avcn.py ... --side both

Outputs:
  RESULTS/abr_tmp/output_{pop}_{stem}_angle{A}_{side}/population_dipole.h5
  RESULTS/abr_tmp/output_avcn_{stem}_angle{A}_{sidespec}/ABR.h5   (per-pop + composite)
  .../figures/avcn_abr.png                                        (SBC/GBC/composite)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import hybridLFPy

from recon_core import head_model, params as P, paths
from recon_core.io_utils import save_dipole_record
from recon_core.mpi_utils import COMM, RANK, broadcast_from_root, reduce_sum
from recon_core.signal_utils import derive

# AVCN populations + geometry (importing this also loads the mechanisms).
from LFP_reconstruction.main_reconstruct_avcn import (
    AVCNPopulation, ELLIPSE_RADIUS_X, ELLIPSE_RADIUS_Y, LAYER_BOUNDARIES, K_YXL,
    _decorate, SYN_DELAY_LOC, SYN_DELAY_SCALE, AXON_TARGET,
)
from LFP_reconstruction import main_reconstruct_sbc as sbc      # SBC deltas
from LFP_reconstruction.main_reconstruct import _extract_spikes

DT, TSTOP, SRATE = P.DT, P.TSTOP, P.SRATE
V_INIT = P.GBC_V_INIT
N_GBC_TOTAL, N_ENDBULBS = P.N_GBC_TOTAL, P.GBC_ENDBULBS

# The rotation is side-specific so the GBC axon, aligned ventromedially in the
# model frame by set_rotations (AXON_TARGET = [-1, 0, -1]), crosses the midline
# on both sides. It also carries the ~32.5 deg outward rostral tilt of the
# human cochlear nucleus (Moore/Osen). _build_rotation(tilt) rebuilds it for
# --avcn-tilt-deg.
from recon_core.head_geometry import (
    AVCN_ROSTRAL_TILT_DEG, build_avcn_rotation as _build_rotation,
)

AVCN_POS_UM = P.AVCN_POS_UM
SBC_POS_UM = P.SBC_POS_UM
ROTATION = P.ROTATION_AVCN

_pic_stem = paths.pic_stem


def superpose_sources(sources, electrode_names, hi=3000., lo=150.):
    """Turn (label, side, p_head, r) sources into band-passed potentials + srate."""
    return (head_model.superpose_sources(
        [(label, p_head, r) for label, _side, p_head, r in sources],
        electrode_names, SRATE, lo=lo, hi=hi), SRATE)


# ---------------------------------------------------------------------------
# Why the dipole is not split into synaptic and axonal parts
#
# Computing a somatodendritic dipole at the AVCN and a separate axonal dipole
# displaced along the tract is invalid with dipole-only tooling, and measurably
# so:
#
#   sum of i_membrane over all segments = 3.0 nA (the injected current; the
#                                         cell as a whole conserves charge)
#   sum over the axonal segments alone  = 3.2 nA (not zero)
#
# A dipole moment p = sum(r * i) is translation invariant, and so well defined,
# only when the group's net current is zero. Each sub-group carries a large net
# current, so its dipole is origin dependent, and displacing the two groups to
# different positions drops the monopole terms. With q_axon = -q_syn separated
# by d ~ 3.5 mm, the dropped term is itself a dipole of moment q*d, which is
# large. Splitting produced a spurious anti-phase cancellation.
#
# The correct treatment with FourSphereVolumeConductor, which models a current
# dipole rather than monopoles, is one dipole per cell over all segments:
# sum(i) = 0 makes it well defined, and it already contains the axonal
# travelling wave because CurrentDipoleMoment sums every segment, extended axon
# included. The remaining approximation is the far-field point-dipole placement
# of a ~4 mm source at 79 mm. A distributed treatment would need monopole
# support in the head model.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Population registry: everything that differs between GBC and SBC. Defaults in
# AVCNPopulation reproduce the GBC pipeline, so the GBC entry mostly names them.
#
# 'parts' lists the dipole sources a population contributes, each with its own
# head position. Both contribute one whole-cell dipole (see above) and differ
# in position (SBC 1.5 mm rostral) and in morphology: the GBC uses the extended
# active axon, so its dipole carries the travelling volley.
# ---------------------------------------------------------------------------
GBC_EXTENDED_HOC = os.path.join(paths.AVCN_MODELS_DIR, 'morphology',
                                'extended', 'VCN_c09_extended_axon.hoc')

POPULATIONS = {
    'gbc': dict(
        label='GBC', morphology=GBC_EXTENDED_HOC, decorate=_decorate,
        n_post_total=N_GBC_TOTAL, n_endbulbs=N_ENDBULBS, endbulb_weights=None,
        per_pop_syn=AVCNPopulation.PER_POP_SYN, k_yxl=K_YXL,
        ellipse_y=ELLIPSE_RADIUS_Y, seed=44,
        parts=[('GBC', None, AVCN_POS_UM)],   # whole-cell dipole, axon included
    ),
    'sbc': dict(
        label='SBC', morphology=sbc.HOC_FILE_SBC, decorate=sbc._decorate_sbc,
        n_post_total=sbc.N_SBC_TOTAL, n_endbulbs=sbc.N_ENDBULBS,
        endbulb_weights=sbc.ENDBULB_WEIGHTS_SBC, per_pop_syn=sbc.PER_POP_SYN_SBC,
        k_yxl=sbc.K_YXL, ellipse_y=sbc.ELLIPSE_RADIUS_Y, seed=45,
        parts=[('SBC', None, SBC_POS_UM)],   # None means plain CurrentDipoleMoment
    ),
}


# ---------------------------------------------------------------------------
# Per-(population, side) simulation, giving a population dipole (3, T) nA.µm
# ---------------------------------------------------------------------------
def _run_one_source(pop_name, side, args, meta):
    from lfpykit import CurrentDipoleMoment
    cfg = POPULATIONS[pop_name]

    X_pops       = [f'ANF_{side}']
    k_yxl_local  = cfg['k_yxl']
    j_yx_local   = [cfg['per_pop_syn']['ANF']['weight']]
    tau_yx_local = [cfg['per_pop_syn']['ANF']['tau2']]

    stem       = _pic_stem(paths.resolve_pic(args.pic_file))
    cond_val, cond_label = paths.condition_key(args.angle, args.itd_us,
                                              args.ild_db)
    spikes_dir = paths.spikes_dir_for(stem, cond_val, side)
    output_dir = paths.make_output_dirs(
        paths.output_dir_for('abr', stem, cond_label, side,
                             prefix=pop_name),
        subdirs=('figures',))

    k_arr         = np.array(k_yxl_local)
    n_syn_per_pop = {X: int(k_arr[:, j].sum()) for j, X in enumerate(X_pops)}

    networkSim = hybridLFPy.CachedNetwork(
        simtime=TSTOP, dt=DT,
        spike_output_path=spikes_dir,
        label='spikes', ext='gdf',
        GIDs={X: [meta[X]['first_gid'], meta[X]['n_neurons']] for X in X_pops},
        X=X_pops,
    )

    # one probe per dipole part (somatodendritic, axonal, or a single plain one)
    probes, part_keys = [], []
    for part_label, probe_cls, _pos in cfg['parts']:
        cls = probe_cls or CurrentDipoleMoment
        probes.append(cls(None))            # cell set by hybridLFPy in cellsim()
        part_keys.append((part_label, cls.__name__))

    pop_label = f'{cfg["label"]}_{side}'
    pop = AVCNPopulation(
        n_syn_per_pop=n_syn_per_pop,
        axon_target=AXON_TARGET,   # fixed ventromedial, laterality via X_pops
        n_post_total=cfg['n_post_total'],
        n_endbulbs=cfg['n_endbulbs'],
        endbulb_weights=cfg['endbulb_weights'],
        per_pop_syn=cfg['per_pop_syn'],
        y=pop_label,
        cellParams={
            'morphology': cfg['morphology'], 'passive': False, 'v_init': V_INIT,
            'dt': DT, 'tstart': 0., 'tstop': TSTOP,
            'nsegs_method': 'lambda_f', 'lambda_f': 100,
            'custom_fun': [cfg['decorate']], 'custom_fun_args': [{}],
        },
        rand_rot_axis=[],   # deterministic ventromedial axon orientation instead
        simulationParams={'rec_imem': True},
        populationParams={
            'number':   args.n_cells,
            'radius':   cfg['ellipse_y'],
            'radius_x': ELLIPSE_RADIUS_X,
            'radius_y': cfg['ellipse_y'],
            'z_min': 0.0, 'z_max': 0.0, 'min_cell_interdist': 1.0,
            'min_r': np.array([[0.], [0.]]),
        },
        layerBoundaries=LAYER_BOUNDARIES,
        probes=probes,
        savelist=['somapos'],
        savefolder=output_dir,
        dt_output=DT,
        POPULATIONSEED=cfg['seed'],
        X=X_pops,
        networkSim=networkSim,
        k_yXL=k_yxl_local,
        synParams={'section': 'allsec', 'syntype': 'Exp2Syn'},
        synDelayLoc=SYN_DELAY_LOC,
        synDelayScale=SYN_DELAY_SCALE,
        J_yX=j_yx_local,
        tau_yX=tau_yx_local,
    )

    pop.run()
    COMM.Barrier()

    # Sum per-cell dipole moments (3, T) nA.µm on this rank, per part
    n_t = int(round(TSTOP / DT)) + 1
    dipoles = {}
    for part_label, out_key in part_keys:
        local = None
        for i in pop.RANK_CELLINDICES:
            d = pop.output[i][out_key].astype(np.float64)
            local = d.copy() if local is None else local + d
        if local is None:
            local = np.zeros((3, n_t), dtype=np.float64)
        dipoles[part_label] = reduce_sum(local)
    COMM.Barrier()
    return dipoles, output_dir


# ---------------------------------------------------------------------------
# Head-model projection and saving (rank 0)
# ---------------------------------------------------------------------------
def _project_and_save(side, dipoles, output_dir):
    """Rotate each part's dipole into head coords and save, as {part: p_head}."""
    srate = SRATE
    out = {}
    with h5py.File(os.path.join(output_dir, 'population_dipole.h5'), 'w') as f:
        for part, p_model in dipoles.items():
            p_head = head_model.rotate_to_head(p_model, ROTATION[side])
            f.create_dataset(part, data=p_head)
            out[part] = p_head
        f.create_dataset('srate', data=srate)
        f.attrs['axes']  = 'x=mediolateral, y=anteroposterior, z=inferosuperior'
        f.attrs['units'] = 'nA·µm'
        f.attrs['parts'] = ','.join(out)
    return out


def _apply_head_model(sources, output_dir, electrode_names):
    """Project each source dipole from its own position and sum at the scalp.

    Thin wrapper over main_abr.superpose_sources, the one 4-sphere
    superposition also used by the cross-nucleus composite in main_abr_full.py,
    that additionally writes this run's ABR.h5. sources is a list of
    (part_label, side, p_head, r_dipole); sources sharing a part_label, such as
    both sides of one generator, are summed.

    Returns (V_by_key, srate), where V_by_key maps each part label and
    'composite' to a band-passed (n_electrodes, n_t) array in µV.
    """
    V_out, srate = superpose_sources(sources, electrode_names, hi=3000., lo=150.)

    with h5py.File(os.path.join(output_dir, 'ABR.h5'), 'w') as f:
        for key, V_uV in V_out.items():
            f.create_dataset(key, data=V_uV)
        f.create_dataset('srate', data=srate)
        f.create_dataset('electrode_names', data=np.array(electrode_names, dtype='S'))
        f.attrs['units'] = 'µV'
        f.attrs['keys']  = ','.join(V_out)
    print(f'ABR saved to {os.path.join(output_dir, "ABR.h5")}  keys={list(V_out)}')
    return V_out, srate


_derivation = derive


def _plot_abr(output_dir, V_out, electrode_names, srate, cond_label, side,
              n_cells, derivation='Cz-M1'):
    """Top: Cz per population and composite. Bottom: composite derivation."""
    cz_idx = electrode_names.index('Cz')
    any_V  = next(iter(V_out.values()))
    tvec   = np.arange(any_V.shape[1]) / srate * 1e3
    colours = {'GBC': 'seagreen', 'SBC': 'darkorange', 'composite': 'black'}

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    for key, V_uV in V_out.items():
        ax0.plot(tvec, V_uV[cz_idx], lw=1.1 if key == 'composite' else 0.9,
                 color=colours.get(key, None),
                 label=f'{key} (Cz)', zorder=3 if key == 'composite' else 2)
    ax0.axhline(0, color='k', lw=0.4, ls=':')
    ax0.set_ylabel('Cz potential (µV)')
    ax0.set_title(f'AVCN ABR (SBC + GBC) | {cond_label} | side {side} | N={n_cells}')
    ax0.legend(fontsize=9)

    diff, lbl = _derivation(V_out['composite'], electrode_names, derivation)
    ax1.plot(tvec, diff, color='darkorchid', lw=1.0, label=f'composite {lbl}')
    ax1.axhline(0, color='k', lw=0.4, ls=':')
    ax1.set_xlabel('Time (ms)'); ax1.set_ylabel('Amplitude (µV)')
    ax1.set_title(f'{lbl}  (vertex-positive upward)')
    ax1.legend(fontsize=9)

    path = os.path.join(output_dir, 'figures', 'avcn_abr.png')
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f'ABR figure saved to {path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cond_value(args):
    """Raw stimulus key the spike cache is stored under (angle, seconds or dB)."""
    return paths.condition_key(args.angle, args.itd_us, args.ild_db)[0]


def _cond_label(args):
    """Readable stimulus label for the output directory name."""
    return paths.condition_key(args.angle, args.itd_us, args.ild_db)[1]


def main():
    import argparse
    parser = argparse.ArgumentParser(description='AVCN (SBC + GBC) ABR reconstruction')
    parser.add_argument('--pic-file',    type=str, default=None, dest='pic_file')
    parser.add_argument('--angle',       type=int, default=0)
    parser.add_argument('--side',        type=str, default='L',
                        choices=['L', 'R', 'both'])
    parser.add_argument('--n-cells',     type=int, default=200, dest='n_cells')
    parser.add_argument('--itd-us', type=float, default=None, dest='itd_us',
                        help='select an artificial-ITD condition (µs); overrides '
                             '--angle. The pic key is looked up in seconds.')
    parser.add_argument('--ild-db', type=float, default=None, dest='ild_db',
                        help='select an artificial-ILD condition (dB); overrides '
                             '--itd-us and --angle.')
    parser.add_argument('--generators', type=str, default='both',
                        choices=['gbc', 'sbc', 'both'], dest='generators',
                        help='which bushy-cell generators to model. They are '
                             'distinct cell types, so "both" SUMS them at the scalp.')
    parser.add_argument('--derivation',  type=str, default='Cz-M1',
                        choices=['Cz-M1', 'Cz-M2', 'Cz-avg'])
    parser.add_argument('--gbc-hoc', type=str, default=None, dest='gbc_hoc',
                        help='override the GBC morphology (default: extended '
                             'active axon; pass the plain VCN_c* EM hoc to '
                             'measure the axonal travelling-wave contribution)')
    parser.add_argument('--avcn-tilt-deg', type=float, default=AVCN_ROSTRAL_TILT_DEG,
                        dest='avcn_tilt_deg',
                        help='outward rostral tilt of the CN about the head '
                             'vertical axis (deg); 0 = untilted (default %(default)s)')
    args = parser.parse_args()

    global ROTATION
    ROTATION = _build_rotation(args.avcn_tilt_deg)

    if args.gbc_hoc:
        POPULATIONS['gbc']['morphology'] = args.gbc_hoc

    sides = ['L', 'R'] if args.side == 'both' else [args.side]
    pops  = ['gbc', 'sbc'] if args.generators == 'both' else [args.generators]
    stem  = _pic_stem(paths.resolve_pic(args.pic_file))

    # Extract presynaptic spikes once per side; both populations share the ANF drive.
    meta_by_side = {
        side: broadcast_from_root(
            lambda side=side: _extract_spikes(_cond_value(args), side,
                                              pic_file=args.pic_file))
        for side in sides
    }

    sources = []   # (part_label, side, p_head, r_dipole) on rank 0
    for pop_name in pops:
        for side in sides:
            dipoles, output_dir = _run_one_source(pop_name, side, args,
                                                  meta_by_side[side])
            if RANK == 0:
                p_heads = _project_and_save(side, dipoles, output_dir)
                for part_label, _probe_cls, pos_map in POPULATIONS[pop_name]['parts']:
                    sources.append((part_label, side, p_heads[part_label],
                                    pos_map[side]))
                    save_dipole_record(stem, _cond_label(args), 'AVCN',
                                       part_label, side,
                                       p_heads[part_label], pos_map[side],
                                       POPULATIONS[pop_name]['n_post_total'],
                                       args.n_cells, SRATE)

    if RANK == 0:
        tag = 'avcn' if len(pops) > 1 else pops[0]
        final_dir = paths.make_output_dirs(
            paths.output_dir_for('abr', stem, _cond_label(args), args.side,
                                 prefix=tag),
            subdirs=('figures',))

        electrode_names = list(P.ELECTRODES)
        V_out, srate = _apply_head_model(sources, final_dir, electrode_names)
        _plot_abr(final_dir, V_out, electrode_names, srate,
                  _cond_label(args), args.side, args.n_cells,
                  derivation=args.derivation)


if __name__ == '__main__':
    main()
