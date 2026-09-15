#!/usr/bin/env python3
"""
HybridLFPy LFP reconstruction for the AVCN globular bushy cell (GBC) population.

Unlike the MSO/LSO scripts (stick morphologies with Exp2Syn), the GBC is a
detailed morphology decorated with ported cnmodel channels (klt, kht, ihvcn,
leak, nacncoop; XM13_nacncoop mouse Type II) and driven by auditory-nerve
endbulbs of Held.

Morphology: the Dryad EM reconstruction under models/avcn/morphology/.
Biophysics is applied at cell build time through LFPy custom_fun =
gbc_biophysics.decorate_gbc, which adapts to any morphology.

Presynaptic drive is ANF_{side} only, since the cochlear nucleus is
ipsilateral to its ear. 20 endbulbs per GBC, matching the NEST ANFs2GBCs
convergence, placed 70% soma, 20% proximal dendrite and hubs, 10% hillock and
AIS (gbc_biophysics.weighted_endbulb_idx).

CLI (single or MPI):
  python LFP_reconstruction/main_reconstruct_avcn.py --pic-file RESULTS/x.pic \
      --angle 0 --side L --n-cells 100
  mpiexec -n 4 python LFP_reconstruction/main_reconstruct_avcn.py ... --n-cells 3600

Outputs go to RESULTS/lfp_tmp/output_avcn_{stem}_angle{A}_{S}/figures/:
  avcn_lfp_reconstruction.png
  avcn_lfp_single_cells.png
"""

import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import LFPy
import lfpykit.models as lfpykit_models
import hybridLFPy

from recon_core import params as P, paths, tree_dipole
from recon_core.population import ReconstructionPopulation
from LFP_reconstruction import figures
from recon_core.mpi_utils import (COMM, RANK, broadcast_from_root,
                                  load_mechanisms, set_temperature)

AVCN_DIR = paths.AVCN_MODELS_DIR
sys.path.insert(0, AVCN_DIR)
import gbc_biophysics  # noqa: E402

# AVCN mechanisms only: their SUFFIXes collide with models/mso.
load_mechanisms(AVCN_DIR)
set_temperature(P.BODY_TEMPERATURE_C)

STICK_HOC = os.path.join(AVCN_DIR, 'morphology', 'bushy_stick.hoc')
# Default is the Dryad EM reconstruction (mesh-inflated, accurate surface areas).
# --hoc-file picks another VCN_c* cell, or STICK_HOC for a fast smoke test.
HOC_FILE = os.path.join(AVCN_DIR, 'morphology', 'dryad',
                        'VCN_c09_Full_MeshInflate.hoc')


def _decorate(cell):
    """LFPy custom_fun: apply cnmodel GBC channel densities. nseg set by LFPy."""
    gbc_biophysics.decorate_gbc(cell, set_nseg=False)


DT, TSTOP = P.DT, P.TSTOP
V_INIT = P.GBC_V_INIT
N_CELLS = 100                     # default representative count for a quick run
N_GBC_TOTAL = P.N_GBC_TOTAL
N_ENDBULBS = P.GBC_ENDBULBS

ELLIPSE_RADIUS_X, ELLIPSE_RADIUS_Y = P.AVCN_RADIUS_X, P.GBC_RADIUS_Y

N_CH = P.N_CH
PROBE_Z = P.probe_z(P.AVCN_PROBE_HALF_SPAN)
PROBE_X = np.zeros(N_CH)
PROBE_Y = np.zeros(N_CH)
SIGMA = P.SIGMA_EXTRACELLULAR

AXON_TARGET = P.AVCN_AXON_TARGET
LAYER_BOUNDARIES = P.AVCN_LAYERS
K_YXL = P.GBC_CONVERGENCE
SYN_DELAY_LOC = P.GBC_DELAYS
SYN_DELAY_SCALE = P.SYN_DELAY_SCALE

# blank_onset_ms: the detailed EM morphology leaves a one-sample capacitive
# transient at finitialize that would otherwise set the whole colour scale.
FIGURE_STYLE = figures.FigureStyle(name='AVCN (GBC)', file_prefix='avcn',
                                   trace_gain=60.0, blank_onset_ms=0.2)


# ---------------------------------------------------------------------------
# AVCNPopulation subclass
# ---------------------------------------------------------------------------
class AVCNPopulation(ReconstructionPopulation):
    """Bushy cells driven by ANF endbulbs of Held.

    Serves both bushy types: the globular cell (20 modified endbulbs, EM
    morphology) and the spherical cell (3 large axosomatic endbulbs, its own
    morphology and channel densities). They differ only in the constructor
    arguments, so main_reconstruct_sbc.py is a parameter set, not a class.
    """

    PER_POP_SYN = P.GBC_SYNAPSES
    N_POST_TOTAL = N_GBC_TOTAL

    def __init__(self, n_syn_per_pop=None, axon_target=None,
                 n_post_total=N_GBC_TOTAL, n_endbulbs=N_ENDBULBS,
                 endbulb_weights=None, per_pop_syn=None, **kwargs):
        self.axon_target = axon_target        # unit vector in model coords, or None
        self.N_POST_TOTAL = n_post_total
        self.DEFAULT_N_SRC = n_endbulbs
        self.n_endbulbs = n_endbulbs
        self.endbulb_weights = endbulb_weights   # compartment split; None = GBC default
        super().__init__(n_syn_per_pop=n_syn_per_pop, per_pop_syn=per_pop_syn, **kwargs)

    def set_rotations(self):
        """Give every cell the same rotation, aligning its axon to axon_target.

        Replaces hybridLFPy's random per-cell rotation. The GBC axons all cross
        the midline in one direction, so their axial currents have to summate
        rather than average away; a random spin would cancel the dipole this
        population exists to produce. With axon_target=None the parent
        behaviour (identity) is restored.
        """
        from time import time
        if self.axon_target is None:
            return super().set_rotations()

        tic = time()
        if RANK == 0:
            trial_params = {k: v for k, v in self.cellParams.items()
                            if k not in ('custom_fun', 'custom_fun_args')}
            cell = LFPy.Cell(**trial_params)
            native = gbc_biophysics.native_axon_direction(cell)
            if native is None:
                print('[AVCN] no axon compartment found; skipping axon alignment')
                rot = {}
            else:
                rot = gbc_biophysics.lfpy_align_angles(native, self.axon_target)
                cell.set_rotation(**rot)
                achieved = gbc_biophysics.native_axon_direction(cell)
                print(f'[AVCN] axon aligned: native={np.round(native, 2)} -> '
                      f'{np.round(achieved, 2)} (target {np.round(self.axon_target, 2)})')
            rotations = [dict(rot) for _ in range(self.POPULATION_SIZE)]
            print('found cell rotations in %.2f s' % (time() - tic))
        else:
            rotations = None
        return COMM.bcast(rotations, root=0)

    def select_synapse_idx(self, cell, pop_type, idx, layer):
        """Endbulbs land on the soma and proximal dendrite, per endbulb_weights."""
        return gbc_biophysics.weighted_endbulb_idx(cell, len(idx),
                                                   weights=self.endbulb_weights)

    def draw_rand_pos(self, radius_x=ELLIPSE_RADIUS_X, radius_y=ELLIPSE_RADIUS_Y,
                      z_min=0.0, z_max=0.0, min_cell_interdist=1.0, **kwargs):
        """Fill the AVCN elliptic cylinder, ordered tonotopically along x."""
        return self.rejection_sample_ellipse(
            extents={'x': (-radius_x, radius_x), 'y': (-radius_y, radius_y),
                     'z': (z_min, z_max)},
            sample_order=('x', 'y', 'z'), ellipse_axes=('x', 'y'),
            min_cell_interdist=min_cell_interdist, sort_axis='x')



# ---------------------------------------------------------------------------
# Split GBC generator: three grouped dipoles instead of one lumped whole-cell
# one. See recon_core/tree_dipole.py for why grouping EDGES is valid where
# grouping SEGMENTS is not.
# ---------------------------------------------------------------------------
class GroupedDipoleMixin:
    """Write per-compartment-group dipoles into self.output.

    hybridLFPy keys probe output by CLASS NAME and computes it as `M @ imem`
    (population.py:367-370, 1283-1320) -- a form no edge-grouped decomposition
    can take, since M multiplies segment currents by positions. So cellsim is
    overridden.

    ⚠ This is a FORK of hybridLFPy.Population.cellsim (population.py:1244-1326).
    Two parity details matter and are kept deliberately: the `ss.decimate` call
    runs even at q=1 (a Chebyshev-I filter the existing records have been
    through) and the float32 cast. Dropping either makes "sum of groups == the
    old lumped record" fail at the filter level rather than at float precision.
    """

    #: {output key: tuple of compartment classes}
    GROUPS = {}

    #: Blank this much of the start of every cell's imem, before anything else.
    #: NEURON's finitialize leaves the membrane off equilibrium, so the first
    #: timestep carries ~0.42 nA of UNBALANCED charge. That breaks the premise
    #: of the edge decomposition, and in `pos.T @ imem` it is multiplied by the
    #: cell's displacement from the origin (mean 324 um in the AVCN), producing
    #: a spurious 515 nA.um artefact LARGER than the 340 nA.um real signal --
    #: which `ss.decimate`'s IIR filter then smears forward over ~10 ms.
    #: Blanking here, per cell and before decimation, removes it at source.
    #: The window is DERIVED: no synaptically driven current can precede the
    #: shortest endbulb delay, so everything before it is provably non-neural.
    SETTLE_MS = min(P.GBC_DELAYS[0], P.SBC_DELAYS[0])

    def _tree(self, cell):
        """(parents, order, classes, pos_mid), recomputed per cell.

        NEURON sections are rebuilt for every cell, so the cached arrays would
        dangle; measured at 3 ms against a ~1 s simulation, so not worth caching.
        """
        parents = tree_dipole.segment_parents(cell)
        order = tree_dipole.topological_order(parents)
        classes = gbc_biophysics.segment_compartment_classes(cell)
        pos_mid = np.c_[cell.x.mean(-1), cell.y.mean(-1), cell.z.mean(-1)]
        return parents, order, classes, pos_mid

    def cellsim(self, cellindex, return_just_cell=False):
        import scipy.signal as ss
        cell = LFPy.Cell(**self.cellParams)
        cell.set_pos(**self.pop_soma_pos[cellindex])
        cell.set_rotation(**self.rotations[cellindex])
        if return_just_cell:
            return cell

        self.insert_all_synapses(cellindex, cell)
        for probe in self.probes:
            probe.cell = cell
        cell.simulate(**self.simulationParams)          # rec_imem=True
        cell.imem[:, :int(round(self.SETTLE_MS / cell.dt))] = 0.0

        # Any probes are still honoured, so a plain CurrentDipoleMoment can be
        # passed alongside as the whole-cell reference: summing the groups must
        # reproduce it exactly. That identity is the gate on this class.
        for probe in self.probes:
            probe.data = probe.get_transformation_matrix() @ cell.imem
            probe.data = ss.decimate(probe.data,
                                     q=self.decimatefrac).astype(np.float32)
            probe.cell = None
            self.output[cellindex][probe.__class__.__name__] = probe.data.copy()

        parents, order, classes, pos_mid = self._tree(cell)
        moments, centroids, weights = tree_dipole.group_dipoles(
            cell.imem, pos_mid, parents, classes, self.GROUPS, order)

        for key in self.GROUPS:
            self.output[cellindex][key] = ss.decimate(
                moments[key], q=self.decimatefrac).astype(np.float32)
            # weighted so the population centroid is a weighted mean after
            # summing; divided by the summed weight on rank 0.
            self.output[cellindex][key + '__c'] = (
                centroids[key] * weights[key]).astype(np.float64)
            self.output[cellindex][key + '__w'] = np.float64(weights[key])

        for attrbt in self.savelist:
            attr = getattr(cell, attrbt)
            if isinstance(attr, np.ndarray):
                self.output[cellindex][attrbt] = attr.astype('float32')
            else:
                try:
                    self.output[cellindex][attrbt] = attr
                except BaseException:
                    self.output[cellindex][attrbt] = str(attr)
        self.output[cellindex]['srate'] = 1E3 / self.dt_output

        cell.__del__()


class GBCSynapticPopulation(GroupedDipoleMixin, AVCNPopulation):
    """Run A: ANF endbulbs, sodium ABSENT, so the dipole is purely postsynaptic.

    The canonical hybridLFPy treatment (Hagen 2016 drives passive cells). Known
    one-directional bias: with no AP there is no AHP and no spike-driven
    KHT/KLT, so nothing truncates the EPSP and this OVERESTIMATES the post-spike
    envelope. Stated, not testable away.
    """

    GROUPS = {'GBCsyn': gbc_biophysics.GBC_SYN_CLASSES}


class GBCSpikingPopulation(GroupedDipoleMixin, AVCNPopulation):
    """Run B: the cell's own NEST spike train fires its AIS; no ANF input."""

    PER_POP_SYN = P.GBC_SPIKING_SYNAPSES
    GROUPS = {'GBCspike': gbc_biophysics.GBC_SPIKE_CLASSES,
              'GBCtrunk': gbc_biophysics.GBC_TRUNK_CLASSES}

    def select_synapse_idx(self, cell, pop_type, idx, layer):
        """One suprathreshold synapse on the axon initial segment.

        ⚠ NOT cell.get_idx('Axon_Initial_Segment'): the baked extended-axon hoc
        names its sections `sections[N]`, so LFPy's section-name matching
        returns an EMPTY array and the synapse would silently land on segment 0.
        The compartment-class map is the only way in for this morphology.
        """
        segs = gbc_biophysics.seg_idx_for_classes(cell, ('initialsegment',))
        if len(segs) == 0:
            segs = gbc_biophysics.seg_idx_for_classes(cell, ('hillock',))
        if len(segs) == 0:
            segs = gbc_biophysics.seg_idx_for_classes(cell, ('soma',))
        if len(segs) == 0:
            return idx
        return np.random.choice(segs, size=len(idx),
                                replace=True).astype('int32')


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description='AVCN (GBC) LFP reconstruction')
    parser.add_argument('--pic-file', type=str, default=None, dest='pic_file')
    parser.add_argument('--angle',    type=int, default=0)
    parser.add_argument('--side',     type=str, default='L', choices=['L', 'R'])
    parser.add_argument('--n-cells',  type=int, default=N_CELLS, dest='n_cells')
    parser.add_argument('--n-single', type=int, default=5, dest='n_single')
    parser.add_argument('--hoc-file', type=str, default=HOC_FILE, dest='hoc_file',
                        help='GBC morphology hoc (default: VCN_c09 EM reconstruction)')
    args = parser.parse_args()

    side     = args.side
    pic_file = paths.resolve_pic(args.pic_file)

    from LFP_reconstruction.main_reconstruct import _extract_spikes

    meta = broadcast_from_root(
        lambda: _extract_spikes(args.angle, side, pic_file=pic_file))

    X_pops        = [f'ANF_{side}']
    k_yxl_local   = K_YXL
    j_yx_local    = [AVCNPopulation.PER_POP_SYN['ANF']['weight']]
    tau_yx_local  = [AVCNPopulation.PER_POP_SYN['ANF']['tau2']]

    stem       = paths.pic_stem(pic_file)
    spikes_dir = paths.spikes_dir_for(stem, args.angle, side)
    output_dir = paths.make_output_dirs(paths.output_dir_for(
        'lfp', stem, f'angle{args.angle}', side, prefix='avcn'))

    k_arr         = np.array(k_yxl_local)
    n_syn_per_pop = {X: int(k_arr[:, j].sum()) for j, X in enumerate(X_pops)}

    networkSim = hybridLFPy.CachedNetwork(
        simtime=TSTOP, dt=DT,
        spike_output_path=spikes_dir,
        label='spikes', ext='gdf',
        GIDs={X: [meta[X]['first_gid'], meta[X]['n_neurons']] for X in X_pops},
        X=X_pops,
    )
    probe = lfpykit_models.PointSourcePotential(
        cell=None, x=PROBE_X, y=PROBE_Y, z=PROBE_Z, sigma=SIGMA,
    )
    pop_label = f'AVCN_{side}'
    pop = AVCNPopulation(
        n_syn_per_pop=n_syn_per_pop,
        axon_target=AXON_TARGET,   # fixed ventromedial; laterality set by X_pops
        y=pop_label,
        cellParams={
            'morphology': args.hoc_file, 'passive': False, 'v_init': V_INIT,
            'dt': DT, 'tstart': 0., 'tstop': TSTOP,
            'nsegs_method': 'lambda_f', 'lambda_f': 100,
            'custom_fun': [_decorate], 'custom_fun_args': [{}],
        },
        rand_rot_axis=[],   # deterministic ventromedial axon orientation instead
        simulationParams={'rec_imem': True},
        populationParams={
            'number':   args.n_cells,
            'radius':   ELLIPSE_RADIUS_Y,
            'radius_x': ELLIPSE_RADIUS_X,
            'radius_y': ELLIPSE_RADIUS_Y,
            'z_min': 0.0, 'z_max': 0.0, 'min_cell_interdist': 1.0,
            'min_r': np.array([[0.], [0.]]),
        },
        layerBoundaries=LAYER_BOUNDARIES,
        probes=[probe],
        savelist=['somapos'],
        savefolder=output_dir,
        dt_output=DT,
        POPULATIONSEED=44,
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

    n_grab = min(args.n_single, len(pop.RANK_CELLINDICES))
    cell_indices = random.sample(list(pop.RANK_CELLINDICES), n_grab) if n_grab else []
    single_contribs = (np.stack([pop.output[i]['PointSourcePotential'] * 1e3
                                 for i in cell_indices], axis=0)
                       if cell_indices else None)
    soma_pos = (np.array([[pop.pop_soma_pos[i]['x'], pop.pop_soma_pos[i]['y'],
                           pop.pop_soma_pos[i]['z']] for i in cell_indices])
                if cell_indices else None)

    pop.collect_data()
    COMM.Barrier()

    postproc = hybridLFPy.PostProcess(
        y=[pop_label], dt_output=DT,
        mapping_Yy=[(pop_label, pop_label)],
        savelist=['somapos'], probes=[probe],
        savefolder=output_dir,
    )
    if RANK == 0:
        postproc.run()
    COMM.Barrier()

    if RANK == 0:
        figures.plot_all(output_dir, (PROBE_X, PROBE_Y, PROBE_Z), side, args.angle,
                         args.n_cells, FIGURE_STYLE,
                         stimulus_freq=meta.get('stim_freq_hz'),
                         single_contribs=single_contribs, soma_pos=soma_pos,
                         cell_gids=cell_indices, dt_ms=DT)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
