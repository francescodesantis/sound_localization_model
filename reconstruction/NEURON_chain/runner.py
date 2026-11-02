#!/usr/bin/env python3
"""One nucleus/side job of the NEURON chain: build, run, project, emit, measure.

The build-and-project loop is owned here (the ABR entry points build their
spike directory from the .pic and hardcode the population, rotation and
settling window); NUCLEI makes those explicit per nucleus.

Rules:
1. Every MPI collective is called on every rank; only the use of its result
   is restricted to rank 0 (reduce_sum returns None elsewhere).
2. The full-N gate runs before simulating; --no-emit allows small-N runs of
   the dipole path.
3. Delays come from NEURON_chain.params (imported as C), never from
   recon_core.config.params.

  mpiexec -n 4 python reconstruction/NEURON_chain/runner.py \
      --nucleus GBC --side L --n-cells 3600 --pic-file RESULTS/<f>.pic \
      --angle 0 --expect-ranks 4

Outputs under the chain record set:
  RESULTS/abr_tmp/dipoles/<stem>_<stimulus>_chain/<NUCLEUS>__<generator>__<side>__<condition>.h5
  RESULTS/abr_tmp/dipoles/<stem>_<stimulus>_chain/metrics__<NUCLEUS>__<side>.json
  RESULTS/lfp_tmp/chain_spikes_<stem>_angle<cond>_<side>/spikes-<label>-0.gdf
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import h5py

import hybridLFPy

from recon_core.physics import head_model
from recon_core.utils import paths
from recon_core.utils.io_utils import save_dipole_record
from recon_core.utils.mpi_utils import (COMM, RANK, SIZE, broadcast_from_root,
                                  reduce_sum)

# Chain delays; everything else falls through to recon_core.config.params.
from NEURON_chain import params as C
from NEURON_chain import instrument, spike_io
from NEURON_chain.instrument import StageTimer
from NEURON_chain.populations import (ChainGBCPopulation, ChainSBCPopulation,
                                      ChainMNTBPopulation, ChainMSOPopulation,
                                      ChainLSOPopulation)

# populations.py has already loaded the AVCN, MSO/LSO and MNTB mechanisms.
from LFP_reconstruction.main_reconstruct_avcn import (
    AVCNPopulation, AXON_TARGET, ELLIPSE_RADIUS_X, ELLIPSE_RADIUS_Y, K_YXL,
    LAYER_BOUNDARIES, _decorate)
from LFP_reconstruction.main_reconstruct import _extract_spikes
# SBC specifics (aliased: the GBC import above binds the same names).
from LFP_reconstruction.main_reconstruct_sbc import (
    HOC_FILE_SBC, _decorate_sbc, ENDBULB_WEIGHTS_SBC, PER_POP_SYN_SBC,
    K_YXL as K_YXL_SBC, N_ENDBULBS as N_ENDBULBS_SBC,
    ELLIPSE_RADIUS_Y as ELLIPSE_RADIUS_Y_SBC)
# MNTB specifics (aliased).
from LFP_reconstruction.main_reconstruct_mntb import (
    ELLIPSE_RADIUS_X as MNTB_RADIUS_X, ELLIPSE_RADIUS_Y as MNTB_RADIUS_Y,
    LAYER_BOUNDARIES as MNTB_LAYERS, HOC_FILE as MNTB_HOC)
from LFP_reconstruction.main_reconstruct_calyx import CalyxPopulation, CALYX_HOC
# MSO/LSO specifics (aliased).
from LFP_reconstruction.main_reconstruct import (
    HOC_FILE as MSO_HOC, ELLIPSE_RADIUS_X as MSO_RADIUS_X,
    ELLIPSE_RADIUS_Y as MSO_RADIUS_Y, LAYER_BOUNDARIES as MSO_LAYERS)
from LFP_reconstruction.main_reconstruct_lso import (
    HOC_STUB as LSO_HOC_STUB, ELLIPSE_RADIUS_X as LSO_RADIUS_X,
    ELLIPSE_RADIUS_Y as LSO_RADIUS_Y, LAYER_BOUNDARIES as LSO_LAYERS)
# left_ear/right_ear -> ipsilateral/contralateral for a given side.
from ABR_reconstruction.main_abr import _side_condition
from lfpykit import CurrentDipoleMoment

DT, TSTOP, SRATE = C.DT, C.TSTOP, C.SRATE

GBC_EXTENDED_HOC = os.path.join(paths.AVCN_MODELS_DIR, 'morphology', 'extended',
                                'VCN_c09_extended_axon.hoc')

#: Stage of each nucleus in the dependency graph (written into the metrics).
STAGE_OF = {'GBC': 1, 'SBC': 1, 'MNTB': 2, 'MSO': 3, 'LSO': 3}

#: Mechanisms every chain cell needs, checked at run time (the AVCN, MSO/LSO
#: and MNTB mechanisms are loaded into one process).
REQUIRED_MECHANISMS = ('klt', 'kht', 'ih', 'nacncoop',
                       'kltbc', 'khtbc', 'ihvcn', 'namntb')


def check_mechanisms():
    """Assert every mechanism the chain's cells insert is registered.

    A collision between two .mod directories sharing a SUFFIX does not
    necessarily raise -- `load_mechanisms` swallows the benign 'already exists'
    -- so the failure mode is a cell silently missing a conductance. This turns
    that into an exception before any cell is built.
    """
    from neuron import h

    available = set()
    mt = h.MechanismType(0)                     # 0 = density mechanisms
    name = h.ref('')
    for i in range(int(mt.count())):
        mt.select(i)
        mt.selected(name)
        available.add(name[0])
    missing = [m for m in REQUIRED_MECHANISMS if m not in available]
    if missing:
        raise RuntimeError(
            f'mechanisms missing after the combined import: {missing}. The '
            'AVCN, MSO/LSO and MNTB .mod directories share SUFFIXes and '
            'load_mechanisms swallows the benign duplicate-load error, so a '
            'collision shows up as an absent conductance rather than a crash.')
    return sorted(available & set(REQUIRED_MECHANISMS))


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Per-nucleus builders (the populations do not share a constructor signature)
# ---------------------------------------------------------------------------
def _build_gbc(side, n_cells, network_sim, output_dir, spec):
    """ChainGBCPopulation: endbulb-driven, one run, two edge groups.

    Identical to the `gbc_grouped` registry entry of main_abr_avcn.py except
    that (a) the class is the chain subclass, so a spike recorder is attached at
    the distal heminode, and (b) the delays come from NEURON_chain.params.
    """
    k_arr = np.array(K_YXL)
    x_pops = spec['input_pops'](side)
    n_syn_per_pop = {X: int(k_arr[:, j].sum()) for j, X in enumerate(x_pops)}

    pop = ChainGBCPopulation(
        n_syn_per_pop=n_syn_per_pop,
        axon_target=AXON_TARGET[side],      # atlas chord to the contra MNTB
        n_post_total=spec['n_total'],
        n_endbulbs=C.GBC_ENDBULBS,
        endbulb_weights=None,
        per_pop_syn=AVCNPopulation.PER_POP_SYN,
        y=f'GBC_{side}',
        cellParams={
            'morphology': GBC_EXTENDED_HOC, 'passive': False,
            'v_init': C.GBC_V_INIT, 'dt': DT, 'tstart': 0., 'tstop': TSTOP,
            'nsegs_method': 'lambda_f', 'lambda_f': 100,
            'custom_fun': [_decorate], 'custom_fun_args': [{}],
        },
        rand_rot_axis=[],       # deterministic ventromedial axon orientation
        simulationParams={'rec_imem': True},
        populationParams={
            'number':   n_cells,
            'radius':   ELLIPSE_RADIUS_Y,
            'radius_x': ELLIPSE_RADIUS_X,
            'radius_y': ELLIPSE_RADIUS_Y,
            'z_min': 0.0, 'z_max': 0.0, 'min_cell_interdist': 1.0,
            'min_r': np.array([[0.], [0.]]),
        },
        layerBoundaries=LAYER_BOUNDARIES,
        probes=[],                          # the groups replace probe output
        savelist=['somapos'],
        savefolder=output_dir,
        dt_output=DT,
        POPULATIONSEED=spec['seed'],
        X=x_pops,
        networkSim=network_sim,
        k_yXL=K_YXL,
        synParams={'section': 'allsec', 'syntype': 'Exp2Syn'},
        synDelayLoc=C.GBC_DELAYS,           # chain params, not recon_core
        synDelayScale=C.SYN_DELAY_SCALE,
        J_yX=[AVCNPopulation.PER_POP_SYN['ANF']['weight']],
        tau_yX=[AVCNPopulation.PER_POP_SYN['ANF']['tau2']],
    )
    return pop


class SBC(CurrentDipoleMoment):
    """CurrentDipoleMoment renamed 'SBC': hybridLFPy keys pop.output by the probe
    class name, which reduce_groups reads as the generator label.
    """


def _build_sbc(side, n_cells, network_sim, output_dir, spec):
    """ChainSBCPopulation: endbulb-driven, one whole-cell dipole recorded at the
    soma (the SBC axon is truncated). Configuration from main_reconstruct_sbc.
    """
    k_arr = np.array(K_YXL_SBC)
    x_pops = spec['input_pops'](side)
    n_syn_per_pop = {X: int(k_arr[:, j].sum()) for j, X in enumerate(x_pops)}

    pop = ChainSBCPopulation(
        n_syn_per_pop=n_syn_per_pop,
        axon_target=AXON_TARGET[side],      # unused by the truncated SBC axon,
                                             # passed for parity with main_re-
                                             # construct_sbc.py's own call
        n_post_total=spec['n_total'],
        n_endbulbs=N_ENDBULBS_SBC,
        endbulb_weights=ENDBULB_WEIGHTS_SBC,
        per_pop_syn=PER_POP_SYN_SBC,
        y=f'SBC_{side}',
        cellParams={
            'morphology': HOC_FILE_SBC, 'passive': False,
            'v_init': C.SBC_V_INIT, 'dt': DT, 'tstart': 0., 'tstop': TSTOP,
            'nsegs_method': 'lambda_f', 'lambda_f': 100,
            'custom_fun': [_decorate_sbc], 'custom_fun_args': [{}],
        },
        rand_rot_axis=[],       # deterministic ventromedial axon orientation
        simulationParams={'rec_imem': True},
        populationParams={
            'number':   n_cells,
            'radius':   ELLIPSE_RADIUS_Y_SBC,
            'radius_x': ELLIPSE_RADIUS_X,
            'radius_y': ELLIPSE_RADIUS_Y_SBC,
            'z_min': 0.0, 'z_max': 0.0, 'min_cell_interdist': 1.0,
            'min_r': np.array([[0.], [0.]]),
        },
        layerBoundaries=LAYER_BOUNDARIES,
        probes=[SBC(None)],                  # cell set by hybridLFPy in cellsim
        savelist=['somapos'],
        savefolder=output_dir,
        dt_output=DT,
        POPULATIONSEED=spec['seed'],
        X=x_pops,
        networkSim=network_sim,
        k_yXL=K_YXL_SBC,
        synParams={'section': 'allsec', 'syntype': 'Exp2Syn'},
        synDelayLoc=C.SBC_DELAYS,           # chain params, not recon_core
        synDelayScale=C.SYN_DELAY_SCALE,
        J_yX=[PER_POP_SYN_SBC['ANF']['weight']],
        tau_yX=[PER_POP_SYN_SBC['ANF']['tau2']],
    )
    return pop


class principal(CurrentDipoleMoment):
    """Renamed CurrentDipoleMoment (see SBC): generator 'MNTB:principal'."""


class calyx(CurrentDipoleMoment):
    """Renamed CurrentDipoleMoment (see SBC): generator 'MNTB:calyx'."""


def _build_mntb(side, n_cells, spikes_dir, meta, output_dir, spec):
    """MNTB principal cell and calyx, both driven by the chained contralateral GBC.

    Two populations keyed by generator name, each with its own CachedNetwork.
    The principal cell (ChainMNTBPopulation) emits MNTBC_{side} for the MSO/LSO
    stage; the calyx emits nothing. Configuration from
    main_reconstruct_{mntb,calyx}; delays MNTB_DELAYS and CALYX_DELAYS.
    """
    x_pops = spec['input_pops'](side)
    gids = {X: [meta[X]['first_gid'], meta[X]['n_neurons']] for X in x_pops}
    k_arr = np.array(C.MNTB_CONVERGENCE)
    n_syn_per_pop = {X: int(k_arr[:, j].sum()) for j, X in enumerate(x_pops)}

    def _net():
        return hybridLFPy.CachedNetwork(
            simtime=TSTOP, dt=DT, spike_output_path=spikes_dir,
            label='spikes', ext='gdf', GIDs=gids, X=x_pops)

    common = dict(
        n_syn_per_pop=n_syn_per_pop,
        # No random rotation: coherent mediolateral calyx dipoles, as in main_abr_mntb.
        rand_rot_axis=[],
        simulationParams={'rec_imem': True},
        populationParams={
            'number':   n_cells,
            'radius':   MNTB_RADIUS_Y,
            'radius_x': MNTB_RADIUS_X,
            'radius_y': MNTB_RADIUS_Y,
            'z_min': 0.0, 'z_max': 0.0, 'min_cell_interdist': 1.0,
            'min_r': np.array([[0.], [0.]]),
        },
        layerBoundaries=MNTB_LAYERS,
        savelist=['somapos'],
        savefolder=output_dir,
        dt_output=DT,
        X=x_pops,
        k_yXL=C.MNTB_CONVERGENCE,
        synParams={'section': 'allsec', 'syntype': 'Exp2Syn'},
        synDelayScale=[None],
    )

    principal_syn = ChainMNTBPopulation.PER_POP_SYN['GBC']
    principal_pop = ChainMNTBPopulation(
        y=f'MNTBprincipal_{side}',
        cellParams={'morphology': MNTB_HOC, 'passive': False,
                    'v_init': C.MNTB_V_INIT, 'dt': DT, 'tstart': 0.,
                    'tstop': TSTOP, 'nsegs_method': None},
        probes=[principal(None)],
        POPULATIONSEED=46,          # as main_abr_mntb.py's 'principal' entry
        networkSim=_net(),
        synDelayLoc=C.MNTB_DELAYS,           # chain params, not recon_core
        J_yX=[principal_syn['weight']],
        tau_yX=[principal_syn['tau2']],
        **common,
    )

    calyx_syn = CalyxPopulation.PER_POP_SYN['GBC']
    calyx_pop = CalyxPopulation(
        y=f'MNTBcalyx_{side}',
        cellParams={'morphology': CALYX_HOC, 'passive': False,
                    'v_init': C.MNTB_V_INIT, 'dt': DT, 'tstart': 0.,
                    'tstop': TSTOP, 'nsegs_method': None},
        probes=[calyx(None)],
        POPULATIONSEED=47,          # as main_abr_mntb.py's 'calyx' entry
        networkSim=_net(),
        synDelayLoc=C.CALYX_DELAYS,  # chain params: zero, driven at heminode
        J_yX=[calyx_syn['weight']],
        tau_yX=[calyx_syn['tau2']],
        **common,
    )

    return {'principal': principal_pop, 'calyx': calyx_pop}


class postsynaptic(CurrentDipoleMoment):
    """Renamed CurrentDipoleMoment (see SBC): generator 'MSO:postsynaptic'."""


class synaptic(CurrentDipoleMoment):
    """Renamed CurrentDipoleMoment (see SBC): generator 'LSO:synaptic'."""


class _MergedNetworkSim:
    """CachedNetwork-shaped object spanning several real ones.

    A CachedNetwork reads one spike directory, and only `.nodes[X]` / `.dbs[X]`
    are read downstream, so merging those dicts is sufficient. The MSO inputs
    live in three directories: this side's chain directory (SBC_ipsi, MNTBC),
    the other side's (SBC_contra) and the NEST cache (LNTBC).
    """

    def __init__(self):
        self.nodes, self.dbs, self.GIDs, self.X = {}, {}, {}, []
        self.simtime, self.dt = TSTOP, DT

    def add(self, net):
        for x in net.X:
            self.nodes[x] = net.nodes[x]
            self.dbs[x] = net.dbs[x]
            self.GIDs[x] = net.GIDs[x]
            self.X.append(x)


def _one_source_network(spikes_dir, x_pops, missing_hint):
    """One real CachedNetwork over ONE directory, with the metadata check
    resolve_input normally does, since MSO/LSO's builders below bypass it."""
    meta_path = os.path.join(spikes_dir, 'metadata.json')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f'{meta_path} is missing: {missing_hint}')
    meta = spike_io.read_metadata(spikes_dir)
    missing = [X for X in x_pops if X not in meta]
    if missing:
        raise KeyError(f'{missing} absent from the metadata in {spikes_dir}; '
                       'the input drive is incomplete')
    return hybridLFPy.CachedNetwork(
        simtime=TSTOP, dt=DT, spike_output_path=spikes_dir,
        label='spikes', ext='gdf',
        GIDs={X: [meta[X]['first_gid'], meta[X]['n_neurons']] for X in x_pops},
        X=x_pops)


def _mso_network_builder(stem, cond_val, side, pic_file):
    """Merge the MSO's three source directories (see _MergedNetworkSim)."""
    contra_side = 'R' if side == 'L' else 'L'

    own_dir = spike_io.chain_spikes_dir(stem, cond_val, side)
    net_own = _one_source_network(
        own_dir, [f'SBC_{side}', f'MNTBC_{side}'],
        f'run the chain SBC and MNTB stages for side {side} first')

    contra_dir = spike_io.chain_spikes_dir(stem, cond_val, contra_side)
    net_contra = _one_source_network(
        contra_dir, [f'SBC_{contra_side}'],
        f'run the chain SBC stage for side {contra_side} first')

    nest_dir = paths.spikes_dir_for(stem, cond_val, side)
    nest_meta = broadcast_from_root(
        lambda: _extract_spikes(cond_val, side, pic_file=pic_file))
    lntbc = f'LNTBC_{side}'
    if lntbc not in nest_meta:
        raise KeyError(f'{lntbc} absent from the NEST metadata in {nest_dir}')
    net_nest = hybridLFPy.CachedNetwork(
        simtime=TSTOP, dt=DT, spike_output_path=nest_dir,
        label='spikes', ext='gdf',
        GIDs={lntbc: [nest_meta[lntbc]['first_gid'], nest_meta[lntbc]['n_neurons']]},
        X=[lntbc])

    merged = _MergedNetworkSim()
    merged.add(net_own)
    merged.add(net_contra)
    merged.add(net_nest)
    return merged


def _lso_network_builder(stem, cond_val, side, pic_file):
    """Wrap the LSO's one source directory (both LSO inputs are ipsilateral)
    in the merged-network shape the Population expects."""
    own_dir = spike_io.chain_spikes_dir(stem, cond_val, side)
    net_own = _one_source_network(
        own_dir, [f'SBC_{side}', f'MNTBC_{side}'],
        f'run the chain SBC and MNTB stages for side {side} first')
    merged = _MergedNetworkSim()
    merged.add(net_own)
    return merged


def _build_mso(side, n_cells, network_sim, output_dir, spec):
    """ChainMSOPopulation: contra/ipsi SBC excitation on the two dendrites,
    MNTBC/LNTBC inhibition on the soma; one whole-cell dipole. Under a monaural
    condition the columns driven by the silent ear are zeroed, as in main_abr.
    """
    contra_side = 'R' if side == 'L' else 'L'
    x_pops = [f'SBC_{contra_side}', f'SBC_{side}', f'MNTBC_{side}',
              f'LNTBC_{side}']
    k_yxl_local = [row[:] for row in C.MSO_CONVERGENCE]
    side_cond = _side_condition(spec.get('_condition', 'binaural'), side)
    _SILENCED = {'ipsilateral': (0, 2), 'contralateral': (1, 3)}
    for col in _SILENCED.get(side_cond, ()):
        for row in k_yxl_local:
            row[col] = 0
    k_arr = np.array(k_yxl_local)
    n_syn_per_pop = {X: int(k_arr[:, j].sum()) for j, X in enumerate(x_pops)}

    pop = ChainMSOPopulation(
        n_syn_per_pop=n_syn_per_pop,
        y=f'MSO_{side}',
        cellParams={
            'morphology': MSO_HOC, 'passive': False, 'v_init': C.MSO_V_INIT,
            'dt': DT, 'tstart': 0., 'tstop': TSTOP, 'nsegs_method': None,
        },
        rand_rot_axis=['z'],
        simulationParams={'rec_imem': True},
        populationParams={
            'number':   n_cells,
            'radius':   MSO_RADIUS_Y,
            'radius_x': MSO_RADIUS_X,
            'radius_y': MSO_RADIUS_Y,
            'z_min': -100.0, 'z_max': 100.0, 'min_cell_interdist': 1.0,
            'min_r': np.array([[0.], [0.]]),
        },
        layerBoundaries=MSO_LAYERS,
        probes=[postsynaptic(None)],
        savelist=['somapos'],
        savefolder=output_dir,
        dt_output=DT,
        POPULATIONSEED=spec['seed'],
        X=x_pops,
        networkSim=network_sim,
        k_yXL=k_yxl_local,
        synParams={'section': 'allsec', 'syntype': 'Exp2Syn'},
        synDelayLoc=C.MSO_DELAYS,            # chain params, not recon_core
        synDelayScale=[None] * len(x_pops),
        J_yX=C.MSO_J_YX,
        tau_yX=C.MSO_TAU_YX,
    )
    return pop


def _build_lso(side, n_cells, network_sim, output_dir, spec):
    """ChainLSOPopulation: ipsi SBC excitation on the dendrites, ipsi MNTBC
    inhibition on the soma; one whole-cell dipole.
    """
    x_pops = [f'SBC_{side}', f'MNTBC_{side}']
    side_cond = _side_condition(spec.get('_condition', 'binaural'), side)
    k_sbc, k_mntbc = C.LSO_CONVERGENCE[0]
    if side_cond == 'ipsilateral':          # ipsi ear: keep SBC, drop MNTBC
        k_mntbc = 0
    elif side_cond == 'contralateral':      # contra ear: drop SBC, keep MNTBC
        k_sbc = 0
    k_yxl_local = [[k_sbc, k_mntbc]]
    k_arr = np.array(k_yxl_local)
    n_syn_per_pop = {X: int(k_arr[:, j].sum()) for j, X in enumerate(x_pops)}

    pop = ChainLSOPopulation(
        n_syn_per_pop=n_syn_per_pop,
        y=f'LSO_{side}',
        cellParams={
            'morphology': LSO_HOC_STUB[side], 'passive': False,
            'v_init': C.LSO_V_INIT, 'dt': DT, 'tstart': 0., 'tstop': TSTOP,
            'nsegs_method': None,
        },
        rand_rot_axis=[],
        simulationParams={'rec_imem': True},
        populationParams={
            'number':   n_cells,
            'radius':   LSO_RADIUS_Y,
            'radius_x': LSO_RADIUS_X,
            'radius_y': LSO_RADIUS_Y,
            'z_min': 0.0, 'z_max': 0.0, 'min_cell_interdist': 1.0,
            'min_r': np.array([[0.], [0.]]),
        },
        layerBoundaries=LSO_LAYERS,
        probes=[synaptic(None)],
        savelist=['somapos'],
        savefolder=output_dir,
        dt_output=DT,
        POPULATIONSEED=spec['seed'],
        X=x_pops,
        networkSim=network_sim,
        k_yXL=k_yxl_local,
        synParams={'section': 'allsec', 'syntype': 'Exp2Syn'},
        synDelayLoc=C.LSO_DELAYS,             # chain params, not recon_core
        synDelayScale=[None, None],
        J_yX=C.LSO_J_YX,
        tau_yX=C.LSO_TAU_YX,
    )
    return pop


#: Everything that differs between nuclei.
NUCLEI = {
    'GBC': dict(
        record_nucleus='AVCN',              # dipole records are filed as AVCN
        build=_build_gbc,
        n_total=C.N_GBC_TOTAL,
        seed=44,                            # as main_abr_avcn's gbc entries
        emits=True,
        emit_label='GBC_{side}',
        input_kind='anf',
        input_pops=lambda side: [f'ANF_{side}'],
        groups=tuple(ChainGBCPopulation.GROUPS),      # GBCsoma, GBCaxon
        rotation=C.ROTATION_AVCN,
        origin=C.AVCN_POS_UM,
        # Settling window: no synaptic current can precede the endbulb delay.
        settle_ms=min(C.GBC_DELAYS[0], C.SBC_DELAYS[0]),
    ),
    'SBC': dict(
        record_nucleus='AVCN',              # dipole records are filed as AVCN
        build=_build_sbc,
        n_total=C.N_SBC_TOTAL,              # 28,000 -- NOT 3,600, see module top
        seed=45,                            # as main_reconstruct_sbc.py / the
                                            # 'sbc' entry of main_abr_avcn.py
        emits=True,
        emit_label='SBC_{side}',
        input_kind='anf',
        input_pops=lambda side: [f'ANF_{side}'],
        groups=('SBC',),                    # one whole-cell dipole, no split
        # One whole-cell dipole at the SBC atlas position (no measured centroid).
        has_centroid=False,
        rotation=C.ROTATION_AVCN,
        origin=C.SBC_POS_UM,
        settle_ms=min(C.GBC_DELAYS[0], C.SBC_DELAYS[0]),
    ),
    'MNTB': dict(
        record_nucleus='MNTB',
        build=_build_mntb,
        multi=True,                        # two populations, not two edge-groups
        emit_generator='principal',        # the postsynaptic sink stands for MNTBC
        n_total=C.N_MNTB_TOTAL,
        emits=True,
        emit_label='MNTBC_{side}',
        input_kind='chain',
        # First chained input: side L reads the contralateral GBC_R.
        input_pops=lambda side: [f'GBC_{"R" if side == "L" else "L"}'],
        groups=('principal', 'calyx'),
        has_centroid=False,                # co-located at the fixed atlas MNTB
                                            # position, as main_abr_mntb.py
        rotation=C.ROTATION_MNTB,
        origin=C.MNTB_POS_UM,
        # Settling window: the GBC settling floor plus this generator's delay
        # (0.3 ms principal, 0 ms calyx).
        settle_ms=(min(C.GBC_DELAYS[0], C.SBC_DELAYS[0])
                  + min(C.MNTB_DELAYS[0], C.CALYX_DELAYS[0])),
    ),
    'MSO': dict(
        record_nucleus='MSO',
        build=_build_mso,
        n_total=C.N_MSO_TOTAL,
        seed=42,                            # as main_abr.py
        # Terminal stage: emits no spikes. The four inputs span three directories
        # (this side's chain dir, the other side's, the NEST cache), so
        # network_builder replaces resolve_input.
        input_kind='mixed',
        input_pops=lambda side: [f'SBC_{"R" if side == "L" else "L"}',
                                 f'SBC_{side}', f'MNTBC_{side}',
                                 f'LNTBC_{side}'],
        network_builder=_mso_network_builder,
        groups=('postsynaptic',),
        has_centroid=False,                 # fixed atlas MSO position
        rotation=C.ROTATION_MSO,
        origin=C.MSO_POS_UM,
        # Settling window: the earliest arrival over the active inputs
        # (SBC contra/ipsi, MNTBC; LNTBC has weight 0).
        settle_ms=min(
            min(C.GBC_DELAYS[0], C.SBC_DELAYS[0]) + C.MSO_DELAYS[0],
            min(C.GBC_DELAYS[0], C.SBC_DELAYS[0]) + C.MSO_DELAYS[1],
            (min(C.GBC_DELAYS[0], C.SBC_DELAYS[0])
             + min(C.MNTB_DELAYS[0], C.CALYX_DELAYS[0])) + C.MSO_DELAYS[2],
        ),
    ),
    'LSO': dict(
        record_nucleus='LSO',
        build=_build_lso,
        n_total=C.N_LSO_TOTAL,
        seed=43,                            # as main_abr_lso.py
        # Terminal stage, as the MSO.
        input_kind='mixed',
        input_pops=lambda side: [f'SBC_{side}', f'MNTBC_{side}'],
        network_builder=_lso_network_builder,
        groups=('synaptic',),
        has_centroid=False,                 # fixed atlas LSO position
        rotation=C.ROTATION_LSO,
        origin=C.LSO_POS_UM,
        # Settling window: earliest arrival of SBC_ipsi and MNTBC_ipsi.
        settle_ms=min(
            min(C.GBC_DELAYS[0], C.SBC_DELAYS[0]) + C.LSO_DELAYS[0],
            (min(C.GBC_DELAYS[0], C.SBC_DELAYS[0])
             + min(C.MNTB_DELAYS[0], C.CALYX_DELAYS[0])) + C.LSO_DELAYS[1],
        ),
    ),
}


# ---------------------------------------------------------------------------
# Input resolution
# ---------------------------------------------------------------------------
def resolve_input(spec, stem, cond_val, side, pic_file):
    """(spikes_dir, metadata) of this nucleus's presynaptic drive.

    'anf': the NEST extraction cache (rank 0 builds it from the .pic).
    'chain': NEURON_chain's own directory, keyed by the input population's side
    (the MNTB on side L reads GBC_R from the R directory).
    """
    if spec['input_kind'] == 'anf':
        spikes_dir = paths.spikes_dir_for(stem, cond_val, side)
        # Rank 0 alone may touch the .pic and write the cache; the others take
        # the metadata by broadcast. This is a collective: all ranks call it.
        meta = broadcast_from_root(
            lambda: _extract_spikes(cond_val, side, pic_file=pic_file))
        return spikes_dir, meta

    x_pops = spec['input_pops'](side)
    input_sides = {X.rsplit('_', 1)[-1] for X in x_pops}
    if len(input_sides) != 1:
        raise ValueError(
            f'{x_pops}: a chained nucleus must read every input from ONE '
            f'side\'s chain directory; got sides {sorted(input_sides)}')
    input_side = input_sides.pop()
    spikes_dir = spike_io.chain_spikes_dir(stem, cond_val, input_side)
    meta_path = os.path.join(spikes_dir, 'metadata.json')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f'{meta_path} is missing: the upstream chain stage has not run. '
            'A chained nucleus reads NEURON_chain\'s own GDF directory, never '
            'the NEST extraction cache.')
    return spikes_dir, spike_io.read_metadata(spikes_dir)


def assert_full_n(spec, side, n_cells, no_emit):
    """Refuse a partial-N EMITTING run before any cell is built.

    `spike_io.write_gdf` refuses it too, but after the simulation. The reason is
    not bookkeeping: the unsimulated cells would be written as silent, and
    downstream that is indistinguishable from genuine silence.
    """
    if no_emit or not spec.get('emits'):
        return
    n_total = spec['n_total']
    if n_cells != n_total:
        label = spec['emit_label'].format(side=side)
        raise ValueError(
            f'{label}: simulated {n_cells} of {n_total} cells. A partial-N run '
            'must not emit spikes: the missing cells would be written as '
            'silent, and downstream that is indistinguishable from genuine '
            'silence. Re-run at full N, or pass --no-emit to exercise the '
            'dipole path alone.')


# ---------------------------------------------------------------------------
# Reduction and projection
# ---------------------------------------------------------------------------
def reduce_groups(pop, groups, timer, has_centroid=True):
    """Sum each group's dipole across ranks, and its weighted centroid when the
    population measures one (has_centroid: the GBC edge groups); otherwise the
    offset is zero and the dipole sits at the atlas position. Only rank 0 uses
    the results.
    """
    import time

    n_t = int(round(TSTOP / DT)) + 1
    moments, centroids = {}, {}
    for g in groups:
        loc_p = loc_c = None
        loc_w = 0.0
        for i in pop.RANK_CELLINDICES:
            d = pop.output[i][g].astype(np.float64)
            loc_p = d.copy() if loc_p is None else loc_p + d
            if has_centroid:
                c = pop.output[i][g + '__c']
                loc_c = c.copy() if loc_c is None else loc_c + c
                loc_w += float(pop.output[i][g + '__w'])
        if loc_p is None:
            loc_p = np.zeros((3, n_t))
        if loc_c is None:
            loc_c = np.zeros(3)

        t0 = time.perf_counter()
        moments[g] = reduce_sum(loc_p)
        if has_centroid:
            tot_c = reduce_sum(loc_c)
            tot_w = reduce_sum(np.array([loc_w]))
        timer.record_collective(time.perf_counter() - t0)

        centroids[g] = None
        if RANK == 0:
            centroids[g] = ((tot_c / tot_w[0]) if (has_centroid and tot_w[0] > 0)
                            else np.zeros(3))
    COMM.Barrier()
    return moments, centroids


def project_and_save(spec, side, moments, centroids, output_dir):
    """Blank the transient, rotate into the head frame, write population_dipole.h5.

    Rank 0 only -- `moments` is None elsewhere. The SAME rotation matrix is
    applied to the moment and to its centroid, which is the whole reason
    rotate_to_head and rotate_offset_to_head sit beside each other.
    """
    n_settle = int(round(spec['settle_ms'] / DT))
    rotation, origin = spec['rotation'][side], spec['origin'][side]
    p_heads, r_heads = {}, {}
    with h5py.File(os.path.join(output_dir, 'population_dipole.h5'), 'w') as f:
        for g, p_model in moments.items():
            p_model = np.asarray(p_model, dtype=float).copy()
            p_model[:, :n_settle] = 0.0
            p_head = head_model.rotate_to_head(p_model, rotation)
            r_head = head_model.rotate_offset_to_head(centroids[g], rotation,
                                                      origin)
            f.create_dataset(g, data=p_head)
            f.create_dataset(f'r__{g}', data=r_head)
            p_heads[g], r_heads[g] = p_head, r_head
        f.create_dataset('srate', data=float(SRATE))
        f.attrs['axes'] = 'x=mediolateral, y=anteroposterior, z=inferosuperior'
        f.attrs['units'] = 'nA·µm'
        f.attrs['parts'] = ','.join(p_heads)
        f.attrs['variant'] = C.CHAIN_VARIANT
    return p_heads, r_heads


def emit(pop, spec, side, spikes_dir, timer):
    """Publish this population's spikes as GDF and update the metadata.

    `emit_spikes` is called on every rank, unconditionally: it gathers before
    any rank branch, so calling it under `if RANK == 0` would hang the others at
    the next collective. Only its RETURN VALUE is rank-0 material.
    """
    import time

    label = spec['emit_label'].format(side=side)
    pop.EMIT_LABEL = label
    t0 = time.perf_counter()
    meta = pop.emit_spikes(spikes_dir, spec['n_total'])      # every rank
    elapsed = time.perf_counter() - t0
    if RANK != 0:
        return None

    gdf = os.path.join(spikes_dir, f'spikes-{label}-0.gdf')
    timer.record_gdf_io(os.path.getsize(gdf), elapsed)
    # Merge under a lock: jobs of one stage share this directory and run
    # concurrently (spike_io.metadata_lock).
    with spike_io.metadata_lock(spikes_dir):
        existing = {}
        if os.path.exists(os.path.join(spikes_dir, 'metadata.json')):
            existing = spike_io.read_metadata(spikes_dir)
        existing[label] = meta
        spike_io.write_metadata(spikes_dir, existing)
    print(f'[chain] emitted {meta["n_spikes"]} spikes for {label} -> {gdf}')
    return meta


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------
def run_stage(nucleus, side, args):
    """Build, run, project, emit and measure ONE nucleus/side stage.

    Returns a dict on rank 0 (record paths, the metrics path and the emitted
    metadata) and None elsewhere, following reduce_sum's contract.
    """
    if nucleus not in NUCLEI:
        raise ValueError(
            f'{nucleus} is not registered in runner.NUCLEI (have '
            f'{sorted(NUCLEI)}). Register it rather than approximating it with '
            'another nucleus\'s builder -- the populations do not share a '
            'constructor signature.')
    spec = NUCLEI[nucleus]

    pic_file = paths.resolve_pic(args.pic_file)
    stem = paths.pic_stem(pic_file)
    cond_val, cond_label = paths.condition_key(args.angle, args.itd_us,
                                               args.ild_db)
    n_cells = args.n_cells

    # Cheap gates first, and the full-N one BEFORE any NEURON work.
    assert_full_n(spec, side, n_cells, args.no_emit)
    if args.expect_ranks is not None and SIZE != args.expect_ranks:
        raise RuntimeError(f'running on {SIZE} ranks but --expect-ranks '
                           f'{args.expect_ranks} was given')
    mechs = check_mechanisms()
    if RANK == 0:
        print(f'[chain] mechanisms present: {mechs}')

    dipoles_dir = paths.dipoles_dir_for(stem, cond_label,
                                        variant=C.CHAIN_VARIANT)
    output_dir = paths.make_output_dirs(
        paths.output_dir_for('abr', stem, cond_label, side,
                            prefix=f'chain_{nucleus.lower()}',
                            suffix=('' if getattr(args, 'condition', 'binaural')
                                    == 'binaural'
                                    else f'_{args.condition}')),
        subdirs=('figures',))

    # One population reduced into groups, or (MNTB, spec['multi']) a dict of
    # populations, one per generator.
    multi = spec.get('multi', False)

    # Read by _build_mso/_build_lso.
    spec['_condition'] = getattr(args, 'condition', 'binaural')

    with StageTimer(nucleus, side, stage=STAGE_OF[nucleus]) as timer:
        with timer.phase('setup'):
            drive_desc = None
            if multi:
                spikes_dir, meta = resolve_input(spec, stem, cond_val, side,
                                                args.pic_file)
                x_pops = spec['input_pops'](side)
                missing = [X for X in x_pops if X not in meta]
                if missing:
                    raise KeyError(f'{missing} absent from the metadata in '
                                   f'{spikes_dir}; the input drive is incomplete')
                # Each generator builds its own CachedNetwork (see _build_mntb).
                pops = spec['build'](side, n_cells, spikes_dir, meta,
                                     output_dir, spec)
                timer.record_cells(
                    sum(len(p.RANK_CELLINDICES) for p in pops.values()))
                drive_desc = f'{x_pops} from {spikes_dir}'
            elif spec.get('network_builder'):
                # MSO/LSO: inputs span several directories; the builder checks them.
                x_pops = spec['input_pops'](side)
                network_sim = spec['network_builder'](stem, cond_val, side,
                                                      args.pic_file)
                pop = spec['build'](side, n_cells, network_sim, output_dir,
                                    spec)
                timer.record_cells(len(pop.RANK_CELLINDICES))
                drive_desc = f'{x_pops} (merged multi-directory network)'
            else:
                spikes_dir, meta = resolve_input(spec, stem, cond_val, side,
                                                args.pic_file)
                x_pops = spec['input_pops'](side)
                missing = [X for X in x_pops if X not in meta]
                if missing:
                    raise KeyError(f'{missing} absent from the metadata in '
                                   f'{spikes_dir}; the input drive is incomplete')
                network_sim = hybridLFPy.CachedNetwork(
                    simtime=TSTOP, dt=DT, spike_output_path=spikes_dir,
                    label='spikes', ext='gdf',
                    GIDs={X: [meta[X]['first_gid'], meta[X]['n_neurons']]
                          for X in x_pops},
                    X=x_pops)
                pop = spec['build'](side, n_cells, network_sim, output_dir,
                                    spec)
                timer.record_cells(len(pop.RANK_CELLINDICES))
                drive_desc = f'{x_pops} from {spikes_dir}'
            if RANK == 0:
                print(f'[chain] {nucleus}_{side}: {n_cells} cells over {SIZE} '
                      f'ranks, drive {drive_desc}')

        with timer.phase('simulate'):
            if multi:
                for p in pops.values():
                    p.run()
            else:
                pop.run()
            COMM.Barrier()

        with timer.phase('reduce'):
            if multi:
                moments, centroids = {}, {}
                for g in spec['groups']:
                    m, c = reduce_groups(
                        pops[g], (g,), timer,
                        has_centroid=spec.get('has_centroid', True))
                    moments.update(m)
                    centroids.update(c)
            else:
                moments, centroids = reduce_groups(
                    pop, spec['groups'], timer,
                    has_centroid=spec.get('has_centroid', True))

        result = None
        with timer.phase('project'):
            if RANK == 0:
                p_heads, r_heads = project_and_save(spec, side, moments,
                                                    centroids, output_dir)
                # 'binaural' for GBC, SBC, MNTB; MSO/LSO take --condition.
                records = [
                    save_dipole_record(stem, cond_label, spec['record_nucleus'],
                                       g, side, p_heads[g], r_heads[g],
                                       spec['n_total'], n_cells, SRATE,
                                       condition=spec['_condition'],
                                       directory=dipoles_dir)
                    for g in spec['groups']]
                result = {'records': records, 'output_dir': output_dir,
                          'dipoles_dir': dipoles_dir}

        emitted = None
        if spec.get('emits') and not args.no_emit:
            with timer.phase('emit'):
                # Always the chain's own directory: for ANF-driven nuclei spikes_dir
                # is the NEST cache, whose GBC_/SBC_ files have the same names.
                emit_dir = spike_io.chain_spikes_dir(stem, cond_val, side)
                # Only the postsynaptic population emits (MNTB: 'principal'); the
                # calyx has no downstream consumer.
                emit_pop = (pops[spec['emit_generator']] if multi else pop)
                emitted = emit(emit_pop, spec, side, emit_dir, timer)

    # StageTimer.write is a collective too: every rank calls it, only rank 0
    # gets a path back.
    metrics_path = timer.write(instrument.metrics_path_for(dipoles_dir,
                                                           nucleus, side,
                                                           spec['_condition']))
    if RANK != 0:
        return None

    data = instrument.load_metrics(metrics_path)
    missing = instrument.missing_metrics(data)
    if missing:
        raise RuntimeError(f'metrics record {metrics_path} is incomplete: '
                           f'{missing}')
    # Check n_ranks against what mpiexec was asked for.
    if args.expect_ranks is not None and data['n_ranks'] != args.expect_ranks:
        raise AssertionError(f'metrics n_ranks={data["n_ranks"]} but '
                             f'--expect-ranks {args.expect_ranks}')
    print(f'[chain] {nucleus}_{side} wall {data["wall_s"]:.1f} s over '
          f'{data["n_ranks"]} ranks; metrics -> {metrics_path}')
    result = result or {}
    result.update(metrics=metrics_path, emitted=emitted)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='one NEURON-chain nucleus/side stage')
    parser.add_argument('--nucleus', required=True, choices=sorted(STAGE_OF))
    parser.add_argument('--side', default='L', choices=['L', 'R', 'both'])
    parser.add_argument('--n-cells', type=int, required=True, dest='n_cells')
    parser.add_argument('--pic-file', default=None, dest='pic_file')
    parser.add_argument('--angle', type=int, default=0)
    parser.add_argument('--itd-us', type=float, default=None, dest='itd_us')
    parser.add_argument('--ild-db', type=float, default=None, dest='ild_db')
    parser.add_argument('--condition', default='binaural',
                        choices=['binaural', 'left_ear', 'right_ear'],
                        help='Acoustic condition. Only the MSO and LSO read '
                             'this (post-hoc column silencing, mirroring '
                             'main_abr.py/main_abr_lso.py); every other '
                             'nucleus ignores it and always records binaural.')
    parser.add_argument('--no-emit', action='store_true', dest='no_emit',
                        help='skip the GDF publication, which is what lets the '
                             'dipole path be smoke-tested at partial N. A run '
                             'WITH emission is refused below full N.')
    parser.add_argument('--expect-ranks', type=int, default=None,
                        dest='expect_ranks',
                        help='assert this many MPI ranks, and assert the '
                             'metrics record agrees. The only place n_ranks '
                             'has real ground truth.')
    args = parser.parse_args()

    sides = ['L', 'R'] if args.side == 'both' else [args.side]
    for side in sides:
        run_stage(args.nucleus, side, args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
