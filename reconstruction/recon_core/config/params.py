"""
Reconstruction parameters for the LFP and ABR pipelines.

Two kinds of value live here:

1. Mirrored from the NEST network: population sizes, convergences, synaptic
   weights and delays, reversal and resting potentials, synaptic time
   constants. They are read live from simulate/models/BrainstemModel/params.py
   through recon_core.config.brainstem and are never retyped here.
2. Reconstruction only: NEURON timestep, morphology geometry, probe layout,
   ABR filter band, conduction delays the point-neuron network cannot carry.

Synaptic conductances are the NEST weights converted to µS,
weight_uS = |NEST weight in nS| * 1e-3. The magnitude is taken because NEST
signs a synapse by its weight and NEURON by its reversal potential. A pathway
with weight 0 in NEST is off here too.

Head frame (µm): x dorsoventral, y rostrocaudal, z mediolateral. The geometry
lives in recon_core.config.head_geometry and is re-exported at the end.
"""

import numpy as np

from recon_core.config import brainstem
from recon_core.utils.signal_utils import srate_from_dt

# Live NEST parameter set. Attribute access mirrors params.py exactly:
# BRAINSTEM.POP_NUM.n_MSOs, BRAINSTEM.E_L.MSO, BRAINSTEM.POP_CONV.SBCs2MSOs
BRAINSTEM = brainstem.parameters()


# ===========================================================================
# Synaptic conductances
# ===========================================================================
NS_TO_US = 1e-3


def syn_weight(pathway):
    """NEURON conductance in µS for one pathway, from the live NEST weight
    (0.0 when the pathway is off in NEST)."""
    nest_ns = getattr(BRAINSTEM.SYN_WEIGHTS, pathway.replace('2calyx', '2MNTBCs')
                      if pathway == 'GBCs2calyx' else pathway)
    return abs(float(nest_ns)) * NS_TO_US


# ===========================================================================
# Simulation (reconstruction-only)
# ===========================================================================
# 0.026 ms (38.5 kHz) resolves the 3 kHz ABR band and sub-ms wave latencies.
DT = 0.026        # ms   NEURON timestep, shared by every pipeline
# Every dipole record is TSTOP long and the composite sums them on one time
# axis: changing TSTOP requires regenerating every record. 30 ms leaves ~20 ms
# after the ~7 ms settling of the 150 Hz high-pass.
TSTOP = 30.0      # ms   simulated duration
SRATE = srate_from_dt(DT)   # Hz

# Settling window discarded from the LSO dipole: the active cell starts out of
# equilibrium.
SETTLE_MS = 3.0

# NEURON global temperature. The AVCN (cnmodel), MSO/LSO and MNTB mechanisms
# all scale their gating rates with celsius (Q10, see each .mod file).
BODY_TEMPERATURE_C = 37.0


# ===========================================================================
# Scalp recording
# ===========================================================================
ELECTRODES = ('Cz', 'M1', 'M2')       # vertex and both mastoids
DERIVATIONS = ('Cz-M1', 'Cz-M2', 'Cz-avg', 'FPz-M1', 'FPz-M2')

# Scalp band-pass applied to every ABR output: clinical BAEP band, 4th-order Butterworth.
BAND_DEFAULT = (150., 3000.)    # Hz


# ===========================================================================
# Probe (near-field LFP)
# ===========================================================================
N_CH = 16                 # linear probe channels, spanning z
SIGMA_EXTRACELLULAR = 0.3   # S/m


def probe_z(half_span_um):
    """Channel depths for a linear probe spanning ±`half_span_um` along z."""
    return np.linspace(-half_span_um, half_span_um, N_CH)


def probe_positions(half_span_um, axis='z'):
    """(x, y, z) channel coordinates for a linear probe along `axis`.

    The probe runs along the dendritic axis, the direction the current dipole
    points, so it sees the sink/source pair as a depth profile. Perpendicular
    to it the probe would sample the tonotopic gradient and little dipole.
    """
    span = np.linspace(-half_span_um, half_span_um, N_CH)
    zero = np.zeros(N_CH)
    return {'x': (span, zero, zero),
            'y': (zero, span, zero),
            'z': (zero, zero, span)}[axis]


# ===========================================================================
# MSO, medial superior olive (binaural coincidence detector)
# ===========================================================================
N_MSO_TOTAL = BRAINSTEM.POP_NUM.n_MSOs
# Two values differ from the NEST point neuron, for the MSO cell
# (models/mso/mso_model.hoc, KLT + Ih, no Na):
#  * rest -60 mV (NEST E_L -51): MSO principal cells rest at -60 mV, held by Ih
#    against KLT (Magnusson et al. 2005); the cell's conductances are fitted to it.
#  * glycinergic reversal -90 mV (NEST -70), from MSO whole-cell data.
MSO_V_INIT = -60.0
MSO_INH_REV = -90.0

# Elliptic cylinder insertion volume.
MSO_RADIUS_X = 255.0     # µm  tonotopic half-axis
MSO_RADIUS_Y = 2845.0    # µm  rostrocaudal half-axis
MSO_PROBE_HALF_SPAN = 400.0   # µm

# Bipolar morphology: contra SBC on the medial dendrite, ipsi SBC on the
# lateral one, inhibition on the soma (Cant & Hyson 1992; Joris 1998).
MSO_LAYERS = [
    [10., 160.],     # medial dendrite
    [-160., -10.],   # lateral dendrite
    [-10., 10.],     # soma
]

# Input band. The MSO is restricted to low CFs, mapped onto its tonotopic
# index through the ERB scale (see tonotopic_index).
MSO_FREQ_MIN = 200.0     # Hz
MSO_FREQ_MAX = 4000.0    # Hz

MSO_SYNAPSES = {
    'SBC': {
        'syntype': 'Exp2Syn',
        'tau1': BRAINSTEM.TAUS_EX_RISE.MSO,
        'tau2': BRAINSTEM.TAUS_EX_DECAY.MSO,
        'e': BRAINSTEM.EXC_REV.MSO,
        'weight': syn_weight('SBCs2MSO'),
    },
    'MNTBC': {
        'syntype': 'Exp2Syn',
        'tau1': BRAINSTEM.TAUS_IN_RISE.MSO,
        'tau2': BRAINSTEM.TAUS_IN_DECAY.MSO,
        'e': MSO_INH_REV,                    # not NEST's -70: see MSO_INH_REV
        'weight': syn_weight('MNTBCs2MSO'),
    },
    'LNTBC': {
        'syntype': 'Exp2Syn',
        'tau1': BRAINSTEM.TAUS_IN_RISE.MSO,
        'tau2': BRAINSTEM.TAUS_IN_DECAY.MSO,
        'e': MSO_INH_REV,
        'weight': syn_weight('LNTBCs2MSO'),       # 0 in the NEST model
    },
}

# Synapses per presynaptic population per layer, matching MSO_LAYERS.
# Column order: SBC_contra, SBC_ipsi, MNTBC_ipsi, LNTBC_ipsi.
# One table for both pipelines, all counts from POP_CONV.
MSO_CONVERGENCE = [
    [BRAINSTEM.POP_CONV.SBCs2MSOs, 0, 0, 0],    # medial dendrite, contra SBC
    [0, BRAINSTEM.POP_CONV.SBCs2MSOs, 0, 0],    # lateral dendrite, ipsi SBC
    [0, 0, BRAINSTEM.POP_CONV.MNTBCs2MSOs,      # soma, inhibition
     BRAINSTEM.POP_CONV.LNTBCs2MSOs],
]

# ANF_CN_CONDUCTION_MS is not added: MSO and LSO use the NEST delays as they are.
MSO_DELAYS = [BRAINSTEM.SYN_DELAYS.SBCs2MSOcontra,
              BRAINSTEM.SYN_DELAYS.SBCs2MSOipsi,
              BRAINSTEM.SYN_DELAYS.MNTBCs2MSO,
              BRAINSTEM.SYN_DELAYS.LNTBCs2MSO]

# hybridLFPy bookkeeping. J_yX and tau_yX go into the population metadata but
# never reach the cell: insert_all_synapses overrides both from the *_SYNAPSES
# tables above. Derived here so the metadata matches what was simulated.
def _bookkeeping(synapses, order):
    """(J_yX, tau_yX) for hybridLFPy, read off the synapse table."""
    return ([synapses[pop]['weight'] for pop in order],
            [synapses[pop]['tau2'] for pop in order])


MSO_J_YX, MSO_TAU_YX = _bookkeeping(
    MSO_SYNAPSES, ['SBC', 'SBC', 'MNTBC', 'LNTBC'])


# ===========================================================================
# LSO, lateral superior olive (ILD coder)
# ===========================================================================
N_LSO_TOTAL = BRAINSTEM.POP_NUM.n_LSOs
LSO_V_INIT = BRAINSTEM.E_L.LSO

# Human dimensions, reoriented: long axis rostrocaudal, tonotopy mediolateral
# (lateral is low CF).
LSO_HALF_HEIGHT_Y = 1400.0   # µm  rostrocaudal half-height (2.8 mm span)
LSO_RADIUS_X = 400.0         # µm  dorsoventral semi-axis
LSO_RADIUS_Z = 600.0         # µm  mediolateral / tonotopic semi-axis
# LSO dendrites run along y, perpendicular to the tonotopic axis, so the probe
# runs along y as well. 500 µm brackets the 413.5 µm dendrite tips.
LSO_PROBE_AXIS = 'y'
LSO_PROBE_HALF_SPAN = 500.0  # µm

# One layer covering everything. hybridLFPy assigns synapses by absolute z but
# the reoriented nucleus spreads somas along z, so placement is by section name
# instead (see LSOPopulation).
LSO_LAYERS = [[-1.0e4, 1.0e4]]

LSO_SYNAPSES = {
    'SBC': {
        'syntype': 'Exp2Syn',
        'tau1': BRAINSTEM.TAUS_EX_RISE.LSO,
        'tau2': BRAINSTEM.TAUS_EX_DECAY.LSO,
        'e': BRAINSTEM.EXC_REV.LSO,
        'weight': syn_weight('SBCs2LSO'),
    },
    'MNTBC': {
        'syntype': 'Exp2Syn',
        'tau1': BRAINSTEM.TAUS_IN_RISE.LSO,
        'tau2': BRAINSTEM.TAUS_IN_DECAY.LSO,
        'e': BRAINSTEM.INH_REV.LSO,
        'weight': syn_weight('MNTBCs2LSO'),
    },
}
LSO_CONVERGENCE = [[BRAINSTEM.POP_CONV.SBCs2LSOs, BRAINSTEM.POP_CONV.MNTBCs2LSOs]]
LSO_DELAYS = [BRAINSTEM.SYN_DELAYS.SBCs2LSO, BRAINSTEM.SYN_DELAYS.MNTBCs2LSO]
LSO_J_YX, LSO_TAU_YX = _bookkeeping(LSO_SYNAPSES, ['SBC', 'MNTBC'])

# LSO principal-cell axon conduction velocity (models/mso/build_lso_axon.py):
# central myelinated fibre, 3.0-3.5 m/s per µm outer diameter at 2.3 µm.
LSO_AXON_CONDUCTION_VELOCITY_M_S = 7.4


# ===========================================================================
# AVCN, bushy cells (globular and spherical)
# ===========================================================================
# Both are driven by ANF endbulbs of Held and share one Population class. They
# differ in morphology, channel densities and endbulb count.
GBC_V_INIT = -65.0    # mV  leak reversal of the cnmodel XM13_nacncoop
SBC_V_INIT = -65.0    #     decoration, not a NEST resting potential

N_GBC_TOTAL = BRAINSTEM.POP_NUM.n_GBCs
N_SBC_TOTAL = BRAINSTEM.POP_NUM.n_SBCs
GBC_ENDBULBS = BRAINSTEM.POP_CONV.ANFs2GBCs   # 20 modified endbulbs
SBC_ENDBULBS = BRAINSTEM.POP_CONV.ANFs2SBCs   # 3 large axosomatic endbulbs

AVCN_RADIUS_X = 400.0    # µm  tonotopic half-axis
GBC_RADIUS_Y = 600.0     # µm  rostrocaudal half-axis (caudal AVCN)
SBC_RADIUS_Y = 875.0     # µm  the spherical-cell area spans the rostral ~1.75 mm
AVCN_PROBE_HALF_SPAN = 300.0   # µm  bushy cells are smaller than MSO/LSO

AVCN_LAYERS = [[-100.0, 100.0]]   # placement is by section name, not by depth

# Every GBC axon is aligned to one direction so their axial currents summate;
# see AVCN_AXON_HEAD_DIR in head_geometry.
# Few large axosomatic endbulbs on the SBC, so placement is mostly somatic.
SBC_ENDBULB_WEIGHTS = {'soma': 0.85, 'primarydendrite': 0.15}

GBC_SYNAPSES = {
    'ANF': {
        'syntype': 'Exp2Syn',
        'tau1': BRAINSTEM.TAUS_EX_RISE.GBC,
        'tau2': BRAINSTEM.TAUS_EX_DECAY.GBC,
        'e': BRAINSTEM.EXC_REV.GBC,
        'weight': syn_weight('ANFs2GBCs'),
    },
}
SBC_SYNAPSES = {
    'ANF': {
        'syntype': 'Exp2Syn',
        'tau1': BRAINSTEM.TAUS_EX_RISE.SBC,
        'tau2': BRAINSTEM.TAUS_EX_DECAY.SBC,
        'e': BRAINSTEM.EXC_REV.SBC,
        'weight': syn_weight('ANFs2SBCs'),
    },
}
GBC_CONVERGENCE = [[GBC_ENDBULBS]]
SBC_CONVERGENCE = [[SBC_ENDBULBS]]
# GBC_DELAYS, SBC_DELAYS: ANF_AVCN_DELAY_MS, defined with the ANF conduction below.
SYN_DELAY_SCALE = [None]      # no jitter on the endbulb delay

#: The GBC dipole groups: one run, recorded as two edge groups that partition
#: the cell (soma + dendrites + AIS, and the axon) and sum to the whole-cell moment.
GBC_GROUPED_GROUPS = ('GBCsoma', 'GBCaxon')


# ===========================================================================
# MNTB, medial nucleus of the trapezoid body
# ===========================================================================
N_MNTB_TOTAL = BRAINSTEM.POP_NUM.n_MNTBCs
MNTB_V_INIT = BRAINSTEM.E_L.MNTBC

MNTB_RADIUS_X = 200.0     # µm  tonotopic half-axis
MNTB_RADIUS_Y = 2000.0    # µm  rostrocaudal half-axis, ~4 mm as for the MSO
MNTB_PROBE_HALF_SPAN = 300.0   # µm  brackets the ±67 µm dendrite tips

# Must match the section extents in models/mntb/mntb_model_active.hoc.
MNTB_LAYERS = [
    [10.0, 70.0],     # dend_A (+z)
    [-70.0, -10.0],   # dend_B (-z)
    [-10.0, 10.0],    # soma, where the axosomatic calyx lands
]

# The calyx of Held is the fastest synapse in the model.
MNTB_SYNAPSES = {
    'GBC': {
        'syntype': 'Exp2Syn',
        'tau1': BRAINSTEM.TAUS_EX_RISE.MNTBC,
        'tau2': BRAINSTEM.TAUS_EX_DECAY.MNTBC,
        'e': BRAINSTEM.EXC_REV.MNTBC,
        'weight': syn_weight('GBCs2MNTBCs'),
    },
}
# The presynaptic terminal, driven suprathreshold to generate the extracellular
# prespike. Co-located with the principal cell.
CALYX_SYNAPSES = {
    'GBC': {
        'syntype': 'Exp2Syn',
        'tau1': BRAINSTEM.TAUS_EX_RISE.MNTBC,
        'tau2': BRAINSTEM.TAUS_EX_DECAY.MNTBC,
        'e': BRAINSTEM.EXC_REV.MNTBC,
        'weight': syn_weight('GBCs2calyx'),
    },
}
MNTB_CONVERGENCE = [[0], [0], [BRAINSTEM.POP_CONV.GBCs2MNTBCs]]   # soma only

# --- GBC -> MNTB delay ------------------------------------------------------
# NEST's single SYN_DELAYS.GBCs2MNTBCs stands for both the axonal crossing and
# the calyx synapse. The reconstruction models the axon, so it splits them:
#   conduction = GBC_AXON_LENGTH_UM / GBC_MNTB_CONDUCTION_VELOCITY_M_S
#   synaptic   = 0.30 ms (calyx of Held at 37 degC, presynaptic AP to
#                postsynaptic threshold, 0.25-0.35 ms)
# 30 m/s is the conduction velocity of an 8 µm central myelinated fibre, and
# models/avcn/validate_gbc_axon.py checks the baked axon conducts at it.
# MSO and LSO read the NEST MNTBC spike trains and are not affected.
from recon_core.config.head_geometry import GBC_AXON_LENGTH_UM as _GBC_AXON_LENGTH_UM  # noqa: E402
GBC_MNTB_CONDUCTION_VELOCITY_M_S = 30.0
GBC_MNTB_CONDUCTION_MS = (_GBC_AXON_LENGTH_UM * 1e-3) / GBC_MNTB_CONDUCTION_VELOCITY_M_S
GBC_MNTB_SYNAPTIC_MS = 0.30

# --- NetCon spike-detection thresholds, one per recording site -------------
#   soma  -20 mV: bushy-cell and MNTB somatic APs (SBC soma APs peak near -9 mV,
#                 so 0 mV would miss most of them).
#   node  -10 mV: nodes of Ranvier on the GBC axon (peaks 40-50 mV).
SOMA_SPIKE_THRESHOLD_MV = -20.0
NODE_SPIKE_THRESHOLD_MV = -10.0

# Replaces NEST's SYN_DELAYS.GBCs2MNTBCs for the reconstructed MNTB; the NEST
# value still governs the NEST network.
MNTB_DELAYS = [GBC_MNTB_CONDUCTION_MS + GBC_MNTB_SYNAPTIC_MS]
MNTB_J_YX, MNTB_TAU_YX = _bookkeeping(MNTB_SYNAPSES, ['GBC'])
CALYX_J_YX, CALYX_TAU_YX = _bookkeeping(CALYX_SYNAPSES, ['GBC'])
# The calyx fires when the AP reaches the heminode, after the axonal crossing
# and before the synapse: conduction term only, so the prespike leads the
# postsynaptic sink by GBC_MNTB_SYNAPTIC_MS. Invasion of the terminal emerges
# from the calyx cable.
CALYX_DELAYS = [GBC_MNTB_CONDUCTION_MS]


# ===========================================================================
# Cochlear tonotopy
# ===========================================================================
# The cochlear model's CF range, from simulate/utils/cochlea_utils.py.
CF_MIN_HZ = 125.0
CF_MAX_HZ = 20000.0
N_ANF_TOTAL = BRAINSTEM.n_ANFs

# Auditory-nerve generator.
ANF_N_BINS = 25          # ~1 mm bins along the ~24 mm nerve
# Central axon diameter of the default fibre (human central process, Rattay et
# al. 2013); the myelin is derived from it (anf_biophysics.central_myelin_cm).
ANF_FIBER_DIAM = 3.5     # µm

#: Wave I representations of the auditory nerve. They are alternatives and are
#: never summed:
#:   'fibre'    every bin of the nerve (--anf-source full)
#:   'distal'   boundary reading: only the distal bins, collapsed to an
#:              aperture source at the modiolus (--anf-source boundary
#:              --wave1-source aperture). A travelling AP self-cancels in the
#:              far field; what reaches the scalp is the distal CAP plus the
#:              stationary potentials at conductivity junctions (Martin,
#:              Pratt & Schwegler 1995).
#:   'modiolus' boundary source at the canal entry (default;
#:              --anf-source boundary --wave1-source modiolus)
ANF_DISTRIBUTED_GROUP = 'fibre'
ANF_DISTAL_GROUP = 'distal'
ANF_MODIOLUS_GROUP = 'modiolus'

#: Wave I representation -> the --anf-source mode that selects it. Any ANF
#: generator that is not a stationary junction must be listed here, or
#: main_abr_full refuses to assemble.
ANF_WAVE1_GROUPS = {
    ANF_DISTRIBUTED_GROUP: 'full',          # --anf-source full
    ANF_DISTAL_GROUP:      'boundary',      # --anf-source boundary --wave1-source aperture
    ANF_MODIOLUS_GROUP:    'boundary',      # --anf-source boundary --wave1-source modiolus
}

#: Stationary junction potentials, present in every mode (Martin, Pratt &
#: Schwegler 1995): porus = I_n, brainstem entry = II_p.
ANF_STATIONARY_GROUPS = ('porus', 'brainstem')

# Inter-fibre dispersion: five central-axon diameters spanning the human range
# (2.0-3.75 µm), equally weighted.
ANF_DIAM_SET = (2.0, 2.4, 2.9, 3.35, 3.75)    # µm
ANF_DIAM_WEIGHTS = (1.0, 1.0, 1.0, 1.0, 1.0)

#: Node diameter as a fraction of fibre diameter.
ANF_NODE_DIAM_RATIO = 2.0 / 3.0
ANF_PERIPH_DIAM = 1.5    # um  peripheral dendrite (human: 1.0-2.0 um)
ANF_NODE_DIAM = 2.0      # um
ANF_NODE_LEN = 1.0       # um
ANF_INTERNODE_LEN = 300.0  # um  ~100x fibre diameter
ANF_SOMA_DIAM = 22.0     # um  human SGN soma, unmyelinated

# Human peripheral dendrite: myelinated except for the terminal at the organ of
# Corti and short pre- and post-somatic segments; the soma is unmyelinated.
ANF_TERMINAL_LEN = 10.0     # um  unmyelinated terminal, organ of Corti
ANF_PRESOMA_LEN = 100.0     # um  unmyelinated pre-somatic segment
ANF_POSTSOMA_LEN = 5.0      # um  unmyelinated post-somatic segment
ANF_PERIPH_N_NODES = 6      # nodes on the myelinated peripheral dendrite
ANF_PERIPH_NODE_DIAM = 1.0  # um  node diameter there (fibre is 1.5 um)
ANF_V_INIT = -65.0       # mV
# Conduction velocity range of this repository's ANF cable, measured per fibre
# diameter in ANF_DIAM_SET over the 24 mm path (terminal to most central node):
#     diam µm   2.00    2.40    2.90    3.35    3.75
#     CV m/s    10.03   11.68   13.78   15.13   16.48
ANF_CV_RANGE_MS = (10.03, 16.48)   # m/s


def _erb_number(freq_hz):
    """Glasberg & Moore (1990) ERB-rate scale."""
    return 21.3 * np.log10(1.0 + freq_hz / 229.0)


def tonotopic_index(freq_hz, n_total):
    """Index of a characteristic frequency in a tonotopic population.

    Populations are laid out linearly on the ERB scale, as the cochlear model
    distributes its characteristic frequencies.
    """
    span = _erb_number(CF_MAX_HZ) - _erb_number(CF_MIN_HZ)
    frac = (_erb_number(freq_hz) - _erb_number(CF_MIN_HZ)) / span
    return int(round(float(np.clip(frac, 0.0, 1.0)) * (n_total - 1)))


# ===========================================================================
# Head model (re-exported; defined in recon_core/config/head_geometry.py)
# ===========================================================================
from recon_core.config.head_geometry import (          # noqa: E402,F401
    FOUR_SPHERE_RADII, FOUR_SPHERE_SIGMAS, ELECTRODE_POS,
    NUCLEUS_POS_UM, MSO_POS_UM, LSO_POS_UM, AVCN_POS_UM, SBC_POS_UM, MNTB_POS_UM,
    AVCN_AXON_TARGET, AVCN_AXON_HEAD_DIR,
    GBC_MNTB_CHORD_UM, GBC_MNTB_TORTUOSITY, GBC_AXON_LENGTH_UM,
    ROTATION_MSO, ROTATION_LSO, ROTATION_AVCN, ROTATION_MNTB,
    ANF_PATH_UM, MODIOLUS_POS_UM, anf_path_binned,
)


# ===========================================================================
# ANF -> cochlear-nucleus axonal conduction
# ===========================================================================
# NEST generates ANF spikes at the spiral ganglion and its point neurons have
# no spatial extent, so the nerve's conduction time to the cochlear nucleus is
# not in the network. ANF_PATH_UM ends at the cochlear nucleus.
def anf_path_length_mm(side='L'):
    """Arc length of the modelled auditory-nerve path, mm."""
    path = np.asarray(ANF_PATH_UM[side], dtype=float)
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) / 1e3


def anf_mean_cv():
    """Mean conduction velocity over ANF_CV_RANGE_MS, m/s."""
    return sum(ANF_CV_RANGE_MS) / len(ANF_CV_RANGE_MS)


#: Cable-derived conduction time (path length / mean CV), for reference.
ANF_CN_CONDUCTION_DERIVED_MS = anf_path_length_mm() / anf_mean_cv()
#: Reference ANF -> cochlear-nucleus conduction time; no delay list applies it
#: (the AVCN uses ANF_AVCN_DELAY_MS, MSO/LSO/MNTB the NEST delays).
ANF_CN_CONDUCTION_MS = 1.538

# ANF -> GBC and ANF -> SBC delay: the mean cable transit of the ANF fibre set
# (1.846 ms), used directly in place of NEST's ANFs2GBCs/ANFs2SBCs synaptic
# delay. MNTB, calyx, MSO and LSO keep their delays as defined above.
ANF_AVCN_DELAY_MS = 1.846
GBC_DELAYS = [ANF_AVCN_DELAY_MS]
SBC_DELAYS = [ANF_AVCN_DELAY_MS]
