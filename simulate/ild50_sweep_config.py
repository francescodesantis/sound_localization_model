"""
ILD50 parameter-sweep definitions (edit here).

A sweep is a list of configurations; each configuration is a dict of
overrides applied on top of the defaults in models/BrainstemModel/params.py,
keyed by dotted path ("SYN_WEIGHTS.SBCs2LSO" -> p.SYN_WEIGHTS.SBCs2LSO).
Every configuration is simulated with the same seeds (0..N_SEEDS-1), so the
ANF responses are shared across configurations (common random numbers) and
ILD50 differences between configurations are paired comparisons.

Configuration names are built from the overrides (not from list position), so
adding/removing configurations never invalidates results already on disk.
"""
import itertools

import numpy as np

# =============================================================================
# Stimulus / protocol (shared by all sweeps)
# =============================================================================

TIME_SIMULATION = 60  # ms
TIME_ON = 50  # ms
RAMP_MS = 5
IPSI_LEVEL = 60  # dB, left ear (ipsi to the analysed left LSO)
TONE_FREQ_HZ = 16000
# Contralateral (right) level; the Zilany "artificial_ild" mode sets the right
# level to this absolute value while the left stays at IPSI_LEVEL.
CONTRA_LEVELS = np.linspace(0, 90, 19)

# Spike counts saved per run (per neuron, per cue). LSO is what ILD50 needs;
# MNTBC is kept to diagnose whether a parameter acts through inhibition.
SAVE_POPS = ["LSO", "MNTBC"]

# Timing recording (sweeps with "timing": True): input tree of the analysed left LSO neuron.
# name -> (side, population, tree node it projects onto); built in this order.
TIMING_TREE = {
    "SBC": ("L", "SBC", "LSO"),
    "MNTBC": ("L", "MNTBC", "LSO"),
    "GBC": ("R", "GBC", "MNTBC"),  # contra GBCs drive the left MNTB
    "ANF_ipsi": ("L", "parrot_ANF", "SBC"),
    "ANF_contra": ("R", "parrot_ANF", "GBC"),
}
TIMING_POPS = {"LSO": ("L", "LSO"), **{k: v[:2] for k, v in TIMING_TREE.items()}}
TIMING_INTERVAL_MS = 0.02  # multimeter sampling of g_ex/g_in

# =============================================================================
# Helpers to build sweeps
# =============================================================================

BASELINE = {}  # defaults of params.py


def oat(path, values):
    """One-at-a-time: vary `path` over explicit `values`, everything else at baseline."""
    return [{path: v} for v in values]


def scaled(base, factors):
    return [base * f for f in factors]


def shifted(base, offsets):
    return [base + o for o in offsets]


def grid(**axes):
    """Full factorial over dotted paths, e.g. grid(**{"SYN_WEIGHTS.SBCs2LSO": [...], ...})."""
    paths = list(axes.keys())
    return [dict(zip(paths, combo)) for combo in itertools.product(*axes.values())]


# Baseline values, duplicated here only to build the value lists below.
# The driver checks that they match params.py so they cannot silently drift.
_B = {
    "SYN_WEIGHTS.SBCs2LSO": 5,
    "SYN_WEIGHTS.MNTBCs2LSO": -80,
    "POP_CONV.SBCs2LSOs": 40,
    "POP_CONV.MNTBCs2LSOs": 8,
    "SYN_DELAYS.MNTBCs2LSO": 0.78,
    "SYN_WEIGHTS.ANFs2GBCs": 5,
    "SYN_WEIGHTS.GBCs2MNTBCs": 30.0,
    "G_LEAK.LSO": 18.75,
    "MEMB_CAPS.LSO": 12.0,
    "V_TH.LSO": -45.0,
    "INH_REV.LSO": -90,
    "TAUS_EX_DECAY.LSO": 0.5,
    "TAUS_IN_DECAY.LSO": 0.5,
    "SYN_DELAYS.SBCs2LSO": 2.0,
    "SYN_DELAYS.ANFs2SBCs": 0.5,
    "SYN_DELAYS.ANFs2GBCs": 0.5,
    "SYN_DELAYS.GBCs2MNTBCs": 0.5,
    "SYN_WEIGHTS.ANFs2SBCs": 12.0,
    "E_L.LSO": -63.0,
    "V_RESET.LSO": -65.0,
    "T_REF.LSO": 0.6,
    "EXC_REV.LSO": 0,
    "TAUS_EX_RISE.LSO": 0.2,
    "TAUS_IN_RISE.LSO": 0.2,
}

_MUL = [0.5, 0.75, 1.25, 1.5, 2.0]


def dt_config(dt, shift=0.0):
    """Delays giving dt_syn = (inhibitory - excitatory path synaptic delays) onto the LSO, in ms.

    Convention: dt < 0 inhibition leads, dt > 0 excitation leads. Synaptic delays only: the
    integration times of SBC/GBC/MNTB and the ANF latencies are added by the network and are
    measured from the timing recordings. Only the final synapse of one branch is lengthened
    (never shortened below baseline): later inhibition -> MNTBCs2LSO, later excitation -> SBCs2LSO.
    `shift` is added to both branches and must leave the LSO response unchanged.
    """
    d_exc = _B["SYN_DELAYS.ANFs2SBCs"] + _B["SYN_DELAYS.SBCs2LSO"]  # 2.5 ms
    d_inh = _B["SYN_DELAYS.ANFs2GBCs"] + _B["SYN_DELAYS.GBCs2MNTBCs"] + _B["SYN_DELAYS.MNTBCs2LSO"]  # 1.78 ms
    extra = round(dt - (d_inh - d_exc), 3)  # inhibition later than at baseline (dt_syn = -0.72 ms)
    return {"SYN_DELAYS.SBCs2LSO": round(_B["SYN_DELAYS.SBCs2LSO"] + max(-extra, 0) + shift, 3),
            "SYN_DELAYS.MNTBCs2LSO": round(_B["SYN_DELAYS.MNTBCs2LSO"] + max(extra, 0) + shift, 3)}


_DT_GRID = np.round(np.arange(-2.0, 1.2501, 0.125), 3)

# =============================================================================
# Sweeps
# =============================================================================

SWEEPS = {
    # Baseline only, many seeds: decide how many seeds the real sweeps need.
    "pilot": {
        "n_seeds": 30,
        "configs": [BASELINE],
    },
    # Stage 1: one-at-a-time screening around the baseline.
    "oat": {
        "n_seeds": 10,
        "configs": [BASELINE]
        + oat("SYN_WEIGHTS.SBCs2LSO", scaled(_B["SYN_WEIGHTS.SBCs2LSO"], _MUL))
        + oat("SYN_WEIGHTS.MNTBCs2LSO", scaled(_B["SYN_WEIGHTS.MNTBCs2LSO"], _MUL))
        + oat("POP_CONV.SBCs2LSOs", [20, 30, 50, 60, 80])
        + oat("POP_CONV.MNTBCs2LSOs", [2, 4, 6, 12, 16])
        + oat("SYN_DELAYS.MNTBCs2LSO", shifted(_B["SYN_DELAYS.MNTBCs2LSO"], [-0.5, -0.25, 0.25, 0.5, 1.0]))
        + oat("SYN_WEIGHTS.ANFs2GBCs", scaled(_B["SYN_WEIGHTS.ANFs2GBCs"], _MUL))
        + oat("SYN_WEIGHTS.GBCs2MNTBCs", scaled(_B["SYN_WEIGHTS.GBCs2MNTBCs"], _MUL))
        + oat("G_LEAK.LSO", scaled(_B["G_LEAK.LSO"], _MUL))
        + oat("MEMB_CAPS.LSO", scaled(_B["MEMB_CAPS.LSO"], _MUL))
        + oat("V_TH.LSO", shifted(_B["V_TH.LSO"], [-6, -3, 3, 6]))
        + oat("INH_REV.LSO", shifted(_B["INH_REV.LSO"], [-10, -5, 5, 10, 20]))
        + oat("TAUS_EX_DECAY.LSO", scaled(_B["TAUS_EX_DECAY.LSO"], _MUL))
        + oat("TAUS_IN_DECAY.LSO", scaled(_B["TAUS_IN_DECAY.LSO"], _MUL)),
    },
    # Stage 1b: one-at-a-time on the parameters not covered by "oat" (no convergence).
    "oat2": {
        "n_seeds": 10,
        "configs": [BASELINE]
        # delays: excitatory vs inhibitory branch timing
        + oat("SYN_DELAYS.SBCs2LSO", shifted(_B["SYN_DELAYS.SBCs2LSO"], [-1.5, -1.0, -0.5, 0.5, 1.0]))
        + oat("SYN_DELAYS.ANFs2SBCs", shifted(_B["SYN_DELAYS.ANFs2SBCs"], [-0.25, 0.25, 0.5, 1.0]))
        + oat("SYN_DELAYS.ANFs2GBCs", shifted(_B["SYN_DELAYS.ANFs2GBCs"], [-0.25, 0.25, 0.5, 1.0]))
        + oat("SYN_DELAYS.GBCs2MNTBCs", shifted(_B["SYN_DELAYS.GBCs2MNTBCs"], [-0.25, 0.25, 0.5, 1.0]))
        # weights: SBC drive, and finer steps where "oat" showed failure (ANFs2GBCs<=3)
        # or saturation (GBCs2MNTBCs>=37.5)
        + oat("SYN_WEIGHTS.ANFs2SBCs", scaled(_B["SYN_WEIGHTS.ANFs2SBCs"], _MUL))
        + oat("SYN_WEIGHTS.ANFs2GBCs", [3.5, 4.0, 4.5, 5.5, 8.0])
        + oat("SYN_WEIGHTS.GBCs2MNTBCs", [20.0, 25.0, 27.5, 32.5, 35.0])
        # LSO neuron (G_LEAK, MEMB_CAPS, V_TH, INH_REV, decay taus already in "oat")
        + oat("E_L.LSO", shifted(_B["E_L.LSO"], [-6, -3, 3, 6]))
        + oat("V_RESET.LSO", shifted(_B["V_RESET.LSO"], [-6, -3, 3, 6]))
        + oat("T_REF.LSO", scaled(_B["T_REF.LSO"], [0.5, 1.5, 2.0, 3.0]))
        + oat("EXC_REV.LSO", shifted(_B["EXC_REV.LSO"], [-20, -10, 10]))
        # rise must stay below the 0.5 ms decay (iaf_cond_beta)
        + oat("TAUS_EX_RISE.LSO", scaled(_B["TAUS_EX_RISE.LSO"], [0.25, 0.5, 1.5, 2.0]))
        + oat("TAUS_IN_RISE.LSO", scaled(_B["TAUS_IN_RISE.LSO"], [0.25, 0.5, 1.5, 2.0])),
    },
    # Relative exc/inh timing: ILD50 vs dt_syn on a fine grid (baseline dt_syn = -0.72 ms).
    "dt": {
        "n_seeds": 10,
        "timing": True,
        "configs": [BASELINE] + [dt_config(t) for t in _DT_GRID],
    },
    # Control: a common shift of both branches must give spike counts identical to shift 0.
    "dt_invariance": {
        "n_seeds": 10,
        "timing": True,
        "configs": [dt_config(t, c) for t in (-1.72, -0.72, 0.28) for c in (0.0, -0.25, 0.25, 0.5, 1.0)],
    },
    # Stage 2: E/I grid (tests whether ILD50 depends only on the I/E ratio).
    # Edit the axes after looking at the OAT results.
    "ei_grid": {
        "n_seeds": 10,
        "configs": grid(**{
            "SYN_WEIGHTS.SBCs2LSO": scaled(_B["SYN_WEIGHTS.SBCs2LSO"], [0.5, 0.75, 1.0, 1.5, 2.0]),
            "SYN_WEIGHTS.MNTBCs2LSO": scaled(_B["SYN_WEIGHTS.MNTBCs2LSO"], [0.5, 0.75, 1.0, 1.5, 2.0]),
        }),
    },
}
