"""
Delays of the NEURON chain, composed from recon_core.config.params.

Rule: the A -> B conduction term is zero where NEURON carries the action
potential from A to where B is driven (GBC -> MNTB, calyx), and kept in full
where no axon is modelled (SBC -> MSO/LSO, MNTB -> MSO/LSO). No delay list
carries ANF_CN_CONDUCTION_MS.

Every value is an expression over recon_core constants. Anything not
redefined here falls through to recon_core.config.params.
"""

from recon_core.config import params as _P

#: Suffix on the dipole-record directory, so chain records never mix with NEST
#: ones.
CHAIN_VARIANT = 'chain'

# ANF -> GBC/SBC: the input still comes from the cochlear model.
GBC_DELAYS = list(_P.GBC_DELAYS)
SBC_DELAYS = list(_P.SBC_DELAYS)

# GBC -> MNTB: the GBC axon is modelled up to the heminode, so only the
# synaptic delay remains.
MNTB_DELAYS = [_P.GBC_MNTB_SYNAPTIC_MS]

# The calyx is driven at the recorded heminode: zero delay. The prespike leads
# the postsynaptic sink by GBC_MNTB_SYNAPTIC_MS.
CALYX_DELAYS = [0.0]

# SBC/MNTB -> MSO and LSO: no axon is modelled, so the NEST delays are kept in
# full. Columns as MSO_CONVERGENCE: SBC_contra, SBC_ipsi, MNTBC_ipsi, LNTBC_ipsi
# (LNTBC is still a NEST train; its weight is 0).
MSO_DELAYS = [_P.MSO_DELAYS[0],           # SBC contra, chained
              _P.MSO_DELAYS[1],           # SBC ipsi,   chained
              _P.MSO_DELAYS[2],           # MNTBC,      chained
              _P.MSO_DELAYS[3]]           # LNTBC,      NEST

LSO_DELAYS = list(_P.LSO_DELAYS)               # SBC, MNTBC -- both chained


def __getattr__(name):
    """Fall through to recon_core.config.params for everything not defined here."""
    return getattr(_P, name)
