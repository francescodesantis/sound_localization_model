#!/usr/bin/env python3
"""
Assert that recon_core.params really mirrors the NEST network definition.

These checks fail if a literal from simulate/models/BrainstemModel/params.py is
retyped into the reconstruction, or if the two pipelines start disagreeing
about the same synapse.

    python tests/test_params.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recon_core import brainstem, params as P

FAILURES = []


def check(name, got, want):
    ok = got == want
    print(f'  {"OK  " if ok else "FAIL"}  {name:<34s} {got!r}'
          + ('' if ok else f'   expected {want!r}'))
    if not ok:
        FAILURES.append(name)


def main():
    B = brainstem.parameters()

    print('\npopulation sizes  (POP_NUM)')
    check('N_MSO_TOTAL', P.N_MSO_TOTAL, B.POP_NUM.n_MSOs)
    check('N_LSO_TOTAL', P.N_LSO_TOTAL, B.POP_NUM.n_LSOs)
    check('N_GBC_TOTAL', P.N_GBC_TOTAL, B.POP_NUM.n_GBCs)
    check('N_SBC_TOTAL', P.N_SBC_TOTAL, B.POP_NUM.n_SBCs)
    check('N_MNTB_TOTAL', P.N_MNTB_TOTAL, B.POP_NUM.n_MNTBCs)
    check('N_ANF_TOTAL', P.N_ANF_TOTAL, B.n_ANFs)

    print('\nconvergence  (POP_CONV)')
    check('MSO_CONVERGENCE', P.MSO_CONVERGENCE, [
        [B.POP_CONV.SBCs2MSOs, 0, 0, 0],
        [0, B.POP_CONV.SBCs2MSOs, 0, 0],
        [0, 0, B.POP_CONV.MNTBCs2MSOs, B.POP_CONV.LNTBCs2MSOs],
    ])
    check('LSO_CONVERGENCE', P.LSO_CONVERGENCE,
          [[B.POP_CONV.SBCs2LSOs, B.POP_CONV.MNTBCs2LSOs]])
    check('MNTB_CONVERGENCE', P.MNTB_CONVERGENCE,
          [[0], [0], [B.POP_CONV.GBCs2MNTBCs]])
    check('GBC_ENDBULBS', P.GBC_ENDBULBS, B.POP_CONV.ANFs2GBCs)
    check('SBC_ENDBULBS', P.SBC_ENDBULBS, B.POP_CONV.ANFs2SBCs)

    print('\nsynaptic delays  (SYN_DELAYS)')
    check('MSO_DELAYS', P.MSO_DELAYS,
          [B.SYN_DELAYS.SBCs2MSOcontra, B.SYN_DELAYS.SBCs2MSOipsi,
           B.SYN_DELAYS.MNTBCs2MSO, B.SYN_DELAYS.LNTBCs2MSO])
    check('LSO_DELAYS', P.LSO_DELAYS,
          [B.SYN_DELAYS.SBCs2LSO, B.SYN_DELAYS.MNTBCs2LSO])
    check('MNTB_DELAYS', P.MNTB_DELAYS, [B.SYN_DELAYS.GBCs2MNTBCs])
    check('GBC_DELAYS', P.GBC_DELAYS, [B.SYN_DELAYS.ANFs2GBCs])
    check('SBC_DELAYS', P.SBC_DELAYS, [B.SYN_DELAYS.ANFs2SBCs])

    print('\nresting potentials  (E_L)')
    check('MSO_V_INIT', P.MSO_V_INIT, B.E_L.MSO)
    check('LSO_V_INIT', P.LSO_V_INIT, B.E_L.LSO)
    check('MNTB_V_INIT', P.MNTB_V_INIT, B.E_L.MNTBC)

    print('\nsynaptic kinetics and reversals  (TAUS_*, EXC_REV, INH_REV)')
    for label, syn, rise, decay, rev in (
            ('MSO/SBC', P.MSO_SYNAPSES['SBC'],
             B.TAUS_EX_RISE.MSO, B.TAUS_EX_DECAY.MSO, B.EXC_REV.MSO),
            ('MSO/MNTBC', P.MSO_SYNAPSES['MNTBC'],
             B.TAUS_IN_RISE.MSO, B.TAUS_IN_DECAY.MSO, B.INH_REV.MSO),
            ('MSO/LNTBC', P.MSO_SYNAPSES['LNTBC'],
             B.TAUS_IN_RISE.MSO, B.TAUS_IN_DECAY.MSO, B.INH_REV.MSO),
            ('LSO/SBC', P.LSO_SYNAPSES['SBC'],
             B.TAUS_EX_RISE.LSO, B.TAUS_EX_DECAY.LSO, B.EXC_REV.LSO),
            ('LSO/MNTBC', P.LSO_SYNAPSES['MNTBC'],
             B.TAUS_IN_RISE.LSO, B.TAUS_IN_DECAY.LSO, B.INH_REV.LSO),
            ('GBC/ANF', P.GBC_SYNAPSES['ANF'],
             B.TAUS_EX_RISE.GBC, B.TAUS_EX_DECAY.GBC, B.EXC_REV.GBC),
            ('SBC/ANF', P.SBC_SYNAPSES['ANF'],
             B.TAUS_EX_RISE.SBC, B.TAUS_EX_DECAY.SBC, B.EXC_REV.SBC),
            ('MNTB/GBC', P.MNTB_SYNAPSES['GBC'],
             B.TAUS_EX_RISE.MNTBC, B.TAUS_EX_DECAY.MNTBC, B.EXC_REV.MNTBC),
            ('CALYX/GBC', P.CALYX_SYNAPSES['GBC'],
             B.TAUS_EX_RISE.MNTBC, B.TAUS_EX_DECAY.MNTBC, B.EXC_REV.MNTBC)):
        check(f'{label} tau1', syn['tau1'], rise)
        check(f'{label} tau2', syn['tau2'], decay)
        check(f'{label} e_rev', syn['e'], rev)

    print('\nhybridLFPy bookkeeping matches the synapse tables it describes')
    check('MSO_J_YX', P.MSO_J_YX, [P.MSO_SYNAPSES[k]['weight']
                                   for k in ('SBC', 'SBC', 'MNTBC', 'LNTBC')])
    check('MSO_TAU_YX', P.MSO_TAU_YX, [P.MSO_SYNAPSES[k]['tau2']
                                       for k in ('SBC', 'SBC', 'MNTBC', 'LNTBC')])
    check('LSO_J_YX', P.LSO_J_YX,
          [P.LSO_SYNAPSES[k]['weight'] for k in ('SBC', 'MNTBC')])
    check('LSO_TAU_YX', P.LSO_TAU_YX,
          [P.LSO_SYNAPSES[k]['tau2'] for k in ('SBC', 'MNTBC')])

    # Reconstruction-only parameters, deliberately absent from params.py (see the
    # params module docstring). The check is that they still exist and are sane,
    # so nobody wires them to a NEST value they are not comparable to.
    print('\nreconstruction-only parameters (deliberately NOT from params.py)')
    check('DT positive', P.DT > 0, True)
    check('Nyquist above the ABR band',
          P.SRATE / 2 > P.BAND_CLINICAL[1], True)
    check('synapse weights are µS, not the nS of SYN_WEIGHTS',
          all(0 < syn['weight'] < 1
              for table in (P.MSO_SYNAPSES, P.LSO_SYNAPSES, P.GBC_SYNAPSES,
                            P.SBC_SYNAPSES, P.MNTB_SYNAPSES, P.CALYX_SYNAPSES)
              for syn in table.values()), True)

    print()
    if FAILURES:
        print(f'{len(FAILURES)} FAILED: {", ".join(FAILURES)}')
        return 1
    print('recon_core.params mirrors simulate/models/BrainstemModel/params.py')
    return 0


if __name__ == '__main__':
    sys.exit(main())
