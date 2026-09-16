#!/usr/bin/env python3
"""
This runs 3 stages: per-nucleus ABR -> composite ABR -> usual figures

Edit the CONFIG block below and run it from the repository root:
    mamba activate sl_env
    cd <repo root>
    python reconstruction/run_abr.py
# it will run mpirun 4

Everything will be found under:
    reconstruction/abr_results/<pic stem>_<condition>/
        ABR_full.h5                        the composite + every generator signal
        cz_m1_m2_<nucleus>.png             per nucleus, Cz / M1 / M2
        full_abr_electrodes.png            composite + decomposition, per electrode
        full_abr_contributions_<mont>.png  composite + decomposition, one montage
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recon_core import bootstrap, io_utils, paths  # noqa: E402

# ===========================================================================
#  CONFIG - everything that we may want to change for a new run
# ===========================================================================

# Diotic click: mode 'artificial_itd' at 0 -> the SAME waveform in both ears,
# so no HRTF, ITD = 0 and ILD = 0 by construction.  Regenerate it with
#   python reconstruction/main.py abr delivery-pic --delivery diotic
# (its HRTF twin is .../click_70dB_HRTF/click_70dB_HRTF_a0_0.05ms_seed42.pic)
PIC_FILE = 'RESULTS/validation/click_70dB_diotic/click_70dB_diotic_0.05ms_seed42.pic'

# where the sound goes: 'binaural' | 'left_ear' | 'right_ear'
CONDITION = 'binaural'

'''
* MSO and LSO take `--condition`: they are binaural, so a monaural condition is
  produced by silencing the inputs the absent ear would have driven.
* AVCN and MNTB do NOT take `--condition`: they are monaural nuclei, and
  main_abr_full.py simply SELECTS the ear-driven hemisphere (left ear -> AVCN L,
  MNTB R).  One binaural run serves every condition, which is why switching
  condition re-runs only MSO and LSO.
'''

# How the stimulus is indexed inside the .pic
#   ('--itd-us', '0')     artificial ITD in microseconds
#   ('--ild-db', '-10')   artificial ILD in dB
#   ('--angle',  '0')     HRTF azimuth in degrees
SELECTOR = ('--itd-us', '0')

# Cells per population. NO AVCN bc it has two populations
#   (I need to find a way to specify them separately)
N_CELLS = {'mso': 15500, 'lso': 5600, 'mntb': 3600}

MPI_RANKS = 4 #8

# True  = re-simulate every nucleus even when its dipole records already exist.
# False = simulate only what is missing (safe to re-run after a crash, or to
#         redraw figures without paying for the nuclei again OR to run monaural after a binaural run).
FORCE_RERUN = True
DRY_RUN = False

'''
IF DRY_RUN == True -> the script will print every command it would run
then exits without executing anything
-> may be needed to check which nuclei are already present and which will be re-run
useful to make an estimate of how long the run will take
'''
# ===========================================================================

PYTHON = sys.executable
MAIN = os.path.join(bootstrap.PACKAGE_ROOT, 'main.py')
RESULTS_ROOT = os.path.join(bootstrap.PACKAGE_ROOT, 'abr_results')

LSO_GENERATOR = 'synaptic'          # just bc I started working on spiking LSO in NEURON
SIDES = ('L', 'R')

# nucleus -> the dipole records it must leave behind, as (generator, ...).
# AVCN and MNTB always file theirs under 'binaural' (they are monaural nuclei);
# MSO and LSO file theirs under the acoustic condition.
RECORDS = {
    'avcn': (('GBC', 'SBC'), 'binaural'),
    'mntb': (('principal', 'calyx'), 'binaural'),
    'mso': (('postsynaptic',), None),
    'lso': ((LSO_GENERATOR,), None),
}

NUCLEUS_UPPER = {'avcn': 'AVCN', 'mntb': 'MNTB', 'mso': 'MSO', 'lso': 'LSO'}

IPSILATERAL = {'left_ear': 'Cz-M1', 'right_ear': 'Cz-M2'} # clinical montage wants the ipsilateral mastoid


def run(args, what):
    """Run one dispatcher command, echoing it first."""
    cmd = [PYTHON, MAIN, *args]
    if MPI_RANKS > 1 and args[0] == 'abr' and args[1] in NUCLEUS_UPPER:
        cmd = ['mpiexec', '-n', str(MPI_RANKS), *cmd]
    print(f'\n--- {what}\n    {" ".join(cmd)}', flush=True)
    if DRY_RUN:
        return
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f'FAILED ({result.returncode}): {what}')


def record_paths(nucleus, dipoles_dir):
    """It says which dipole record we need to produce for this condition."""
    '''
    REMEMBER that dipoles are named: <NUCLEUS>__<generator>__<side>__<condition>.h5
    avcn  -> AVCN__GBC__L__binaural.h5      AVCN__GBC__R__binaural.h5
         AVCN__SBC__L__binaural.h5      AVCN__SBC__R__binaural.h5
    mntb  -> MNTB__principal__{L,R}__binaural.h5   MNTB__calyx__{L,R}__binaural.h5
    mso   -> MSO__postsynaptic__L__left_ear.h5     ...__R__left_ear.h5
    lso   -> LSO__synaptic__L__left_ear.h5         ...__R__left_ear.h5
    '''

    generators, fixed_condition = RECORDS[nucleus]
    condition = fixed_condition or CONDITION
    return [os.path.join(dipoles_dir,
                         f'{NUCLEUS_UPPER[nucleus]}__{gen}__{side}__{condition}.h5')
            for gen in generators for side in SIDES]


def nucleus_commands(nucleus):
    """The dispatcher arguments for one nucleus."""
    base = ['abr', nucleus, '--pic-file', PIC_FILE, *SELECTOR, '--side', 'both']
    if nucleus == 'avcn':
        return base + ['--generators', 'both', '--full-n']
    if nucleus == 'mntb':
        return base + ['--generators', 'both', '--n-cells', str(N_CELLS['mntb'])]
    if nucleus == 'mso':
        return base + ['--n-cells', str(N_CELLS['mso']), '--condition', CONDITION]
    return base + ['--n-cells', str(N_CELLS['lso']),
                   '--generators', LSO_GENERATOR, '--condition', CONDITION]


def partial_records(paths_to_check):
    """Records that exist but were simulated at less than their full population.
    Existence alone is NOT enough to skip a nucleus.
    """
    out = []
    for path in paths_to_check:
        if not os.path.exists(path):
            continue
        attrs, _, _ = io_utils.read_dipole_record(path)
        n_cells, n_total = attrs.get('n_cells'), attrs.get('n_total')
        if n_cells is not None and n_total is not None and n_cells != n_total:
            out.append((os.path.basename(path), n_cells, n_total))
    return out


def check_full_n(paths_to_check):
    """
    Dipole records are superposed with NO scaling, so a partial-N nucleus enters
    the composite under-weighted: the wave ORDER and LATENCIES stay right, the
    amplitude RATIOS do not.  This is the check that catches a shared --n-cells
    being applied to populations of different size.
    """
    partial = partial_records(paths_to_check)
    if partial:
        print('\n!! these records are NOT at full N -- the composite will be '
              'weighted wrongly:')
        for name, n_cells, n_total in partial:
            print(f'     {name:<38} {n_cells} of {n_total} '
                  f'({100 * n_cells / n_total:.1f}%)')
        print('   amplitude ratios between nuclei are not physical; wave order '
              'and latencies still are.')
    return not partial


def main():
    if CONDITION not in ('binaural', 'left_ear', 'right_ear'):
        sys.exit(f'CONDITION must be binaural/left_ear/right_ear, not {CONDITION!r}')

    stem = paths.pic_stem(PIC_FILE)
    # label built as f'angle{angle}', so an angle must stay an int
    name, cast = {'--angle': ('angle', int),
                  '--itd-us': ('itd_us', float),
                  '--ild-db': ('ild_db', float)}[SELECTOR[0]]
    _, cond_label = paths.condition_key(**{name: cast(SELECTOR[1])})
    dipoles_dir = paths.dipoles_dir_for(stem, cond_label)
    dest = os.path.join(RESULTS_ROOT, f'{stem}_{CONDITION}')

    print(f'stimulus   {stem}  ({cond_label})')
    print(f'condition  {CONDITION}')
    print(f'LSO        {LSO_GENERATOR}   (forced for now)')
    print(f'output     {dest}')
    if DRY_RUN:
        print('DRY RUN -- nothing will be executed')
    if not DRY_RUN:
        os.makedirs(dest, exist_ok=True)

    # -- 1. per-nucleus dipole records -------------------------------------
    expected = []
    for nucleus in ('avcn', 'mso', 'lso', 'mntb'):
        wanted = record_paths(nucleus, dipoles_dir)
        expected += wanted
        have = [p for p in wanted if os.path.exists(p)]
        partial = partial_records(wanted)
        if have and len(have) == len(wanted) and not partial and not FORCE_RERUN:
            print(f'\n--- {nucleus.upper()}: {len(have)} records already present '
                  f'at full N -> skip (FORCE_RERUN=True to redo)')
            continue
        if partial:
            print(f'\n--- {nucleus.upper()}: re-running, these records are '
                  f'below full N:')
            for pname, n_cells, n_total in partial:
                print(f'      {pname:<38} {n_cells} of {n_total}')
        run(nucleus_commands(nucleus), f'{nucleus.upper()} ABR')

    # -- 2. full-N gate ----------------------------------------------------
    if not DRY_RUN:
        check_full_n(expected)

    # -- 3. composite, written straight into the destination ---------------
    run(['abr', 'full', '--pic-file', PIC_FILE, *SELECTOR, '--side', 'both',
         '--condition', CONDITION, '--lso-generator', LSO_GENERATOR,
         '--out-dir', dest], 'composite ABR')

    # -- 4. figures, also straight into the destination ---------------------
    run(['plot', 'electrodes', '--pic-file', PIC_FILE, *SELECTOR,
         '--condition', CONDITION, '--lso-generator', LSO_GENERATOR,
         '--out', dest], 'per-nucleus Cz/M1/M2')

    run(['plot', 'full-abr', '--dir', dest, '--layout', 'electrodes',
         '--out', os.path.join(dest, 'full_abr_electrodes.png')],
        'composite + decomposition, per electrode')

    # Binaural has no ipsilateral side, so both clinical montages are drawn;
    # a monaural run gets only the montage ipsilateral to the stimulated ear.
    montages = (['Cz-M1', 'Cz-M2'] if CONDITION == 'binaural'
                else [IPSILATERAL[CONDITION]])
    for montage in montages:
        run(['plot', 'full-abr', '--dir', dest, '--layout', 'contributions',
             '--derivation', montage,
             '--out', os.path.join(dest, f'full_abr_contributions_{montage}.png')],
            f'composite + decomposition, {montage}')

    if not DRY_RUN:
        print(f'\n=== {dest}')
        for name in sorted(os.listdir(dest)):
            if not os.path.isdir(os.path.join(dest, name)):
                print(f'    {name}')


if __name__ == '__main__':
    main()
