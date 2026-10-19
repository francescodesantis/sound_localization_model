"""
Every filesystem location the reconstruction pipelines read or write.

Output directory names are part of the pipelines' contract: a producer writes
a directory and a consumer (a plot script, main_abr_full) finds it again by
rebuilding the same name, so both sides call the helpers here.

Layout under RESULTS/::

    lfp_tmp/spikes_<stem>_angle<cond>_<side>/      GDF spike files + metadata.json
    lfp_tmp/output_[<pfx>_]<stem>_<label>_<side>/  near-field LFP (hybridLFPy)
    abr_tmp/output_[<pfx>_]<stem>_<label>_<side>/  scalp ABR
    abr_tmp/dipoles/<stem>_<cond_label>/           per-generator head dipoles
    regression/                                    golden-output fingerprints
"""

import os
import re

from recon_core.bootstrap import PACKAGE_ROOT, REPO_ROOT   # noqa: F401  (re-exported)

# RESULTS/ lives at the repository root: simulate writes the .pic files there
# and the reconstruction writes its own outputs alongside them.
RESULTS_DIR = os.path.join(REPO_ROOT, 'RESULTS')
LFP_TMP_DIR = os.path.join(RESULTS_DIR, 'lfp_tmp')
ABR_TMP_DIR = os.path.join(RESULTS_DIR, 'abr_tmp')
DIPOLES_DIR = os.path.join(ABR_TMP_DIR, 'dipoles')

# The complete brainstem ABR gets its own top-level folder rather than sitting
# in abr_tmp/ among the per-nucleus run directories.
FULL_ABR_DIR = os.path.join(RESULTS_DIR, 'full_abr')

DEFAULT_PIC = os.path.join(RESULTS_DIR, 'baseline_simulation.pic')

# NEURON models (morphologies and compiled .mod mechanisms), under
# reconstruction/models/. mso/ also holds the LSO morphologies and is the .mod
# source the MNTB borrows klt/kht/ih from; renaming it would touch every .hoc.
MODELS_DIR = os.path.join(PACKAGE_ROOT, 'models')
MSO_MODELS_DIR = os.path.join(MODELS_DIR, 'mso')
AVCN_MODELS_DIR = os.path.join(MODELS_DIR, 'avcn')
MNTB_MODELS_DIR = os.path.join(MODELS_DIR, 'mntb')


def pic_stem(pic_file):
    """Sanitised basename of a `.pic` path, safe to embed in a directory name."""
    return re.sub(r'[^a-zA-Z0-9_-]', '_',
                  os.path.splitext(os.path.basename(pic_file))[0])


def resolve_pic(pic_file):
    """A `--pic-file` argument, defaulting to the baseline simulation."""
    return pic_file or DEFAULT_PIC


def spikes_dir_for(stem, cond_value, side):
    """GDF spike cache for one (stimulus, condition, side).

    cond_value is the raw condition key (an int angle, seconds of ITD, dB of
    ILD), not the readable label, since the cache is keyed by whatever was
    looked up in the .pic.
    """
    return os.path.join(LFP_TMP_DIR, f'spikes_{stem}_angle{cond_value}_{side}')


def output_dir_for(kind, stem, cond_label, side, prefix=None, suffix=''):
    """Result directory for one run.

    kind       : 'lfp' or 'abr', selects RESULTS/{lfp_tmp,abr_tmp}
    prefix     : nucleus tag ('lso', 'avcn', 'mntb_calyx'); None for MSO, whose
                 directories are historically unprefixed
    cond_label : readable stimulus label ('angle0', 'itd500us', 'ild-10dB')
    suffix     : accumulated run tags ('_left_ear', '_active', '_noinh')
    """
    base = {'lfp': LFP_TMP_DIR, 'abr': ABR_TMP_DIR}[kind]
    head = f'output_{prefix}' if prefix else 'output'
    return os.path.join(base, f'{head}_{stem}_{cond_label}_{side}{suffix}')


def dipoles_dir_for(stem, cond_label):
    """Directory holding every nucleus's dipole records for one stimulus.

    cond_label is the readable label from condition_key ('angle0', 'itd500us',
    'ild-10dB'), so two runs that differ only in ITD or ILD keep separate
    records instead of overwriting each other.
    """
    return os.path.join(DIPOLES_DIR, f'{stem}_{cond_label}')


def make_output_dirs(output_dir, subdirs=('cells', 'figures', 'populations')):
    """Create a result directory and the sub-folders hybridLFPy expects."""
    for sub in subdirs:
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)
    return output_dir


def condition_key(angle=0, itd_us=None, ild_db=None):
    """Resolve the stimulus selector to (pic_key, readable_label).

    Precedence: --ild-db, then --itd-us, then --angle. The pic key is the raw
    value the .pic was indexed by (seconds for ITD, dB for ILD, int degrees for
    angle); the label is what goes into directory names.
    """
    if ild_db is not None:
        return float(ild_db), f'ild{ild_db:g}dB'
    if itd_us is not None:
        return itd_us * 1e-6, f'itd{itd_us:g}us'
    return angle, f'angle{angle}'
