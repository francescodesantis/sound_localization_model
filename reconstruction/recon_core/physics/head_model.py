"""
Projection of current dipole moments onto the scalp through the 4-sphere model.

Every generator is a dipole at its own position and the scalp potential is the
linear superposition

    Phi_total(r_e) = sum_g  L(r_e, r_g) . p_g(t)

so each nucleus is simulated once, its head-frame dipole stored, and composites
assembled later (main_abr_full).

Units (lfpykit): positions µm, dipole moment nA·µm, potential mV; saved and
plotted ABRs are µV.
"""

from collections import defaultdict

import numpy as np

from recon_core.config import head_geometry as hg
from recon_core.utils.signal_utils import bandpass


def volume_conductor(electrode_names):
    """FourSphereVolumeConductor for the given electrodes, plus their positions."""
    from lfpykit.eegmegcalc import FourSphereVolumeConductor

    r_elec = np.stack([hg.ELECTRODE_POS[e] for e in electrode_names])
    fsc = FourSphereVolumeConductor(
        r_electrodes=r_elec,
        radii=hg.FOUR_SPHERE_RADII,
        sigmas=hg.FOUR_SPHERE_SIGMAS,
    )
    return fsc, r_elec


def project_sources(sources, electrode_names):
    """Sum unfiltered scalp potentials, in mV, grouped by source label.

    sources is an iterable of (label, p_head (3, T), r_dipole (3,)). Sources
    sharing a label are summed, which is how the L and R populations of one
    generator collapse into a single trace.
    """
    fsc, _ = volume_conductor(electrode_names)
    V_mV = defaultdict(float)
    for label, p_head, r_dipole in sources:
        V_mV[label] = V_mV[label] + fsc.get_dipole_potential(
            np.asarray(p_head, dtype=float), np.asarray(r_dipole, dtype=float))
    return dict(V_mV)


def superpose_sources(sources, electrode_names, srate, lo=150., hi=3000.):
    """Project, band-pass and superpose: the core of every ABR script.

    sources entries are (label, p_head, r_dipole). Returns
    {label: (n_electrodes, T) uV} with an extra 'composite' key holding the sum
    of every label.

    Per-label traces and the composite are both filtered and stay consistent
    because the filter is linear, so callers can group them further (per-nucleus
    totals, monaural differences) without re-filtering.
    """
    V_mV = project_sources(sources, electrode_names)

    out, total = {}, 0.0
    for label, v in V_mV.items():
        total = total + v
        out[label] = bandpass(v * 1e3, fs=srate, lo=lo, hi=hi)   # mV -> uV
    out['composite'] = bandpass(np.asarray(total) * 1e3, fs=srate, lo=lo, hi=hi)
    return out


def project_by_side(p_head_by_side, position_by_side, electrode_names, srate,
                    lo=150., hi=3000.):
    """Wrapper for a single generator simulated on one or both sides.

    Each side's dipole is projected from its own position. The sides are
    mirrored about the midline, which is what makes a lateralised generator
    give different potentials at the left and right mastoids.

    Returns (V_uV (n_electrodes, T), srate).
    """
    sources = [('generator', p_head, position_by_side[side])
               for side, p_head in p_head_by_side.items()]
    return superpose_sources(sources, electrode_names, srate, lo=lo, hi=hi)['composite'], srate


def rotate_to_head(p_model, rotation):
    """Rotate a model-frame dipole (3, T) into the head frame."""
    return np.asarray(rotation) @ np.asarray(p_model, dtype=float)


def rotate_offset_to_head(offset_model_um, rotation, origin_head_um):
    """Place a model-frame offset (3,) at a head-frame origin, with the same
    rotation as the dipole (used for tree_dipole sub-group centroids).
    """
    return (np.asarray(origin_head_um, dtype=float)
            + np.asarray(rotation) @ np.asarray(offset_model_um, dtype=float))
