#!/usr/bin/env python3
"""Stationary far-field potentials where the auditory nerve crosses a
conductivity boundary.

A propagating action potential in a uniform conductor is a moving tripole whose
far field nearly cancels. With lambda(s) the lead field projected on the fibre
tangent at arc position s, V(t) = integral I_ax(s,t) lambda(s) ds; where lambda
changes, the cancellation fails and a stationary source remains,

    p(t) = kappa * alpha * L * I_ax(s_j, t)

(lead-field jump x axial current at the crossing). It is the secondary source
of a piecewise-homogeneous conductor and adds to the distributed primary source.

Human basis: Martin, Pratt & Schwegler (1995), Electroenceph clin Neurophysiol
96:357-370. With the exposed eighth nerve suspended in air instead of CSF, the
latencies of I_n and II_p shortened while I_p, III and V did not.

alpha comes from the 4-sphere conductivities or the canal geometry, I_ax from
the fibre simulation, the orientation from the head-frame path. Latency and
polarity do not depend on kappa or L; amplitude does.
"""

import numpy as np

from recon_core.config.head_geometry import FOUR_SPHERE_SIGMAS

#: S/m, from the 4-sphere table.
SIGMA_BRAIN, SIGMA_CSF, SIGMA_BONE, _SIGMA_SCALP = FOUR_SPHERE_SIGMAS

#: Media either side of each junction, travelling peripheral -> central.
JUNCTION_MEDIA = {
    'porus': (SIGMA_BONE, SIGMA_CSF),
    'brainstem': (SIGMA_CSF, SIGMA_BRAIN),
}


def contrast(sigma_in, sigma_out):
    """Planar-interface conductivity contrast in (-1, 1), travelling
    peripheral -> central: alpha = (sigma_in - sigma_out) / (sigma_in + sigma_out)
    (two infinite half-spaces, perpendicular crossing; Plonsey & Barr).
    """
    return (sigma_in - sigma_out) / (sigma_in + sigma_out)


# ===========================================================================
# Wave attribution (Martin, Pratt & Schwegler 1995, intraoperative, n = 6)
# ===========================================================================
# One volley crossing two boundaries:
#   I_n   negative trough between waves I and II  <- porus acusticus
#   II_p  wave II, positive peak                  <- brainstem entry (CSF -> brain)
# Latencies with the nerve in CSF: I_p 1.84, I_n 2.60, II_p 3.16 ms.
#
# The two junctions are different kinds of source:
#   porus      geometric constriction at constant conductivity: the internal
#              auditory canal is CSF-filled, and inside it the bony wall forces
#              the return current axially through a narrow annulus that opens
#              at the porus. Driving term: the area fill factor of the canal.
#   brainstem  material conductivity step, CSF -> neural tissue: contrast().
# ===========================================================================

#: Internal auditory canal (IAC) lumen and eighth-nerve bundle diameters, mm
#: (IAC mid-canal diameter ~5.7-5.9 mm, Erkoc et al. 2012).
IAC_LUMEN_DIAM_MM = 4.5      # canal lumen
IAC_NERVE_DIAM_MM = 2.7      # combined nerve bundle
IAC_LENGTH_MM = 8.5          # modiolus -> porus


def _area(d_mm):
    return np.pi * (0.5 * d_mm) ** 2


#: Fraction of the canal lumen occupied by the nerve bundle (5.73/15.90 = 0.360).
AREA_FILL_FACTOR = _area(IAC_NERVE_DIAM_MM) / _area(IAC_LUMEN_DIAM_MM)

#: Modiolar boundary source (--wave1-source modiolus): wave I as an entry-
#: aperture source, p = kappa * alpha * L * I_ax. Values at the low end of
#: alpha 0.40-0.50, kappa 1.0-1.2, L 1.5-2.5 mm.
MODIOLUS_FILL_FACTOR = 0.40      # alpha, area fill at the canal mouth
MODIOLUS_KAPPA = 1.0             # aperture shape prefactor
MODIOLUS_L_UM = 1_500.0          # modiolar convergence zone

#: Junctions whose dipole follows the canal axis rather than the local fibre
#: tangent: an aperture source is formed by extracellular current flowing along
#: the bony channel, whereas a material step is driven by the intracellular
#: axial current, which follows the fibre.
ORIENT_ALONG_CANAL = ('modiolus',)

ALPHA = {
    # Positive: entering the canal the extracellular space narrows (r_e rises),
    # so the entry aperture is a far-field source: vertex-positive wave I.
    'modiolus': +MODIOLUS_FILL_FACTOR,
    # Negative: at the porus r_e drops (d r_e/dz < 0), so with V_e = -I_ax r_e
    # the exit acts as an extracellular sink: vertex-negative I_n.
    'porus': -AREA_FILL_FACTOR,
    # Material contrast CSF -> brain from the 4-sphere conductivities.
    'brainstem': contrast(*JUNCTION_MEDIA['brainstem']),
}

#: Aperture-shape prefactor per junction: 1.0-1.5 for a cylindrical channel
#: flaring into a semi-infinite volume; 1.0 where there is no aperture.
KAPPA = {'modiolus': MODIOLUS_KAPPA, 'porus': 1.2, 'brainstem': 1.0}

#: Axial length over which the lead field changes at each junction, µm: the L
#: of p = kappa * alpha * L * I_ax. The exact form is the integral
#: p(t) = integral I_ax(z,t) dW(z)/dz dz, which reduces to the product when
#: I_ax varies little across the transition (see transition_kernel).
LEAD_FIELD_TRANSITION_UM = {
    'modiolus': MODIOLUS_L_UM,
    # Aperture spreading length at the porus mouth, (pi/4) r_IAC = 1.77 mm. The
    # canal length is not included: the canal is the separation between the
    # modiolus and porus sources, already expressed by their positions.
    'porus': 1e3 * (0.25 * np.pi * 0.5 * IAC_LUMEN_DIAM_MM),
    # Root entry zone: between the myelin transition width and the AP's spatial
    # half-wavelength (20 m/s x 0.5 ms / 2 = 5 mm); 3.5-4.5 mm.
    'brainstem': 4_000.0,
}


def ap_extent_um(arcs_um, i_ax, dt_ms):
    """Spatial extent of the AP along the fibre, µm: propagation speed between
    two junctions times the FWHM of the axial-current pulse. Needs >= 2 junctions.
    """
    arcs_um = np.asarray(arcs_um, dtype=float)
    i_ax = np.asarray(i_ax, dtype=float)
    if len(arcs_um) < 2:
        raise ValueError('need >=2 junctions to measure the AP speed; '
                         'refusing to assume a coupling length')
    pk = np.array([int(np.argmax(np.abs(row))) for row in i_ax])
    order = np.argsort(arcs_um)
    a, b = order[0], order[-1]
    dt = (pk[b] - pk[a]) * dt_ms
    if dt <= 0:
        raise ValueError('axial-current peaks are not ordered along the fibre; '
                         'cannot measure a propagation speed')
    speed_um_per_ms = (arcs_um[b] - arcs_um[a]) / dt
    # FWHM of the first junction's pulse, in ms.
    row = np.abs(i_ax[a])
    half = 0.5 * row[pk[a]]
    above = np.flatnonzero(row > half)
    fwhm_ms = (above[-1] - above[0] + 1) * dt_ms
    return float(speed_um_per_ms * fwhm_ms)


def propagation_speed_um_per_ms(arcs_um, i_ax, dt_ms):
    """Volley speed along the fibre, µm/ms, between the two most central junctions."""
    arcs_um = np.asarray(arcs_um, dtype=float)
    i_ax = np.asarray(i_ax, dtype=float)
    if len(arcs_um) < 2:
        raise ValueError('need >=2 junctions to measure the volley speed')
    pk = np.array([int(np.argmax(np.abs(row))) for row in i_ax])
    order = np.argsort(arcs_um)
    # The two most central junctions: the central-axon speed, excluding the soma delay.
    a, b = (order[-2], order[-1]) if len(order) >= 3 else (order[0], order[-1])
    dt = (pk[b] - pk[a]) * dt_ms
    if dt <= 0:
        raise ValueError('axial-current peaks are not ordered along the fibre')
    return float((arcs_um[b] - arcs_um[a]) / dt)


def transition_kernel(dt_ms, speed_um_per_ms, l_trans_um):
    """Unit-area temporal kernel: the lead-field transition profile seen at the
    volley's speed.

    The equivalent dipole is p(t) = integral I_ax(t - z/v) dW(z)/dz dz; with
    tau = z/v it becomes a temporal convolution,

        p(t) = kappa * alpha * L * (I_ax * phi)(t),   phi(tau) = v phi_z(v tau)

    with phi_z a unit-area profile of width L. It preserves area, lowers the
    peak and broadens the pulse; with no smoothing it reduces to the product
    form. The profile is a centred Gaussian with FWHM = l_trans_um.
    """
    sigma_um = l_trans_um / 2.354820045   # FWHM -> sigma
    sigma_ms = sigma_um / float(speed_um_per_ms)
    half = int(np.ceil(4.0 * sigma_ms / dt_ms))
    if half < 1:
        return np.array([1.0])            # kernel narrower than dt: no smoothing
    tau = np.arange(-half, half + 1) * dt_ms
    k = np.exp(-0.5 * (tau / sigma_ms) ** 2)
    return k / k.sum()


def stationary_dipole(i_ax, alpha, tangent, k_um, kappa, kernel=None):
    """(T,) axial current nA -> (3, T) current dipole moment nA·µm along `tangent`.

    `kappa` is per junction (KAPPA[name]). `kernel` from transition_kernel()
    gives the convolution form; without it, the product form.
    """
    t = np.asarray(tangent, dtype=float)
    t = t / np.linalg.norm(t)
    i = np.asarray(i_ax, dtype=float)
    if kernel is not None and len(kernel) > 1:
        i = np.convolve(i, np.asarray(kernel, dtype=float), mode='same')
    return kappa * alpha * k_um * np.outer(t, i)


def canal_axis(side):
    """Unit head-frame vector along the cochlear nerve's canal, modiolus->porus."""
    from recon_core.config import head_geometry as hg
    path = np.asarray(hg.ANF_PATH_UM[side], dtype=float)
    arcs = np.concatenate([[0.0], np.cumsum(
        np.linalg.norm(np.diff(path, axis=0), axis=1))])
    i = int(np.argmin(np.abs(arcs - hg.junction_arcs_um(side)['porus'])))
    v = path[i] - path[0]
    return v / np.linalg.norm(v)


def aperture_source(p_bins, r_bins, side):
    """Distal-zone bins -> one aperture dipole at the modiolus: (1,3,T), (1,3).

    The wave I generator sits inside the otic capsule, whose bone
    (~0.006-0.02 S/m) encloses the scalae (~1.5 S/m). Return currents are
    channelled along Rosenthal's canal and the internal auditory canal and leave
    through the bony apertures, so the ensemble is represented as one dipole at
    the modiolus along the canal axis. The 4-sphere model has no temporal bone,
    so this lead-field effect is expressed at the source.

    The summed moment is projected on the canal axis (sign preserving):
        p_ap(t) = axis * (axis . sum_k p_k(t))
    """
    p_sum = np.asarray(p_bins, dtype=float).sum(axis=0)      # (3,T)
    axis = canal_axis(side)
    from recon_core.config import head_geometry as hg
    r_ap = np.asarray(hg.MODIOLUS_POS_UM[side], dtype=float)
    return np.outer(axis, axis @ p_sum)[None, :, :], r_ap[None, :]


# ---------------------------------------------------------------------------
# Zone weighting of the distributed primary source
# ---------------------------------------------------------------------------
# A source inside a spherical inclusion of conductivity sigma_i in a background
# sigma_e couples to a distant electrode with weight w = 3 sigma_e /
# (2 sigma_e + sigma_i). The cisternal nerve lies in CSF (1.79 S/m in 0.33 S/m):
# w = 0.40. The other zones keep weight 1.
ZONE_WEIGHTS = {
    'distal': 1.0,
    'intracanalicular': 1.0,
    'cisternal': 3.0 * SIGMA_BRAIN / (2.0 * SIGMA_BRAIN + SIGMA_CSF),
    'proximal': 1.0,
}

#: Zone boundaries as arc positions, µm. 'distal' is the wave I generator
#: (terminal, soma and the first part of the canal); 'proximal' surrounds the
#: brainstem junction.
DISTAL_END_UM = 6_000.0
PROXIMAL_START_UM = 20_000.0


def zone_of(arc_um, porus_um):
    """Which conductivity zone an arc position falls in."""
    if arc_um < DISTAL_END_UM:
        return 'distal'
    if arc_um < porus_um:
        return 'intracanalicular'
    if arc_um < PROXIMAL_START_UM:
        return 'cisternal'
    return 'proximal'


def bin_weights(bin_arcs_um, porus_um, weights=None):
    """(K,) weight per bin, from its arc position's zone."""
    w = ZONE_WEIGHTS if weights is None else weights
    return np.array([w[zone_of(a, porus_um)] for a in np.asarray(bin_arcs_um)])
