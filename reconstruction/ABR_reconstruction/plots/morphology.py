#!/usr/bin/env python3
"""
Schematic figure per nucleus: population placement inside the model's insertion
volume (elliptic disk or elliptic cylinder), plus a zoomed single-cell
morphology showing the actual compartments, parsed from the .hoc files rather
than simulated.

Nuclei covered: MSO, AVCN-GBC, AVCN-SBC, MNTB (principal cell), LSO.

Run:
  python ABR_reconstruction/plot_nuclei_morphology.py
Output:
  ABR_reconstruction/figures/{mso,gbc,sbc,mntb,lso}_population_morphology.png
"""

import os
import sys
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from recon_core import params as P, paths                       # noqa: E402

REPO_ROOT = paths.REPO_ROOT
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'figures')
os.makedirs(FIG_DIR, exist_ok=True)

np.random.seed(0)

# ---------------------------------------------------------------------------
# Generic .hoc morphology parser (pt3dadd points, per section, with category)
# ---------------------------------------------------------------------------
def parse_hoc(path):
    text = open(path).read()
    sections, category = {}, {}

    pat_b = re.compile(
        r'access\s+sections\[(\d+)\]\s*\n'
        r'((?:\s*\w+\.append\(\)\s*\n)+)'
        r'(?:\s*connect[^\n]*\n)*'
        r'\s*sections\[\1\]\s*\{([^}]*)\}',
    )
    matches = list(pat_b.finditer(text))
    if matches:
        for m in matches:
            idx, appends, body = m.groups()
            cat = re.findall(r'(\w+)\.append\(\)', appends)[0]
            pts = re.findall(r'pt3dadd\(([^)]*)\)', body)
            coords = [[float(v) for v in p.split(',')[:4]] for p in pts]
            name = f'sections[{idx}]'
            sections[name], category[name] = coords, cat
        return sections, category

    create_m = re.search(r'create\s+([^\n]*)', text)
    names = [n.strip().split('[')[0] for n in create_m.group(1).split(',')]
    for name in names:
        body_m = re.search(rf'(?<!\w){re.escape(name)}\s*\{{([^}}]*)\}}', text)
        if not body_m:
            continue
        pts = re.findall(r'pt3dadd\(([^)]*)\)', body_m.group(1))
        coords = [[float(v) for v in p.split(',')[:4]] for p in pts]
        sections[name], category[name] = coords, name
    return sections, category


NAMED_CATEGORY_GROUPS = {
    # collapse hand-written per-branch names into a coarse category for colouring
    'soma': 'soma',
    'dend_A': 'dendrite', 'dend_A1': 'dendrite', 'dend_A2': 'dendrite',
    'dend_B': 'dendrite', 'dend_B1': 'dendrite', 'dend_B2': 'dendrite',
    'dend_medial': 'dendrite', 'dend_lateral': 'dendrite',
    'axon': 'axon', 'ais': 'AIS',
    'precalyx_axon': 'precalyx axon', 'calyx': 'calyx terminal',
}

CATEGORY_COLORS = {
    'soma': 'black',
    'dendrite': 'tab:green',
    'Proximal_Dendrite': 'tab:green',
    'Distal_Dendrite': 'yellowgreen',
    'Dendritic_Hub': 'darkgreen',
    'Dendritic_Swelling': 'olive',
    'axon': 'tab:red',
    'AIS': 'tab:red',
    'Axon_Initial_Segment': 'tab:red',
    'Axon_Hillock': 'firebrick',
    'Myelinated_Axon': 'tab:orange',
    'precalyx axon': 'tab:red',
    'calyx terminal': 'tab:purple',
}


def _category_label(cat):
    return NAMED_CATEGORY_GROUPS.get(cat, cat)


def _plot_morphology(ax, sections, category, plane=('x', 'z'), lw_scale=0.6, lw_max=6.0):
    """2-D projection of every section, coloured by compartment category."""
    ax_map = {'x': 0, 'y': 1, 'z': 2}
    i, j = ax_map[plane[0]], ax_map[plane[1]]
    seen = {}
    all_u, all_v = [], []
    for name, pts in sections.items():
        cat = _category_label(category[name])
        color = CATEGORY_COLORS.get(cat, 'gray')
        pts = np.array(pts)
        u, v, d = pts[:, i], pts[:, j], pts[:, 3]
        lw = min(max(d.mean() * lw_scale, 0.6), lw_max)
        ax.plot(u, v, color=color, linewidth=lw, solid_capstyle='round', alpha=0.95)
        seen.setdefault(cat, color)
        all_u.append(u); all_v.append(v)
    ax.set_aspect('equal')

    # Degenerate axis guard: a purely 1-D stick (the MSO, where everything
    # varies along z) leaves the other axis' range near 0, which with
    # aspect='equal' collapses the subplot. Give it a fixed span instead.
    u_range = np.ptp(np.concatenate(all_u)) if all_u else 1.0
    v_range = np.ptp(np.concatenate(all_v)) if all_v else 1.0
    if v_range < 0.05 * max(u_range, 1.0):
        v_mid = np.mean(np.concatenate(all_v)) if all_v else 0.0
        half = max(u_range * 0.15, 20.0)
        ax.set_ylim(v_mid - half, v_mid + half)

    # Fixed-width proxy handles so a fat soma doesn't blow up the legend glyph.
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=col, lw=2.5) for col in seen.values()]
    ax.legend(handles, list(seen.keys()), fontsize=7, loc='best', ncol=1)
    return seen


# ---------------------------------------------------------------------------
# Population insertion-volume helpers
# ---------------------------------------------------------------------------
def sample_flat_disk(n, radius_x, radius_y, z_half=0.0, seed=0):
    """Flat elliptic disk (z = 0, or a thin z slab): AVCN, MNTB, MSO."""
    rng = np.random.default_rng(seed)
    x, y = np.empty(0), np.empty(0)
    while len(x) < n:
        m = (n - len(x)) * 2
        xi = (rng.random(m) - 0.5) * 2 * radius_x
        yi = (rng.random(m) - 0.5) * 2 * radius_y
        keep = (xi / radius_x) ** 2 + (yi / radius_y) ** 2 <= 1
        x = np.concatenate([x, xi[keep]])
        y = np.concatenate([y, yi[keep]])
    x, y = x[:n], y[:n]
    z = (rng.random(n) - 0.5) * 2 * z_half if z_half > 0 else np.zeros(n)
    return x, y, z


def sample_elliptic_cylinder(n, radius_x, radius_z, half_height_y, seed=0):
    """True elliptic cylinder, axis along y, the LSO convention."""
    rng = np.random.default_rng(seed)
    x, z = np.empty(0), np.empty(0)
    while len(x) < n:
        m = (n - len(x)) * 2
        xi = (rng.random(m) - 0.5) * 2 * radius_x
        zi = (rng.random(m) - 0.5) * 2 * radius_z
        keep = (xi / radius_x) ** 2 + (zi / radius_z) ** 2 <= 1
        x = np.concatenate([x, xi[keep]])
        z = np.concatenate([z, zi[keep]])
    x, z = x[:n], z[:n]
    y = (rng.random(n) - 0.5) * 2 * half_height_y
    return x, y, z


def _draw_disk_outline(ax, radius_x, radius_y, z=0.0, color='steelblue'):
    theta = np.linspace(0, 2 * np.pi, 100)
    xe = radius_x * np.cos(theta)
    ye = radius_y * np.sin(theta)
    ze = np.full_like(xe, z)
    ax.plot(xe, ye, ze, color=color, lw=1.5, alpha=0.7)
    verts = [list(zip(xe, ye, ze))]
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    ax.add_collection3d(Poly3DCollection(verts, facecolor=color, alpha=0.08))


def _draw_flat_cylinder_outline(ax, radius_x, radius_y, height, color='steelblue'):
    """Squat elliptic cylinder, axis along z (cross-section in x-y), the
    MSO/AVCN/MNTB convention drawn with a visible height instead of a flat disk.
    Cosmetic only; it has no bearing on the simulation geometry."""
    theta = np.linspace(0, 2 * np.pi, 100)
    xe = radius_x * np.cos(theta)
    ye = radius_y * np.sin(theta)
    half = height / 2.0
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    for z in (-half, half):
        ze = np.full_like(xe, z)
        ax.plot(xe, ye, ze, color=color, lw=1.2, alpha=0.6)
        verts = [list(zip(xe, ye, ze))]
        ax.add_collection3d(Poly3DCollection(verts, facecolor=color, alpha=0.06))
    # a few vertical struts connecting top and bottom rims
    for t in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        x0, y0 = radius_x * np.cos(t), radius_y * np.sin(t)
        ax.plot([x0, x0], [y0, y0], [-half, half], color=color, lw=0.6, alpha=0.35)


def _draw_cylinder_outline(ax, radius_x, radius_z, half_height_y, color='steelblue'):
    theta = np.linspace(0, 2 * np.pi, 60)
    xe = radius_x * np.cos(theta)
    ze = radius_z * np.sin(theta)
    for y in (-half_height_y, half_height_y):
        ax.plot(xe, np.full_like(xe, y), ze, color=color, lw=1.2, alpha=0.6)
    # a few longitudinal lines
    for t in np.linspace(0, 2 * np.pi, 10, endpoint=False):
        x0, z0 = radius_x * np.cos(t), radius_z * np.sin(t)
        ax.plot([x0, x0], [-half_height_y, half_height_y], [z0, z0],
                color=color, lw=0.6, alpha=0.35)


def _draw_probe(ax, fixed_a, fixed_b, values, axis='z', color='black'):
    """Linear multi-channel electrode shank, as configured in the matching
    LFP_reconstruction/main_reconstruct*.py (PROBE_X/PROBE_Y/PROBE_Z).
    axis selects which coordinate the channels spread along; the other two are
    held fixed at (fixed_a, fixed_b) in the remaining axis order."""
    fa = np.full_like(values, fixed_a)
    fb = np.full_like(values, fixed_b)
    if axis == 'z':
        px, py, pz = fa, fb, values
    elif axis == 'y':
        px, py, pz = fa, values, fb
    else:  # 'x'
        px, py, pz = values, fa, fb
    ax.plot(px, py, pz, color=color, lw=2.0, alpha=0.9, zorder=6)
    ax.scatter(px, py, pz, color=color, s=18, marker='s', alpha=0.95,
              zorder=7, label=f'electrode probe ({len(values)} ch)')


# ---------------------------------------------------------------------------
# Per-nucleus figure builder
# ---------------------------------------------------------------------------
def build_figure(title, out_name, geometry, hoc_path, morph_plane, morph_title,
                  n_display=60, pop_kind='disk', pop_view=(20, -60), probe=None):
    fig = plt.figure(figsize=(14, 6.5))
    ax_pop = fig.add_subplot(1, 2, 1, projection='3d')
    ax_morph = fig.add_subplot(1, 2, 2)

    # --- Panel A: population inserted in the geometry --------------------
    if pop_kind == 'disk':
        rx, ry = geometry['radius_x'], geometry['radius_y']
        z_half = geometry.get('z_half', 0.0)
        vis_height = geometry.get('vis_height', 0.0)
        x, y, z = sample_flat_disk(n_display, rx, ry, z_half=z_half)
        if vis_height > 0:
            _draw_flat_cylinder_outline(ax_pop, rx, ry, vis_height)
            zlim = max(vis_height * 1.5, 20.0)
        else:
            _draw_disk_outline(ax_pop, rx, ry)
            zlim = max(rx, ry) * 0.3
        if probe is not None and probe.get('axis', 'z') == 'z':
            zlim = max(zlim, probe['half'] * 1.15)
        ax_pop.set_zlim(-zlim, zlim)
        extent = max(rx, ry) * 1.15
        ax_pop.set_xlim(-extent, extent); ax_pop.set_ylim(-extent, extent)
    else:
        rx, rz, hy = geometry['radius_x'], geometry['radius_z'], geometry['half_height_y']
        x, y, z = sample_elliptic_cylinder(n_display, rx, rz, hy)
        _draw_cylinder_outline(ax_pop, rx, rz, hy)
        extent = max(rx, rz, hy) * 1.1
        ylim = hy * 1.15
        if probe is not None and probe.get('axis', 'z') == 'y':
            ylim = max(ylim, probe['half'] * 1.15)
        ax_pop.set_xlim(-extent, extent); ax_pop.set_ylim(-ylim, ylim)
        ax_pop.set_zlim(-extent, extent)

    ax_pop.scatter(x, y, z, s=14, color='tab:blue', depthshade=True, alpha=0.85,
                  label='population')
    if probe is not None:
        axis = probe.get('axis', 'z')
        vals = np.linspace(-probe['half'], probe['half'], probe['n_ch'])
        coord = {'x': probe.get('x', 0.0), 'y': probe.get('y', 0.0), 'z': probe.get('z', 0.0)}
        fixed_a, fixed_b = (coord[c] for c in ('x', 'y', 'z') if c != axis)
        _draw_probe(ax_pop, fixed_a, fixed_b, vals, axis=axis)
    ax_pop.set_xlabel('x [µm]'); ax_pop.set_ylabel('y [µm]'); ax_pop.set_zlabel('z [µm]')
    ax_pop.set_title(f'Population insertion volume\n(N={n_display} of '
                      f'{geometry["n_total"]:,} shown)', fontsize=10)
    ax_pop.view_init(elev=pop_view[0], azim=pop_view[1])
    ax_pop.legend(fontsize=7, loc='upper left')

    # --- Panel B: zoomed single-cell morphology ---------------------------
    sections, category = parse_hoc(hoc_path)
    _plot_morphology(ax_morph, sections, category, plane=morph_plane)
    ax_morph.set_title(morph_title, fontsize=10)
    ax_morph.set_xlabel(f'{morph_plane[0]} [µm]')
    ax_morph.set_ylabel(f'{morph_plane[1]} [µm]')
    ax_morph.axhline(0, color='gray', lw=0.4, ls=':')
    ax_morph.axvline(0, color='gray', lw=0.4, ls=':')

    n_sections = len(sections)
    fig.suptitle(f'{title}   —   {n_sections} compartments (sections) in this morphology',
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = os.path.join(FIG_DIR, out_name)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {out_path}  ({n_sections} sections)')


def main():
    # MSO bipolar principal cell
    build_figure(
        title='MSO principal neuron',
        out_name='mso_population_morphology.png',
        geometry={'radius_x': P.MSO_RADIUS_X, 'radius_y': P.MSO_RADIUS_Y,
                  'z_half': 100.0, 'vis_height': 200.0, 'n_total': P.N_MSO_TOTAL},
        hoc_path=os.path.join(paths.MSO_MODELS_DIR, 'mso_model.hoc'),
        morph_plane=('z', 'y'),
        morph_title='Hand-built bipolar MSO cell (passive)\nsoma + dend_medial/lateral',
        pop_kind='disk',
        n_display=60,
        probe={'x': 0.0, 'y': 0.0, 'axis': 'z', 'half': P.MSO_PROBE_HALF_SPAN, 'n_ch': P.N_CH},
    )

    build_figure(
        title='AVCN: Globular Bushy Cell (GBC)',
        out_name='gbc_population_morphology.png',
        geometry={'radius_x': P.AVCN_RADIUS_X, 'radius_y': P.GBC_RADIUS_Y,
                  'n_total': P.N_GBC_TOTAL},
        hoc_path=os.path.join(paths.AVCN_MODELS_DIR, 'morphology', 'dryad',
                              'VCN_c09_Full_MeshInflate.hoc'),
        morph_plane=('x', 'y'),
        morph_title='VCN_c09 EM reconstruction (Dryad)\nsoma/dendrite/hillock/AIS/myelinated axon',
        pop_kind='disk',
        probe={'x': 0.0, 'y': 0.0, 'axis': 'z', 'half': P.AVCN_PROBE_HALF_SPAN, 'n_ch': P.N_CH},
    )

    # AVCN SBC
    build_figure(
        title='AVCN: Spherical Bushy Cell (SBC)',
        out_name='sbc_population_morphology.png',
        geometry={'radius_x': P.AVCN_RADIUS_X, 'radius_y': P.SBC_RADIUS_Y,
                  'n_total': P.N_SBC_TOTAL},
        hoc_path=os.path.join(paths.AVCN_MODELS_DIR, 'morphology', 'neuromorpho',
                              'SBC_S113.hoc'),
        morph_plane=('x', 'y'),
        morph_title='SBC_S113 EM reconstruction (NeuroMorpho, Atoh7+)\nsoma/dendrite/hillock/AIS/myelinated axon',
        pop_kind='disk',
        probe={'x': 0.0, 'y': 0.0, 'axis': 'z', 'half': P.AVCN_PROBE_HALF_SPAN, 'n_ch': P.N_CH},
    )

    # MNTB principal cell
    build_figure(
        title='MNTB: principal neuron',
        out_name='mntb_population_morphology.png',
        geometry={'radius_x': P.MNTB_RADIUS_X, 'radius_y': P.MNTB_RADIUS_Y,
                  'n_total': P.N_MNTB_TOTAL},
        hoc_path=os.path.join(paths.MNTB_MODELS_DIR, 'mntb_model_active.hoc'),
        morph_plane=('z', 'y'),
        morph_title='Hand-built principal cell (Kulesza 2015)\nsoma + tufted dend_A/B + AIS + axon',
        pop_kind='disk',
        n_display=60,
        probe={'x': 0.0, 'y': 0.0, 'axis': 'z', 'half': P.MNTB_PROBE_HALF_SPAN, 'n_ch': P.N_CH},
    )

    # LSO
    build_figure(
        title='LSO principal neuron',
        out_name='lso_population_morphology.png',
        geometry={'radius_x': P.LSO_RADIUS_X, 'radius_z': P.LSO_RADIUS_Z,
                  'half_height_y': P.LSO_HALF_HEIGHT_Y, 'n_total': P.N_LSO_TOTAL},
        hoc_path=os.path.join(paths.MSO_MODELS_DIR, 'lso_model_active.hoc'),
        morph_plane=('z', 'x'),
        morph_title='Hand-built LSO cell (human dims)\nsoma + dend_A/B/C + short axon',
        pop_kind='cylinder',
        n_display=80,
        pop_view=(15, -50),
        probe={'x': 0.0, 'z': 0.0, 'axis': 'y', 'half': P.LSO_HALF_HEIGHT_Y * 1.15, 'n_ch': P.N_CH},   # cosmetic: drawn along y, slightly longer than HALF_HEIGHT_Y
    )


if __name__ == '__main__':
    main()
