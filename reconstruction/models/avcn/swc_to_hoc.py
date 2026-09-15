#!/usr/bin/env python3
"""
Offline converter: NeuroMorpho SWC to syGlass-style compartmental .hoc.

A one-time build tool, not part of the simulation path. It takes a NeuroMorpho
reconstruction (SWC: type 1 soma, 2 axon, 3 dendrite) and emits a NEURON .hoc
in the convention used by the figshare EM globular-bushy-cell files in
models/avcn/morphology/dryad/ (create sections[N] plus SectionList objects
named soma, Axon_Hillock, Axon_Initial_Segment, Myelinated_Axon,
Proximal_Dendrite, Distal_Dendrite).

Those SectionList names are exactly the keys in gbc_biophysics.COMPARTMENT_OF,
so the converted cell is decorated by decorate_gbc and oriented by
native_axon_direction with no code changes, exactly like the real EM cells.

Geometry source
---------------
The 3-D geometry comes from NEURON's own Import3d reader, the same code LFPy
uses to load SWC, rather than being re-derived from the raw SWC points. That
matters: Import3d applies SWC-specific handling (3-point spherical soma,
diameters at branch points) that reproduces NeuroMorpho's reported membrane
area, whereas a naive point-by-point cylinder reconstruction inflates it by
about 1.5x. Each imported section is then only re-partitioned into runs of a
single compartment class, splitting at existing 3-D points, which preserves
total area exactly (a split at point k duplicates the point, never the frustum).

Compartment classification (per 3-D point, by SWC type and path distance):

  type 1 soma  -> soma
  type 2 axon  path-dist < HILLOCK_UM (3)    -> Axon_Hillock
               HILLOCK_UM..AIS_UM (3..30)    -> Axon_Initial_Segment
               > AIS_UM (30)                 -> Myelinated_Axon
  type 3 dend  path-dist <= PROX_UM (40)     -> Proximal_Dendrite
               > PROX_UM                     -> Distal_Dendrite

Thresholds default to values measured on the EM GBC VCN_c09 (hillock 1.8 µm,
AIS 24 µm, proximal dendrite and hub ~40 µm) and can be overridden on the CLI.

The .hoc carries geometry only (pt3dadd), with no insert, gbar, Ra, cm or nseg,
just like the EM files. All biophysics stays in gbc_biophysics.decorate_gbc.

Usage
-----
  python models/avcn/swc_to_hoc.py \
      --swc S113.CNG.swc \
      --out models/avcn/morphology/neuromorpho/SBC_S113.hoc
"""

import argparse
import os
from collections import Counter, defaultdict

import numpy as np


# syGlass SectionList names, matching models/avcn/morphology/dryad/*.hoc and the
# keys of gbc_biophysics.COMPARTMENT_OF.
SL_SOMA = 'soma'
SL_HILL = 'Axon_Hillock'
SL_AIS  = 'Axon_Initial_Segment'
SL_MYEL = 'Myelinated_Axon'
SL_PROX = 'Proximal_Dendrite'
SL_DIST = 'Distal_Dendrite'
SECTIONLIST_NAMES = [SL_SOMA, SL_MYEL, SL_HILL, SL_PROX, SL_DIST, SL_AIS]

# --- Fixed myelinated-axon truncation (SBC) ---------------------------------
# The spherical bushy cell's long reconstructed axon is decorated near-passive
# and is not used for an axonal travelling-wave dipole as the GBC's is, yet it
# is about 60% of the compartments. Only a short proximal stub is kept: the AIS
# plus this much myelinated axon (µm of path distance beyond the AIS). The stub
# preserves a robust ventromedial orientation reference and the near-soma
# axonal field; the excitability change from the removed axonal capacitance is
# absorbed by re-tuning the endbulb weight. Set to None for the full axon.
KEEP_MYELIN_UM = 50.0


def _section_kind(sec):
    """soma / axon / dend from a NEURON section name (Import3d array name)."""
    nm = sec.name().split('.')[-1].split('[')[0]
    if nm.startswith('soma'):
        return 'soma'
    if nm.startswith('axon'):
        return 'axon'
    return 'dend'   # dend / apic / anything else


def _classify_point(kind, pdist, hillock_um, ais_um, prox_um):
    if kind == 'soma':
        return SL_SOMA
    if kind == 'axon':
        return (SL_HILL if pdist < hillock_um
                else SL_AIS if pdist < ais_um
                else SL_MYEL)
    return SL_PROX if pdist <= prox_um else SL_DIST


def import3d_cell(swc_path):
    """Load the SWC via NEURON Import3d, the path LFPy uses. Returns h."""
    from neuron import h
    h.load_file('stdlib.hoc')
    h.load_file('import3d.hoc')
    rdr = h.Import3d_SWC_read()
    rdr.input(swc_path)
    gui = h.Import3d_GUI(rdr, 0)
    gui.instantiate(None)
    return h


def build(swc_path, hillock_um, ais_um, prox_um, myelin_cutoff_um=None):
    """Return (out_sections, summary).

    out_sections: list of dicts {cls, pts:[(x,y,z,diam)], parent, parent_loc}
    ordered so every parent precedes its children; section 0 is the soma.

    myelin_cutoff_um : float or None
        If set, drop every myelinated-axon point whose path distance from the
        soma exceeds this cutoff, along with the whole distal subtree, keeping
        a fixed proximal stub. None keeps the full axon.
    """
    from neuron import h
    import3d_cell(swc_path)

    secs = list(h.allsec())
    somas = [s for s in secs if _section_kind(s) == 'soma']
    if not somas:
        raise RuntimeError('no soma section found in SWC')
    soma0 = somas[0]

    # --- per-section 3-D points, and topological order from the soma ----------
    def pts_of(sec):
        n = int(h.n3d(sec=sec))
        return [(h.x3d(i, sec=sec), h.y3d(i, sec=sec),
                 h.z3d(i, sec=sec), h.diam3d(i, sec=sec)) for i in range(n)]

    children = defaultdict(list)
    parent_of = {}
    for s in secs:
        ps = s.parentseg()
        if ps is not None:
            p = ps.sec
            children[p.name()].append(s)
            parent_of[s.name()] = (p, float(ps.x))

    # cumulative path distance from soma to the START of each section
    base_dist = {soma0.name(): 0.0}
    order = [soma0]
    stack = [soma0]
    seen = {soma0.name()}
    while stack:
        cur = stack.pop()
        cpts = pts_of(cur)
        # path length along this section (to its end)
        seglen = sum(float(np.linalg.norm(np.subtract(cpts[i + 1][:3], cpts[i][:3])))
                     for i in range(len(cpts) - 1))
        for ch in children[cur.name()]:
            if ch.name() in seen:
                continue
            seen.add(ch.name())
            _p, x = parent_of[ch.name()]
            base_dist[ch.name()] = base_dist[cur.name()] + seglen * x
            order.append(ch)
            stack.append(ch)

    # --- re-partition each section into single-class runs ---------------------
    out = []
    # map (source section name, point index) to the output section index of the
    # run containing that point, so children can find their parent run.
    run_of_point = {}
    pruned = set()   # source sections dropped by the myelin cutoff (subtree too)

    for sec in order:
        # skip any section whose parent was pruned (keeps the tree connected)
        if sec.name() in parent_of and parent_of[sec.name()][0].name() in pruned:
            pruned.add(sec.name())
            continue

        kind = _section_kind(sec)
        pts = pts_of(sec)
        # cumulative distance per point from soma
        cum = [base_dist[sec.name()]]
        for i in range(1, len(pts)):
            cum.append(cum[-1] +
                       float(np.linalg.norm(np.subtract(pts[i][:3], pts[i - 1][:3]))))
        classes = [_classify_point(kind, cum[i], hillock_um, ais_um, prox_um)
                   for i in range(len(pts))]

        # myelin-cutoff truncation: keep points up to the first myelinated point
        # beyond the cutoff; drop the remainder of this section and its subtree.
        if myelin_cutoff_um is not None:
            tcut = next((i for i in range(len(pts))
                         if classes[i] == SL_MYEL and cum[i] > myelin_cutoff_um), None)
            if tcut is not None:
                pruned.add(sec.name())          # subtree beyond the stub is dropped
                if tcut == 0:                    # whole section is past the cutoff
                    continue
                pts, cum, classes = pts[:tcut], cum[:tcut], classes[:tcut]

        # parent output-section for the FIRST run of this source section
        if sec.name() in parent_of:
            psec, ploc = parent_of[sec.name()]
            # nearest 3-D point index on the parent to the connection location
            pn = int(h.n3d(sec=psec))
            pidx = min(pn - 1, max(0, int(round(ploc * (pn - 1)))))
            first_parent = run_of_point[(psec.name(), pidx)]
            first_parent_loc = 1.0
        else:
            first_parent = -1
            first_parent_loc = 1.0

        # split into runs of equal class
        start = 0
        prev_out = first_parent
        prev_loc = first_parent_loc
        while start < len(pts):
            cls = classes[start]
            end = start
            while end + 1 < len(pts) and classes[end + 1] == cls:
                end += 1
            run_pts = pts[start:end + 1]
            # geometric continuity: a non-first run repeats the previous run's
            # last point as its first, which duplicates a point but adds no area.
            prepend = None
            if start > 0:
                prepend = pts[start - 1]
            out.append({'cls': cls, 'pts': run_pts, 'prepend': prepend,
                        'parent': prev_out, 'parent_loc': prev_loc})
            out_idx = len(out) - 1
            for i in range(start, end + 1):
                run_of_point[(sec.name(), i)] = out_idx
            prev_out = out_idx
            prev_loc = 1.0
            start = end + 1

    summary = _summarise(out)
    return out, summary


def _summarise(out):
    cnt = Counter()
    length = defaultdict(float)
    for s in out:
        cnt[s['cls']] += 1
        p = s['pts']
        for a, b in zip(p[:-1], p[1:]):
            length[s['cls']] += float(np.linalg.norm(np.subtract(a[:3], b[:3])))
    return cnt, length


def write_hoc(out, out_path, src_name):
    n = len(out)
    L = []
    L.append('//')
    L.append('//  Converted from NeuroMorpho SWC by models/avcn/swc_to_hoc.py')
    L.append(f'//    Source = {src_name}')
    L.append('//    Geometry from NEURON Import3d; section names match dryad/*.hoc')
    L.append('//    Geometry only (pt3dadd); biophysics via gbc_biophysics.decorate_gbc')
    L.append('//')
    L.append('')
    for sl in SECTIONLIST_NAMES:
        L.append(f'objref {sl}')
        L.append(f'{sl} = new SectionList()')
    L.append(f'create sections[{n}]')
    L.append('')
    for i, s in enumerate(out):
        L.append(f'access sections[{i}]')
        L.append(f'{s["cls"]}.append()')
        if s['parent'] >= 0:
            loc = s['parent_loc']
            loc_s = '0' if loc == 0.0 else '1' if loc == 1.0 else f'{loc}'
            L.append(f'connect sections[{i}](0), sections[{s["parent"]}]({loc_s})')
        L.append(f'sections[{i}] {{')
        if s['prepend'] is not None:
            x, y, z, d = s['prepend']
            L.append(f'    pt3dadd({x:.6f}, {y:.6f}, {z:.6f}, {d:.6f}) // link')
        for (x, y, z, d) in s['pts']:
            L.append(f'    pt3dadd({x:.6f}, {y:.6f}, {z:.6f}, {d:.6f})')
        L.append('}')
        L.append('')
    L.append('access sections[0]')
    L.append('')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('\n'.join(L))
    return n


def main():
    ap = argparse.ArgumentParser(description='NeuroMorpho SWC -> syGlass-style hoc')
    ap.add_argument('--swc', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--hillock-um', type=float, default=3.0)
    ap.add_argument('--ais-um',     type=float, default=30.0)
    ap.add_argument('--prox-um',    type=float, default=40.0)
    args = ap.parse_args()

    cutoff = (args.ais_um + KEEP_MYELIN_UM) if KEEP_MYELIN_UM is not None else None
    out, (cnt, length) = build(args.swc, args.hillock_um, args.ais_um, args.prox_um,
                               myelin_cutoff_um=cutoff)
    n = write_hoc(out, args.out, os.path.basename(args.swc))
    if cutoff is not None:
        print(f'  (myelinated axon truncated at {cutoff:.0f} µm path distance = '
              f'AIS {args.ais_um:.0f} + stub {KEEP_MYELIN_UM:.0f} µm)')
    print(f'Wrote {n} sections -> {args.out}')
    print('  class                       sections   length(µm)')
    for sl in SECTIONLIST_NAMES:
        if cnt[sl]:
            print(f'  {sl:24s}  {cnt[sl]:8d}   {length[sl]:9.1f}')


if __name__ == '__main__':
    main()
